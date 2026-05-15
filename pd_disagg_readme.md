# vLLM PD (Prefill-Decode) 分离架构详解

## 1. 为什么需要 PD 分离?

LLM 推理有两个阶段：

- **Prefill (预填充)**：处理完整的 prompt，计算所有 token 的 KV Cache。**计算密集型**，需要大量算力。
- **Decode (解码)**：逐 token 生成，每次只处理一个 token。**显存带宽密集型**，需要快速访问 KV Cache。

将两个阶段混在同一个实例中会导致：
1. Prefill 打断 Decode → 尾部延迟 (tail ITL) 抖动大
2. 无法独立调优 TTFT 和 ITL（不同的并行策略 TP/PP 无法分别应用）
3. Chunked Prefill 虽可缓解，但很难找到最优 chunk size

PD 分离的核心思路：**用两个独立的 vLLM 实例分别做 Prefill 和 Decode**，中间通过 KV Cache 传输机制连接。

> **注意**：PD 分离不会提升吞吐量，它优化的是延迟的可控性和稳定性。

---

## 2. 整体架构

```
                          ┌─────────────────────────────────┐
                          │         Proxy / Router          │
                          │  (请求路由: Prefill → Decode)     │
                          └──────────┬──────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
     ┌──────────────────────────┐      ┌──────────────────────────┐
     │    Prefill Instance      │      │    Decode Instance       │
     │    (KV Producer)         │      │    (KV Consumer)         │
     │                          │      │                          │
     │  ┌──────────────────┐    │      │  ┌──────────────────┐    │
     │  │ Scheduler        │    │      │  │ Scheduler        │    │
     │  │ + SchedulerConn  │    │      │  │ + SchedulerConn  │    │
     │  └────────┬─────────┘    │      │  └────────┬─────────┘    │
     │           │              │      │           │              │
     │  ┌────────▼─────────┐    │      │  ┌────────▼─────────┐    │
     │  │ Worker           │    │      │  │ Worker           │    │
     │  │ + WorkerConn     │    │      │  │ + WorkerConn     │    │
     │  │   start_load_kv()│    │      │  │   start_load_kv()│    │
     │  │   save_kv_layer()│────┼──KV──┼─►│   wait_layer()   │    │
     │  │   wait_for_save()│    │Cache │  │   save_kv_layer()│    │
     │  └──────────────────┘    │      │  └──────────────────┘    │
     └──────────────────────────┘      └──────────────────────────┘
```

**每个实例内部**，KV Connector 分为两个角色：
- **Scheduler Connector**：在调度器进程中，负责元数据/计划（哪些请求需要加载/保存 KV）
- **Worker Connector**：在每个 Worker 进程中，负责实际的 KV 传输操作

---

## 3. 核心抽象

### 3.1 三层抽象体系

PD 分离的实现基于三层抽象（位于 `vllm/distributed/kv_transfer/`）：

| 层次 | 抽象 | 核心 API | 说明 |
|------|------|----------|------|
| 底层 | **KV Pipe** | `send_tensor` / `recv_tensor` | 单向 FIFO 管道，传输 tensor |
| 中层 | **LookupBuffer** | `insert` / `drop_select` | 类 SQL 的 KV 查找缓冲区 |
| 顶层 | **KV Connector** | `start_load_kv` / `save_kv_layer` 等 | 与 vLLM 引擎的集成接口 |

**为什么需要 LookupBuffer？** Prefill 和 Decode 实例处理请求的顺序可能不同。比如 Prefill 处理顺序是 A→B→C，但 Decode 可能先处理 C。FIFO Pipe 无法处理这种乱序，LookupBuffer 通过 token 匹配来实现乱序查找。

**层级可以跳过**：如果你的传输层已经支持 KV 查找（如 Redis、RDMA database），可以跳过 Pipe 和 LookupBuffer，直接实现 Connector 层。

### 3.2 KV Connector 的两个角色

```python
class KVConnectorRole(enum.Enum):
    SCHEDULER = 0   # 在 Scheduler 进程中运行
    WORKER = 1      # 在 Worker 进程中运行
```

#### Scheduler 侧方法（调度逻辑）

| 方法 | 作用 |
|------|------|
| `get_num_new_matched_tokens()` | 查询远程有多少 token 的 KV Cache 可用 |
| `update_state_after_alloc()` | 块分配后更新 Connector 状态 |
| `build_connector_meta()` | 构建本步的元数据（Scheduler → Worker） |
| `update_connector_output()` | 从 Worker 输出更新状态 |
| `request_finished()` | 请求完成时回调，Connector 可异步持有 blocks |
| `take_events()` | 获取 KV Cache 事件 |

#### Worker 侧方法（传输逻辑）

| 方法 | 作用 | 调用时机 |
|------|------|----------|
| `start_load_kv()` | 开始异步加载 KV Cache | Forward pass **之前** |
| `wait_for_layer_load(layer_name)` | 阻塞等待某层的 KV 加载完成 | Attention 层**入口** |
| `save_kv_layer(layer_name, kv_layer, ...)` | 开始异步保存某层的 KV | Attention 层**出口** |
| `wait_for_save()` | 阻塞等待所有保存完成 | Forward pass **之后** |
| `get_finished()` | 返回异步传输完成的请求 ID | Forward pass 结束后 |
| `register_kv_caches()` | 预注册 KV Cache（如用于 RDMA） | 初始化时 |

---

## 4. 端到端运行流程

### 4.1 请求的完整生命周期

```
Client                Proxy              Prefill Instance         Decode Instance
  │                    │                       │                        │
  │──POST /completions▶│                       │                        │
  │                    │                       │                        │
  │                    │──POST (max_tokens=1)──▶│                        │
  │                    │   request_id 编码      │                        │
  │                    │   P/D 的 KV 地址       │                        │
  │                    │                       │                        │
  │                    │                  ┌─────┴──────┐                 │
  │                    │                  │ Scheduler: │                 │
  │                    │                  │ get_matched│                 │
  │                    │                  │ _tokens()  │                 │
  │                    │                  │ = 0 (新请求)│                │
  │                    │                  └─────┬──────┘                 │
  │                    │                       │                        │
  │                    │                  ┌─────┴──────┐                 │
  │                    │                  │ Worker:    │                 │
  │                    │                  │ 模型前向传播 │                │
  │                    │                  │ 每层:       │                │
  │                    │                  │ save_kv_   │                 │
  │                    │                  │ layer()    │─── NCCL/NIXL ──▶│
  │                    │                  │            │   KV 传输开始   │
  │                    │                  │ wait_for_  │                 │
  │                    │                  │ save()     │                 │
  │                    │                  └─────┬──────┘                 │
  │                    │                       │                        │
  │                    │◀──200 OK───────────────┤                        │
  │                    │   (prefill 完成)       │                        │
  │                    │                       │                        │
  │                    │──POST (原始参数)───────┼───────────────────────▶│
  │                    │                       │                  ┌─────┴──────┐
  │                    │                       │                  │ Scheduler: │
  │                    │                       │                  │ get_matched│
  │                    │                       │                  │ _tokens()  │
  │                    │                       │                  │ = N (远程KV)│
  │                    │                       │                  │            │
  │                    │                       │                  │ 请求进入    │
  │                    │                       │                  │ WAITING_   │
  │                    │                       │                  │ FOR_REMOTE │
  │                    │                       │                  │ _KVS       │
  │                    │                       │                  └─────┬──────┘
  │                    │                       │                        │
  │                    │                       │                  ┌─────┴──────┐
  │                    │                       │                  │ Worker:    │
  │                    │                       │                  │ start_load │
  │                    │                       │                  │ _kv()      │
  │                    │                       │                  │ 每层:       │
  │                    │                       │                  │ wait_for_  │
  │                    │                       │                  │ layer_load │
  │                    │                       │                  │ ()         │
  │                    │                       │                  │ 模型解码步   │
  │                    │                       │                  └─────┬──────┘
  │                    │                       │                        │
  │◀──stream tokens───│◀──stream tokens───────┼────────────────────────┤
  │                    │                       │                        │
```

### 4.2 阶段详解

#### Phase 1: Prefill 阶段 (KV Producer)

1. **Proxy 接收请求**，构造特殊 `request_id`，编码了 Prefill 和 Decode 实例的 KV 地址：
   ```
   ___prefill_addr_10.0.0.1:14579___decode_addr_10.0.0.2:14580_<uuid>
   ```

2. **Proxy 发送到 Prefill 实例**，设置 `max_tokens=1`（只做 Prefill，不生成）

3. **Prefill Scheduler**：
   - 调用 `connector.get_num_new_matched_tokens()` → 返回 0（新请求，无远程 KV）
   - 构建调度计划，分配 KV Cache blocks
   - 调用 `connector.build_connector_meta()` 生成元数据

4. **Prefill Worker**（模型执行）：
   ```
   ┌──────────────────────────────────────────┐
   │ bind_connector_metadata()                 │
   │ start_load_kv()         # Producer: no-op │
   │                                          │
   │ for layer in model.layers:               │
   │     wait_for_layer_load() # Producer: no-op│
   │     attention_forward()                   │
   │     save_kv_layer()      # 异步保存 KV    │
   │                                          │
   │ wait_for_save()         # 等待保存完成     │
   │ get_finished()          # 返回完成的 req   │
   └──────────────────────────────────────────┘
   ```

5. **KV Cache 传输**：Connector (NCCL/NIXL/...) 将 KV 数据从 Prefill GPU 传输到 Decode GPU

#### Phase 2: Decode 阶段 (KV Consumer)

6. **Proxy 发送原始请求到 Decode 实例**（保留原始 `max_tokens`）

7. **Decode Scheduler**：
   - 调用 `connector.get_num_new_matched_tokens()` → 返回 N（远程已有 N 个 token 的 KV）
   - 如果返回 `None`，请求被延迟（Connector 需要更多时间）
   - 请求进入 `WAITING_FOR_REMOTE_KVS` 状态
   - 传输完成后，请求被提升（promote）为 `WAITING` → 可被调度
   - 构建 `kv_connector_metadata` 用于加载操作

8. **Decode Worker**：
   ```
   ┌──────────────────────────────────────────┐
   │ bind_connector_metadata()                 │
   │ start_load_kv()         # 异步加载远程 KV │
   │                                          │
   │ for layer in model.layers:               │
   │     wait_for_layer_load() # 阻塞等该层 KV │
   │     attention_forward()                   │
   │     save_kv_layer()      # Consumer: no-op│
   │                                          │
   │ wait_for_save()                          │
   │ get_finished()          # 返回接收完成的req│
   └──────────────────────────────────────────┘
   ```

9. **Decode 完成后**，Scheduler 释放 blocks，输出结果返回 Client

---

## 5. 模型执行中的逐层传输

关键实现在 `kv_transfer_utils.py` 中的 `maybe_transfer_kv_layer` 装饰器：

```python
# 每个 Attention 层都被这个装饰器包装
@maybe_transfer_kv_layer
def forward(self, layer_name, ...):
    # 进入时: wait_for_layer_load(layer_name)
    #         ↓ 阻塞直到该层 KV 就绪
    # 执行:   正常的 attention 计算
    # 退出时: save_kv_layer(layer_name, kv_cache, attn_metadata)
    #         ↓ 异步开始保存该层 KV
```

这实现了 **逐层流水线传输**：
- Producer 端：计算完第 i 层 → 立即开始传输第 i 层 KV → 同时计算第 i+1 层
- Consumer 端：等待第 i 层 KV 就绪 → 使用已加载的 KV 计算 → 等待第 i+1 层

```
Producer:  [Layer0: compute → save] [Layer1: compute → save] [Layer2: ...]
                │ async send              │ async send
                ▼                         ▼
Consumer:  [wait Layer0 → use]  [wait Layer1 → use]  [wait Layer2 → ...]
```

---

## 6. 请求状态流转

在 Decode 实例的 Scheduler 中，请求有以下与 KV 传输相关的状态：

```
新请求到达
    │
    ▼
get_num_new_matched_tokens()
    │
    ├─ 返回 None ──▶ 请求延迟，下一轮再查
    │
    ├─ 返回 0 ──▶ 无远程 KV，正常调度（本地 Prefill）
    │
    └─ 返回 N > 0 ──▶ 有远程 KV 可用
                        │
                        ▼
                  WAITING_FOR_REMOTE_KVS  ◀── 异步 KV 加载中
                        │
                        │ (KV 传输完成)
                        ▼
                  WAITING  ──▶ 可被调度 ──▶ RUNNING
                        │
                        │ (请求完成)
                        ▼
                  request_finished()
                        │
                        ├─ 返回 False ──▶ 立即释放 blocks
                        └─ 返回 True  ──▶ Connector 异步持有 blocks
                                           （等 get_finished() 通知后才释放）
```

---

## 7. 配置

### 7.1 KVTransferConfig 核心字段

通过 `--kv-transfer-config` 传入 JSON：

```json
{
    "kv_connector": "P2pNcclConnector",    // Connector 类型
    "kv_role": "kv_producer",               // 角色: kv_producer / kv_consumer / kv_both
    "kv_rank": 0,                            // 实例编号: 0=Prefill, 1=Decode
    "kv_parallel_size": 2,                   // KV 传输并行数 (P2pNccl: 2)
    "kv_ip": "127.0.0.1",                    // KV 连接 IP
    "kv_port": 14579,                        // KV 连接端口
    "kv_buffer_device": "cuda",              // 缓冲设备: cuda / cpu / xpu
    "kv_buffer_size": 1e9,                   // 缓冲区大小 (bytes)
    "kv_load_failure_policy": "recompute",   // 加载失败策略: recompute / fail
    "kv_connector_extra_config": {}          // Connector 特有配置
}
```

### 7.2 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | localhost | NIXL 侧通道主机 |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | 5600 | NIXL 侧通道端口 |
| `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` | 480 | NIXL 请求超时 (秒) |

---

## 8. 可用 Connector 列表

| Connector | 传输方式 | 适用场景 | 生产就绪 |
|-----------|----------|----------|----------|
| **NixlConnector** | NVIDIA NIXL (RDMA) | GPU 零拷贝传输，性能最优 | 是 |
| **P2pNcclConnector** | NCCL + ZMQ | 点对点 GPU 直传 | 是 |
| **MooncakeConnector** | Mooncake Transfer Engine | KV Cache 传输引擎 | 是 |
| **LMCacheConnectorV1** | LMCache + NIXL | KV Cache 分布式缓存 | 是 |
| **FlexKVConnectorV1** | FlexKV 分布式存储 | 超大规模 KV Cache 管理 | 实验性 |
| **OffloadingConnector** | CPU 内存卸载 | 单机 KV 卸载到 CPU | 是 |
| **SimpleCPUOffloadConnector** | CPU 内存卸载 | 最小化 CPU 卸载实现 | 实验性 |
| **HF3FSKVConnector** | HuggingFace 3FS | 分布式文件系统存储 | 实验性 |
| **MoRIIOConnector** | MoRIIO 引擎 | 自定义传输引擎 | 实验性 |
| **MultiConnector** | 组合多个 Connector | 混合传输策略 | 实验性 |
| **ExampleConnector** | 磁盘 (safetensors) | 调试/测试 | 仅调试 |
| **DecodeBenchConnector** | - | Decode 基准测试 | 仅测试 |

---

## 9. Proxy / Router 服务器

Proxy 负责将客户端请求路由到正确的 Prefill 和 Decode 实例。

### 9.1 基础 Proxy (1P1D)

**文件**：`benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py`

基于 Quart 的简单代理，工作流程：
1. 接收请求
2. 发送到 Prefill 实例（`max_tokens=1`），request_id 编码 KV 地址
3. Prefill 完成后，发送到 Decode 实例（原始参数）
4. 流式返回 Decode 结果

```
request_id = "___prefill_addr_{P_KV_ADDR}___decode_addr_{D_KV_ADDR}_{uuid}"
```

### 9.2 XpYd Proxy (动态扩展)

**文件**：`examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py`

基于 ZMQ 注册机制，支持：
- Prefill/Decode 实例动态注册/注销
- 心跳检测
- Round-robin / 随机选择 P/D 配对

### 9.3 多实例 Proxy

**文件**：`examples/online_serving/disaggregated_serving/disagg_proxy_demo.py`

基于 FastAPI，支持多个 Prefill 和 Decode 实例的负载均衡。

---

## 10. 关键源码文件索引

### 核心框架

| 文件路径 | 作用 |
|----------|------|
| `vllm/config/kv_transfer.py` | KVTransferConfig 配置定义 |
| `vllm/distributed/kv_transfer/kv_connector/v1/base.py` | KVConnectorBase_V1 抽象基类 |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | Connector 工厂（懒加载注册表） |
| `vllm/distributed/kv_transfer/kv_transfer_state.py` | 全局 Connector 单例管理 |
| `vllm/distributed/kv_transfer/__init__.py` | 公共 API 导出 |

### Connector 实现

| 文件路径 | Connector |
|----------|-----------|
| `vllm/distributed/kv_transfer/kv_connector/v1/nixl/` | NixlConnector (RDMA) |
| `vllm/distributed/kv_transfer/kv_connector/v1/p2p/` | P2pNcclConnector (NCCL+ZMQ) |
| `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/` | MooncakeConnector |
| `vllm/distributed/kv_transfer/kv_connector/v1/example_connector.py` | ExampleConnector (磁盘) |
| `vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py` | OffloadingConnector |
| `vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py` | MultiConnector |

### 引擎集成

| 文件路径 | 作用 |
|----------|------|
| `vllm/v1/worker/kv_connector_model_runner_mixin.py` | ModelRunner 的 KV Connector Mixin |
| `vllm/v1/worker/gpu/kv_connector.py` | GPU ModelRunner 的 ActiveKVConnector |
| `vllm/model_executor/layers/attention/kv_transfer_utils.py` | 逐层传输装饰器 |
| `vllm/v1/core/sched/scheduler.py` | Scheduler 中的 KV 传输调度逻辑 |
| `vllm/v1/outputs.py` | KVConnectorOutput / ModelRunnerOutput |

### 协议与代理

| 文件路径 | 作用 |
|----------|------|
| `vllm/entrypoints/serve/disagg/protocol.py` | P/D 间的请求/响应协议 (GenerateRequest/Response) |
| `benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py` | 基础 1P1D 代理 |
| `examples/online_serving/disaggregated_prefill.sh` | 完整启动示例 |

### 编码器缓存传输 (EC Transfer)

| 文件路径 | 作用 |
|----------|------|
| `vllm/distributed/ec_transfer/ec_connector/base.py` | ECConnectorBase 基类 |
| `vllm/distributed/ec_transfer/ec_connector/factory.py` | EC Connector 工厂 |
| `examples/online_serving/disaggregated_encoder/` | 编码器分离示例 |

支持 3 路分离：**Encoder → Prefill → Decode**

---

## 11. 快速上手示例

### 启动 1P1D 分离服务

```bash
# 1. 启动 Prefill 实例 (GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8100 \
    --kv-transfer-config '{
        "kv_connector":"P2pNcclConnector",
        "kv_role":"kv_producer",
        "kv_rank":0,
        "kv_parallel_size":2
    }'

# 2. 启动 Decode 实例 (GPU 1)
CUDA_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 --port 8200 \
    --kv-transfer-config '{
        "kv_connector":"P2pNcclConnector",
        "kv_role":"kv_consumer",
        "kv_rank":1,
        "kv_parallel_size":2
    }'

# 3. 启动 Proxy
python benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py \
    --prefill-url http://localhost:8100 \
    --decode-url http://localhost:8200

# 4. 发送请求
curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","prompt":"Hello","max_tokens":10}'
```

### 使用 NixlConnector (RDMA)

```bash
--kv-transfer-config '{
    "kv_connector":"NixlConnector",
    "kv_role":"kv_producer",
    "kv_buffer_device":"cuda",
    "kv_connector_extra_config":{"backends":["UCX","GDS"]}
}'
```

---

## 12. Uniform KV Cache (跨层优化)

某些 Connector (如 NixlConnector) 支持将所有层的 KV Cache 分配在一个连续内存区域中：

```
普通布局:  [Layer0 KV Blocks] [Layer1 KV Blocks] ... [LayerN KV Blocks]
           每层独立 tensor，传输时需要逐层复制

跨层布局:  [Block0: L0+L1+...+LN] [Block1: L0+L1+...+LN] ...
           一个连续 tensor，传输一个 block 就包含所有层数据
```

启用条件（三个同时满足）：
1. Connector 的 `prefer_cross_layer_blocks` 返回 True
2. 所有层具有相同的 page size（单一 Attention Group）
3. Attention 后端支持 `num_layers` 维度

相关代码：`kv_connector_model_runner_mixin.py` 中的 `use_uniform_kv_cache()` 和 `allocate_uniform_kv_caches()`

---

## 13. KV 加载失败处理

当 Decode 实例从远端加载 KV 失败时，有两种策略（`kv_load_failure_policy`）：

- **`fail`**（默认）：请求直接失败，返回错误
- **`recompute`**：将失败的 block 标记为 `invalid_block_ids`，重新调度请求在本地重新计算这些 block 的 KV

---

## 14. 通信机制总结

| Connector | 数据通道 | 控制通道 | 特点 |
|-----------|----------|----------|------|
| NixlConnector | RDMA (零拷贝) | ZMQ 侧通道 | 高性能，GPU Direct |
| P2pNcclConnector | NCCL P2P | ZMQ 控制面 | 成熟稳定 |
| MooncakeConnector | Mooncake Engine | ZMQ/HTTP | 第三方引擎 |
| ExampleConnector | 磁盘 I/O | 无 | 仅调试用 |

---

## 15. 测试入口

| 测试目录 | 内容 |
|----------|------|
| `tests/v1/kv_connector/unit/` | 各 Connector 的单元测试 |
| `tests/v1/kv_connector/nixl_integration/` | NIXL 集成测试（精度、边界、多 Connector） |
| `tests/v1/ec_connector/integration/` | EC Connector 集成测试 |
| `tests/v1/engine/test_abort_final_step.py` | 含 KV 传输的引擎级中断测试 |
