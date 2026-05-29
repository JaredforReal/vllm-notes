# GLM5Next PD 分离部署指南

## 环境信息

| 项目 | 值 |
|------|-----|
| **Prefill 节点** | node070 (10.2.1.70 / 10.1.50.70) |
| **Decode 节点** | node239 (10.2.2.111 / 10.1.51.111) |
| **GPU** | 每节点 8× H100, NV18 全互联 |
| **RDMA** | 19× mlx5 设备, RoCE |
| **NIXL** | v0.10.1 (cu12) |
| **vLLM 路径** | `/opensource/guohong/vllm` |

---

## Step 1: 验证跨节点 RDMA 连通性

在两个节点上分别运行：

```bash
apt-get update && apt-get install -y perftest iproute2 rdma-core ibverbs-utils infiniband-diags

# 快速 RDMA 连通性测试（用 ib_write_bw）
# 在 node070 上先启动 server：
ib_write_bw -d mlx5_0 --size=65536 --iters=1000

# 在 node239 上运行 client（使用 node070 的某个 IP）：
ib_write_bw -d mlx5_0 --size=65536 --iters=1000 10.1.50.70
```

如果 RDMA 测试失败（跨节点 NIC 不在同一 L2），NIXL 可以走 **UCX TCP fallback**。性能会差一些，但功能正常。

---

## Step 2: 启动 Prefill 节点

**在 node070 (10.1.50.70) 上运行**：

```bash
# --- Prefill 实例 ---
export MODEL_PATH=/workspace/yuxuan/GLM-5-Next-0521-FP8/  # 替换为实际模型路径
export VLLM_SSM_CONV_STATE_LAYOUT=DS  # KDA 3-read transfer 必需

# NIXL side channel 配置
export VLLM_NIXL_SIDE_CHANNEL_HOST=10.1.50.70
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --no-async-scheduling \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_producer",
    "kv_rank": 0,
    "kv_parallel_size": 2,
    "kv_buffer_device": "cuda",
    "kv_buffer_size": 1e10,
    "kv_connector_extra_config": {
      "backends": ["UCX"]
    }
  }'
```

**关键参数说明**：
- `kv_role=kv_producer`: 该节点是 KV 生产者（只做 prefill）
- `kv_rank=0`: 该节点在 PD 对中的编号
- `kv_parallel_size=2`: P+D 共 2 个节点
- `VLLM_SSM_CONV_STATE_LAYOUT=DS`: KDA 状态传输需要 DS 布局
- `--no-disable-hybrid-kv-cache-manager`: Hybrid 模型必需（启用 HMA）
- `backends: ["UCX"]`: 使用 UCX 作为 NIXL 后端（支持 RDMA 和 TCP fallback）

---

## Step 3: 启动 Decode 节点

**在 node239 (10.1.51.111) 上运行**：

```bash
# --- Decode 实例 ---
# kv_role=kv_consumer 表示这是 KV 消费者

export MODEL_PATH=/workspace/yuxuan/GLM-5-Next-0521-FP8/  # 与 Prefill 节点相同的模型路径
export VLLM_SSM_CONV_STATE_LAYOUT=DS

# NIXL side channel 连接到 Prefill 节点
export VLLM_NIXL_SIDE_CHANNEL_HOST=10.1.51.111
export VLLM_NIXL_SIDE_CHANNEL_PORT=5600

vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 8200 \
  --tensor-parallel-size 8 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_consumer",
    "kv_rank": 1,
    "kv_parallel_size": 2,
    "kv_buffer_device": "cuda",
    "kv_buffer_size": 1e10,
    "kv_connector_extra_config": {
      "backends": ["UCX"]
    }
  }'
```

**关键差异**：
- `kv_role=kv_consumer`: 该节点是 KV 消费者（做 decode）
- `kv_rank=1`: 第二个节点
- `VLLM_NIXL_SIDE_CHANNEL_HOST=10.1.50.70`: 连接到 Prefill 节点的 side channel

---

## Step 5: 启动 Proxy

Proxy 可以运行在任一节点或第三方机器上。需要先安装依赖：

**在 node070 上运行**：

```bash
# 替换 IP 为实际可达的 IP
python benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py \
  --prefill-url http://10.1.50.70:8100 \
  --decode-url http://10.1.51.111:8200 \
  --port 8000
```

```bash
vllm-router --policy round_robin \
    --vllm-pd-disaggregation \
    --prefill http://10.1.50.70:8100 \
    --decode http://10.1.51.111:8200 \
    --host 0.0.0.0 \
    --port 8000 \
    --intra-node-data-parallel-size 1
```

**注意**：如果 Proxy 运行在外部机器（172.18.65.118），确保该机器能访问两个节点的端口。

---

## Step 6: 发送测试请求

```bash
# 通过 Proxy 发送请求
curl -X POST http://172.18.65.118:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<MODEL_NAME>",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 1.0
  }'

```

---

### 启动顺序
1. **先启动 Prefill 节点** (Step 3) — 等待模型加载完成
2. **再启动 Decode 节点** (Step 4) — 会自动连接 Prefill 的 side channel
3. **最后启动 Proxy** (Step 5)
4. 等待看到 NIXL handshake 成功的日志后，发送请求

---

## 可选: 开启 MTP Speculative Decoding

如果要在 Decode 端开启 MTP：

```bash
# Decode 节点添加 spec decode 参数
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
vllm serve $MODEL_PATH \
  --host 0.0.0.0 \
  --port 8200 \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --speculative-config '{
    "method": "draft_model",
    "num_speculative_tokens": 3,
    "draft_model_config": {
      "model": "$MODEL_PATH"
    }
  }' \
  --kv-transfer-config '{
    "kv_connector": "NixlConnector",
    "kv_role": "kv_consumer",
    "kv_rank": 1,
    "kv_parallel_size": 2,
    "kv_buffer_device": "cuda",
    "kv_buffer_size": 1e10,
    "kv_connector_extra_config": {
      "backends": ["UCX"]
    }
  }' \
  --no-disable-hybrid-kv-cache-manager
```

**注意**: Prefill 节点不需要 `--speculative-config`。但两端 `num_spec` 的差异可能导致
KDA conv_state 形状不匹配。如果遇到 compatibility hash 问题，两端都需要配置相同
的 speculative config（Prefill 端加上但不实际使用 MTP）。
