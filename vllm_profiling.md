# vLLM Profiling 系统详解

## 概述

vLLM 内置了两套 profiling 工具：**PyTorch Profiler** 和 **Nsight Systems**。本文重点分析 PyTorch Profiler 的完整运行机制。

## 1. 配置入口

### CLI 参数（v0.13.0+）

`--profiler-config` 是一个嵌套的 JSON 配置，不再是简单的 `--profile` 布尔开关。

```bash
# 基本用法
vllm serve <model> \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/traces"}'

# 完整配置
vllm serve <model> \
  --profiler-config '{
    "profiler": "torch",
    "torch_profiler_dir": "/tmp/traces",
    "torch_profiler_with_stack": true,
    "torch_profiler_with_flops": false,
    "torch_profiler_record_shapes": false,
    "torch_profiler_with_memory": false,
    "torch_profiler_use_gzip": true,
    "torch_profiler_dump_cuda_time_total": true,
    "delay_iterations": 0,
    "max_iterations": 0,
    "warmup_iterations": 0,
    "active_iterations": 5,
    "wait_iterations": 0
  }'
```

也可以用 dot notation：

```bash
vllm serve <model> \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/tmp/traces
```

### ProfilerConfig 字段说明

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `profiler` | `"torch"` / `"cuda"` / `None` | `None` | Profiler 后端类型 |
| `torch_profiler_dir` | `str` | `""` | Trace 输出目录（`profiler=torch` 时必填）|
| `torch_profiler_with_stack` | `bool` | `true` | 记录 Python 调用栈 |
| `torch_profiler_with_flops` | `bool` | `false` | 估算 FLOPS |
| `torch_profiler_record_shapes` | `bool` | `false` | 记录 tensor shapes |
| `torch_profiler_with_memory` | `bool` | `false` | 追踪内存分配/释放 |
| `torch_profiler_use_gzip` | `bool` | `true` | 压缩 trace 文件 |
| `torch_profiler_dump_cuda_time_total` | `bool` | `true` | 停止时打印 CUDA 时间汇总表 |
| `ignore_frontend` | `bool` | `false` | 跳过 AsyncLLM 前端 CPU profiling |
| `delay_iterations` | `int` | `0` | 延迟多少个 iteration 再开始 |
| `max_iterations` | `int` | `0` | 最多 profile 多少个 iteration（0=无限）|
| `warmup_iterations` | `int` | `0` | 预热步数（数据丢弃）|
| `active_iterations` | `int` | `5` | 实际记录的步数 |
| `wait_iterations` | `int` | `0` | 等待步数（profiler 关闭，零开销）|

### 配置传播链

```
CLI args (arg_utils.py)
  └─ EngineArgs.profiler_config: ProfilerConfig
      └─ VllmConfig.profiler_config
          ├─ AsyncLLM (前端 profiler)
          ├─ EngineCore → Executor → Worker (GPU profiler)
          └─ 每个组件通过 vllm_config 访问
```

## 2. 两层 Profiler 架构

vLLM 的 profiling 分为**前端**和**Worker**两层，同时运行：

```
┌──────────────────────────────────────────────────────┐
│                    AsyncLLM (前端进程)                  │
│  ┌─────────────────────────────────────────────┐     │
│  │ torch.profiler.profile(activities=[CPU])     │     │
│  │ 捕获：请求调度、队列管理、API 处理              │     │
│  └─────────────────────────────────────────────┘     │
└───────────────────┬──────────────────────────────────┘
                    │ EngineCore.profile()
                    │ collective_rpc("profile")
                    ▼
┌──────────────────────────────────────────────────────┐
│              GPU Workers (TP/RANK 0..N)                │
│  ┌─────────────────────────────────────────────┐     │
│  │ TorchProfilerWrapper                         │     │
│  │ torch.profiler.profile(activities=[CPU,CUDA])│     │
│  │ 捕获：全部 GPU kernel + CPU 端 PyTorch 操作    │     │
│  └─────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────┘
```

### 前端 Profiler

文件：`vllm/v1/engine/async_llm.py`

- 仅捕获 **CPU** 活动（`ProfilerActivity.CPU`）
- 记录前端的请求调度和编排逻辑
- 通过 `asyncio.to_thread()` 异步启动/停止，避免阻塞事件循环
- Trace 命名：`{hostname}_{pid}.async_llm`

### Worker Profiler

文件：`vllm/profiler/wrapper.py`

- 捕获 **CPU + CUDA**（或 XPU）活动
- 由 `TorchProfilerWrapper`（torch profiler）或 `CudaProfilerWrapper`（nsys）封装
- **延迟创建**：首次调用 `start_profile` 时才实例化，不在启动时
- Trace 命名：`{prefix}_dp{X}_pp{Y}_tp{Z}_rank{R}`

## 3. Profiling 触发方式

Profiling **不会自动开始**，需要手动触发。有三种方式：

### 方式一：HTTP API（在线服务）

```bash
# 1. 启动服务时配置 profiler（但不开始收集）
vllm serve <model> \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/tmp/traces"}'

# 2. 发请求前开始 profiling
curl -X POST http://localhost:8000/start_profile

# 3. 发请求...
curl -X POST http://localhost:8000/v1/chat/completions ...

# 4. 停止 profiling（会 flush trace 文件）
curl -X POST http://localhost:8000/stop_profile
```

API 路由定义在 `vllm/entrypoints/serve/profile/api_router.py`，始终挂载。

### 方式二：Python API（离线推理）

```python
from vllm import LLM

llm = LLM(model="<model>",
          profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir="/tmp/traces"))

llm.start_profile()           # 开始
outputs = llm.generate(...)    # 执行推理
llm.stop_profile()             # 停止
```

### 方式三：Benchmark 脚本

```bash
vllm bench latency \
  --model <model> \
  --profile \
  --batch-size 16 \
  --input-len 512 \
  --output-len 8

# 或 serve benchmark
vllm bench serve \
  --backend vllm \
  --model <model> \
  --profile \
  --num-prompts 2
```

## 4. 控制流：从 API 到 Worker

以 HTTP API 为例的完整控制流：

```
POST /start_profile
  │
  ▼
AsyncLLM.start_profile()                    # v1/engine/async_llm.py:903
  ├─ asyncio.to_thread(self.profiler.start)  # 启动前端 CPU profiler
  └─ engine_core.profile_async(True, prefix) # 通知 EngineCore
       │
       ▼
EngineCore.profile(is_start=True)            # v1/engine/core.py:607
  └─ model_executor.profile(True, prefix)
       │
       ▼
AbstractExecutor.profile()                   # v1/executor/abstract.py:260
  └─ collective_rpc("profile", args=(True, prefix))
       │                                     # 广播到所有 Worker
       ▼
Worker.profile(is_start=True)                # v1/worker/gpu_worker.py:876
  ├─ 首次调用：创建 TorchProfilerWrapper
  └─ profiler.start()
       └─ torch.profiler.profile 开始记录
```

停止时反向：`POST /stop_profile` → `Worker.profile(False)` → `profiler.stop()` → flush trace 文件。

## 5. 每次 Iteration 的标注

Worker 在每次 `execute_model()` 调用中自动标注 iteration 信息：

```python
# v1/worker/gpu_worker.py:842-844
with self.annotate_profile(scheduler_output):
    output = self.model_runner.execute_model(...)
```

`annotate_profile` 做两件事：

1. **调用 `profiler.step()`**：推进 profiler 的 schedule（wait → warmup → active）
2. **创建 `record_function` 上下文**：标注类似：
   ```
   execute_context_5(1024)_generation_3(256)
   ```
   含义：5 个 prefill 请求（1024 tokens）+ 3 个 decode 请求（256 tokens）

这些标注在 Chrome trace / Perfetto 中会显示为 named ranges，方便定位 prefill vs decode iteration。

### Schedule 控制

当设置 `warmup_iterations`/`wait_iterations` 时，内部使用 `torch.profiler.schedule`：

```
wait_iterations=2, warmup_iterations=1, active_iterations=5

Iteration:  0  1  2  3  4  5  6  7  8  ...
State:     WAIT WAIT WARM ACT ACT ACT ACT ACT off  ...
                      ↑   ↑  ↑  ↑  ↑
                    discard  ←  记录  →
```

- **WAIT**：profiler 关闭，零开销
- **WARMUP**：profiler 开启但丢弃数据
- **ACTIVE**：profiler 开启并记录数据
- Schedule 只执行一次（`repeat=1`）

### delay_iterations 和 max_iterations

独立于 schedule，在 `WorkerProfiler.step()` 中控制：

```python
# vllm/profiler/wrapper.py
def step(self):
    # 跳过 delay 阶段
    if self._delay_iters > 0:
        self._delay_iters -= 1
        if self._delay_iters == 0:
            self._call_start()   # delay 结束后自动 start
        return

    # 正常 step
    if self._schedule:
        self._profiler_step()

    # 检查 max_iterations
    if self._max_iters > 0 and self._active_count >= self._max_iters:
        self._call_stop()        # 达到上限自动 stop
```

## 6. 输出文件

### Trace 文件

由 `torch.profiler.tensorboard_trace_handler` 写入 `torch_profiler_dir`：

```
/tmp/traces/
├── {hostname}_{pid}.async_llm.{ext}              # 前端 CPU trace
├── {prefix}_dp0_pp0_tp0_rank0.{ext}              # Worker 0 trace
├── {prefix}_dp0_pp0_tp1_rank1.{ext}              # Worker 1 trace
├── ...
└── profiler_out_0.txt                             # CUDA 时间汇总（rank 0）
```

### 汇总表

停止时自动生成 `profiler_out_{rank}.txt`，按 CUDA self time 排序：

```
---------------------------  ----------------  ------------
Name                         Self CUDA Time    Calls
---------------------------  ----------------  ------------
void cutlass::device_kernel  10,327,352 us     17,505
void vllm::act_and_mul...     1,119,749 us     18,912
void flash::FlashAttnFwd...     916,662 us     21,312
...
```

仅在 rank 0 打印到 stdout，所有 rank 写入各自的 `.txt` 文件。

### 查看 Trace

1. **Perfetto UI**（推荐）：访问 https://ui.perfetto.dev/，拖入 trace 文件
2. **Chrome Tracing**：打开 `chrome://tracing/`，Load trace 文件
3. **TensorBoard**：`tensorboard --logdir=<torch_profiler_dir>`

> 注意：trace 文件可以是 gzip 压缩的，Perfetto 和 Chrome 都支持直接加载。

## 7. NVTX 标注（可选增强）

除了自动的 iteration 标注，还可以开启更细粒度的 NVTX 标注：

### 方式一：Layerwise NVTX Hooks

通过 observability config 开启，给每个 `nn.Module` 的 forward 加 NVTX range：

```bash
--observability-config '{"enable_layerwise_nvtx_tracing": true}'
```

- 在 `GPUModelRunner._register_layerwise_nvtx_hooks()` 中注册
- 每个 Module 的 forward 前后 push/pop NVTX range
- 包含模块名、参数 shape、输入输出 tensor shape
- **不支持 CUDA Graph**（会 graph break）

### 方式二：自定义 Scope

环境变量控制模型代码中的 scope 标注：

```bash
# 使用 torch.profiler.record_function
VLLM_CUSTOM_SCOPES_FOR_PROFILING=1

# 使用 nvtx.annotate
VLLM_NVTX_SCOPES_FOR_PROFILING=1
```

这些 scope 在 `vllm/v1/utils.py:420-440` 中通过 `record_function_or_nullcontext()` 懒初始化。

## 8. CUDA Profiler 模式（配合 Nsight Systems）

```bash
# 服务端
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end repeat \
  vllm serve <model> --profiler-config.profiler cuda

# 客户端
curl -X POST http://localhost:8000/start_profile
# ... 发请求 ...
curl -X POST http://localhost:8000/stop_profile
```

Worker 端使用 `CudaProfilerWrapper`，调用 `torch.cuda.profiler.start()/stop()` + `torch.cuda.nvtx.range()`。

分析用 Nsight Systems GUI 或 `nsys stats`。

## 9. 实用建议

### 减少开销

- 只发少量请求（trace 文件会很大）
- 用 `delay_iterations` 跳过 warmup 阶段
- 用 `max_iterations` 限制采集量
- 不需要 `record_shapes` 和 `profile_memory` 时不要开

### 超时处理

Flush trace 文件可能很慢（70B 模型 100 请求 ~10 分钟），设大 RPC 超时：

```bash
export VLLM_RPC_TIMEOUT=1800000  # 30 分钟
```

### 常见配置模式

```bash
# 模式 1：只看 decode 阶段（跳过前 10 个 iteration 的 prefill）
--profiler-config '{
  "profiler": "torch",
  "torch_profiler_dir": "/tmp/traces",
  "delay_iterations": 10,
  "max_iterations": 5
}'

# 模式 2：Schedule 精确控制（等待 5 步、预热 2 步、记录 3 步）
--profiler-config '{
  "profiler": "torch",
  "torch_profiler_dir": "/tmp/traces",
  "wait_iterations": 5,
  "warmup_iterations": 2,
  "active_iterations": 3
}'

# 模式 3：最详细（内存 + shapes + flops，开销最大）
--profiler-config '{
  "profiler": "torch",
  "torch_profiler_dir": "/tmp/traces",
  "torch_profiler_record_shapes": true,
  "torch_profiler_with_memory": true,
  "torch_profiler_with_flops": true
}'
```

## 10. 关键代码文件索引

| 功能 | 文件路径 |
|------|---------|
| CLI 参数注册 | `vllm/engine/arg_utils.py` |
| ProfilerConfig 定义 | `vllm/config/profiler.py` |
| Profiler 封装层 | `vllm/profiler/wrapper.py` |
| Layerwise profiling | `vllm/profiler/layerwise_profile.py` |
| 前端 profiler 初始化 | `vllm/v1/engine/async_llm.py` |
| Worker profiler 控制 | `vllm/v1/worker/gpu_worker.py` |
| NVTX hooks 注册 | `vllm/v1/worker/gpu_model_runner.py` |
| NVTX PyTorch hooks | `vllm/utils/nvtx_pytorch_hooks.py` |
| HTTP API 路由 | `vllm/entrypoints/serve/profile/api_router.py` |
| Python API | `vllm/entrypoints/llm.py` |
| 官方文档 | `vllm/docs/contributing/profiling.md` |
| 示例脚本 | `vllm/examples/features/profiling/` |
