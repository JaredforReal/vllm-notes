# vLLM CUDA Graph 深度解析

> 基于源码的原理分析与代码导读，聚焦 V1 架构。

## 目录

1. [为什么需要 CUDA Graph](#1-为什么需要-cuda-graph)
2. [CUDA Graph 模式体系](#2-cuda-graph-模式体系)
3. [核心数据结构：BatchDescriptor 与 BatchExecutionDescriptor](#3-核心数据结构)
4. [两套子系统：Explicit vs Compiled](#4-两套子系统)
5. [初始化与 Capture 流程](#5-初始化与-capture-流程)
6. [运行时 Dispatch 与 Replay](#6-运行时-dispatch-与-replay)
7. [输入 Padding 与 Bucketing 策略](#7-输入-padding-与-bucketing-策略)
8. [Attention Backend 兼容性](#8-attention-backend-兼容性)
9. [内存管理](#9-内存管理)
10. [Speculative Decoding 与 CUDA Graph](#10-speculative-decoding-与-cuda-graph)
11. [Vision Encoder CUDA Graph](#11-vision-encoder-cuda-graph)
12. [代码阅读顺序指南](#12-代码阅读顺序指南)

---

## 1. 为什么需要 CUDA Graph

### 问题：Kernel Launch Overhead

每次调用一个 CUDA kernel，CPU 需要向 GPU 提交一个任务（kernel launch），这涉及：
- CPU 端的 API 调用开销（~5-20us/kernel）
- CPU-GPU 之间的同步点
- CPU 成为瓶颈时 GPU 出现气泡（bubble）

一个 LLM 推理 step 可能涉及数百个 kernel（矩阵乘法、LayerNorm、Attention、激活函数、量化等），在 decode 阶段（batch size 小、计算量少），kernel launch overhead 占比可达 30-50%。

### 解决方案：CUDA Graph

CUDA Graph 允许将一系列 GPU 操作**录制**（capture）为一个图，然后**重放**（replay）这个图。重放时，CPU 只需一个 API 调用就能提交整个图的所有操作，消除了逐个 kernel launch 的开销。

**关键约束**：
- CUDA Graph 中操作的输入/输出内存地址在 capture 和 replay 之间必须**完全一致**
- 操作的控制流和数据形状必须**完全一致**（不能有动态分支）
- 这意味着：batch size 不同 = 需要不同的 graph → 需要**预录制多个尺寸的 graph**

---

## 2. CUDA Graph 模式体系

**文件**: `vllm/config/compilation.py:53-103`

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0                    # 不使用 CUDA Graph
    PIECEWISE = 1               # 分段录制：排除 attention op，只录制其他层
    FULL = 2                    # 完整录制：包括 attention 在内的所有操作
    FULL_DECODE_ONLY = (FULL, NONE)       # decode 用 FULL，mixed 用 eager
    FULL_AND_PIECEWISE = (FULL, PIECEWISE) # decode 用 FULL，mixed 用 PIECEWISE（默认）
```

### 模式之间的关系

```
配置模式 (CUDAGraphMode)              运行时实际模式
─────────────────────────            ─────────────────
NONE                                 → 所有情况: NONE
PIECEWISE                            → 所有情况: PIECEWISE
FULL                                 → 所有情况: FULL
FULL_DECODE_ONLY = (FULL, NONE)      → uniform decode: FULL
                                     → mixed prefill+decode: NONE
FULL_AND_PIECEWISE = (FULL, PW)      → uniform decode: FULL
                                     → mixed prefill+decode: PIECEWISE
```

关键方法（`compilation.py:65-90`）：
- `decode_mode()` (行 65)：取 tuple 第一个元素，即 decode 场景的模式
- `mixed_mode()` (行 68)：取 tuple 第二个元素，即 mixed 场景的模式
- `separate_routine()` (行 89)：返回 `True` 如果 value 是 tuple（区分 decode 和 mixed）

### FULL vs PIECEWISE 的本质区别

| 特性 | FULL | PIECEWISE |
|------|------|-----------|
| 录制范围 | 整个模型 forward | 每个编译子图单独录制 |
| Attention | 包含在图中 | 排除在外（attention 使用 eager） |
| Request padding | 需要（`num_reqs` 固定） | 不需要（`num_reqs=None`） |
| 捕获机制 | `CudaGraphManager` 显式管理 | `CUDAGraphWrapper` 自动管理 |
| 内存效率 | graph pool 共享，效率高 | 使用 `weak_ref` 节省内存 |
| 限制 | batch shape 必须完全匹配 | 更灵活 |

---

## 3. 核心数据结构

### 3.1 BatchDescriptor（Compiled 路径的 key）

**文件**: `vllm/forward_context.py:30-59`

```python
@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int           # padded 后的 token 数
    num_reqs: int | None = None  # None 表示 PIECEWISE（无需 request padding）
    uniform: bool = False     # 是否所有请求 token 数相同
    has_lora: bool = False    # 是否有 LoRA
    num_active_loras: int = 0 # 活跃 LoRA 数量
```

这是**编译路径**（`CUDAGraphWrapper`）的 dispatch key。两个 `BatchDescriptor` 相同 = 使用同一个 captured graph。

PIECEWISE 模式下 `num_reqs=None`，意味着同一个 graph 可以处理不同数量的请求——因为 PIECEWISE 不包含 attention（不需要固定的 block table 形状）。

### 3.2 BatchExecutionDescriptor（Explicit 路径的 key）

**文件**: `vllm/v1/worker/gpu/cudagraph_utils.py:35-43`

```python
@dataclass(frozen=True)
class BatchExecutionDescriptor:
    cg_mode: CUDAGraphMode         # FULL 或 PIECEWISE
    num_tokens: int                # padded token 数
    num_reqs: int | None           # None = PIECEWISE
    uniform_token_count: int | None = None  # uniform decode 时每个请求的 token 数
```

这是**显式路径**（`CudaGraphManager`）的 capture/replay key。比 `BatchDescriptor` 多了 `cg_mode` 和 `uniform_token_count`，因为显式路径需要根据这些信息决定如何 capture。

### 3.3 兼容性检查

**文件**: `cudagraph_utils.py:46-61`

```python
def _is_compatible(desc, num_reqs, num_tokens, uniform_token_count):
    return (
        # PIECEWISE（uniform_token_count=None）可以处理任何 uniform_token_count
        (desc.uniform_token_count is None
         or desc.uniform_token_count == uniform_token_count)
        # PIECEWISE（num_reqs=None）不需要 request padding
        and (desc.num_reqs is None or desc.num_reqs >= num_reqs)
        # token 数必须 >= 实际需要（padding 后）
        and desc.num_tokens >= num_tokens
    )
```

---

## 4. 两套子系统

vLLM V1 有**两套互补的** CUDA Graph 管理系统：

### 4.1 Explicit CUDA Graph（ModelCudaGraphManager）

**文件**: `vllm/v1/worker/gpu/cudagraph_utils.py:263-395`

用于 **FULL 模式**。在模型外部显式管理整个模型的 CUDA Graph。

```
ModelCudaGraphManager
├── capture():  为每个 BatchExecutionDescriptor 录制整个 model forward
├── run_fullgraph():  重放已录制的 graph
├── hidden_states:  持久化输出 buffer
└── graphs: dict[BatchExecutionDescriptor, torch.cuda.CUDAGraph]
```

**工作方式**：
1. Capture 时创建 dummy inputs（使用持久化 buffer 的地址）
2. 用 `torch.cuda.graph()` 录制整个 forward pass
3. Replay 时，只需将新数据写入持久化 buffer，然后调用 `graph.replay()`
4. 从持久化 `hidden_states` 中读取输出

### 4.2 Compiled CUDA Graph（CUDAGraphWrapper）

**文件**: `vllm/compilation/cuda_graph.py:145-357`

用于 **PIECEWISE 模式**。嵌入在编译后的模型图中，每个子图（subgraph）独立录制。

```
CUDAGraphWrapper（包裹一个 Inductor 子图）
├── runtime_mode: FULL 或 PIECEWISE
├── concrete_cudagraph_entries: dict[BatchDescriptor, CUDAGraphEntry]
└── __call__():
    ├── 从 ForwardContext 读取 runtime_mode 和 batch_descriptor
    ├── mode 不匹配 → 直接调用 runnable
    ├── key 不存在 → 首次遇到，capture 新 graph
    └── key 存在 → replay 已有 graph
```

**嵌套设计**（`cuda_graph.py:149-167` 的文档说明）：
- FULL wrapper 包裹整个模型
- PIECEWISE wrapper 包裹每个编译子图
- Runtime 时，根据 dispatch 决定激活哪一层 wrapper

### 4.3 两套系统如何协同

```
运行时 dispatch 结果 = FULL
    │
    └─→ ModelCudaGraphManager.run_fullgraph()
        （重放整个模型的 captured graph）
        （内部的 CUDAGraphWrapper 看到 runtime_mode=FULL 但自己的 mode=PIECEWISE → pass through）

运行时 dispatch 结果 = PIECEWISE
    │
    └─→ 调用 model forward（不使用显式 graph）
        内部的 CUDAGraphWrapper 看到 runtime_mode=PIECEWISE 匹配自己的 mode
        → 每个子图独立 capture/replay
```

---

## 5. 初始化与 Capture 流程

### 5.1 初始化顺序

```
GPUModelRunner.__init__()
  └── CudagraphDispatcher(vllm_config)                   [cudagraph_dispatcher.py:34]
        └── 初始化空的 cudagraph_keys

GPUModelRunner.initialize_kv_cache()
  └── initialize_attn_backend()                          [gpu_model_runner.py:6165]
        └── 查询每个 attention backend 的 CG 支持级别
        └── resolve_cudagraph_mode_and_sizes()            [compilation.py:1268]
              └── 根据后端支持调整 CG 模式和 capture sizes
        └── CudagraphDispatcher.initialize_cudagraph_keys()  [cudagraph_dispatcher.py:165]
              └── 生成所有有效的 BatchDescriptor key
```

### 5.2 Capture 入口

**文件**: `gpu_model_runner.py:5989-6068`

```python
def capture_model(self) -> int:
    set_cudagraph_capturing_enabled(True)
    with self._freeze_gc(), graph_capture(device=self.device):
        # 按 PIECEWISE → FULL 顺序 capture
        for runtime_mode, batch_descs in self.cudagraph_dispatcher.get_capture_descs():
            self._capture_cudagraphs(batch_descs, runtime_mode)

        # Capture vision encoder graphs（如果启用）
        if self.encoder_cudagraph_manager is not None:
            self.encoder_cudagraph_manager.capture()
```

### 5.3 单个 BatchDescriptor 的 Capture

**文件**: `gpu_model_runner.py:6080-6113`

```python
def _warmup_and_capture(self, desc, cudagraph_runtime_mode, ...):
    # 1. Warmup：多次运行以触发 JIT 编译和内存分配
    for _ in range(num_warmups):
        self._dummy_run(desc.num_tokens, cudagraph_runtime_mode=CUDAGraphMode.NONE, ...)

    # 2. Capture：最后一次运行时录制 graph
    self._dummy_run(desc.num_tokens,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    is_graph_capturing=True, ...)
```

### 5.4 ModelCudaGraphManager 的 Capture

**文件**: `cudagraph_utils.py:176-230`

```python
def capture(self, create_forward_fn, ...):
    with graph_capture(device=self.device):
        # PIECEWISE 先，FULL 后——PIECEWISE activation 更大
        for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
            for desc in descs:
                forward_fn = create_forward_fn(desc)

                # Warmup（mode=NONE）
                forward_fn(CUDAGraphMode.NONE)

                if desc.cg_mode == CUDAGraphMode.PIECEWISE:
                    # PIECEWISE：触发 CUDAGraphWrapper 内部 capture
                    forward_fn(CUDAGraphMode.PIECEWISE)
                else:
                    # FULL：显式 torch.cuda.graph() capture
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, self.pool):
                        forward_fn(CUDAGraphMode.NONE)
                    self.graphs[desc] = graph
```

### 5.5 CUDAGraphWrapper 的 Capture

**文件**: `cuda_graph.py:265-339`

```python
def __call__(self, *args, **kwargs):
    # 从 ForwardContext 读取 dispatch 决策
    cudagraph_runtime_mode = forward_context.cudagraph_runtime_mode

    # 模式不匹配 → pass through
    if cudagraph_runtime_mode != self.runtime_mode:
        return self.runnable(*args, **kwargs)

    entry = self.concrete_cudagraph_entries[batch_descriptor]

    if entry.cudagraph is None:
        # 首次遇到此 key：capture
        cudagraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cudagraph, pool=self.graph_pool):
            output = self.runnable(*args, **kwargs)
            if self.cudagraph_options.weak_ref_output:
                output = weak_ref_tensors(output)  # 释放强引用节省内存

        entry.output = weak_ref_tensors(output)
        entry.cudagraph = cudagraph
        return output
```

**重要特性**：PIECEWISE 模式下，graph 不是在启动时一次性全部 capture 的，而是在**运行时首次遇到**新的 `BatchDescriptor` 时才 capture（lazy capture）。

### 5.6 准备 Capture 用的 Input

**文件**: `cudagraph_utils.py:398-435`

```python
def prepare_inputs_to_capture(num_reqs, num_tokens, model_state, input_buffers,
                               block_tables, attn_groups, kv_cache_config):
    # 创建 dummy InputBatch（使用持久化 buffer 地址）
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)

    # 获取持久化的 block tables 和 slot mappings（相同内存地址）
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    slot_mappings = block_tables.get_dummy_slot_mappings(num_tokens)

    # 构建 attention metadata
    attn_metadata = model_state.prepare_attn(
        input_batch, CUDAGraphMode.NONE,
        input_block_tables, slot_mappings, attn_groups, kv_cache_config,
        for_capture=True,
    )
    return attn_metadata, slot_mappings_by_layer
```

**关键**：所有输入 tensor 使用**持久化 buffer 的地址**。Capture 时这些地址被固化到 graph 中，replay 时只需更新 buffer 内容。

---

## 6. 运行时 Dispatch 与 Replay

### 6.1 Dispatch 决策

**文件**: `cudagraph_dispatcher.py:234-323`

```python
def dispatch(self, num_tokens, uniform_decode, has_lora, num_active_loras,
             valid_modes, invalid_modes):
    # 计算 padded num_tokens（向上取整到最近的 capture size）
    batch_desc = self._create_padded_batch_descriptor(...)

    # 优先级：FULL > PIECEWISE > NONE
    if CUDAGraphMode.FULL in allowed_modes:
        if batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
            return CUDAGraphMode.FULL, batch_desc

    if CUDAGraphMode.PIECEWISE in allowed_modes:
        # PIECEWISE 放宽约束：num_reqs=None, uniform=False
        batch_desc_relaxed = replace(batch_desc, num_reqs=None, uniform=False)
        if batch_desc_relaxed in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, batch_desc_relaxed

    # 没有匹配的 graph
    return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
```

### 6.2 在 ModelRunner 中调用

**文件**: `gpu_model_runner.py:3545-3640`

```python
# 在 prepare_inputs 中决定 CG 模式
cudagraph_mode, batch_descriptor = dispatch_cudagraph(
    num_tokens_padded,
    disable_full=use_cascade_attn or has_encoder_output
)
```

### 6.3 FULL 模式 Replay

**文件**: `cudagraph_utils.py:247-260`

```python
def run_fullgraph(self, desc):
    # Sync offloader（确保之前的异步拷贝完成）
    get_offloader().sync_prev_onload()
    # 重放 graph
    self.graphs[desc].replay()
```

`ModelCudaGraphManager.run_fullgraph()`（行 382-395）额外从持久化 `hidden_states` 中切片返回结果：

```python
def run_fullgraph(self, desc):
    super().run_fullgraph(desc)
    return self.hidden_states[:desc.num_tokens]
```

### 6.4 PIECEWISE 模式 Replay

PIECEWISE 模式在 `set_forward_context()` 中设置 `cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE`，然后调用普通的 model forward。模型内部的每个 `CUDAGraphWrapper` 实例自行决定 replay：

```python
# ForwardContext 中设置 runtime_mode
with set_forward_context(attn_metadata, ..., cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE):
    model(**model_inputs)  # 每个 CUDAGraphWrapper 自动 replay

# CUDAGraphWrapper.__call__() 内部
entry.cudagraph.replay()
return entry.output  # 返回 weak_ref 的输出
```

---

## 7. 输入 Padding 与 Bucketing 策略

### 7.1 为什么需要 Padding

CUDA Graph 要求每次 replay 的操作完全相同。如果 batch size 不同，GPU kernel 的 grid size 不同，就不是同一个 graph。因此需要：
1. 预先为一系列固定的 batch size 录制 graph（bucketing）
2. 运行时将实际 batch size **padding** 到最近的 captured size

### 7.2 Capture Sizes

**文件**: `compilation.py`（`CompilationConfig`）

默认的 capture sizes：
```python
# 大致为: [1, 2, 4, 8, 16, ..., 248, 256, 272, ..., max_size]
# 即: [1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)
```

小 batch 密集覆盖（每 8 一个），大 batch 稀疏覆盖（每 16 一个），平衡 capture 时间和内存使用。

### 7.3 Padding 映射

**文件**: `cudagraph_dispatcher.py:71-90`

```python
def _compute_bs_to_padded_graph_size(self):
    # 对每个 batch size，预计算它应该被 pad 到哪个 capture size
    # 例如: capture_sizes = [1, 2, 4, 8, 16, ...]
    # bs=5 → pad to 8
    # bs=10 → pad to 16
    for end, start in zip(capture_sizes + [max_size+1], [0] + capture_sizes):
        for bs in range(start, end):
            if bs == start:
                self._bs_to_padded_graph_size[bs] = start
            else:
                self._bs_to_padded_graph_size[bs] = end
```

### 7.4 Seq Lens 和 Slot Mapping 的 Padding

**文件**: `vllm/v1/worker/gpu/input_batch.py:80-146`

```python
@classmethod
def make_dummy(cls, num_reqs, num_tokens, input_buffers):
    # 真实请求的 seq_lens
    input_buffers.seq_lens[:num_reqs] = num_tokens // num_reqs
    # FULL 模式：将 padding 位置的 seq_lens 设为 0
    input_buffers.seq_lens[num_reqs:] = 0

    # query_start_loc: padding 位置设为 num_tokens
    input_buffers.query_start_loc[num_reqs + 1:] = num_tokens
```

**文件**: `vllm/v1/worker/gpu/block_table.py`

Slot mapping padding：超出实际 token 数的位置填充 `PAD_SLOT_ID`，使 attention kernel 忽略这些位置。

```python
# block_table.py:234-243 (Triton kernel 中)
# 最后一个 thread block 负责将剩余 slot 填充为 PAD_SLOT_ID
if pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) >= n_slots:
    slot_mappings[off] = PAD_SLOT_ID
```

---

## 8. Attention Backend 兼容性

**文件**: `vllm/v1/attention/backend.py:479`

```python
class AttentionCGSupport(enum.IntEnum):
    ALWAYS = 3                # 支持混合 prefill-decode
    UNIFORM_BATCH = 2         # 支持 uniform batch（spec-decode OK）
    UNIFORM_SINGLE_TOKEN_DECODE = 1  # 仅 query_len==1 的 decode
    NEVER = 0                 # 不支持 CUDA Graph
```

**文件**: `vllm/v1/worker/gpu/attn_utils.py:43-124`

`init_attn_backend()` 查询每个 attention group 的 CG 支持级别，取**最低**的作为全局支持级别。

| Backend | 支持级别 | 影响 |
|---------|---------|------|
| FlashAttention 3 | ALWAYS | 可以使用 FULL 模式（包括 mixed prefill-decode） |
| FlashInfer | UNIFORM_SINGLE_TOKEN_DECODE | 只能在 uniform single-token decode 时使用 FULL |
| 其他 | NEVER | 不支持 CUDA Graph |

这个信息在 `resolve_cudagraph_mode_and_sizes()` 中被用来**降级** CG 模式：
- 如果 backend 不支持 FULL → 降级为 PIECEWISE 或 NONE
- 如果 backend 只支持 single-token decode → FULL 只能用于 decode，mixed 使用 NONE/PIECEWISE

---

## 9. 内存管理

### 9.1 Graph Pool

**所有 FULL graph 共享一个全局内存池**（`cudagraph_utils.py:101`）：

```python
self.pool = current_platform.get_global_graph_pool()
```

`torch.cuda.graph()` 使用这个 pool 分配中间 activation 内存。由于所有 graph 共享 pool，小 graph 可以复用大 graph 释放的内存。

### 9.2 Capture 顺序：PIECEWISE 先，FULL 后

**文件**: `cudagraph_utils.py:191-194`

```python
# Capture in order: PIECEWISE first, then FULL.
# PIECEWISE has larger activations so FULL activations
# should fit in already allocated buffers in the graph pool.
for mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL]:
```

PIECEWISE 的 activation 更大（包含 attention），先 capture 使其分配较大的内存，后续 FULL 的 activation 可以**复用**这些内存。

### 9.3 持久化 Buffer

**文件**: `vllm/v1/worker/gpu/input_batch.py:12-32`

```python
class InputBuffers:
    input_ids: torch.Tensor       # [max_num_tokens, int32]
    positions: torch.Tensor       # [max_num_tokens, int64]
    query_start_loc: torch.Tensor # [max_num_reqs+1, int32]
    seq_lens: torch.Tensor        # [max_num_reqs, int32]
```

这些 buffer 在整个模型生命周期内存在，地址不变。Capture 时 graph 记录这些地址，replay 时只需更新内容。

### 9.4 输出 Buffer

**文件**: `cudagraph_utils.py:274-278, 358-366`

```python
class ModelCudaGraphManager:
    self.hidden_states: torch.Tensor | None = None
    self.aux_hidden_states: list[torch.Tensor] = []

# capture 时分配（只分配一次，使用最大尺寸）
if self.hidden_states is None:
    self.hidden_states = torch.empty_like(hidden_states)
self.hidden_states[:num_tokens] = hidden_states  # 切片写入
```

### 9.5 Weak Reference 输出

**文件**: `cuda_graph.py:320-331`

PIECEWISE 模式下，中间子图的输出使用 `weak_ref_tensors()`：

```python
if self.cudagraph_options.weak_ref_output:
    output = weak_ref_tensors(output)  # 立即释放强引用
```

这避免了中间 activation 在 graph pool 中累积，节省大量内存。只对最后一个子图的输出保留强引用。

### 9.6 UVA Buffer

**文件**: `vllm/v1/worker/gpu/buffer_utils.py:36-43`

```python
class UvaBuffer:
    """CPU tensor with GPU-accessible view (Unified Virtual Addressing)."""
```

某些数据（如 `all_token_ids`）使用 UVA buffer，避免在 GPU 上分配内存。CPU tensor 通过 UVA 机制被 GPU kernel 直接访问。

---

## 10. Speculative Decoding 与 CUDA Graph

### 10.1 Spec Decode 对 CG 的影响

Speculative decoding 时，decode 阶段每个请求的 query_len = `1 + num_speculative_tokens`（不再为 1）。这影响：
- CUDA Graph 的 capture sizes 需要是 `uniform_decode_query_len` 的倍数
- 所有 decode graph 的 `num_tokens` = `num_reqs * uniform_decode_query_len`

**文件**: `compilation.py`（`adjust_cudagraph_sizes_for_spec_decode()`）

```python
# 将 capture sizes 调整为 uniform_decode_query_len 的倍数
# 例如 uniform_decode_query_len=5 (1 + 4 spec tokens)
# capture_sizes = [1, 2, 4, 8, 16, ...]
# → 调整后 = [5, 10, 15, 20, 25, ...]
```

### 10.2 EAGLE Speculator CUDA Graph

**文件**: `vllm/v1/worker/gpu/spec_decode/eagle/cudagraph.py:21-82`

EAGLE speculator 有独立的 `EagleCudaGraphManager`，使用**独立的 graph pool** 避免与主模型内存冲突：

```python
class EagleCudaGraphManager(CudaGraphManager):
    def __init__(self, ...):
        # 独立的 graph pool
        self.pool = torch.cuda.graph_pool_handle()
```

---

## 11. Vision Encoder CUDA Graph

**文件**: `vllm/v1/worker/encoder_cudagraph.py:50-598`

Vision encoder 的 CUDA Graph 使用 **budget-based**（基于输出 token 预算）而非 batch-size-based 的策略：

```
EncoderCudaGraphManager
├── BudgetGraphMetadata: 按 output token 预算索引
│   ├── graph: torch.cuda.CUDAGraph
│   ├── input_buffer, output_buffer
│   └── metadata_buffers
├── _capture_budget_graph(): 按预算录制
├── _run_budget_graph():     按 budget replay
└── _execute_local():        贪心打包多个 image 到同一个 budget graph
```

`_execute_local()` 的贪心策略（行 277-404）：
1. 按 output token 数排序 images
2. 贪心地将 images 打包到最小的可容纳 budget
3. 最大化 graph 利用率

---

## 12. 代码阅读顺序指南

### 第一阶段：理解配置和模式（自底向上）

1. **`vllm/config/compilation.py:53-103`** — `CUDAGraphMode` 枚举
   - 理解 5 种模式及其 decode/mixed 分解
   - `separate_routine()` 是理解模式体系的关键

2. **`vllm/forward_context.py:30-59`** — `BatchDescriptor`
   - 理解 dispatch key 的结构
   - `num_reqs=None` 对 PIECEWISE 的意义

3. **`vllm/v1/worker/gpu/cudagraph_utils.py:35-61`** — `BatchExecutionDescriptor` + `_is_compatible()`
   - 显式路径的 key 和兼容性检查

### 第二阶段：理解 Dispatch 决策

4. **`vllm/v1/cudagraph_dispatcher.py:15-349`** — `CudagraphDispatcher`
   - `__init__()` (行 34)：数据结构
   - `initialize_cudagraph_keys()` (行 165)：生成所有有效 key
   - `dispatch()` (行 234)：运行时 dispatch 逻辑（FULL > PIECEWISE > NONE）
   - `get_capture_descs()` (行 325)：返回 capture 描述符（PIECEWISE 先，FULL 后）

### 第三阶段：理解 Capture 流程

5. **`vllm/v1/worker/gpu_model_runner.py`**
   - `capture_model()` (行 5989)：capture 入口
   - `_capture_cudagraphs()` (行 6115)：按 mode 遍历 capture
   - `_warmup_and_capture()` (行 6080)：单个 batch descriptor 的 warmup + capture

6. **`vllm/v1/worker/gpu/cudagraph_utils.py:80-260`** — `CudaGraphManager`（基类）
   - `__init__()` (行 81)：初始化 graph 存储、pool、candidates
   - `_init_candidates()` (行 108)：构建优先级候选列表
   - `capture()` (行 176)：核心 capture 逻辑（PIECEWISE 先，FULL 后）
   - `dispatch()` (行 232)：运行时查找匹配的 descriptor
   - `run_fullgraph()` (行 247)：replay

7. **`vllm/v1/worker/gpu/cudagraph_utils.py:263-435`** — `ModelCudaGraphManager`
   - `capture()` (行 280)：模型特定 capture（创建 forward_fn 闭包）
   - `run_fullgraph()` (行 382)：replay 并切片返回 hidden_states
   - `prepare_inputs_to_capture()` (行 398)：准备 capture 用的输入

### 第四阶段：理解 CUDAGraphWrapper（Compiled 路径）

8. **`vllm/compilation/cuda_graph.py:145-357`** — `CUDAGraphWrapper`
   - `__init__()` (行 178)：初始化，分配 graph pool
   - `__call__()` (行 233)：核心 dispatch + capture/replay 逻辑
   - 重点：行 240-241 从 `ForwardContext` 读取 dispatch 决策
   - 重点：行 283-339 capture 流程（包括 weak_ref 输出）
   - 重点：行 341-356 replay 流程（包括 input address 验证）

### 第五阶段：理解 Buffer 管理

9. **`vllm/v1/worker/gpu/input_batch.py:12-146`**
   - `InputBuffers` (行 12)：持久化 GPU buffer
   - `InputBatch.make_dummy()` (行 80)：capture 用 dummy batch，含 padding 逻辑

10. **`vllm/v1/worker/gpu/block_table.py:13-170`** — `BlockTables`
    - `get_dummy_block_tables()` (行 126)：返回持久化地址的 block tables
    - `get_dummy_slot_mappings()` (行 160)：填充 PAD_SLOT_ID

11. **`vllm/v1/worker/gpu/buffer_utils.py`** — 各种 buffer 工具
    - `UvaBuffer` (行 36)：统一虚拟地址 buffer
    - `StagedWriteTensor` (行 101)：分阶段写入机制（CPU stage → Triton kernel scatter to GPU）

### 第六阶段：理解 Attention 兼容性

12. **`vllm/v1/attention/backend.py:479`** — `AttentionCGSupport` 枚举

13. **`vllm/v1/worker/gpu/attn_utils.py:27-124`**
    - `AttentionCGSupportInfo` (行 27)：全局最小支持级别
    - `init_attn_backend()` (行 43)：初始化并查询后端支持

### 第七阶段：进阶主题

14. **`vllm/v1/worker/gpu/spec_decode/eagle/cudagraph.py:21-82`** — EAGLE speculator CUDA graph（独立 pool）

15. **`vllm/v1/worker/encoder_cudagraph.py:50-598`** — Vision encoder CUDA graph（budget-based）

16. **`vllm/v1/worker/gpu/dp_utils.py:22-123`** — Data Parallel 下的 CG 协调（all-reduce 同步 CG mode 和 padding）

17. **`vllm/v1/worker/gpu/warmup.py:22-150`** — Warmup 流程（Triton kernel JIT 编译）

---

## 附录：核心流程图

### Capture 时序

```
GPUModelRunner.capture_model()                        [gpu_model_runner.py:5989]
  │
  ├── CudagraphDispatcher.get_capture_descs()          [cudagraph_dispatcher.py:325]
  │     └── 返回 [(PIECEWISE, [descs...]), (FULL, [descs...])]
  │
  ├── for (mode, batch_descs) in descs:
  │     └── _capture_cudagraphs(batch_descs, mode)     [gpu_model_runner.py:6115]
  │           └── for desc in batch_descs:
  │                 └── _warmup_and_capture(desc, mode) [gpu_model_runner.py:6080]
  │                       ├── N × _dummy_run(NONE)     # Warmup
  │                       └── _dummy_run(mode, is_graph_capturing=True)
  │                             │
  │                             ├── prepare_inputs_to_capture()  [cudagraph_utils.py:398]
  │                             │     ├── InputBatch.make_dummy()  (持久化 buffer)
  │                             │     ├── BlockTables.get_dummy_*  (持久化地址)
  │                             │     └── model_state.prepare_attn()
  │                             │
  │                             ├── set_forward_context(mode, batch_descriptor)
  │                             │
  │                             └── model(**inputs)
  │                                   │
  │                                   ├── [PIECEWISE 路径]
  │                                   │     CUDAGraphWrapper.__call__()
  │                                   │       └── torch.cuda.graph() capture 每个子图
  │                                   │
  │                                   └── [FULL 路径]
  │                                         torch.cuda.graph(graph, pool) capture 整个模型
  │
  └── EncoderCudaGraphManager.capture()    [encoder_cudagraph.py]
```

### Runtime Replay 时序

```
GPUModelRunner.execute_model()
  │
  ├── 计算 num_tokens, num_reqs, uniform_decode
  │
  ├── CudagraphDispatcher.dispatch(num_tokens, ...)   [cudagraph_dispatcher.py:234]
  │     ├── 计算 padded BatchDescriptor
  │     ├── 检查 FULL keys → 匹配? 返回 FULL
  │     ├── 检查 PIECEWISE keys（放宽约束）→ 匹配? 返回 PIECEWISE
  │     └── 否则返回 NONE
  │
  ├── if FULL:
  │     └── ModelCudaGraphManager.run_fullgraph(desc)  [cudagraph_utils.py:247]
  │           ├── get_offloader().sync_prev_onload()
  │           └── graph.replay()   ← 一个 API 调用重放整个模型
  │           └── 返回 hidden_states[:num_tokens]
  │
  ├── if PIECEWISE:
  │     └── set_forward_context(PIECEWISE, batch_descriptor)
  │     └── model(**inputs)
  │           └── 每个 CUDAGraphWrapper 独立:
  │                 ├── 检查 runtime_mode == PIECEWISE ✓
  │                 ├── 查找 BatchDescriptor → entry
  │                 ├── entry.cudagraph.replay()  ← 重放此子图
  │                 └── 返回 entry.output (weak_ref)
  │
  └── if NONE:
        └── model(**inputs)  ← 普通 eager 执行
```

---

## 关键概念速查表

| 概念 | 源码位置 | 说明 |
|------|----------|------|
| CG 模式枚举 | `compilation.py:53` CUDAGraphMode | NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE |
| Dispatch key (compiled) | `forward_context.py:30` BatchDescriptor | num_tokens + num_reqs + uniform + lora |
| Dispatch key (explicit) | `cudagraph_utils.py:35` BatchExecutionDescriptor | cg_mode + num_tokens + num_reqs + uniform_token_count |
| Dispatch 决策 | `cudagraph_dispatcher.py:234` dispatch() | FULL > PIECEWISE > NONE 优先级 |
| Key 生成 | `cudagraph_dispatcher.py:165` initialize_cudagraph_keys() | 根据 capture_sizes 和 lora cases 生成所有有效 key |
| Padding 映射 | `cudagraph_dispatcher.py:71` _compute_bs_to_padded_graph_size() | 实际 batch → 最近 capture size |
| Capture 入口 | `gpu_model_runner.py:5989` capture_model() | PIECEWISE 先 FULL 后 |
| 显式 Capture | `cudagraph_utils.py:176` CudaGraphManager.capture() | torch.cuda.graph() 录制 |
| 编译 Capture | `cuda_graph.py:233` CUDAGraphWrapper.__call__() | 运行时 lazy capture |
| FULL Replay | `cudagraph_utils.py:247` run_fullgraph() | graph.replay() + 切片输出 |
| PIECEWISE Replay | `cuda_graph.py:341-356` | 每个 wrapper 独立 replay |
| Attention 兼容性 | `attention/backend.py:479` AttentionCGSupport | ALWAYS/UNIFORM_BATCH/UNIFORM_SINGLE_TOKEN_DECODE/NEVER |
| Graph Pool | `cudagraph_utils.py:101` | 所有 graph 共享全局内存池 |
| 持久化 Input | `input_batch.py:12` InputBuffers | 地址不变的 GPU buffer |
| 持久化 Output | `cudagraph_utils.py:275` hidden_states | FULL 模式输出 buffer |
| Weak Ref 输出 | `cuda_graph.py:327` weak_ref_tensors() | PIECEWISE 节省内存 |
| Capture Padding | `input_batch.py:104,115` | seq_lens pad to 0, query_start_loc pad to num_tokens |
| Slot Pad | `block_table.py` (Triton) | 填充 PAD_SLOT_ID |
