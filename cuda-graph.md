# vLLM CUDA Graph 全生命周期：从配置到 Replay 的完整链路

> 基于 `vllm/` 当前源码的 CUDA Graph 代码导读，聚焦 V1 架构下的
> **模式解析、Graph Key 生成、Capture、运行时 Dispatch、Replay、Padding、Attention 兼容性**
> 以及多模态 encoder CUDA Graph。
>
> 核对基准：当前工作区 `vllm` 源码。V1 GPU 执行链路里同时存在两套 runner：
> `vllm/vllm/v1/worker/gpu_model_runner.py` 是较完整的主路径；
> `vllm/vllm/v1/worker/gpu/model_runner.py` 是拆分后的 runner，使用
> `ModelCudaGraphManager` 显式管理 FULL graph。本文以主路径讲清运行机制，并在相关位置单独标注
> 拆分后 runner 的差异。

## 目录

1. [全局概览](#1-全局概览)
2. [Stage 1：配置解析与模式体系](#2-stage-1配置解析与模式体系)
3. [Stage 2：编译切分与 CUDAGraphWrapper 注入](#3-stage-2编译切分与-cudagraphwrapper-注入)
4. [Stage 3：Attention 兼容性与模式降级](#4-stage-3attention-兼容性与模式降级)
5. [Stage 4：Graph Key 初始化与 Bucketing](#5-stage-4graph-key-初始化与-bucketing)
6. [Stage 5：启动期 Capture](#6-stage-5启动期-capture)
7. [Stage 6：运行时 Dispatch 与 Padding](#7-stage-6运行时-dispatch-与-padding)
8. [Stage 7：ForwardContext 驱动 Replay](#8-stage-7forwardcontext-驱动-replay)
9. [Stage 8：拆分后 runner 的 ModelCudaGraphManager 路径](#9-stage-8拆分后-runner-的-modelcudagraphmanager-路径)
10. [Stage 9：多模态 Encoder CUDA Graph](#10-stage-9多模态-encoder-cuda-graph)
11. [横切关注点](#11-横切关注点)
12. [代码阅读顺序](#12-代码阅读顺序)

---

## 1. 全局概览

CUDA Graph 在 vLLM 里解决的是 **decode 阶段 CPU kernel launch overhead**：
一轮 decode 的 GPU 计算通常由很多小 kernel 组成，逐个 launch 会让 CPU 成为瓶颈。CUDA Graph
把一串 GPU 操作录制成图，之后 `graph.replay()` 一次提交整段计算。

但 CUDA Graph 有两个硬约束：

- replay 时输入、输出、中间 buffer 的 **地址必须和 capture 时一致**。
- replay 时 shape、kernel grid、控制流必须和 capture 时一致。

因此 vLLM 做了三件事：

1. 启动期按若干 `cudagraph_capture_sizes` 录制 graph。
2. 运行时把真实 batch pad 到最接近的 capture size。
3. 用 `ForwardContext` 把 “这一步用 FULL / PIECEWISE / NONE，以及 graph key 是什么” 传进模型。

### CUDA Graph 主链路总图

```
┌──────────────────────────────────────────────────────────────────────────┐
│  配置阶段                                                                 │
│                                                                          │
│  VllmConfig                                                              │
│     │                                                                    │
│     ├─ CompilationConfig.cudagraph_mode                                  │
│     ├─ _set_cudagraph_sizes() 生成 cudagraph_capture_sizes                │
│     └─ set_splitting_ops_for_v1() 决定 attention / KV cache 是否切出去      │
│                                                                          │
│  编译 / 装配阶段                                                           │
│     │                                                                    │
│     ├─ maybe_use_cudagraph_partition_wrapper()                           │
│     ├─ wrap_with_cudagraph_if_needed()                                   │
│     └─ CUDAGraphWrapper 包住 piecewise 子图，或包住 FULL 模型外层           │
│                                                                          │
│  Worker 初始化阶段                                                        │
│     │                                                                    │
│     ├─ init_attn_backend() 查询每个 attention backend 的 CG 支持等级         │
│     ├─ resolve_cudagraph_mode_and_sizes() 可能降级 FULL / PIECEWISE         │
│     └─ CudagraphDispatcher.initialize_cudagraph_keys()                    │
│                                                                          │
│  Capture 阶段                                                             │
│     │                                                                    │
│     ├─ GPUModelRunner.capture_model()                                    │
│     ├─ cudagraph_dispatcher.get_capture_descs()                           │
│     ├─ _warmup_and_capture()                                              │
│     └─ _dummy_run(..., is_graph_capturing=True)                           │
│            └─ CUDAGraphWrapper.__call__() 首次 capture                     │
│                                                                          │
│  每个 EngineCore.step() 的模型执行阶段                                      │
│     │                                                                    │
│     ├─ GPUModelRunner.execute_model()                                    │
│     ├─ _determine_batch_execution_and_padding()                           │
│     ├─ CudagraphDispatcher.dispatch() → FULL / PIECEWISE / NONE + key      │
│     ├─ set_forward_context(cudagraph_runtime_mode, batch_descriptor, ...) │
│     └─ model forward                                                     │
│          ├─ PIECEWISE: 子图 CUDAGraphWrapper replay                       │
│          ├─ FULL:     外层 CUDAGraphWrapper replay                         │
│          └─ NONE:     eager / compiled 普通执行                             │
└──────────────────────────────────────────────────────────────────────────┘
```

### 三个核心概念

- `CUDAGraphMode`：用户配置 + 运行时模式。配置态可以是组合模式；运行时只会是
  `NONE`、`PIECEWISE`、`FULL`。
- `BatchDescriptor`：运行时 CUDA Graph dispatch key，描述 padded 后的 token/request 形状和 LoRA 状态。
- `ForwardContext`：把 dispatch 结果带进模型内部，`CUDAGraphWrapper` 只相信它，不重新做调度判断。

### FULL vs PIECEWISE 一眼区分

| 维度 | FULL | PIECEWISE |
|---|---|---|
| 录制范围 | 整个模型 forward | 编译切分后的若干子图 |
| Attention | 通常包含在 graph 内 | 通常作为 cudagraph-unsafe op 留在 graph 外 |
| request padding | 需要固定 `num_reqs` | `num_reqs=None`，只关心 `num_tokens` |
| 典型场景 | uniform decode，尤其纯 decode | mixed prefill-decode / 后端不支持 full attention graph |
| 调度优先级 | 优先尝试 | FULL 不匹配时再尝试 |
| 源码主路径 | 外层 `CUDAGraphWrapper(runtime_mode=FULL)` | 子图 `CUDAGraphWrapper(runtime_mode=PIECEWISE)` |

---

## 2. Stage 1：配置解析与模式体系

### 入口文件

**文件**: `vllm/vllm/config/compilation.py`

`CUDAGraphMode` 定义了 5 种配置值：

```python
NONE = 0
PIECEWISE = 1
FULL = 2
FULL_DECODE_ONLY = (FULL, NONE)
FULL_AND_PIECEWISE = (FULL, PIECEWISE)
```

其中 `NONE`、`PIECEWISE`、`FULL` 同时也是运行时 concrete mode；
`FULL_DECODE_ONLY`、`FULL_AND_PIECEWISE` 是配置态组合模式，通过：

- `decode_mode()`：uniform decode 走什么模式。
- `mixed_mode()`：mixed prefill-decode 走什么模式。
- `separate_routine()`：是否区分 decode 和 mixed 两条路径。

### 配置态到运行态

```
用户配置 / CompilationConfig.cudagraph_mode
  │
  ├─ NONE
  │    └─ 所有 batch → NONE
  │
  ├─ PIECEWISE
  │    └─ 所有可匹配 batch → PIECEWISE
  │
  ├─ FULL
  │    └─ 所有可匹配 batch → FULL
  │
  ├─ FULL_DECODE_ONLY = (FULL, NONE)
  │    ├─ uniform decode → FULL
  │    └─ mixed prefill-decode → NONE
  │
  └─ FULL_AND_PIECEWISE = (FULL, PIECEWISE)
       ├─ uniform decode → FULL
       └─ mixed prefill-decode → PIECEWISE
```

### Capture sizes 初始化

**文件**: `vllm/vllm/config/vllm.py`

`VllmConfig._set_cudagraph_sizes()` 负责把用户配置和 scheduler 上限合成最终的
`compilation_config.cudagraph_capture_sizes`：

```
VllmConfig.__post_init__()
  └─ _set_cudagraph_sizes()
       ├─ 如果 enforce_eager 或 cudagraph_mode=NONE:
       │    └─ max_cudagraph_capture_size = 0, capture_sizes = []
       │
       ├─ 否则确定 max_cudagraph_capture_size
       │    ├─ 用户显式设置则使用用户值
       │    └─ 默认 min(max_num_seqs * decode_query_len * 2, 512)
       │
       ├─ 生成 capture sizes
       │    ├─ interactivity 模式: 1..min(max_size, 32)
       │    └─ 默认: [1, 2, 4] + range(8, 256, 8) + range(256, max+1, 16)
       │
       ├─ 如果 max_num_batched_tokens 落在 max_size 内，把它也加入 capture sizes
       ├─ 如果启用 sequence parallelism，调整到 TP 兼容大小
       └─ post_init_cudagraph_sizes()
```

几个关键点：

- capture size 是 **token 数**，不是 request 数。
- 对纯 decode 而言，`num_tokens ≈ num_reqs * decode_query_len`，所以它看起来像 batch size。
- speculative decoding 下 `decode_query_len = 1 + num_speculative_tokens`，capture size 需要被调整为它的倍数。

---

## 3. Stage 2：编译切分与 CUDAGraphWrapper 注入

PIECEWISE CUDA Graph 依赖模型被切成多个可录制子图。vLLM 默认会把 attention 类 op
放到 graph 外面，因为很多 attention backend 的 metadata、workspace 或 kernel 形态不适合通用 CUDA Graph。

### 切分规则

**文件**: `vllm/vllm/config/compilation.py`

```
CompilationConfig.set_splitting_ops_for_v1()
  │
  ├─ 如果 mode 不是 VLLM_COMPILE:
  │    └─ 只初始化 splitting_ops，不进入 piecewise compile 主路径
  │
  ├─ 如果 splitting_ops is None:
  │    ├─ 默认加入 attention ops
  │    ├─ 非 inductor partition 路径下追加 unified_kv_cache_update
  │    └─ 这些 op 会留在 piecewise CUDA Graph 外
  │
  ├─ 如果 splitting_ops == [] 且 cudagraph_mode 包含 PIECEWISE:
  │    ├─ PIECEWISE → NONE
  │    └─ FULL_AND_PIECEWISE → FULL
  │
  └─ 一些并行/通信特性可能进一步把 PIECEWISE 降到 FULL 或 NONE
```

### Wrapper 注入调用栈

PIECEWISE 有两种注入点：Dynamo 级 piecewise backend 包装，或者 Inductor graph partition 包装。

```
模型编译阶段
  │
  ├─ Dynamo piecewise split 路径
  │    │
  │    └─ vllm/compilation/backends.py
  │         └─ wrap_with_cudagraph_if_needed(...)
  │              ├─ 检查 cudagraph_mode.has_piecewise_cudagraphs()
  │              ├─ current_platform.get_static_graph_wrapper_cls()
  │              └─ CUDAGraphWrapper(
  │                    runnable=piecewise_backend,
  │                    runtime_mode=CUDAGraphMode.PIECEWISE,
  │                    weak_ref_output=is_last_graph,
  │                    gc_disable=not is_first_graph,
  │                 )
  │
  └─ Inductor graph partition 路径
       │
       └─ vllm/compilation/decorators.py
            └─ maybe_use_cudagraph_partition_wrapper(vllm_config)
                 ├─ torch._inductor.utils.set_customized_partition_wrappers(...)
                 └─ 对每个 partition 创建 CUDAGraphWrapper(runtime_mode=PIECEWISE)
```

FULL 的主路径更简单：模型外层直接包一层 `CUDAGraphWrapper(runtime_mode=FULL)`。

**文件**: `vllm/vllm/v1/worker/gpu_model_runner.py`

```
GPUModelRunner.load_model()
  │
  ├─ 如果 STOCK_TORCH_COMPILE:
  │    └─ self.model.compile(fullgraph=True, backend=...)
  │
  └─ 否则由 vLLM 自己控制 CUDA Graph
       ├─ cudagraph_mode.has_full_cudagraphs() 且未启用 ubatching:
       │    └─ self.model = CUDAGraphWrapper(self.model, runtime_mode=FULL)
       └─ 启用 ubatching:
            └─ self.model = UBatchWrapper(..., CUDAGraphMode.FULL/NONE)
```

### Wrapper 本身的行为

**文件**: `vllm/vllm/compilation/cuda_graph.py`

```
CUDAGraphWrapper.__call__(*args, **kwargs)
  │
  ├─ 没有 ForwardContext:
  │    └─ 直接 runnable(*args, **kwargs)
  │
  ├─ 读取 get_forward_context()
  │    ├─ cudagraph_runtime_mode
  │    └─ batch_descriptor
  │
  ├─ runtime_mode == NONE 或 runtime_mode != self.runtime_mode:
  │    └─ 直接 runnable(*args, **kwargs)
  │
  ├─ 第一次见到 batch_descriptor:
  │    ├─ validate_cudagraph_capturing_enabled()
  │    ├─ 记录输入 tensor 地址（DEBUG 下 replay 会校验）
  │    ├─ torch.cuda.CUDAGraph()
  │    ├─ get_offloader().sync_prev_onload()
  │    ├─ with torch.cuda.graph(...):
  │    │    ├─ output = runnable(*args, **kwargs)
  │    │    ├─ get_offloader().join_after_forward()
  │    │    └─ weak_ref_tensors(output) 释放强引用
  │    └─ 缓存 entry.cudagraph / entry.output
  │
  └─ 已 capture:
       ├─ DEBUG 下检查输入地址不变
       ├─ get_offloader().sync_prev_onload()
       ├─ entry.cudagraph.replay()
       └─ return entry.output
```

容易混淆的一点：`CUDAGraphWrapper` 不负责决定应该用哪个 graph。它只读取
`ForwardContext` 里的 `cudagraph_runtime_mode` 和 `batch_descriptor`。真正的 dispatch
发生在 model runner。

---

## 4. Stage 3：Attention 兼容性与模式降级

### 支持等级

**文件**: `vllm/vllm/v1/attention/backend.py`

```
AttentionCGSupport
  ├─ ALWAYS = 3
  │    └─ 支持 mixed prefill-decode 的 full CUDA Graph
  ├─ UNIFORM_BATCH = 2
  │    └─ 只支持所有 request query_len 相同的 batch，可覆盖 spec decode
  ├─ UNIFORM_SINGLE_TOKEN_DECODE = 1
  │    └─ 只支持 query_len == 1 的纯 decode
  └─ NEVER = 0
       └─ attention 不支持 full CUDA Graph
```

### 初始化调用栈

主路径里 attention 兼容性在 KV cache 初始化后检查。

```
GPUModelRunner.initialize_kv_cache(kv_cache_config)
  │
  ├─ init_attn_backend(...) / _check_and_update_cudagraph_mode(...)
  │    ├─ 遍历 kv_cache_groups
  │    ├─ 获取每个 layer 的 AttentionBackend
  │    ├─ builder_cls.get_cudagraph_support(vllm_config, kv_cache_spec)
  │    └─ 取所有 attention backend 的最低支持等级 min_cg_support
  │
  ├─ CompilationConfig.resolve_cudagraph_mode_and_sizes(...)
  │    ├─ mixed_mode=FULL 但 min_cg_support != ALWAYS:
  │    │    ├─ 如果 NEVER，通常要求改 PIECEWISE 或抛错
  │    │    └─ 否则降级到 FULL_AND_PIECEWISE / FULL_DECODE_ONLY
  │    │
  │    ├─ decode_mode=FULL 但 min_cg_support=NEVER:
  │    │    ├─ attention 已 piecewise compile → PIECEWISE
  │    │    └─ 否则 → NONE
  │    │
  │    ├─ spec decode 且 decode_query_len > 1:
  │    │    └─ 要求至少 UNIFORM_BATCH，否则降级 PIECEWISE / NONE
  │    │
  │    ├─ Mamba + FULL:
  │    │    └─ 检查 max_num_reqs 是否超过可用 Mamba cache blocks
  │    │
  │    └─ adjust_cudagraph_sizes_for_spec_decode(...)
  │
  └─ CudagraphDispatcher.initialize_cudagraph_keys(resolved_mode, ...)
```

### 兼容性对模式的影响

```
用户请求 FULL_AND_PIECEWISE
  │
  ├─ Attention 支持 ALWAYS
  │    └─ decode: FULL, mixed: PIECEWISE
  │
  ├─ Attention 只支持 UNIFORM_SINGLE_TOKEN_DECODE
  │    ├─ 普通 decode: FULL
  │    ├─ spec decode: FULL 可能降级
  │    └─ mixed: PIECEWISE
  │
  └─ Attention 为 NEVER
       ├─ attention 已被 piecewise 切出去: PIECEWISE
       └─ 否则: NONE 或抛错
```

这也是为什么 PIECEWISE 很重要：即使 attention backend 不能进入 CUDA Graph，其他线性层、
norm、激活、MoE 等子图仍然可以被 capture/replay。

---

## 5. Stage 4：Graph Key 初始化与 Bucketing

### BatchDescriptor

**文件**: `vllm/vllm/forward_context.py`

```python
@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int
    num_reqs: int | None = None
    uniform: bool = False
    has_lora: bool = False
    num_active_loras: int = 0
```

字段含义：

- `num_tokens`：padding 后的 token 数，也是 capture size。
- `num_reqs`：padding 后 request 数。PIECEWISE 下可以是 `None`。
- `uniform`：所有 request 是否具有同样 query length。纯 decode 和 spec decode 通常为 true。
- `has_lora` / `num_active_loras`：LoRA 会改变执行路径或 kernel grid，因此可能需要专门 graph。

### CudagraphDispatcher 初始化

**文件**: `vllm/vllm/v1/cudagraph_dispatcher.py`

```
CudagraphDispatcher.initialize_cudagraph_keys(cudagraph_mode, uniform_decode_query_len)
  │
  ├─ cudagraph_mode == NONE:
  │    └─ keys_initialized = True，直接返回
  │
  ├─ _compute_bs_to_padded_graph_size()
  │    └─ 为每个 0..max_size 的 token 数预计算 pad 到哪个 capture size
  │
  ├─ _get_lora_cases()
  │    ├─ 无 LoRA → [0]
  │    ├─ cudagraph_specialize_lora=True → [0] + captured_lora_counts
  │    └─ 否则 → [max_loras + 1]
  │
  ├─ mixed_mode != NONE:
  │    ├─ 遍历 cudagraph_capture_sizes × lora_cases
  │    ├─ _create_padded_batch_descriptor(bs, uniform_decode=False, ...)
  │    ├─ 如果 mixed_mode == PIECEWISE:
  │    │    └─ replace(num_reqs=None, uniform=False)
  │    └─ add_cudagraph_key(mixed_mode, batch_desc)
  │
  └─ decode_mode == FULL 且 separate_routine:
       ├─ 只保留 <= max_num_seqs * uniform_decode_query_len 的 capture size
       ├─ _create_padded_batch_descriptor(bs, uniform_decode=True, ...)
       └─ add_cudagraph_key(FULL, batch_desc)
```

### Padding 映射示例

`_compute_bs_to_padded_graph_size()` 会构造一个数组：

```
capture_sizes = [1, 2, 4, 8, 16, 24, 32, ...]

真实 num_tokens   padded num_tokens
───────────────   ─────────────────
1                 1
2                 2
3                 4
4                 4
5                 8
7                 8
9                 16
17                24
```

这个映射是运行时 dispatch 的基础：真实 batch 不需要刚好等于 capture size，只要能 pad 到某个已录制 size。

---

## 6. Stage 5：启动期 Capture

### Capture 入口

**文件**: `vllm/vllm/v1/worker/gpu_model_runner.py`

```
GPUModelRunner.capture_model()
  │
  ├─ cudagraph_mode == NONE:
  │    └─ 跳过 capture
  │
  ├─ 如果 cudagraph_mm_encoder=True:
  │    └─ 初始化 EncoderCudaGraphManager
  │
  ├─ init_routed_experts_capturer()
  │    └─ MoE routed experts 相关 buffer 地址必须在 graph 前固定
  │
  ├─ set_cudagraph_capturing_enabled(True)
  │
  ├─ with _freeze_gc(), graph_capture(device):
  │    ├─ torch.accelerator.synchronize()
  │    ├─ torch.accelerator.empty_cache()
  │    ├─ for runtime_mode, batch_descs in dispatcher.get_capture_descs():
  │    │    └─ _capture_cudagraphs(batch_descs, runtime_mode)
  │    └─ encoder_cudagraph_manager.capture()
  │
  ├─ set_cudagraph_capturing_enabled(False)
  ├─ torch.accelerator.empty_cache()
  └─ lock_workspace()
```

`dispatcher.get_capture_descs()` 返回顺序固定为 PIECEWISE → FULL，并且每组内部按
`num_tokens` 降序。这么做是为了先让大 graph 分配/占住较大的 graph pool，再让小 graph 复用。

### 单组 Capture 调用栈

```
GPUModelRunner._capture_cudagraphs(batch_descriptors, runtime_mode)
  │
  ├─ 判断是否允许 ubatching capture
  │    └─ 目前只在 FULL + uniform decode + 达到阈值时捕获 ubatched graph
  │
  └─ for batch_desc in batch_descriptors:
       └─ _warmup_and_capture(batch_desc, runtime_mode, allow_microbatching)
            │
            ├─ for _ in cudagraph_num_of_warmups:
            │    └─ _dummy_run(
            │          desc.num_tokens,
            │          cudagraph_runtime_mode=NONE,
            │          force_attention=(runtime_mode == FULL),
            │          uniform_decode=desc.uniform,
            │          num_active_loras=desc.num_active_loras,
            │       )
            │
            └─ _dummy_run(
                 desc.num_tokens,
                 cudagraph_runtime_mode=runtime_mode,
                 uniform_decode=desc.uniform,
                 num_active_loras=desc.num_active_loras,
                 is_graph_capturing=True,
               )
```

### `_dummy_run()` 做了什么

`_dummy_run()` 是 capture 的核心。它伪造一个 batch，让模型用和真实运行相同的 input buffer、
attention metadata、slot mapping、LoRA 状态和 ForwardContext 跑一遍。

```
GPUModelRunner._dummy_run(num_tokens, cudagraph_runtime_mode, ...)
  │
  ├─ 根据 uniform_decode / create_mixed_batch 生成 num_scheduled_tokens
  ├─ _determine_batch_execution_and_padding(...)
  │    └─ 反查 dispatcher，确保得到的 mode/key 和目标 capture desc 一致
  │
  ├─ maybe_create_ubatch_slices(...)
  ├─ _get_slot_mappings(...)
  │    └─ dummy run 没有真实 KV slot，slot mapping 通常填 -1
  │
  ├─ 如果 FULL 或 force_attention:
  │    ├─ 写 seq_lens / query_start_loc 到持久化 buffer
  │    ├─ commit_block_table(...)
  │    └─ _build_attention_metadata(..., for_cudagraph_capture=True)
  │
  ├─ 准备 input_ids / inputs_embeds / positions / model_kwargs
  │
  ├─ maybe_dummy_run_with_lora(...)
  │
  └─ with set_forward_context(...):
       └─ self.model(...)
            ├─ FULL: 外层 CUDAGraphWrapper capture
            └─ PIECEWISE: 内部多个 CUDAGraphWrapper capture
```

### Capture 时序图

```
GPUModelRunner        CudagraphDispatcher       _dummy_run / buffers        ForwardContext        CUDAGraphWrapper
     │                         │                         │                       │                       │
     │ capture_model()          │                         │                       │                       │
     │────────────────────────>│                         │                       │                       │
     │ get_capture_descs()      │                         │                       │                       │
     │<────────────────────────│                         │                       │                       │
     │                         │                         │                       │                       │
     │ _capture_cudagraphs()    │                         │                       │                       │
     │──────────────────────────────────────────────────>│                       │                       │
     │                         │                         │ warmup: mode=NONE     │                       │
     │                         │                         │──────────────────────>│ cudagraph_runtime=NONE │
     │                         │                         │                       │──────────────────────>│ pass through
     │                         │                         │                       │<──────────────────────│
     │                         │                         │<──────────────────────│                       │
     │                         │                         │                       │                       │
     │                         │                         │ capture: mode=PW/FULL │                       │
     │                         │                         │──────────────────────>│ batch_descriptor=desc  │
     │                         │                         │                       │──────────────────────>│ torch.cuda.graph()
     │                         │                         │                       │                       │ runnable forward
     │                         │                         │                       │<──────────────────────│ cache graph entry
     │                         │                         │<──────────────────────│                       │
     │                         │                         │                       │                       │
     │ set_cudagraph_capturing_enabled(False)             │                       │                       │
     │ lock_workspace()          │                         │                       │                       │
```

关键点：

- warmup 用 `CUDAGraphMode.NONE`，用于触发编译、分配 workspace、让后续 capture 避免动态分配。
- capture 时打开全局 guard：`validate_cudagraph_capturing_enabled()` 会阻止意外 lazy capture。
- 输入 buffer 是持久化的，graph 记录的是这些 buffer 的地址。

---

## 7. Stage 6：运行时 Dispatch 与 Padding

### 运行时入口

**文件**: `vllm/vllm/v1/worker/gpu_model_runner.py`

每个调度 step 里，`execute_model()` 根据 scheduler 输出决定这轮 forward 的 CUDA Graph 模式。

```
GPUModelRunner.execute_model(scheduler_output)
  │
  ├─ _update_states(scheduler_output)
  ├─ _prepare_inputs(...)
  ├─ 计算：
  │    ├─ num_reqs
  │    ├─ num_tokens_unpadded
  │    ├─ num_scheduled_tokens_np
  │    └─ max_num_scheduled_tokens
  │
  ├─ _determine_batch_execution_and_padding(...)
  │    ├─ _is_uniform_decode(...)
  │    ├─ 计算 has_lora / num_active_loras
  │    ├─ _pad_for_sequence_parallelism(num_tokens)
  │    ├─ cudagraph_dispatcher.dispatch(...)
  │    ├─ DP 场景 coordinate_batch_across_dp(...)
  │    └─ 返回 cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp
  │
  ├─ num_tokens_padded = batch_desc.num_tokens
  ├─ num_reqs_padded = batch_desc.num_reqs or num_reqs
  ├─ _get_slot_mappings(...)
  ├─ _build_attention_metadata(...)
  ├─ _preprocess(..., num_tokens_padded, ...)
  │
  └─ with set_forward_context(...):
       └─ self._model_forward(...)
```

### Dispatch 决策调用栈

**文件**: `vllm/vllm/v1/cudagraph_dispatcher.py`

```
CudagraphDispatcher.dispatch(
    num_tokens,
    uniform_decode,
    has_lora,
    num_active_loras,
    valid_modes,
    invalid_modes,
)
  │
  ├─ allowed_modes = valid_modes or {NONE, PIECEWISE, FULL}
  ├─ allowed_modes -= invalid_modes
  │
  ├─ 如果未初始化 / mode=NONE / num_tokens > max_size / 只允许 NONE:
  │    └─ return NONE, BatchDescriptor(num_tokens)
  │
  ├─ 归一化 LoRA count
  │    ├─ specialize_active_lora=True:
  │    │    └─ 向上取到已 capture 的 active LoRA 数
  │    └─ cudagraph_specialize_lora=False:
  │         └─ 使用 max_loras + 1 的通用 LoRA graph
  │
  ├─ normalized_uniform = uniform_decode and cudagraph_mode.separate_routine()
  ├─ batch_desc = _create_padded_batch_descriptor(...)
  │    ├─ num_tokens → _bs_to_padded_graph_size[num_tokens]
  │    ├─ FULL uniform decode: num_reqs = padded_tokens // decode_query_len
  │    └─ mixed / PIECEWISE: num_reqs = min(padded_tokens, max_num_seqs)
  │
  ├─ 如果 FULL 被允许且 batch_desc 在 cudagraph_keys[FULL]:
  │    └─ return FULL, batch_desc
  │
  ├─ 如果 PIECEWISE 被允许:
  │    ├─ batch_desc_to_check = replace(batch_desc, num_reqs=None, uniform=False)
  │    └─ 如果在 cudagraph_keys[PIECEWISE]，return PIECEWISE, relaxed_desc
  │
  └─ return NONE, BatchDescriptor(num_tokens)
```

### 运行时 Dispatch 时序图

```
SchedulerOutput          GPUModelRunner             CudagraphDispatcher       Attention Metadata      Model Forward
      │                         │                            │                       │                    │
      │ scheduled tokens         │                            │                       │                    │
      │────────────────────────>│                            │                       │                    │
      │                         │ _determine_batch_execution │                       │                    │
      │                         │───────────────────────────>│ dispatch()            │                    │
      │                         │                            │ pad token count       │                    │
      │                         │                            │ try FULL key          │                    │
      │                         │                            │ try PIECEWISE key     │                    │
      │                         │<───────────────────────────│ mode + BatchDescriptor│                    │
      │                         │                            │                       │                    │
      │                         │ _get_slot_mappings()       │                       │                    │
      │                         │───────────────────────────────────────────────────>│                    │
      │                         │ _build_attention_metadata() │                       │                    │
      │                         │───────────────────────────────────────────────────>│                    │
      │                         │                            │                       │                    │
      │                         │ set_forward_context(mode, batch_desc, metadata)    │                    │
      │                         │────────────────────────────────────────────────────────────────────────>│
      │                         │                            │                       │                    │
      │                         │<────────────────────────────────────────────────────────────────────────│
```

### 为什么 FULL 会禁用

运行时可能临时禁用 FULL：

- cascade attention 被启用：`invalid_modes={FULL}`。
- encoder-decoder 首步存在 encoder output：FULL 不适合该动态路径。
- `calculate_kv_scales=True`：KV scale 计算包含动态操作，直接改为 `NONE`。
- data parallel 需要跨 rank 协调 token 数，可能重 dispatch 到同步后的 mode。

---

## 8. Stage 7：ForwardContext 驱动 Replay

### ForwardContext 内容

**文件**: `vllm/vllm/forward_context.py`

`set_forward_context()` 会把以下信息绑定到当前 forward：

- `attn_metadata`：每层 attention backend 需要的 metadata。
- `slot_mapping`：KV cache 写入位置。
- `num_tokens`：padding 后 token 数。
- `num_tokens_across_dp`：DP/MoE 场景跨 rank token 数。
- `cudagraph_runtime_mode`：`NONE` / `PIECEWISE` / `FULL`。
- `batch_descriptor`：CUDA Graph key。
- `ubatch_slices`：DBO / ubatching 切片。
- `skip_compiled`：某些动态路径跳过编译模型。

### PIECEWISE Replay 调用栈

PIECEWISE 模式下，model runner 仍然调用普通 `self.model(...)`，只是模型内部的子图 wrapper 会 replay。

```
GPUModelRunner.execute_model()
  │
  └─ with set_forward_context(
       cudagraph_runtime_mode=PIECEWISE,
       batch_descriptor=BatchDescriptor(num_tokens=padded, num_reqs=None, ...),
       attn_metadata=真实 attention metadata,
       slot_mapping=slot_mappings,
     ):
       └─ self._model_forward(...)
            └─ self.model(...)
                 ├─ attention op / KV cache update op
                 │    └─ graph 外 eager 执行
                 │
                 ├─ CUDAGraphWrapper(partition 0)
                 │    ├─ runtime_mode 匹配 PIECEWISE
                 │    └─ graph.replay()
                 │
                 ├─ attention op / KV cache update op
                 │    └─ graph 外 eager 执行
                 │
                 └─ CUDAGraphWrapper(partition N)
                      ├─ graph.replay()
                      └─ 返回最后输出
```

PIECEWISE 的特点是 `num_reqs=None`。这意味着同一个 graph 只按 token 数和 LoRA 状态匹配，
不强制 request 数一致；attention 不在 graph 里，所以它自己的 request/block table 动态性不污染子图。

### FULL Replay 调用栈

主路径里 FULL 也是 `CUDAGraphWrapper` 完成 replay：

```
GPUModelRunner.execute_model()
  │
  └─ with set_forward_context(
       cudagraph_runtime_mode=FULL,
       batch_descriptor=BatchDescriptor(num_tokens=padded, num_reqs=padded_reqs, uniform=True, ...),
       attn_metadata=padding 后 attention metadata,
       slot_mapping=slot_mappings,
     ):
       └─ self._model_forward(...)
            └─ self.model(...)  # 外层是 CUDAGraphWrapper(runtime_mode=FULL)
                 ├─ CUDAGraphWrapper.__call__()
                 ├─ runtime_mode 匹配 FULL
                 ├─ DEBUG 下检查输入地址不变
                 ├─ get_offloader().sync_prev_onload()
                 ├─ graph.replay()
                 └─ return captured output
```

拆分后的 runner 里，FULL 不靠外层 wrapper，而是由 `ModelCudaGraphManager.run_fullgraph()` 显式 replay；
见下一节。

### Replay 时序图

```
GPUModelRunner          ForwardContext             CUDAGraphWrapper          CUDA Graph Pool           Attention Backend
     │                         │                           │                       │                         │
     │ set_forward_context()    │                           │                       │                         │
     │────────────────────────>│                           │                       │                         │
     │ model(...)               │                           │                       │                         │
     │────────────────────────────────────────────────────>│                       │                         │
     │                         │ get_forward_context()      │                       │                         │
     │                         │<──────────────────────────│                       │                         │
     │                         │                           │ mode/key match?       │                         │
     │                         │                           │──────────────────────>│ graph.replay()          │
     │                         │                           │                       │────────────────────────>│ kernels already recorded
     │                         │                           │                       │<────────────────────────│
     │                         │                           │<──────────────────────│ output weak refs/buffers │
     │<────────────────────────────────────────────────────│                       │                         │
```

---

## 9. Stage 8：拆分后 runner 的 ModelCudaGraphManager 路径

拆分后的 runner 位于 `vllm/vllm/v1/worker/gpu/model_runner.py`，它把 CUDA Graph
显式管理逻辑集中到 `vllm/vllm/v1/worker/gpu/cudagraph_utils.py`。

### 核心数据结构

**文件**: `vllm/vllm/v1/worker/gpu/cudagraph_utils.py`

```python
@dataclass(frozen=True)
class BatchExecutionDescriptor:
    cg_mode: CUDAGraphMode
    num_tokens: int
    num_reqs: int | None
    uniform_token_count: int | None = None
```

它比 `BatchDescriptor` 更偏执行层：

- `cg_mode`：这张 graph 是 FULL 还是 PIECEWISE。
- `uniform_token_count`：用于区分纯 decode / spec decode 的固定 query length。
- `num_reqs=None`：PIECEWISE 不需要 request padding。

### 初始化与 Capture

```
gpu/model_runner.py
  │
  ├─ initialize_kv_cache()
  │    ├─ init_attn_backend(...)
  │    ├─ resolve_cudagraph_mode_and_sizes(...)
  │    └─ self.cudagraph_manager = ModelCudaGraphManager(...)
  │
  └─ capture_model()
       └─ self.cudagraph_manager.capture(
            model,
            model_state,
            input_buffers,
            intermediate_tensors,
            block_tables,
            attn_groups,
            kv_cache_config,
          )
```

`ModelCudaGraphManager.capture()` 内部：

```
ModelCudaGraphManager.capture(...)
  │
  └─ CudaGraphManager.capture(create_forward_fn)
       ├─ with graph_capture(device):
       │    ├─ for mode in [PIECEWISE, FULL]:
       │    │    └─ for desc in capture_descs[mode]:
       │    │         ├─ create_forward_fn(desc)
       │    │         │    ├─ 绑定 input_buffers 切片
       │    │         │    ├─ prepare_inputs_to_capture(...)
       │    │         │    └─ 返回 forward_fn(cg_mode)
       │    │         │
       │    │         ├─ forward_fn(NONE)       # warmup
       │    │         ├─ 如果 desc.cg_mode == PIECEWISE:
       │    │         │    └─ forward_fn(PIECEWISE)  # 触发子图 wrapper capture
       │    │         └─ 否则 FULL:
       │    │              ├─ graph = torch.cuda.CUDAGraph()
       │    │              └─ with torch.cuda.graph(graph, pool):
       │    │                   └─ forward_fn(NONE)
       │    └─ 保存 graphs[desc]
       └─ 返回 captured_attn_states
```

### FULL Replay

拆分后 runner 的 FULL replay 不再通过外层 `CUDAGraphWrapper`：

```
gpu/model_runner.py execute_model(...)
  │
  ├─ dispatch_cg_and_sync_dp(...)
  │    └─ batch_desc.cg_mode = FULL / PIECEWISE / NONE
  │
  ├─ prepare_inputs(...)
  ├─ prepare_attn(...)
  │
  ├─ if batch_desc.cg_mode == FULL:
  │    ├─ kv_connector.pre_forward(...)
  │    └─ cudagraph_manager.run_fullgraph(batch_desc)
  │         ├─ get_offloader().sync_prev_onload()
  │         ├─ graphs[desc].replay()
  │         └─ 返回 hidden_states[:desc.num_tokens]
  │
  └─ else:
       └─ with set_forward_context(...):
            └─ model(**model_inputs)  # PIECEWISE 或 eager
```

### 两套 runner 的对应关系

| 职责 | `gpu_model_runner.py` 主路径 | `worker/gpu/model_runner.py` 拆分路径 |
|---|---|---|
| 运行时 key | `BatchDescriptor` | `BatchExecutionDescriptor` |
| dispatch | `CudagraphDispatcher.dispatch()` | `CudaGraphManager.dispatch()` / `dispatch_cg_and_sync_dp()` |
| FULL capture | 外层 `CUDAGraphWrapper` 在 `_dummy_run()` 中 capture | `CudaGraphManager.capture()` 显式 `torch.cuda.graph()` |
| FULL replay | 外层 `CUDAGraphWrapper.__call__()` | `ModelCudaGraphManager.run_fullgraph()` |
| PIECEWISE | 子图 `CUDAGraphWrapper` | 子图 `CUDAGraphWrapper` |
| 输入地址保证 | runner 持久化 input buffers + ForwardContext | `InputBuffers` + `prepare_inputs_to_capture()` |

阅读源码时最容易混的是：`cudagraph_utils.py` 的 `CudaGraphManager` 不是老主路径的 dispatcher；
它属于拆分后的 runner。两者概念相同，但对象和调用位置不同。

---

## 10. Stage 9：多模态 Encoder CUDA Graph

多模态 encoder CUDA Graph 是独立子系统，主要用于 vision encoder。它不使用 `BatchDescriptor`，
而是按 **token budget** 捕获。

### 初始化入口

**文件**: `vllm/vllm/v1/worker/gpu_model_runner.py`

```
GPUModelRunner.capture_model()
  │
  └─ 如果 compilation_config.cudagraph_mm_encoder
       且模型 supports_encoder_cudagraph(raw_model):
       └─ EncoderCudaGraphManager(
            vllm_config,
            device,
            dtype,
            model,
          )
```

### 数据结构

**文件**: `vllm/vllm/v1/worker/encoder_cudagraph.py`

```
BudgetGraphMetadata
  ├─ token_budget
  ├─ max_batch_size
  ├─ max_frames_per_batch
  ├─ graph
  ├─ input_buffer
  ├─ metadata_buffers
  └─ output_buffer
```

模型侧通过 `SupportsEncoderCudaGraph` 协议提供：

- `get_encoder_cudagraph_config()`
- `get_encoder_cudagraph_budget_range()`
- `prepare_encoder_cudagraph_capture_inputs()`
- `prepare_encoder_cudagraph_replay_buffers()`
- `encoder_cudagraph_forward()`
- `encoder_eager_forward()`

### Capture 调用栈

```
EncoderCudaGraphManager.capture()
  │
  └─ for token_budget in token_budgets:
       └─ _capture_budget_graph(token_budget)
            ├─ model.prepare_encoder_cudagraph_capture_inputs(...)
            ├─ output = model.encoder_cudagraph_forward(mm_kwargs, buffers)
            ├─ output_buffer = torch.empty_like(output)
            ├─ graph = torch.cuda.CUDAGraph()
            ├─ with torch.cuda.graph(graph):
            │    ├─ output = model.encoder_cudagraph_forward(mm_kwargs, buffers)
            │    └─ output_buffer.copy_(output)
            └─ budget_graphs[token_budget] = BudgetGraphMetadata(...)
```

### Replay 调用栈

**文件**: `vllm/vllm/v1/worker/gpu_model_runner.py`

```
GPUModelRunner._execute_mm_encoder(...)
  │
  └─ 如果 encoder_cudagraph_manager 支持该 modality:
       └─ encoder_cudagraph_manager.execute(mm_kwargs)
            ├─ _execute_local(mm_kwargs)
            │    ├─ 计算每个 item 的 output token 数
            │    ├─ 按 token 数升序 greedy pack
            │    ├─ 找 smallest fitting token_budget
            │    ├─ model.prepare_encoder_cudagraph_replay_buffers(...)
            │    ├─ _run_budget_graph(batch_mm_kwargs, token_budget, replay_buffers)
            │    │    ├─ copy actual input 到 captured input_buffer
            │    │    ├─ zero + copy metadata buffers
            │    │    ├─ graph.replay()
            │    │    └─ return output_buffer
            │    └─ scatter slices，恢复原始 item 顺序
            │
            └─ 如果 mm_encoder_tp_mode == "data":
                 ├─ _dp_shard(...)
                 ├─ 每个 TP rank 本地 execute
                 └─ _dp_gather(...)
```

### Encoder CUDA Graph 时序图

```
GPUModelRunner       EncoderCudaGraphManager        Model Protocol              CUDA Graph
     │                         │                          │                         │
     │ _execute_mm_encoder()    │                          │                         │
     │────────────────────────>│ execute(mm_kwargs)        │                         │
     │                         │                          │                         │
     │                         │ get per-item token counts │                         │
     │                         │ greedy pack by budget     │                         │
     │                         │─────────────────────────>│ prepare replay buffers  │
     │                         │<─────────────────────────│                         │
     │                         │ copy input/metadata buffers                         │
     │                         │────────────────────────────────────────────────────>│
     │                         │ graph.replay()                                     │
     │                         │────────────────────────────────────────────────────>│
     │                         │<────────────────────────────────────────────────────│ output_buffer
     │<────────────────────────│ restore item order         │                         │
```

---

## 11. 横切关注点

### 11.1 内存池与 capture 顺序

CUDA Graph capture 使用 graph pool：

- `current_platform.get_global_graph_pool()`：主 CUDA Graph wrapper / manager 共享的 pool。
- capture 顺序通常是大尺寸优先、PIECEWISE 先于 FULL，以提升 pool 复用。
- `lock_workspace()` 在 capture 后锁住 workspace，避免运行时 workspace resize 破坏 graph 假设。

### 11.2 持久化输入 buffer

CUDA Graph replay 要求地址不变，所以运行时不是创建新 tensor，而是写入已有 buffer：

- `input_ids`
- `positions`
- `seq_lens`
- `query_start_loc`
- `slot_mapping`
- block tables
- 非首个 PP rank 的 `intermediate_tensors`

拆分后 runner 进一步把这些集中在 `InputBuffers` 和 `prepare_inputs_to_capture()`。

### 11.3 Attention padding

FULL 模式下 attention 也在 graph 里，所以 attention metadata 也必须按 padded 形状构建：

- `num_tokens_padded=batch_desc.num_tokens`
- `num_reqs_padded=batch_desc.num_reqs`
- `query_start_loc` padding 部分填到 `num_tokens`
- `seq_lens` padding 部分为 0
- unused `slot_mapping` 填 -1 / `PAD_SLOT_ID`，避免 KV 写入

PIECEWISE 模式下 attention 多数在 graph 外，通常只需要 token padding 给子图。

### 11.4 LoRA graph specialization

LoRA 会影响 kernel 路径和 active adapter 数：

- `cudagraph_specialize_lora=True`：为无 LoRA、有 LoRA，以及若干 active LoRA count 捕获不同 graph。
- `specialize_active_lora=True`：运行时 active LoRA 数向上取到已 capture 的 count。
- `cudagraph_specialize_lora=False`：用一个 “max_loras + 1” 的通用 LoRA graph 覆盖所有 LoRA case。

### 11.5 Speculative decoding

Spec decode 下 uniform decode 的 query length 变成：

```
uniform_decode_query_len = 1 + num_speculative_tokens
```

影响：

- FULL decode graph 的 `num_tokens` 必须是 `uniform_decode_query_len` 的倍数。
- attention backend 至少要支持 `UNIFORM_BATCH`，否则 FULL decode 会降级。
- EAGLE / draft model 有自己的 CUDA Graph 逻辑，主路径里还会根据主模型 mode 决定 drafter 是否使用 graph。

### 11.6 Offloader 与 copy stream

`CUDAGraphWrapper` 和 `CudaGraphManager` 都会在 capture/replay 前后处理 offloader：

- capture 前：`get_offloader().sync_prev_onload()`
- capture 的 forward 后：`get_offloader().join_after_forward()`
- replay 前：再次 sync，避免前一轮 eager/piecewise 异步 copy 和 graph replay 交叉写同一静态 buffer。

### 11.7 观测指标

如果 `observability_config.cudagraph_metrics=True`，主路径会生成 `CUDAGraphStat`：

- `num_unpadded_tokens`
- `num_padded_tokens`
- `num_paddings`
- `runtime_mode`

这些统计会进入 v1 metrics/logger，便于观察 padding 浪费和 graph 命中模式。

---

## 12. 代码阅读顺序

建议按下面顺序读源码：

1. `vllm/vllm/config/compilation.py`
   - 先读 `CUDAGraphMode`、`CompilationConfig` 的 cudagraph 字段。
   - 再读 `resolve_cudagraph_mode_and_sizes()` 和 `adjust_cudagraph_sizes_for_spec_decode()`。

2. `vllm/vllm/config/vllm.py`
   - 读 `_set_cudagraph_sizes()`，理解 capture sizes 怎么从 scheduler 上限推导出来。

3. `vllm/vllm/forward_context.py`
   - 读 `BatchDescriptor` 和 `ForwardContext`，这是 dispatch 和 wrapper 的桥。

4. `vllm/vllm/v1/cudagraph_dispatcher.py`
   - 读 `initialize_cudagraph_keys()`、`_compute_bs_to_padded_graph_size()`、`dispatch()`。

5. `vllm/vllm/compilation/cuda_graph.py`
   - 读 `CUDAGraphWrapper.__call__()`，理解 capture/replay 的实际行为。

6. `vllm/vllm/compilation/backends.py` 和 `vllm/vllm/compilation/decorators.py`
   - 读 `wrap_with_cudagraph_if_needed()` 与 `maybe_use_cudagraph_partition_wrapper()`。

7. `vllm/vllm/v1/worker/gpu_model_runner.py`
   - 读 `load_model()` 中 FULL wrapper 注入。
   - 读 `capture_model()`、`_capture_cudagraphs()`、`_warmup_and_capture()`、`_dummy_run()`。
   - 读 `execute_model()` 和 `_determine_batch_execution_and_padding()`。

8. `vllm/vllm/v1/attention/backend.py`
   - 读 `AttentionCGSupport`。
   - 再挑具体 backend，如 `flash_attn.py`、`flashinfer.py`，看它们如何声明支持等级。

9. 拆分后 runner：
   - `vllm/vllm/v1/worker/gpu/model_runner.py`
   - `vllm/vllm/v1/worker/gpu/cudagraph_utils.py`
   - 对照 `ModelCudaGraphManager.capture()` 和 `run_fullgraph()` 看 FULL 显式 graph 管理。

10. 多模态 encoder：
    - `vllm/vllm/v1/worker/encoder_cudagraph.py`
    - `vllm/vllm/v1/worker/encoder_cudagraph_defs.py`
    - 具体模型实现可看 `qwen3_vl.py` 的 `SupportsEncoderCudaGraph` 方法。
