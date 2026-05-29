# vLLM V1 执行流程详解：从调度到推理

本文档从 `EngineCore.step()` 出发，完整追踪一个推理 step 的执行流程，包括：
1. **Scheduler.schedule()** — 调度请求，分配 KV cache blocks
2. **GPUModelRunner.execute_model()** — 准备输入，运行模型前向传播
3. **GPUModelRunner.sample_tokens()** — 采样 token，运行 speculative decoding
4. **KV Connector** — PD 分离中逐层 KV cache 传输
5. **Speculative Decoding (MTP)** — draft-then-verify 流程

以一个常规 Transformer 模型（如 Llama）为例，不涉及特定模型架构。

---

## 1. 整体架构概览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        EngineCore (独立进程)                              │
│                                                                          │
│   run_busy_loop()                                                        │
│     while not shutdown:                                                  │
│       _process_input_queue()   ← 接收 API 层的新请求                     │
│       _process_engine_step()   ← 核心 step 循环                          │
│                                                                          │
│   ┌─ step() ──────────────────────────────────────────────────────────┐  │
│   │                                                                    │  │
│   │  1. scheduler.schedule()           → SchedulerOutput              │  │
│   │  2. executor.execute_model()       → Future[ModelRunnerOutput]    │  │
│   │  3. executor.sample_tokens()       → ModelRunnerOutput            │  │
│   │  4. scheduler.update_from_output() → EngineCoreOutputs            │  │
│   │  5. post_step() → draft tokens → scheduler.update_draft_tokens()  │  │
│   │                                                                    │  │
│   └────────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │                              │                         │
    SchedulerOutput              scheduler_output           ModelRunnerOutput
         │                         (via Executor)                   │
         ▼                              ▼                           ▼
┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│   Scheduler     │      │    GPU Worker(s)      │      │    Scheduler     │
│ (调度 + KV管理)  │      │  (模型推理)            │      │  (更新状态)       │
└─────────────────┘      └──────────────────────┘      └──────────────────┘
```

---

## 2. EngineCore.step() — 主循环入口

**文件**: `vllm/v1/engine/core.py` 约 402 行

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    if not self.scheduler.has_requests():
        return {}, False

    # 1. 调度
    scheduler_output = self.scheduler.schedule()

    # 2. 执行模型 (non-blocking, 返回 Future)
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 3. 获取 grammar bitmask (与模型执行并行)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 4. 等待模型执行完成
    model_output = future.result()

    # 5. 如果返回 None，说明 forward 完成但 sampling 被延迟
    #    需要单独调用 sample_tokens (两阶段执行)
    if model_output is None:
        model_output = self.model_executor.sample_tokens(grammar_output)

    # 6. 更新调度器状态
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**两阶段执行设计**:
- `execute_model()` 完成前向传播，计算 logits，但**不采样**。将中间状态存在 `self.execute_model_state`，返回 `None`
- `sample_tokens()` 从缓存状态中采样 token，运行 speculative decoding
- 这样设计是为了让 grammar bitmask 计算与模型前向传播并行

---

## 3. Scheduler.schedule() — 请求调度

**文件**: `vllm/v1/core/sched/scheduler.py` 约 310 行

### 3.1 核心设计理念

vLLM V1 的调度器**没有显式的 prefill/decode 阶段划分**。每个请求只有一个统一的 `num_computed_tokens` 计数器：

- **新请求 (prefill)**: `num_computed_tokens = 0`，需要调度所有 prompt tokens
- **正在运行的请求 (decode)**: `num_computed_tokens > 0`，每次 +1
- **Chunked prefill**: 每次 schedule 只调度一部分 prompt tokens
- **Speculative decoding**: draft tokens 被当作额外的 "lookahead tokens" 一同调度

### 3.2 调度流程

```
schedule()
  │
  ├── 1. 初始化预算
  │     token_budget = max_num_scheduled_tokens
  │     encoder_budget, spec_decode_tokens, etc.
  │
  ├── 2. 调度 RUNNING 请求 (decode + spec)
  │     for req in running (priority order):
  │       num_new_tokens = num_tokens_with_spec - num_computed_tokens
  │       kv_cache_manager.allocate_slots(req, num_new_tokens)
  │       if allocation fails:
  │         preempt lowest-priority request
  │       if req.spec_token_ids:
  │         extract into scheduled_spec_decode_tokens
  │
  ├── 3. 调度 WAITING 请求 (prefill)
  │     for req in waiting + skipped_waiting:
  │       get_computed_blocks(req)     ← 本地 prefix cache
  │       connector.get_num_new_matched_tokens(req)  ← 远程 KV (PD分离)
  │       allocate_slots(req, num_new_tokens)
  │       if PD disaggregation + KV loading needed:
  │         req.status = WAITING_FOR_REMOTE_KVS
  │
  ├── 4. 构建 SchedulerOutput
  │     ├── scheduled_new_reqs: 新请求的完整数据
  │     ├── scheduled_cached_reqs: 已调度请求的增量数据
  │     ├── num_scheduled_tokens: 每个请求的 token 数
  │     ├── scheduled_spec_decode_tokens: draft tokens
  │     ├── kv_connector_metadata: KV connector 元数据
  │     └── finished_req_ids, preempted_req_ids, etc.
  │
  └── 5. 更新请求状态
        advance num_computed_tokens for all scheduled requests
```

### 3.3 SchedulerOutput 结构

**文件**: `vllm/v1/core/sched/output.py`

```python
@dataclass
class SchedulerOutput:
    # 新请求 (首次调度，包含完整数据)
    scheduled_new_reqs: list[NewRequestData]

    # 已调度请求 (增量更新)
    scheduled_cached_reqs: CachedRequestData

    # 每个请求的 token 数量
    num_scheduled_tokens: dict[str, int]
    total_num_scheduled_tokens: int

    # Speculative decoding draft tokens
    scheduled_spec_decode_tokens: dict[str, list[int]]

    # 编码器输入 (多模态)
    scheduled_encoder_inputs: dict[str, list[int]]

    # KV Connector 元数据 (PD 分离)
    kv_connector_metadata: KVConnectorMetadata | None

    # 其他
    finished_req_ids: set[str]
    preempted_req_ids: set[str] | None
    num_common_prefix_blocks: list[int]  # cascade attention
```

### 3.4 PD 分离中的调度逻辑

```
                          Scheduler (Decode 端)
                                │
               ┌────────────────┼────────────────┐
               │                │                │
         WAITING 请求     WAITING 请求      WAITING 请求
        (新到 Decode)    (新到 Decode)    (新到 Decode)
               │                │                │
               ▼                ▼                ▼
    connector.get_num_new_matched_tokens()
               │
               ├── 返回 None ──→ 请求延迟，下一轮再查
               │
               ├── 返回 0 ──→ 无远程 KV，本地 prefill
               │
               └── 返回 N ──→ 远程有 N 个 token 的 KV
                                │
                                ▼
                    WAITING_FOR_REMOTE_KVS
                    (异步 KV 加载中...)
                                │
                          [KV 传输完成]
                                │
                                ▼
                          WAITING → 可被调度
```

当 KV connector 发现远程有可用的 KV cache 时，Decode 端不需要重新计算这些 token 的 KV cache。请求进入 `WAITING_FOR_REMOTE_KVS` 状态，等待异步 KV 传输完成后才被调度。

---

## 4. Executor → Worker → ModelRunner — 执行链路

### 4.1 调用链

```
EngineCore.step()
  │
  ├── model_executor.execute_model(scheduler_output)
  │     │
  │     ▼
  │   Executor.collective_rpc("execute_model", args=(scheduler_output,))
  │     │
  │     ▼
  │   Worker.execute_model(scheduler_output)
  │     │
  │     ▼
  │   GPUModelRunner.execute_model(scheduler_output)
  │     │
  │     ▼
  │   [返回 None，延迟采样]
  │
  ├── model_executor.sample_tokens(grammar_output)
  │     │
  │     ▼
  │   Worker.sample_tokens(grammar_output)
  │     │
  │     ▼
  │   GPUModelRunner.sample_tokens(grammar_output)
  │     │
  │     ▼
  │   [返回 ModelRunnerOutput]
  │
  └── scheduler.update_from_output(scheduler_output, model_output)
```

### 4.2 Executor 层

**文件**: `vllm/v1/executor/abstract.py`

```python
class Executor:
    def execute_model(self, scheduler_output, non_block=False):
        output = self.collective_rpc(
            "execute_model", args=(scheduler_output,), non_block=non_block
        )
        return output[0]  # 返回 driver worker 的结果

    def sample_tokens(self, grammar_output):
        output = self.collective_rpc("sample_tokens", args=(grammar_output,))
        return output[0]
```

`collective_rpc()` 通过 RPC 机制（同进程直接调用 / Ray / MP）调用所有 Worker 的同名方法。对于单卡场景 (`UniProcExecutor`)，直接调用 driver worker。

---

## 5. GPUModelRunner.execute_model() — 模型前向传播

**文件**: `vllm/v1/worker/gpu_model_runner.py` 约 3856 行

### 5.1 类层次

```python
class GPUModelRunner(
    LoRAModelRunnerMixin,          # LoRA 支持
    KVConnectorModelRunnerMixin,   # PD 分离 KV 传输
    ECConnectorModelRunnerMixin,   # Encoder Cache 传输
):
```

### 5.2 执行流程

```
execute_model(scheduler_output)
  │
  ├── 1. 前置处理
  │     ├── 处理 ngram_gpu scheduler output
  │     ├── 处理 KV connector preemptions
  │     └── 如果没有 scheduled tokens: 返回空输出或 no-forward path
  │
  ├── 2. 更新 persistent batch 状态
  │     └── _update_states(scheduler_output)
  │           ├── 处理 finished_req_ids: 清理请求
  │           ├── 处理 scheduled_new_reqs: 添加新请求到 batch
  │           ├── 处理 scheduled_cached_reqs: 更新已有请求
  │           └── 更新 block_tables, token_ids, positions
  │
  ├── 3. 准备输入
  │     └── _prepare_inputs()
  │           ├── 拷贝 block_tables 到 GPU
  │           ├── 计算 positions (from num_computed_tokens + query_pos)
  │           ├── 提取 input_ids (torch.index_select)
  │           ├── 计算 discard_request_mask (prefill 请求不采样)
  │           └── 如果有 spec decode tokens:
  │                 _calc_spec_decode_metadata()
  │                   ├── logits_indices: 哪些位置需要采样
  │                   ├── bonus_logits_indices: bonus token 位置
  │                   ├── target_logits_indices: draft token 位置
  │                   └── draft_token_ids: draft token IDs
  │
  ├── 4. 确定执行模式
  │     └── _determine_batch_execution_and_padding()
  │           ├── 选择 CUDA Graph 模式 (decode) 或 eager 模式 (prefill)
  │           ├── 计算填充 (padding)
  │           └── 可能拆分 micro-batch
  │
  ├── 5. 构建 attention metadata
  │     └── _build_attention_metadata()
  │           ├── 计算 slot_mappings (token → KV cache 位置)
  │           └── 构建 FlashInfer / FlashAttention metadata
  │
  ├── 6. 预处理
  │     └── _preprocess()
  │           └── 准备 input_ids, positions, intermediate_tensors, model_kwargs
  │
  ├── 7. 模型前向传播 (在 context manager 中)
  │     with set_forward_context(...), \
  │          maybe_get_kv_connector_output(...) as kv_output:
  │       │
  │       │  [KV Connector: start_load_kv() ← 异步加载远端 KV]
  │       │
  │       └── _model_forward(input_ids, positions, ...)
  │             │
  │             └── self.model(input_ids=..., positions=..., ...)
  │                   │
  │                   ├── embed_tokens(input_ids) → hidden_states
  │                   │
  │                   └── for layer in layers:
  │                         │
  │                         └── layer.forward(hidden_states, ...)
  │                               │
  │                               ├── input_layernorm(hidden_states)
  │                               │
  │                               ├── self_attn(hidden_states, positions)
  │                               │     │
  │                               │     └── @maybe_transfer_kv_layer 装饰器
  │                               │           │
  │                               │           ├── wait_for_layer_load(layer_name)
  │                               │           │     ← 阻塞等待该层 KV 从远端加载完
  │                               │           │
  │                               │           ├── attention_forward(q, k, v, kv_cache)
  │                               │           │
  │                               │           └── save_kv_layer(layer_name, kv_cache)
  │                               │                 ← 异步保存该层 KV 到远端
  │                               │
  │                               ├── residual = hidden_states + attn_output
  │                               │
  │                               ├── post_attention_layernorm(residual)
  │                               │
  │                               └── mlp(...) → hidden_states = residual + mlp_out
  │
  │       [KV Connector: wait_for_save() ← 等待所有保存完成]
  │
  ├── 8. Logits 计算
  │     ├── hidden_states[logits_indices] → sample_hidden_states
  │     └── model.compute_logits(sample_hidden_states) → logits
  │
  └── 9. 存储 state，返回 None
        self.execute_model_state = (logits, ...)
        return None   ← 延迟到 sample_tokens() 处理
```

### 5.3 持久化 Batch 设计

GPUModelRunner 维护一个**持久化 batch**：

```
InputBatch:
  ├── req_ids[MAX_NUM_REQS]
  ├── token_ids[MAX_NUM_REQS, MAX_NUM_TOKENS]
  ├── block_tables[MAX_NUM_REQS, MAX_NUM_BLOCKS]
  ├── num_computed_tokens[MAX_NUM_REQS]
  ├── sampling_params[MAX_NUM_REQS]
  └── ... (positions, lora_ids, etc.)
```

每个 step 只更新变化的条目（新请求加入、已完成请求清理），不需要每个 step 重新构建整个 batch。这减少了 CPU → GPU 的数据传输。

---

## 6. GPUModelRunner.sample_tokens() — 采样与 Speculative Decoding

**文件**: `vllm/v1/worker/gpu_model_runner.py` 约 4207 行

### 6.1 流程

```
sample_tokens(grammar_output)
  │
  ├── 1. 恢复 execute_model 缓存的状态
  │     logits, hidden_states, sample_hidden_states, ...
  │
  ├── 2. 应用 grammar bitmask (如果有结构化输出)
  │
  ├── 3. 采样 token
  │     └── _sample() 或 rejection_sampler()
  │
  │     [正常 decode]                    [speculative decode]
  │     sampler(logits,                  rejection_sampler(
  │       sampling_metadata)               spec_decode_metadata,
  │     → sampled_token_ids                draft_probs, logits,
  │                                        sampling_metadata)
  │                                     → accepted_tokens + bonus_token
  │
  ├── 4. 更新 batch 状态
  │     └── _update_states_after_model_execute()
  │
  ├── 5. Propose draft tokens (如果启用 spec decode)
  │     └── propose_draft_token_ids()
  │           │
  │           ├── EAGLE: 用 hidden_states + draft model 生成
  │           ├── MTP: 用上一层的 hidden_states + draft layer 生成
  │           ├── Ngram: 基于 n-gram 查表
  │           └── Medusa: 多头预测
  │
  ├── 6. CPU bookkeeping
  │     └── _bookkeeping_sync()
  │           ├── 拷贝 sampled_token_ids 到 CPU
  │           ├── 处理 logprobs
  │           └── 如果是 ngram spec: propose_draft_token_ids()
  │
  ├── 7. Finalize KV connector (延迟的 wait_for_save)
  │     └── finalize_kv_connector()
  │
  └── 8. 构建 ModelRunnerOutput
```

### 6.2 Rejection Sampling (Spec Decode 验证)

**文件**: `vllm/v1/sample/rejection_sampler.py`

```
RejectionSampler.forward(metadata, draft_probs, logits, sampling_metadata)
  │
  ├── 1. 采样 bonus token
  │     bonus_token = sample(logits[bonus_logits_indices])
  │
  ├── 2. 处理 target logits
  │     apply_logits_processors(logits[target_logits_indices])
  │     apply_sampling_constraints(top_k, top_p)
  │
  ├── 3. Rejection sampling
  │     for each draft token:
  │       draft_prob = draft_probs[position]
  │       target_prob = target_probs[position]
  │       r = random()
  │       if r < min(1, target_prob / draft_prob):
  │         accept draft token → continue
  │       else:
  │         reject → sample recovered token from adjusted distribution
  │         break
  │
  └── 4. 输出
        accepted_tokens + [bonus_token 或 recovered_token]
```

### 6.3 Spec Decode 的完整循环

```
Step N: 正常 decode → 生成 base token
  │
  └── propose_draft_token_ids() → [draft_1, draft_2, draft_3]
        │
        └── 存入 request.spec_token_ids
              │
Step N+1: scheduler.schedule()
  │
  ├── 将 spec_token_ids 附加到 input tokens
  │     input = [base_token, draft_1, draft_2, draft_3]
  │     一起送入 target model forward
  │
  └── scheduler_output.scheduled_spec_decode_tokens = {req: [draft_1,2,3]}

Step N+1: model forward
  │
  ├── 计算 target model 对所有 token 的 logits
  │
  └── rejection_sampler 验证:
        draft_1: accept ✓
        draft_2: accept ✓
        draft_3: reject ✗ → sample recovered token R

        输出: [draft_1, draft_2, R] + bonus token

Step N+1: scheduler.update_from_output()
  │
  ├── generated = [draft_1, draft_2, R, bonus]
  ├── num_accepted = len(generated) - 1 = 3
  ├── num_rejected = num_draft - num_accepted = 3 - 3 = 0
  └── request.num_computed_tokens -= num_rejected
```

---

## 7. KV Connector — PD 分离的逐层传输

### 7.1 整体架构

```
┌──────────────────────────┐     ┌──────────────────────────┐
│    Prefill Instance (P)   │     │    Decode Instance (D)    │
│    (KV Producer)          │     │    (KV Consumer)          │
│                           │     │                           │
│  ┌─────────────────────┐  │     │  ┌─────────────────────┐  │
│  │ Scheduler            │  │     │  │ Scheduler            │  │
│  │  build_connector_    │  │     │  │  get_num_new_matched │  │
│  │  meta()              │  │     │  │  _tokens()           │  │
│  └──────────┬──────────┘  │     │  └──────────┬──────────┘  │
│             │              │     │             │              │
│  ┌──────────▼──────────┐  │     │  ┌──────────▼──────────┐  │
│  │ Worker               │  │     │  │ Worker               │  │
│  │  save_kv_layer()     │──┼─RDMA┼─►│  start_load_kv()    │  │
│  │  wait_for_save()     │  │     │  │  wait_for_layer_     │  │
│  └──────────────────────┘  │     │  │    load()            │  │
│                            │     │  └──────────────────────┘  │
└────────────────────────────┘     └────────────────────────────┘
```

### 7.2 Scheduler 侧 (元数据构建)

**时机**: `schedule()` 的最后一步

```python
# scheduler.py 约 891 行
if self.connector is not None:
    scheduler_output.kv_connector_metadata = \
        self.connector.build_connector_meta(scheduler_output)
```

connector 的 `build_connector_meta()` 根据 SchedulerOutput 中的 block IDs 计算哪些 KV blocks 需要传输，生成不透明的元数据对象。

### 7.3 Worker 侧 (数据传输)

**文件**: `vllm/v1/worker/kv_connector_model_runner_mixin.py` 约 58 行

```python
@contextmanager
def _get_kv_connector_output(scheduler_output, defer_finalize=False):
    # ── 进入时 ──
    metadata = scheduler_output.kv_connector_metadata
    kv_connector.bind_connector_metadata(metadata)
    kv_connector.start_load_kv(forward_context)
    #    ↑ 异步开始加载远端 KV cache

    yield kv_connector_output

    # ── 退出时 ──
    if not defer_finalize:
        kv_connector.wait_for_save()
        #    ↑ 等待所有 KV 保存完成
    kv_connector_output.finished_sending = kv_connector.get_finished_sending()
    kv_connector_output.finished_recving = kv_connector.get_finished_recving()
    kv_connector.clear_connector_metadata()
```

**注意**: 当 speculative decoding 启用时，`defer_finalize=True`，KV connector 的 `wait_for_save()` 被延迟到 `sample_tokens()` 中的 `finalize_kv_connector()` 调用。这允许 draft model 的 KV cache 也被保存。

### 7.4 逐层传输装饰器

**文件**: `vllm/model_executor/layers/attention/kv_transfer_utils.py`

```python
@maybe_transfer_kv_layer
def unified_attention_with_output(...):
    # 标准的 attention 计算
```

装饰器的行为：

```python
def wrapper(*args, **kwargs):
    if not has_kv_transfer_group():
        return func(*args, **kwargs)  # 非 PD 分离: 直接执行

    connector = get_kv_transfer_group()

    # 1. 阻塞等待该层 KV 从远端加载完
    connector.wait_for_layer_load(layer_name)

    # 2. 执行正常的 attention forward
    result = func(*args, **kwargs)

    # 3. 异步保存该层 KV 到远端
    connector.save_kv_layer(layer_name, kv_cache, attn_metadata)

    return result
```

### 7.5 PD 分离的完整时序

```
Prefill Worker                          Decode Worker
    │                                        │
    │ [执行模型前向传播]                        │ [等待 KV 传输]
    │                                        │
    │ Layer 0: attention forward              │
    │   ├── wait_for_layer_load() [no-op]     │
    │   ├── compute attention                 │
    │   └── save_kv_layer("layer.0")          │
    │         ↓ async save                    │
    │         ╔══════════════════╗             │
    │         ║ RDMA transfer:   ║─────────────►│ start_load_kv()
    │         ║ Layer 0 KV data  ║             │
    │         ╚══════════════════╝             │
    │                                        │
    │ Layer 1: attention forward              │
    │   ├── wait_for_layer_load() [no-op]     │
    │   ├── compute attention                 │   Layer 0:
    │   └── save_kv_layer("layer.1")          │   wait_for_layer_load("layer.0")
    │         ↓ async save                    │     ← 阻塞直到 Layer 0 KV 就绪
    │         ╔══════════════════╗             │   compute attention with loaded KV
    │         ║ RDMA transfer:   ║─────────────►│
    │         ║ Layer 1 KV data  ║             │
    │         ╚══════════════════╝             │
    │         ...                              │   Layer 1:
    │                                        │   wait_for_layer_load("layer.1")
    │ wait_for_save()                        │     ← 阻塞直到 Layer 1 KV 就绪
    │   ↑ 等待所有层保存完成                     │   compute attention
    │                                        │
    │                                        │ wait_for_save() [D 端 no-op]
    │                                        │
```

**关键特性**: 逐层流水线传输
- P 端计算完第 i 层 → 立即开始传输第 i 层 KV → 同时计算第 i+1 层
- D 端等待第 i 层 KV 就绪 → 使用已加载的 KV 计算 → 等待第 i+1 层

---

## 8. MTP (Multi-Token Prediction) Speculative Decoding 流程

### 8.1 MTP 架构

```
Target Model (主模型)
  │
  ├── Layer 0 ~ Layer N-1: 正常 decoder layers
  │     生成 hidden_states
  │
  └── 输出: hidden_states (最后一个 token)

MTP Predictor
  │
  ├── embed_input_ids(input_ids) → inputs_embeds
  │
  ├── inputs_embeds = enorm(inputs_embeds)
  │   previous_hidden = hnorm(previous_hidden_states)
  │   hidden = eh_proj(cat(inputs_embeds, previous_hidden))
  │
  ├── MTP Block (KDA + MoE):
  │   hidden = mtp_block(positions, hidden)
  │
  └── shared_head.norm(hidden) → logits (或 hidden_states 给下一层)

  如果有多个 MTP 层 (num_nextn_predict_layers > 1):
    每层独立预测下一个 token
    第 i 层的输出作为第 i+1 层的 previous_hidden_states
```

### 8.2 Draft Token 提交流程

**文件**: `vllm/v1/worker/gpu_model_runner.py` 约 4586 行

```
propose_draft_token_ids(sampled_token_ids, hidden_states, ...)
  │
  ├── 获取 drafter (根据 spec_config.method)
  │
  ├── 对于 MTP / Draft Model 类型的 drafter:
  │   │
  │   ├── 构建 drafter 的输入:
  │   │     input_ids = sampled_token_ids  (刚采样的 base token)
  │   │     positions = next_positions
  │   │     previous_hidden_states = hidden_states  (target model 最后一层输出)
  │   │
  │   ├── 运行 drafter forward:
  │   │     for spec_step in range(num_spec_tokens):
  │   │       hidden = mtp_model(input_ids, positions, prev_hidden, spec_step_idx)
  │   │       logits = mtp_model.compute_logits(hidden, spec_step_idx)
  │   │       draft_token = argmax(logits)  或 sample(logits)
  │   │       prev_hidden = hidden
  │   │       input_ids = draft_token
  │   │       positions += 1
  │   │
  │   └── 返回 draft_token_ids = [draft_1, draft_2, ..., draft_K]
  │
  └── 存入 self._draft_token_ids
```

### 8.3 Draft Tokens 从 Worker 到 Scheduler 的传递

```
GPU Worker                                EngineCore
    │                                        │
    │ sample_tokens() 完成                     │
    │ self._draft_token_ids = [d1, d2, d3]    │
    │ _copy_draft_token_ids_to_cpu()           │
    │                                        │
    │◄────────── take_draft_token_ids() ──────┤  (post_step)
    │                                        │
    │───── DraftTokenIds(req_ids, tokens) ───►│
    │                                        │
    │                                  scheduler.update_draft_token_ids()
    │                                        │
    │                                  for each request:
    │                                    request.spec_token_ids = [d1, d2, d3]
    │                                        │
    │                                  [下一个 step]
    │                                        │
    │                                  scheduler.schedule()
    │                                    ├── 将 spec_token_ids 附加到 input
    │                                    └── scheduled_spec_decode_tokens = {req: [d1,d2,d3]}
```

### 8.4 Spec Decode + KV Connector 交互

当 Speculative Decoding 和 PD 分离同时使用时：

```
execute_model():
  │
  ├── defer_kv_connector_finalize = True  ← 因为有 spec decode
  │
  └── with maybe_get_kv_connector_output(..., defer_finalize=True):
        │
        ├── start_load_kv()      ← 开始异步加载 KV
        ├── model forward        ← 主模型前向
        │   └── per-layer: wait_for_layer_load() / save_kv_layer()
        │
        └── [不调用 wait_for_save()，因为 defer_finalize=True]

sample_tokens():
  │
  ├── _sample() → sampled tokens
  ├── propose_draft_token_ids() → draft tokens
  │     └── 如果 drafter 是 draft model:
  │           draft_model.forward() ← draft model 也需要 KV cache
  │
  └── finalize_kv_connector()
        └── wait_for_save()       ← 现在才等待保存完成
            (draft model 的 KV 也一起保存)
```

---

## 9. 完整 Step 时序图

以下是一个包含 PD 分离 + Speculative Decoding 的完整 step 时序图：

```
EngineCore                 Scheduler              Worker (GPUModelRunner)
    │                          │                          │
    │  step()                  │                          │
    │──────────────────────────►│                          │
    │                          │                          │
    │                   schedule()                         │
    │                   ├── RUNNING 请求 (decode)           │
    │                   │   allocate_slots()               │
    │                   │   extract spec_token_ids         │
    │                   ├── WAITING 请求 (prefill)          │
    │                   │   connector.get_num_matched()    │
    │                   ├── build SchedulerOutput          │
    │                   └── connector.build_connector_meta()│
    │                          │                          │
    │◄─── SchedulerOutput ─────┤                          │
    │                          │                          │
    │─── execute_model(sched_out) ────────────────────────►│
    │                          │                          │
    │                   [并行] grammar_bitmask             │  execute_model()
    │                          │                          │  ├── _update_states()
    │                          │                          │  ├── _prepare_inputs()
    │                          │                          │  │   └── spec_decode_metadata
    │                          │                          │  ├── _build_attn_metadata()
    │                          │                          │  │
    │                          │                          │  │  with kv_connector_output:
    │                          │                          │  │   ├── bind_connector_metadata()
    │                          │                          │  │   ├── start_load_kv()
    │                          │                          │  │   │
    │                          │                          │  │   └── _model_forward()
    │                          │                          │  │       │
    │                          │                          │  │       └── model(input_ids)
    │                          │                          │  │            │
    │                          │                          │  │            ├── Layer 0:
    │                          │                          │  │            │   wait_for_layer_load()
    │                          │                          │  │            │   attention_forward()
    │                          │                          │  │            │   save_kv_layer()
    │                          │                          │  │            │
    │                          │                          │  │            ├── Layer 1:
    │                          │                          │  │            │   wait_for_layer_load()
    │                          │                          │  │            │   attention_forward()
    │                          │                          │  │            │   save_kv_layer()
    │                          │                          │  │            │
    │                          │                          │  │            └── ... (all layers)
    │                          │                          │  │
    │                          │                          │  │  [deferred: wait_for_save 被延迟]
    │                          │                          │  │
    │                          │                          │  ├── compute_logits()
    │                          │                          │  └── return None (state stashed)
    │                          │                          │
    │◄─── None ────────────────────────────────────────────┤
    │                          │                          │
    │─── sample_tokens(grammar) ──────────────────────────►│
    │                          │                          │
    │                          │                  sample_tokens()
    │                          │                  ├── apply grammar bitmask
    │                          │                  ├── _sample() or rejection_sampler()
    │                          │                  │   └── [接受/拒绝 draft tokens]
    │                          │                  ├── propose_draft_token_ids()
    │                          │                  │   └── drafter.forward() → drafts
    │                          │                  ├── _bookkeeping_sync()
    │                          │                  ├── finalize_kv_connector()
    │                          │                  │   └── wait_for_save() + clear_metadata
    │                          │                  └── return ModelRunnerOutput
    │                          │                          │
    │◄─── ModelRunnerOutput ──────────────────────────────┤
    │                          │                          │
    │  update_from_output()     │                          │
    │──────────────────────────►│                          │
    │                          │                          │
    │                   ├── process generated tokens       │
    │                   ├── handle spec rejections          │
    │                   ├── adjust num_computed_tokens      │
    │                   ├── process kv_connector_output     │
    │                   └── build EngineCoreOutputs         │
    │                          │                          │
    │◄─── EngineCoreOutputs ───┤                          │
    │                          │                          │
    │  post_step()              │                          │
    │── take_draft_token_ids() ───────────────────────────►│
    │◄── DraftTokenIds ───────────────────────────────────┤
    │                          │                          │
    │── update_draft_token_ids()►│                         │
    │                          │                          │
    │                   request.spec_token_ids = drafts     │
    │                          │                          │
    │  [Step 结束，输出到 API]   │                          │
```

---

## 10. 关键文件索引

| 组件 | 文件路径 |
|------|----------|
| Engine Core 主循环 | `vllm/v1/engine/core.py` |
| Scheduler | `vllm/v1/core/sched/scheduler.py` |
| SchedulerOutput 类型 | `vllm/v1/core/sched/output.py` |
| GPUModelRunner | `vllm/v1/worker/gpu_model_runner.py` |
| Worker | `vllm/v1/worker/gpu_worker.py` |
| Executor | `vllm/v1/executor/abstract.py` |
| KV Connector Mixin | `vllm/v1/worker/kv_connector_model_runner_mixin.py` |
| ActiveKVConnector | `vllm/v1/worker/gpu/kv_connector.py` |
| KV Connector Base | `vllm/distributed/kv_transfer/kv_connector/v1/base.py` |
| 逐层传输装饰器 | `vllm/model_executor/layers/attention/kv_transfer_utils.py` |
| Attention (被装饰) | `vllm/model_executor/layers/attention/attention.py` |
| Rejection Sampler | `vllm/v1/sample/rejection_sampler.py` |
| SpecDecodeMetadata | `vllm/v1/spec_decode/metadata.py` |
| Draft Proposer | `vllm/v1/spec_decode/` (各 proposer 实现) |
| ModelRunnerOutput | `vllm/v1/outputs.py` |
| MambaSpec / KV Cache Interface | `vllm/v1/kv_cache_interface.py` |

---

## 11. 调试指南

### 11.1 定位问题的入口

| 问题类型 | 入口 |
|---------|------|
| 调度问题 (请求不被调度, 预算不足) | `scheduler.schedule()` 加断点 |
| KV cache 分配失败 | `kv_cache_manager.allocate_slots()` |
| 模型前向传播错误 | `_model_forward()` → `self.model()` |
| Attention 计算/shape 错误 | 各 layer 的 `forward()` |
| Spec decode 验证错误 | `rejection_sampler.forward()` |
| Draft token 提交错误 | `propose_draft_token_ids()` |
| KV 传输问题 | `wait_for_layer_load()` / `save_kv_layer()` |
| PD 分离 KV 不匹配 | `connector.build_connector_meta()` |
| CUDA Graph 兼容性 | `_determine_batch_execution_and_padding()` |

### 11.2 关键日志点

- `scheduler.schedule()` 会打印 token budget 使用情况
- `execute_model()` 会打印 batch size, padding, CUDA graph 模式
- `@maybe_transfer_kv_layer` 装饰器在启用 KV transfer 时会打印层名
- Spec decode rejection 比例在 `update_from_output()` 中统计

### 11.3 常见调试场景

**场景: Spec decode rejection rate 过高**
1. 检查 `propose_draft_token_ids()` 的输出是否合理
2. 检查 target model logits 与 draft model logits 的分布差异
3. 在 `rejection_sampler.forward()` 中检查逐 token 的接受概率

**场景: PD 分离 KV 传输失败**
1. 检查 P/D 两端的 `NIXL_CONNECTOR_VERSION` 是否一致
2. 检查 compatibility hash 是否匹配
3. 在 `wait_for_layer_load()` 中检查是否超时
4. 检查 block IDs 是否正确映射

**场景: Hybrid 模型 (含 Mamba/KDA) 的 KV 传输问题**
1. 确认 `VLLM_SSM_CONV_STATE_LAYOUT=DS`
2. 确认 `--no-disable-hybrid-kv-cache-manager` 已设置
3. 检查 MambaSpec 的 shapes 是否 P/D 两端一致
4. 在 `_build_mamba_local()` / `_build_mamba_remote()` 中检查 descriptor 注册
