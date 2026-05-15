# vLLM Continuous Batching 深度解析

> 基于源码的原理分析与代码导读，聚焦 V1 架构。

## 目录

1. [什么是 Continuous Batching](#1-什么是-continuous-batching)
2. [整体架构与进程模型](#2-整体架构与进程模型)
3. [核心循环：schedule → execute → update](#3-核心循环schedule--execute--update)
4. [Scheduler：连续批处理的决策引擎](#4-scheduler连续批处理的决策引擎)
5. [请求的生命周期](#5-请求的生命周期)
6. [KV Cache 与 PagedAttention](#6-kv-cache-与-pagedattention)
7. [Worker 侧的持久化批次](#7-worker-侧的持久化批次)
8. [代码阅读顺序指南](#8-代码阅读顺序指南)

---

## 1. 什么是 Continuous Batching

传统的静态批处理（Static Batching）要求同一批次中的所有请求同时开始、同时结束。批次长度受限于最慢的请求，造成严重的资源浪费。

Continuous Batching 的核心思想是：**每个 scheduling step 都可以动态调整批次组成**——新的请求可以随时加入，完成的请求可以立即退出，不需要等其他请求完成。

### vLLM 的统一视角

vLLM 的 Scheduler 没有区分 "prefill 阶段" 和 "decode 阶段"。源码中的注释（`scheduler.py:349-358`）明确说明：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and num_tokens_with_spec.
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec. This is general enough to cover chunked prefills,
# prefix caching, speculative decoding, and the "jump decoding" optimization.
```

每个 request 只有两个关键计数器：
- `num_computed_tokens`：已经过模型计算的 token 数
- `num_tokens_with_spec`：当前需要计算到的目标 token 数（= prompt + output + spec tokens）

Scheduler 的唯一目标就是：**为每个 request 分配 token budget，让 `num_computed_tokens` 追上 `num_tokens_with_spec`**。

这种设计天然覆盖了：
- **Chunked prefill**：长 prompt 被分成多个 chunk，每步计算一部分
- **Prefix caching**：已有缓存的 token 不需要重复计算
- **Speculative decoding**：draft tokens 被当作额外的 spec tokens 处理
- **Decode**：每步只需要计算 1 个新 token

---

## 2. 整体架构与进程模型

vLLM V1 采用多进程架构：

```
┌─────────────────────────────────────────────────────────────┐
│  Front-end Process (API Server)                             │
│                                                             │
│  AsyncLLM                                                   │
│  ├── add_request() / generate()   ← 接收 API 请求          │
│  ├── InputProcessor               ← 处理输入               │
│  ├── EngineCoreClient              ← ZMQ IPC                │
│  └── _run_output_handler()         ← 异步拉取输出           │
│         │                          并推送到 per-request 队列  │
└─────────┼───────────────────────────────────────────────────┘
          │ ZMQ DEALER/ROUTER
          │ + msgpack 序列化
          │ + tensor 通道（多模态数据）
┌─────────▼───────────────────────────────────────────────────┐
│  Engine Core Process                                         │
│                                                             │
│  EngineCoreProc                                             │
│  └── run_busy_loop()            ← 核心循环                  │
│        ├── _process_input_queue()    ← 接收新请求/abort     │
│        └── _process_engine_step()    ← schedule+execute     │
│                                                             │
│  EngineCore                                                 │
│  ├── scheduler: Scheduler            ← 调度决策              │
│  ├── model_executor                  ← GPU 执行              │
│  └── step()                          ← 单步执行              │
└─────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────┐
│  Worker Process(es) (GPU)                                    │
│                                                             │
│  GPUModelRunner                                             │
│  ├── _update_states()          ← 更新持久化批次状态           │
│  ├── _prepare_inputs()         ← 构建 GPU 输入张量           │
│  └── model forward pass        ← 模型前向推理                │
└─────────────────────────────────────────────────────────────┘
```

### 关键设计：Front-end 永远不会阻塞在模型执行上

`AsyncLLM` 和 `EngineCore` 在**不同进程**中运行，通过 ZMQ 通信。Front-end 在 `_run_output_handler()` 中异步轮询输出，不影响新请求的接收。

---

## 3. 核心循环：schedule → execute → update

### 3.1 Busy Loop

**文件**: `vllm/v1/engine/core.py:1160-1219`

```python
def run_busy_loop(self):
    """Core busy loop of the EngineCore."""
    while self._handle_shutdown():
        # 1) 轮询输入队列，直到有工作要做
        self._process_input_queue()
        # 2) 执行一步引擎核心（schedule + execute + output）
        self._process_engine_step()
```

当没有请求时，`_process_input_queue()` 会阻塞等待。一旦有请求到达，循环进入 `_process_engine_step()`，执行一步完整的 schedule-execute-update。

### 3.2 单步执行 step()

**文件**: `vllm/v1/engine/core.py:404-433`

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    # 如果没有任何请求，直接返回
    if not self.scheduler.has_requests():
        return {}, False

    # 1. 调度：决定这个 step 处理哪些请求、每个请求分配多少 token
    scheduler_output = self.scheduler.schedule()

    # 2. 非阻塞执行模型（返回 future）
    future = self.model_executor.execute_model(scheduler_output, non_block=True)

    # 3. 处理 grammar bitmask（结构化输出）
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)

    # 4. 等待模型执行完成
    model_output = future.result()

    # 5. 处理在模型执行期间到达的 abort 请求
    self._process_aborts_queue()

    # 6. 根据模型输出更新调度器状态（追加 token、检查停止条件、释放资源）
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

这个 pipeline 可以总结为：

```
schedule() → execute_model() → update_from_output()
```

每一步都会产生不同的批次组成——这就是 continuous batching 的核心。

### 3.3 Pipeline Parallelism 的 Batch Queue

**文件**: `vllm/v1/engine/core.py:445-`

对于 Pipeline Parallelism，vLLM 使用 `step_with_batch_queue()` 来维护一个 deque，允许多个 batch 同时在 pipeline 中执行，减少 pipeline bubble。

---

## 4. Scheduler：连续批处理的决策引擎

**文件**: `vllm/v1/core/sched/scheduler.py`

### 4.1 核心数据结构

```python
class Scheduler:
    # 所有活跃请求，按 request_id 索引
    self.requests: dict[str, Request] = {}

    # 等待队列（优先级或 FCFS）
    self.waiting: RequestQueue

    # 因异步依赖被跳过的等待请求
    self.skipped_waiting: RequestQueue

    # 当前正在运行的请求列表
    self.running: list[Request] = []

    # 上一步结束后完成的请求 ID 集合
    self.finished_req_ids: set[str] = set()
```

### 4.2 调度约束

```python
# 最大并发运行请求数（来自 max_num_seqs 配置）
self.max_num_running_reqs = self.scheduler_config.max_num_seqs

# 每 step 的总 token budget（来自 max_num_batched_tokens 配置）
self.max_num_scheduled_tokens = self.scheduler_config.max_num_scheduled_tokens

# 最大序列长度
self.max_model_len = vllm_config.model_config.max_model_len
```

还有 KV cache block 可用性（动态检查）、encoder compute budget（多模态）、LoRA slot 限制等。

### 4.3 schedule() 方法详解

**文件**: `scheduler.py:348-`

schedule() 是 continuous batching 的核心算法，分两个阶段执行。

#### Phase 1：调度 RUNNING 请求（行 383-552）

```python
# token_budget 初始值 = max_num_scheduled_tokens
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]

    # 计算这个 request 需要多少新 token
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )

    # 长预填充分块
    if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = self.scheduler_config.long_prefill_token_threshold

    # 不超过剩余 budget
    num_new_tokens = min(num_new_tokens, token_budget)

    # 尝试分配 KV cache blocks
    new_blocks = self.kv_cache_manager.allocate_slots(
        request, num_new_tokens, ...
    )

    if new_blocks is None:
        # 分配失败！需要抢占（preempt）低优先级请求
        preempted_req = self.running.pop()  # 优先级最低的
        self._preempt_request(preempted_req, ...)
        # 抢占后重试分配
        if preempted_req == request:
            break  # 没有更低优先级的了，放弃这个 request

    # 分配成功，记录
    num_scheduled_tokens[request_id] = num_new_tokens
    token_budget -= num_new_tokens
    req_index += 1
```

**关键机制——抢占（Preemption）**：

当 GPU 内存不足以容纳所有 running requests 时，scheduler 会抢占最低优先级的请求：
1. 从 `self.running` 中移除
2. 调用 `kv_cache_manager.free(request)` 释放其 KV cache blocks
3. 将 request 状态设为 `PREEMPTED`，`num_computed_tokens` 重置为 0
4. 将 request 放回 `self.waiting` 队列前端，等待下次重新调度

#### Phase 2：调度 WAITING 请求（行 563-856）

```python
# 只有在没有抢占发生且未暂停时，才调度新请求
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break  # 达到最大并发数

        request = request_queue.peek_request()

        # 1. 检查 prefix cache 命中
        new_computed_blocks, num_new_local_computed_tokens = (
            self.kv_cache_manager.get_computed_blocks(request)
        )

        # 2. 计算需要的新 token 数
        num_new_tokens = request.num_tokens - num_computed_tokens

        # 3. 长预填充分块
        if 0 < threshold < num_new_tokens:
            num_new_tokens = threshold
        num_new_tokens = min(num_new_tokens, token_budget)

        # 4. 可选：检查是否能容纳完整序列（准入控制）
        if not self.kv_cache_manager.can_fit_full_sequence(request, ...):
            break

        # 5. 分配 KV cache blocks
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens, ...
        )
        if new_blocks is None:
            break  # 内存不足，停止调度新请求

        # 6. 分配成功：移入 running，记录为 new 或 resumed request
        self.running.append(request)
```

**关键准入控制**：
- `can_fit_full_sequence()`：防止 chunked prefill 只检查第一个 chunk 就准入，导致后续 chunk 无内存可用
- `allocate_slots()` 返回 `None` 时停止调度——这是最严格的内存门控

### 4.4 update_from_output() 方法详解

**文件**: `scheduler.py:1299-`

模型执行完后，scheduler 处理输出并更新状态：

```python
def update_from_output(self, scheduler_output, model_runner_output):
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        request = self.requests.get(req_id)
        generated_token_ids = sampled_token_ids[req_index]

        # 处理 speculative decoding 的接受/拒绝
        if scheduled_spec_token_ids and generated_token_ids:
            num_accepted = len(generated_token_ids) - 1
            num_rejected = num_draft_tokens - num_accepted
            request.num_computed_tokens -= num_rejected  # 回退被拒绝的 token

        # 追加生成的 token，检查停止条件
        new_token_ids, stopped = self._update_request_with_output(
            request, new_token_ids
        )

        if stopped:
            finished = self._handle_stopped_request(request)
            if finished:
                self._free_request(request)  # 释放所有资源
            stopped_running_reqs.add(request)

    # 从 running 列表中移除已停止的请求
    self.running = remove_all(self.running, stopped_running_reqs)
```

**`_update_request_with_output()`**（行 1631-1647）：
```python
def _update_request_with_output(self, request, new_token_ids):
    stopped = False
    for num_new, output_token_id in enumerate(new_token_ids, 1):
        request.append_output_token_ids(output_token_id)
        stopped = check_stop(request, self.max_model_len)  # EOS、max_tokens、stop strings
        if stopped:
            del new_token_ids[num_new:]  # 裁剪到停止位置
            break
    return new_token_ids, stopped
```

**`_handle_stopped_request()`**（行 1588-1604）：
- 对于普通请求：返回 `True`，释放所有资源
- 对于可恢复（resumable）请求（如 streaming input）：返回 `False`，将请求放回 waiting 队列

### 4.5 _update_after_schedule()

**文件**: `scheduler.py:983-1007`

在 schedule() 返回前调用，提前推进 `num_computed_tokens`：

```python
def _update_after_schedule(self, scheduler_output):
    for req_id, num_scheduled_token in num_scheduled_tokens.items():
        request = self.requests[req_id]
        # 立即推进 computed tokens（乐观更新）
        request.num_computed_tokens += num_scheduled_token
        # 标记是否仍在 prefill chunk
        request.is_prefill_chunk = request.num_computed_tokens < (
            request.num_tokens + request.num_output_placeholders
        )
    # 清空 finished_req_ids（因为已经包含在 scheduler_output 中了）
    self.finished_req_ids = set()
```

这允许同一个 prefill request 在下一个 scheduling step 立即被再次调度（处理下一个 chunk）。

---

## 5. 请求的生命周期

### 5.1 Request 对象

**文件**: `vllm/v1/request.py:59-`

```python
class Request:
    num_computed_tokens: int       # 已计算的 token 数
    num_prompt_tokens: int         # prompt token 数
    _output_token_ids: list[int]   # 已生成的 output token
    status: RequestStatus          # 请求状态
    is_prefill_chunk: bool         # 是否仍在 prefill 阶段
```

### 5.2 RequestStatus 状态机

**文件**: `vllm/v1/request.py:299-`

```
WAITING ───────────────────────────────→ RUNNING
  ↑                                          │
  │                                    ┌─────┴──────┐
  │                                    │             │
  │                              (正常执行)    (内存不足)
  │                                    │             │
  │                              ┌─────┴─────┐  PREEMPTED
  │                              │           │      │
  │                         (继续生成)    (停止)    │
  │                              │           │      │
  │                              │    FINISHED_STOPPED
  │                              │    FINISHED_LENGTH
  │                              │    FINISHED_ABORTED
  │                              │           │
  │                          (streaming   (完成)
  │                           input 续传)
  │                              │
  └──────────────────────────────┘
```

关键状态：
- `WAITING`：在等待队列中，等待被调度
- `RUNNING`：正在被调度执行
- `PREEMPTED`：被抢占（内存不足时），等待重新调度
- `FINISHED_*`：各种完成状态（正常停止、达到最大长度、被 abort 等）

### 5.3 进入批次

```
API 请求 → AsyncLLM.add_request()                    [async_llm.py:282]
         → EngineCoreClient.add_request_async()       [ZMQ IPC]
         → EngineCoreProc.process_input_sockets()     [core.py:1368]
         → EngineCore.add_request()                   [core.py:317]
         → Scheduler.add_request()                    [scheduler.py:1737]
           → 加入 self.waiting 队列 + self.requests 字典
         → 下一个 schedule() 的 Phase 2 从 waiting 中取出
           → 分配 KV cache → 移入 self.running
```

### 5.4 退出批次

```
模型执行完成 → Scheduler.update_from_output()          [scheduler.py:1299]
            → _update_request_with_output()            [scheduler.py:1631]
              → request.append_output_token_ids(token)
              → check_stop() → 检查 EOS / max_tokens / stop strings
            → 如果 stopped:
              → _handle_stopped_request()              [scheduler.py:1588]
                → 普通请求: _free_request() 释放所有资源
                → 可恢复请求: 放回 waiting 队列
              → 从 self.running 中移除
```

### 5.5 批次组成的动态变化

每个 scheduling step 之间，批次组成都可能变化：

| 事件 | 发生位置 | 效果 |
|------|----------|------|
| 新请求进入 | `schedule()` Phase 2 | 从 waiting 移到 running |
| 请求完成 | `update_from_output()` | 从 running 移除 |
| 请求被抢占 | `schedule()` Phase 1 | 从 running 移回 waiting |
| 请求被 abort | `_process_aborts_queue()` | 立即标记为 FINISHED_ABORTED |

---

## 6. KV Cache 与 PagedAttention

### 6.1 KVCacheManager

**文件**: `vllm/v1/core/kv_cache_manager.py:106-`

KVCacheManager 是 continuous batching 内存管理的核心。它的 `allocate_slots()` 方法是请求能否被调度的**最终决定者**。

```python
class KVCacheManager:
    def allocate_slots(self, request, num_new_tokens, ...) -> KVCacheBlocks | None:
        """为请求分配 KV cache blocks。返回 None 表示内存不足。"""
```

关键方法：

| 方法 | 用途 |
|------|------|
| `get_computed_blocks()` | 查找 prefix cache 命中的 blocks |
| `can_fit_full_sequence()` | 准入控制：检查是否有足够 blocks 容纳完整序列 |
| `allocate_slots()` | 分配新 blocks，返回 None 表示失败 |
| `free()` | 释放请求的所有 blocks |

### 6.2 Block Pool 与前缀缓存

**文件**: `vllm/v1/core/block_pool.py`

Block Pool 管理 GPU 内存的 block 分配：
- 每个 block 存储固定数量（`block_size`）的 token 对应的 KV cache
- 支持**前缀缓存**：通过 hash 查找已计算的 blocks，多个请求可以共享相同的 prompt 前缀
- 使用引用计数管理共享 blocks

### 6.3 PagedAttention 在 Batching 中的作用

PagedAttention 将 KV cache 按 block 管理（类似虚拟内存的分页），使得：
1. **内存不需要连续**：请求的 KV cache 可以分布在任意 blocks 中
2. **动态分配释放**：请求进入时分配 blocks，完成时释放，其他请求立即可用
3. **前缀共享**：不同请求共享相同 prompt 前缀的 blocks
4. **抢占可行**：抢占一个请求只需释放其 blocks，不需要整理内存

这些特性是 continuous batching 的基础——没有 PagedAttention，动态调整批次组成在内存管理上会极其复杂。

---

## 7. Worker 侧的持久化批次

**文件**: `vllm/v1/worker/gpu_model_runner.py:3762-`

GPUModelRunner 维护一个**持久化批次（persistent batch）**，避免每步都重新构建所有状态。

### execute_model() 流程

```python
def execute_model(self, scheduler_output):
    # 1. 更新持久化批次状态（增量更新）
    deferred_state_corrections_fn = self._update_states(scheduler_output)

    # 2. 构建输入张量
    logits_indices, spec_decode_metadata = self._prepare_inputs(
        scheduler_output, num_scheduled_tokens_np
    )

    # 3. 模型前向推理
    ...
```

### _update_states() 的增量更新

```python
def _update_states(self, scheduler_output):
    # 移除已完成的请求
    for req_id in scheduler_output.finished_req_ids:
        self.input_batch.remove_request(req_id)

    # 移除被抢占/未调度的请求
    for req_id in unscheduled_req_ids:
        self.input_batch.remove_request(req_id)

    # 添加新请求（带完整状态）
    for new_req_data in scheduler_output.scheduled_new_reqs:
        self.input_batch.add_request(new_req_data)

    # 更新已有请求的 block tables
    for cached_req_data in scheduler_output.scheduled_cached_reqs:
        self.input_batch.update_block_table(cached_req_data)
```

### SchedulerOutput 的新/缓存分离

**文件**: `vllm/v1/core/sched/output.py`

`SchedulerOutput` 将请求分为两类：

```python
@dataclass
class SchedulerOutput:
    scheduled_new_reqs: list[NewRequestData]      # 首次调度的请求，发送完整状态
    scheduled_cached_reqs: CachedRequestData       # 之前已见过的请求，只发送 diff
    num_scheduled_tokens: dict[str, int]           # 每个请求的 token 数
    finished_req_ids: set[str]                     # 已完成的请求 ID
    preempted_req_ids: set[str]                    # 被抢占的请求 ID
```

这是一个关键优化：因为 Worker 缓存了请求状态，新请求才需要发送完整数据（prompt tokens、block IDs、sampling params），已缓存请求只需要 ID 和新的 block IDs。

---

## 8. 代码阅读顺序指南

### 第一阶段：理解数据流（自顶向下）

1. **`vllm/v1/engine/async_llm.py`** — 从 API 层开始
   - `generate()` (行 523)：异步生成器的入口，理解请求如何提交
   - `add_request()` (行 282)：创建 EngineCoreRequest 并通过 IPC 发送
   - `_run_output_handler()` (行 634)：异步轮询输出并推送到 per-request 队列

2. **`vllm/v1/engine/core_client.py`** — IPC 层
   - `EngineCoreClient` (行 69)：抽象基类
   - `AsyncMPClient`：AsyncLLM 使用此实现，理解 ZMQ 通信

3. **`vllm/v1/engine/core.py`** — 核心引擎
   - `EngineCoreProc.run_busy_loop()` (行 1160)：主循环入口
   - `_process_input_queue()` (行 1170)：接收新请求
   - `_process_engine_step()` (行 1201)：调用 step()
   - `EngineCore.step()` (行 404)：schedule → execute → update 的完整 pipeline

### 第二阶段：深入 Scheduler（核心）

4. **`vllm/v1/request.py`** — 理解 Request 对象
   - `Request` 类 (行 59)：关注 `num_computed_tokens`、`num_tokens_with_spec`、`status`
   - `RequestStatus` (行 299)：状态机枚举

5. **`vllm/v1/core/sched/scheduler.py`** — 最核心的文件，建议逐行阅读
   - `__init__()` (行 68)：理解所有数据结构和约束
   - `schedule()` (行 348)：**最重要的方法**
     - Phase 1 (行 383-552)：调度 RUNNING 请求 + 抢占逻辑
     - Phase 2 (行 563-856)：调度 WAITING 请求 + 准入控制
   - `_update_after_schedule()` (行 983)：推进 num_computed_tokens
   - `_preempt_request()` (行 961)：抢占逻辑
   - `update_from_output()` (行 1299)：处理模型输出
   - `_update_request_with_output()` (行 1631)：追加 token + 检查停止
   - `_handle_stopped_request()` (行 1588)：停止后的处理
   - `_free_request()`：释放所有资源

6. **`vllm/v1/core/sched/request_queue.py`** — 请求队列
   - `FCFSRequestQueue` (行 75)：先来先服务
   - `PriorityRequestQueue` (行 131)：优先级队列

7. **`vllm/v1/core/sched/output.py`** — 调度输出类型
   - `SchedulerOutput` (行 179)：理解 new/cached 分离设计
   - `NewRequestData` (行 31)
   - `CachedRequestData` (行 110)

### 第三阶段：KV Cache 管理

8. **`vllm/v1/core/kv_cache_manager.py`** — KV cache 管理器
   - `get_computed_blocks()` (行 176)：前缀缓存查找
   - `can_fit_full_sequence()` (行 218)：准入控制
   - `allocate_slots()` (行 257)：**内存分配的关键门控**
   - `free()` (行 429)：释放 blocks

9. **`vllm/v1/core/block_pool.py`** — Block 池管理
   - 理解 block 分配、释放、前缀缓存查找

### 第四阶段：Worker 侧执行

10. **`vllm/v1/worker/gpu_model_runner.py`** — GPU 模型执行
    - `execute_model()` (行 3762)：整体流程
    - `_update_states()` (行 3808)：持久化批次增量更新
    - `_prepare_inputs()` (行 3850)：构建 GPU 输入张量

11. **`vllm/v1/worker/gpu/input_batch.py`** — 持久化批次
    - `InputBatch`：理解 GPU 侧如何维护跨 step 的请求状态

### 第五阶段：进阶主题

12. **`vllm/v1/core/sched/async_scheduler.py`** — 异步调度（speculative decoding 使用）
    - 理解 `num_output_placeholders` 如何用于异步调度

13. **`vllm/v1/engine/core.py`** — Pipeline Parallelism
    - `step_with_batch_queue()` (行 445)：理解 batch queue 如何减少 pipeline bubble

---

## 附录：核心数据流图

```
API Request
    │
    ▼
AsyncLLM.generate()                    [Front-end 进程]
    ├── InputProcessor.process_inputs()
    ├── EngineCoreClient.add_request_async()  ──ZMQ──►
    │
    │                                         EngineCoreProc.run_busy_loop()  [Core 进程]
    │                                           ├── _process_input_queue()
    │                                           │     └── scheduler.add_request()
    │                                           │           → 加入 waiting 队列
    │                                           │
    │                                           └── _process_engine_step()
    │                                                 │
    │                                                 ▼
    │                                           EngineCore.step()
    │                                                 │
    │                                    ┌────────────┼────────────┐
    │                                    ▼            ▼            ▼
    │                              scheduler     executor     scheduler
    │                              .schedule()  .execute()  .update_from_output()
    │                                    │            │            │
    │                              ┌─────┴──────┐     │     ┌─────┴──────────┐
    │                              │            │     │     │                │
    │                         Phase 1:      Phase 2:   │  追加 token     释放资源
    │                       RUNNING reqs  WAITING reqs  │  检查停止       完成请求
    │                              │            │      │                │
    │                        可能抢占      可能准入     │                │
    │                        (内存不足)    (有空闲)     │                │
    │                              │            │      │                │
    │                              └─────┬──────┘      │                │
    │                                    │             │                │
    │                                    ▼             ▼                │
    │                              SchedulerOutput  ModelOutput          │
    │                                    │             │                │
    │                                    └──────┬──────┘                │
    │                                           │                       │
    │                                           ▼                       │
    │                                    GPUModelRunner                  │
    │                                    .execute_model()                │
    │                                    ├── _update_states()  ◄─────────┘
    │                                    │   (增量更新持久化批次)
    │                                    ├── _prepare_inputs()
    │                                    │   (构建 GPU 输入张量)
    │                                    └── model forward pass
    │                                           │
    │                                           ▼
    │                                    ModelRunnerOutput
    │                                           │
    │                                           ▼
    │                                    scheduler.update_from_output()
    │                                    ├── 追加生成 token
    │                                    ├── check_stop() → EOS/max_len
    │                                    ├── _handle_stopped_request()
    │                                    │   → _free_request() 释放 KV cache
    │                                    └── 从 running 移除
    │                                           │
    │                                           ▼
    │                                    EngineCoreOutputs
    │                                           │
    │                              ◄──ZMQ───────┘
    │
    ▼
_run_output_handler()                  [Front-end 进程]
    ├── OutputProcessor.process_outputs()
    └── 推送到 per-request 队列
        │
        ▼
generate() yields RequestOutput
```

---

## 关键概念速查表

| 概念 | 源码位置 | 说明 |
|------|----------|------|
| 核心循环 | `core.py:1160` run_busy_loop | schedule → execute → update 无限循环 |
| 单步执行 | `core.py:404` step | 一个完整的 schedule-execute-update 周期 |
| 调度算法 | `scheduler.py:348` schedule | Phase 1 调度 RUNNING，Phase 2 调度 WAITING |
| Token Budget | `scheduler.py:367` | 每 step 的总 token 预算，在 RUNNING 和 WAITING 间分配 |
| 抢占 | `scheduler.py:961` _preempt_request | 内存不足时抢占低优先级请求 |
| 准入控制 | `scheduler.py:740-752` | can_fit_full_sequence + allocate_slots 双重检查 |
| KV Cache 分配 | `kv_cache_manager.py:257` allocate_slots | 请求能否进入批次的最终决定 |
| 前缀缓存 | `kv_cache_manager.py:176` get_computed_blocks | 复用已计算的 KV cache blocks |
| 输出处理 | `scheduler.py:1299` update_from_output | 追加 token、检查停止、释放资源 |
| 持久化批次 | `gpu_model_runner.py:3808` _update_states | Worker 侧的增量状态更新 |
| 请求状态机 | `request.py:299` RequestStatus | WAITING → RUNNING → FINISHED/PREEMPTED |
