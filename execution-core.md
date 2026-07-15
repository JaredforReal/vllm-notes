# vLLM V1 执行内核：一轮 step 如何产出下一个 token

> 本文是 [`request.md`](./request.md) 的深入篇。`request.md` 画的是一个 request 从 HTTP 到返回的全旅程；
> 本文则放大 `EngineCore.step()` 内部的四段——**run busy loop → schedule（含 KV cache 规划与分配）→
> model executor / 模型运行 → sampling → spec decode**——把调度、KV cache、forward、采样、推测解码
> 的调用栈、数据对象和算法细节讲透。
>
> 核对基准：`vllm` 仓库 tag **`v0.25.1`**（commit `752a3a504`）。`file:line` 基于该版本核对，可直接点开
> （3.3.14 例外——它标注了基于 `glm5_next` 本地工作树，因为其中的 mamba 均衡分组 + 多层组内存核算是
> 尚未进入 v0.25.1 的本地改动）。默认路径 `VLLM_USE_V2_MODEL_RUNNER=0`
> （走 `vllm/v1/worker/gpu_model_runner.py`）。

## 0. 先跑通一次普通 decode（建议 5 分钟）

这不是一篇按文件目录展开的源码索引，而是一条**从一个请求到下一个 token**的因果链。先把下面的
“无 spec、无多模态、单个正在 decode 的请求”跑通；KV cache、grammar、异步和 spec decode 都只是
在这条链上的局部变化。

```text
① Request 在 waiting / running 队列中
        │
② schedule() 决定：本轮算哪个请求、算几个 token、占哪些 KV block
        │                         └── SchedulerOutput
③ execute_model() 依 SchedulerOutput 准备 GPU 输入并做 forward
        │                         └── logits 暂留 GPU
④ sample_tokens() 在 GPU 上把 logits 变成 token
        │                         └── ModelRunnerOutput
⑤ update_from_output() 把 token 提交给 Request，释放已结束请求的资源
        └── 未结束就回到 ②；下一轮 decode 通常再算 1 个 token
```

### 用四个问题读完整篇

| 问题 | 先看哪一章 | 得到什么答案 |
|---|---|---|
| 引擎什么时候阻塞、什么时候连续跑 step？ | 第 1、2 节 | 输入线程、主线程、输出线程如何协作 |
| 为什么这个请求能在这一轮运行？ | 第 3 节 | token 预算、RUNNING 优先、KV 准入与抢占 |
| GPU 到底执行了什么，CPU 等在哪里？ | 第 4、5 节 | `SchedulerOutput → logits → ModelRunnerOutput`，及唯一同步点 |
| draft token 为什么能复用普通 decode 的路径？ | 第 6 节 | 多调度、一次验证、接受或回滚 |

### 四个不变量

| 字段 | 可以把它理解为 | 阅读时观察它何时变化 |
|---|---|---|
| `num_computed_tokens` | 已被 GPU 计算、可作为 KV 上下文的位置 | 提交输出时增加；draft 被拒时回退 |
| `num_tokens_with_spec` | prompt + 已输出 + 待验证 draft 的总长度 | scheduler 计算工作量的基准 |
| `num_new_tokens` | **这一步实际送入模型的 token 数** | 受 token budget / prefill 阈值裁剪 |
| `num_output_placeholders` | 异步路径预留的输出位置 | 保持 CPU 计账与 GPU 执行一致 |

> **第一次阅读的捷径**：第 1 → 2 → 3（只读主流程）→ 4 → 5 → 6。第 3 节里的“KV cache
> 类型、MLA、Mamba、Hybrid”和第 4 节里的多模态都是进阶分支，先跳过不会影响主线理解。

## 目录

0. [先跑通一次普通 decode](#0-先跑通一次普通-decode建议-5-分钟)
1. [谁在驱动闭环：全局视角](#1-谁在驱动闭环全局视角)
2. [控制循环：run_busy_loop 与 step](#2-控制循环run_busy_loop-与-step)
3. [决定本轮工作：schedule 与 KV cache](#3-决定本轮工作schedule-与-kv-cache)
4. [执行本轮工作：model executor 与 GPU](#4-执行本轮工作model-executor-与-gpu)
5. [把 logits 变成 token：sampling](#5-把-logits-变成-tokensampling)
6. [同一闭环的变体：spec decode](#6-同一闭环的变体spec-decode)
7. [回查：完整时序与数据对象流转](#7-回查完整时序与数据对象流转)
8. [回查：代码阅读顺序](#8-回查代码阅读顺序)

---

## 1. 谁在驱动闭环：全局视角

`request.md` 已经给出 step() 的四阶段总览。本文不再重复 HTTP/API 层，而是从
`EngineCoreProc.run_busy_loop()` 进入，沿着下面这条主干一路下到 GPU 和 spec decode：

```
EngineCoreProc.run_busy_loop()            v1/engine/core.py:1259
  ├─ _process_input_queue()               core.py:1269   取 ADD/ABORT/UTILITY，idle 时阻塞
  └─ _process_engine_step()               core.py:1300
       └─ EngineCore.step()               core.py:479
            ├─ Phase 1: Scheduler.schedule()              v1/core/sched/scheduler.py:388
            │     └─ KVCacheManager.allocate_slots()      v1/core/kv_cache_manager.py:244
            │           └─ BlockPool / Coordinator         v1/core/block_pool.py
            ├─ Phase 2: model_executor.execute_model(non_block=True)   core.py:491
            │     └─ … → GPUModelRunner.execute_model()   v1/worker/gpu_model_runner.py:4056
            │           └─ model forward → logits 暂存到 execute_model_state，返回 None
            ├─        Scheduler.get_grammar_bitmask()     scheduler.py:1440   ← 与 GPU forward 并行
            ├─        future.result()                     core.py:497
            ├─ Phase 3: model_executor.sample_tokens()    core.py:499
            │     └─ GPUModelRunner.sample_tokens()       gpu_model_runner.py:4435
            │           ├─ apply_grammar_bitmask()        v1/structured_output/utils.py:85
            │           ├─ Sampler.forward()              v1/sample/sampler.py:72
            │           ├─ _bookkeeping_sync()            gpu_model_runner.py:3613   ← 唯一 GPU→CPU 同步
            │           └─ propose_draft_token_ids()      gpu_model_runner.py:4864   ← spec decode propose
            └─ Phase 4: Scheduler.update_from_output()    scheduler.py:1464
                  └─ spec decode accept/reject + 回退 num_computed_tokens
```

**三个关键设计决策**（后文反复出现，先记下来）：

1. **execute_model 与 sample_tokens 被刻意拆成两次调用**：`execute_model()` 只做 forward 并把 logits 暂存
   到 `self.execute_model_state`，返回 `None`（`gpu_model_runner.py:4417`）；`sample_tokens()` 再取出采样。
   这让 EngineCore 能在 GPU 跑 forward 的同时用 CPU 算 grammar bitmask（`core.py:491-492`）。
2. **采样全程在 GPU 完成**，唯一的 GPU→CPU 同步是 `_bookkeeping_sync()` 把 `sampled_token_ids` 拷成
   Python list（`gpu_model_runner.py:3613` / `_to_list` `:7504`）。top-k/top-p 用 FlashInfer 或
   Gumbel-max，刻意回避 `torch.multinomial`（会触发同步）。
3. **spec decode 复用同一套 token 记账**：draft token 不是特殊路径，而是被计入
   `num_tokens_with_spec = len(prompt) + len(output) + len(spec_token_ids)`（`scheduler.py:390-399`），
   于是 chunked prefill / prefix cache / spec decode / jump decoding 共用同一套调度逻辑。

---

## 2. 控制循环：run_busy_loop 与 step

### 2.1 进程 / 线程模型

多进程服务时，`EngineCoreProc`（`v1/engine/core.py:896`）在独立进程里运行。它持有三个队列和两个
socket 线程，把 **ZMQ IO、msgpack decode、多模态 tensor IPC、grammar 初始化** 等“输入侧重活”和
**主调度循环**解耦：

```
EngineCoreProc 进程  (v1/engine/core.py:896)
│
├── input_queue   : Queue[(EngineCoreRequestType, Any)]   core.py:915
├── output_queue  : Queue[(client_index, EngineCoreOutputs)]  core.py:916
├── aborts_queue  : Queue[list[str]]                      (EngineCore 基类 core.py:226)
│
├── [线程] process_input_sockets()    core.py:1484
│     ZMQ DEALER ← 前端 ROUTER
│     ├─ ADD:    msgpack decode EngineCoreRequest + preprocess_add_request()
│     │          (多模态 tensor IPC 还原、Request.from_engine_core_request、grammar_init)
│     │          → input_queue.put((ADD, (Request, wave)))
│     ├─ ABORT:  generic decode request_ids
│     │          → aborts_queue.put(ids)   ← 让 step 之间能“急切”处理 abort
│     │          → input_queue.put((ABORT, ids))   ← 同时入 input_queue 保证顺序
│     └─ UTILITY / WAKEUP → input_queue
│
├── [线程] process_output_sockets()   core.py:1589
│     ZMQ PUSH → 前端 PULL
│     ├─ output_queue.get()  阻塞等待
│     ├─ msgpack encode EngineCoreOutputs（zero-copy 抽取 tensor backing buffer）
│     ├─ 遇 ENGINE_CORE_DEAD 哨兵 → 广播后退出
│     └─ MessageTracker 回收 reuse_buffers，避免 tensor 被过早释放
│
└── [主线程] run_busy_loop()          core.py:1259
      while _handle_shutdown():
          _process_input_queue()     ← 取输入、idle 时阻塞
          _process_engine_step()     ← 跑一次 step()，产出 output
```

`process_input_sockets()` 的核心循环在 `core.py:1556-1587`：`poller.poll()` 拿到就绪 socket 后
`recv_multipart(copy=False)`，按 `EngineCoreRequestType` 分发。ADD 走 `add_request_decoder`
（带 `oob_tensor_provider=self.tensor_ipc_receiver`，用于多模态 tensor 的零拷贝跨进程还原），
再调 `preprocess_add_request()`。ABORT 同时写两个队列——`aborts_queue` 让模型执行期间也能尽早
记录 abort，`input_queue` 保证顺序不丢请求（`core.py:1579-1584` 注释明确说“aborting in the
scheduler is idempotent”）。

> **DP 注意**：`DPEngineCoreProc(EngineCoreProc)`（`core.py:1745`）为数据并行重写了 `run_busy_loop`
> 和 `_handle_client_request`，多了一条和 DP coordinator 的 XSUB 协调通道（`core.py:1508-1521`）。
> 单 DP 读代码可先忽略，把它当成普通 `EngineCoreProc`。

### 2.2 run_busy_loop 本体

**文件**: `vllm/v1/engine/core.py:1259`

```python
def run_busy_loop(self):
    while self._handle_shutdown():          # core.py:1324  关闭状态机
        self._process_input_queue()         # core.py:1269  取输入 / idle 阻塞
        self._process_engine_step()         # core.py:1300  跑 step、产出 output
    raise SystemExit
```

就两步循环。关键在于“什么时候阻塞、什么时候连续 step”。

#### _process_input_queue() — core.py:1269

```
_process_input_queue()
  │
  ├─ while not has_work() and is_running():        ← 没活且没停机时
  │     ├─ _notify_idle_state_callbacks()          ← 通知“引擎空闲”回调
  │     ├─ if input_queue.empty():
  │     │     清空 aborts_queue（此时没有在跑的请求，abort 直接丢弃也无害）
  │     │     [DEBUG] "EngineCore waiting for work."
  │     ├─ block = self.process_input_queue_block   ← True：长轮询阻塞；False：立即返回
  │     └─ req = input_queue.get(block=block)
  │           └─ _handle_client_request(*req)       ← ADD/ABORT/UTILITY/WAKEUP 分发
  │
  └─ while not input_queue.empty():                ← 有活后一次性 drain 当前输入
        req = input_queue.get_nowait()
        _handle_client_request(*req)
```

`has_work()` = scheduler 有未完成请求 **或** input_queue 非空。所以循环语义是：

- **空闲时**：阻塞在 `input_queue.get(block=True)`，等第一个请求唤醒。
- **被唤醒后**：`has_work()` 变真，跳出 while，进入 drain 循环把当前积压的输入全部处理掉，
  然后**立刻**进入 `_process_engine_step()`——不等下一个输入。

#### _process_engine_step() — core.py:1300

```
_process_engine_step()
  ├─ outputs, model_executed = self.step_fn()       ← step_fn 绑定 step() 或 step_with_batch_queue()
  ├─ for (client_index, eco) in outputs.items():
  │     output_queue.put_nowait((client_index, eco)) ← 交给 output socket 线程
  ├─ self.post_step(model_executed)                  ← core.py:510  拉 draft token（同步路径）
  └─ if not model_executed and scheduler.has_requests():
        time.sleep(0.001)                            ← 让出 GIL，给后台 KV 传输线程机会
```

`step_fn` 在 `core.py:221-223` 绑定：`max_concurrent_batches > 1`（pipeline parallel）时绑
`step_with_batch_queue`，否则绑 `step`。末尾的 `time.sleep(0.001)` 很关键：当本步没有真正跑模型
（例如 `WAITING_FOR_REMOTE_KV` 状态、或延迟 KV connector free），主动让出 GIL，避免空转饿死
后台传输线程（`core.py:1311-1315` 注释）。

### 2.3 _handle_client_request() — core.py:1372

```
_handle_client_request(request_type, request)
  ├─ WAKEUP  → 直接 return（用于唤醒阻塞的 busy loop，本身无负载）
  ├─ ADD     → _reject_add_in_shutdown() ? return : add_request(req, wave)
  │                                  └─ Scheduler.add_request() → 进 waiting 队列
  ├─ ABORT   → abort_requests(request)         ← 立即处理（scheduler abort 幂等）
  ├─ UTILITY → client_idx, call_id, method_name, args = request
  │            output = UtilityOutput(call_id)
  │            get_result  = lambda: getattr(self, method_name)(*args)
  │            enqueue_output = lambda out: output_queue.put_nowait(
  │                                  (client_idx, EngineCoreOutputs(utility_output=out)))
  │            _invoke_utility_method(...)        ← profile / reset cache / sleep / LoRA / collective RPC
  └─ EXECUTOR_FAILED → raise RuntimeError
```

`UTILITY` 是控制面 RPC（profile、reset prefix cache、sleep/wake、LoRA 加载、collective RPC），
返回值走 `EngineCoreOutputs.utility_output`，前端用 `call_id` 唤醒等待中的 future，不进普通
request 输出路径。

### 2.4 EngineCore.step() — 四阶段

**文件**: `vllm/v1/engine/core.py:479`

```python
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:
    if not self.scheduler.has_requests():          # core.py:488
        return {}, False
    scheduler_output = self.scheduler.schedule(self._should_throttle_prefills())   # 490  Phase 1
    future = self.model_executor.execute_model(scheduler_output, non_block=True)    # 491  Phase 2 (非阻塞)
    grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)           # 492  与 GPU 并行
    with self.log_error_detail(...), self.log_iteration_details(...):
        model_output = future.result()             # 497  等 GPU forward 完成
        if model_output is None:                   # 498  execute_model 暂存了 logits
            model_output = self.model_executor.sample_tokens(grammar_output)        # 499  Phase 3
    self._process_aborts_queue()                   # 503  step 之间急切处理 abort
    engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)  # 504  Phase 4
    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0     # 508
```

`post_step()`（`core.py:510`）在 `_process_engine_step` 里 step 之后调用：**同步路径**下用它把
worker 产出的 draft token 拉回 scheduler（`take_draft_token_ids()` → `update_draft_token_ids()`）；
异步路径下 draft token 直接在 worker 内更新，这里跳过（`core.py:514`）。

`step_with_batch_queue()`（`core.py:519`）是 pipeline parallel 的变体：用 `batch_queue` 把
多步的 `(future, scheduler_output, exec_future)` 排成流水线，允许 `execute_model` 和
`sample_tokens` 都以 `non_block=True` 发出，实现 PP stage 间的重叠。结构同构，本文主线以 `step()` 为准。

### 2.5 单步时序：busy loop 的一次迭代

```
[输入线程]              [主线程 run_busy_loop]              [输出线程]            [Worker/GPU]
  │                            │                                │                      │
  │ recv ADD (ZMQ)             │                                │                      │
  │ preprocess_add_request     │                                │                      │
  │ input_queue.put(ADD,req)   │                                │                      │
  │───────────────────────────>│                                │                      │
  │                            │ _process_input_queue           │                      │
  │                            │  add_request → waiting         │                      │
  │                            │ _process_engine_step           │                      │
  │                            │  step()                        │                      │
  │                            │   ├ schedule() ──────────────────────────────────────>│ (Phase 1 纯 CPU)
  │                            │   │  ← SchedulerOutput         │                      │
  │                            │   ├ execute_model(non_block) ───────────────────────>│ (Phase 2 GPU forward)
  │                            │   ├ get_grammar_bitmask() [CPU, 与 forward 并行]      │
  │                            │   ├ future.result() <─────── forward 完成 ───────────│ logits 暂存, 返回 None
  │                            │   ├ sample_tokens() ─────────────────────────────────>│ (Phase 3 GPU 采样)
  │                            │   │                            │                      │ ← ModelRunnerOutput
  │                            │   ├ _process_aborts_queue      │                      │
  │                            │   └ update_from_output()      │                      │ (Phase 4 CPU)
  │                            │  output_queue.put(eco)         │                      │
  │                            │───────────────────────────────>│                      │
  │                            │  post_step() (拉 draft token)  │                      │
  │                            │                                │ msgpack encode        │
  │                            │                                │ ZMQ PUSH ────────────>│ (前端)
  │                            │ loop again (has_work? yes)     │                      │
  │                            │ _process_input_queue (drain)   │                      │
  │                            │ _process_engine_step ...       │                      │
```

**进程边界总结**：

| 边界 | 传输 | 数据对象 |
|------|------|----------|
| 输入线程 → 主线程 | 进程内 `input_queue` | `(EngineCoreRequestType, Request/ids)` |
| 主线程 → 输出线程 | 进程内 `output_queue` | `(client_index, EngineCoreOutputs)` |
| EngineCore → Worker | `model_executor` 跨进程 RPC（multiproc）或同进程调用（uniproc） | `SchedulerOutput` 下行 / `ModelRunnerOutput` 上行 |
| EngineCore → 前端 | ZMQ `PUSH→PULL` | `EngineCoreOutputs`（msgpack） |

---

## 3. 决定本轮工作：schedule 与 KV cache

`Scheduler.schedule()`（`v1/core/sched/scheduler.py:388`）是每个 iteration 的起点。它不区分
“prefill phase / decode phase”，而是用一个统一的 token 记账把 chunked prefill、prefix cache、
spec decode 全部纳入。

> **本章主线**：先读 3.1、3.2 和 3.5，回答“谁先跑、这一轮算多少 token、输出给 Worker 什么”。
> 3.3.1–3.3.5 是所有模型共用的 KV 分配机制；从 3.3.6 起是缓存格式和混合模型的进阶参考，可按需读。

### 3.1 Scheduler 持有的状态

```
Scheduler (v1/core/sched/scheduler.py)
  ├── self.running : list[Request]      正在生成、每步都会被尝试调度
  ├── self.waiting : RequestQueue       新到 / 被抢占的请求，等准入
  ├── self.kv_cache_manager : KVCacheManager     scheduler.py:254-268 构造
  ├── self.connector : KVConnector      外部 KV（P/D disagg、外部 KV store）
  ├── self.structured_output_manager     grammar 后端
  ├── self.max_num_scheduled_tokens      每步 token 预算（max_num_seqs × max_num_batched_tokens 相关）
  ├── self.num_sampled_tokens_per_step   LLM=1, diffusion=0
  └── self.scheduler_reserve_full_isl    WAITING 准入是否要求整序列能放下
```

`Request` 的几个关键字段（`v1/request.py`）：`num_computed_tokens`（已算到哪）、
`spec_token_ids`（待验证的 draft，`request.py:152`）、`num_output_placeholders`（`request.py:141`，
异步调度用的占位计数）、`block_hashes`（prefix cache 用）、`num_tokens_with_spec` 属性
（`request.py:250-252` = prompt + output + spec）。

### 3.2 schedule() 主流程

**文件**: `vllm/v1/core/sched/scheduler.py:388`

```
schedule(throttle_prefills)                                            scheduler.py:388
  │
  ├─ current_step += 1
  ├─ 初始化累积容器：scheduled_{new,resumed,running}_reqs / preempted_reqs
  │                 req_to_new_blocks / num_scheduled_tokens / token_budget
  │                 scheduled_spec_decode_tokens / scheduled_encoder_inputs        :401-417
  │
  ├─ kv_cache_manager.new_step_starts()                  :422   每步开始：清“本步缓存”标记
  │
  ├─ defer_prefills = throttle_prefills and not prefill_capacity_bound
  │                   and any(running 中有非 prefill_chunk)     :426-428   ← DP prefill 均衡
  │
  ├─ 【Phase A】调度 RUNNING 请求（inline 循环）         :430-623
  │    for request in self.running (req_index 遍历, token_budget>0):
  │      ├─ 跳过：达到 max_tokens / 未到 next_decode_eligible_step / defer_prefills   :435-461
  │      ├─ num_new_tokens = num_tokens_with_spec + num_output_placeholders
  │      │                   - num_computed_tokens                  :463-467
  │      │    ↑ 长预填充分块 (long_prefill_token_threshold)          :468-469
  │      │    ↑ token_budget / max_model_len 截断                    :470-479
  │      ├─ allocate_slots 重试 + 抢占循环（见 3.5）                :523-572
  │      ├─ 记录 scheduled_running_reqs / req_to_new_blocks / num_scheduled_tokens  :574-580
  │      └─ spec：把本轮要验证的 draft token 切出来                  :582-594
  │            num_scheduled_spec_tokens = num_new_tokens + num_computed_tokens
  │                                       - num_tokens - num_output_placeholders
  │            scheduled_spec_decode_tokens[req_id] = spec_token_ids[:n]   :590-594
  │
  ├─ 【Phase B】调度 WAITING 请求（inline 循环）        :625-988
  │    for request in self.waiting:
  │      ├─ prefix cache 命中：get_computed_blocks(request) → (new_computed_blocks,
  │      │    num_new_local_computed_tokens)                          :667-712
  │      │    （hybrid/Mamba/connector 走 find_longest_cache_hit_per_group :675-708）
  │      ├─ connector 外部 token：connector.get_num_new_matched_tokens() :722-743
  │      ├─ num_new_tokens（chunked prefill 切分、长 prefill 阈值）    :782-812
  │      ├─ allocate_slots(全参数：new_computed_blocks, num_external_computed_tokens,
  │      │    full_sequence_must_fit=scheduler_reserve_full_isl, reserved_blocks, ...)  :866-885
  │      ├─ 失败 → free encoder inputs, break（停止准入）              :888-895
  │      └─ 成功 → running.append, status=RUNNING,
  │            num_computed_tokens = num_computed_tokens              :939-960
  │
  ├─ 【收尾】
  │    ├─ get_num_common_prefix_blocks(any_req_id)    :1004-1010   ← cascade attention
  │    ├─ take_new_block_ids()                        :1046-1050   ← 需要清零的新 block
  │    ├─ num_spec_tokens_to_schedule（动态 SD 选 K） :1052-1057
  │    └─ 构造 SchedulerOutput 并返回                 :1064-1075
  │
  └─ return SchedulerOutput
```

**为什么 RUNNING 在 WAITING 之前**：decode 请求每步只要 1 个 token（+ draft），延迟敏感且内存占用小，
优先保证在跑的请求不 stall；新请求的 prefill 占内存大，放到后面，剩余 token 预算和 KV 空间不足时
自然被推迟。

**统一记账的力量**（`scheduler.py:390-399` 注释）：

```
num_tokens_with_spec = len(prompt) + len(output) + len(spec_token_ids)
num_new_tokens       = num_tokens_with_spec + num_output_placeholders - num_computed_tokens
```

- **普通 decode**：`spec_token_ids=[]`，`num_new_tokens = 1`。
- **chunked prefill**：prompt 很长，`num_new_tokens` 被 `long_prefill_token_threshold` / `token_budget`
  截断，分多步算完。
- **prefix cache 命中**：命中的 token 被算进 `num_computed_tokens`，`num_new_tokens` 自动减小。
- **spec decode**：`spec_token_ids` 膨胀 `num_tokens_with_spec`，于是 `num_new_tokens = 1 + num_draft`，
  模型一次前向同时算 `[last_committed, draft_1, …, draft_k]`。

### 3.3 KV cache 规划与分配（深入）

这是本文的核心之一。v1 的 KV cache 栈是**三层委托**：

```
Scheduler
  └─ KVCacheManager                  v1/core/kv_cache_manager.py:110      对 scheduler 的门面
       └─ KVCacheCoordinator          v1/core/kv_cache_coordinator.py:61   抽象基类，3 个具体子类
            │   ├─ NoPrefixCache      kv_cache_coordinator.py:369   关闭缓存 / 0 group
            │   ├─ Unitary            kv_cache_coordinator.py:419   单 group
            │   └─ Hybrid             kv_cache_coordinator.py:506   多 group，定点迭代找最长命中
            ├─ BlockPool              v1/core/block_pool.py:144      空闲链表 + prefix hash 表
            └─ tuple[SingleTypeKVCacheManager ...]
                  ├─ FullAttentionManager        single_type_kv_cache_manager.py:540
                  ├─ SlidingWindowManager        :601
                  ├─ ChunkedLocalAttentionManager :808
                  ├─ MambaManager                :958
                  ├─ CrossAttentionManager       :1293
                  └─ SinkFullAttentionManager    :1356
```

`KVCacheManager` 自己几乎不存逻辑——它存 `kv_cache_config`、`coordinator`、`block_pool`、
`watermark_blocks`（`kv_cache_manager.py:127-179`），把活全转发给 `coordinator`。**没有**
`UnifiedKVCacheManager`（“unified”是 spec 层概念 `UniformTypeKVCacheSpecs`）；**也没有**
`num_blocks_per_alloc` 字段——分配粒度恒为 1 个 block，需要几个由 `ceil(num_tokens/block_size)` 决定。

#### 3.3.1 allocate_slots 十步算法

**文件**: `vllm/v1/core/kv_cache_manager.py:244`

scheduler 对每个请求调一次 `allocate_slots`，决定“给多少 block、能不能放下”。算法（行号在
`kv_cache_manager.py` 内）：

```
allocate_slots(request, num_new_tokens, num_new_computed_tokens=0,
               new_computed_blocks=None, num_lookahead_tokens=0,
               num_external_computed_tokens=0, delay_cache_blocks=False,
               num_encoder_tokens=0, full_sequence_must_fit=False,
               reserved_blocks=0, has_scheduled_reqs=True)
  │
  ├─ ① 算 token 总数                                    :355-361
  │     num_local_computed  = request.num_computed_tokens + num_new_computed_tokens
  │     total_computed      = min(num_local_computed + num_external, max_model_len)
  │
  ├─ ② watermark 门槛                                   :363-370
  │     仅对 WAITING/PREEMPTED 且已有请求在跑时施加 watermark_blocks
  │     （RUNNING 不吃 watermark，避免 decode 被 watermark 卡住）
  │
  ├─ ③ full_sequence_must_fit 准入检查（WAITING）       :372-387
  │     n = coordinator.get_num_blocks_to_allocate(..., apply_admission_cap=True)
  │     if n + watermark_blocks > block_pool.get_num_free_blocks(): return None
  │
  ├─ ④ 先回收 sliding-window / Mamba 的 skipped block    :400-402
  │     coordinator.remove_skipped_blocks(...)
  │
  ├─ ⑤ 重新算本步真实需要的 block 数                     :404-412
  │     num_blocks_to_allocate = coordinator.get_num_blocks_to_allocate(...)
  │
  ├─ ⑥ 空闲容量检查                                     :416-420
  │     available = get_num_free_blocks() - reserved_blocks
  │     if num_blocks_to_allocate + watermark_blocks > available: return None
  │
  ├─ ⑦ 挂上 prefix 命中的 block（touch，拉出空闲链表）  :422-433
  │     coordinator.allocate_new_computed_blocks(...)
  │       └─ manager.add_local_computed_blocks → block_pool.touch()  ref_cnt++
  │
  ├─ ⑧ 分配全新 block                                   :435-440
  │     coordinator.allocate_new_blocks(request_id, num_tokens_need_slot, ...)
  │       └─ manager.allocate_new_blocks → block_pool.get_new_blocks(n)
  │
  ├─ ⑨ 把写满的 block 入 prefix hash 表                 :444-456
  │     num_tokens_to_cache = min(total_computed + num_new_tokens, request.num_tokens)
  │     ↑ 截断到 num_tokens：未验证的 draft token 不缓存
  │     coordinator.cache_blocks(request, num_tokens_to_cache)  (除非 delay_cache_blocks)
  │
  └─ ⑩ return create_kv_cache_blocks(new_blocks)        :458
```

**block 数量怎么算**（`SingleTypeKVCacheManager.get_num_blocks_to_allocate`，
`single_type_kv_cache_manager.py:101`）：

```
num_required_blocks = cdiv(num_tokens, block_size)          :132   ← ceil(num_tokens / block_size)
# running 请求（已 cached 过）的快速路径：
num_new = max(num_required_blocks - num_req_blocks, 0)      :154   ← 处理 draft 回滚后 block 不增
# 新请求：减去已覆盖的 computed/skipped，再加回 evictable（ref_cnt==0 但还在命中表里的）
```

**从空闲链表取 block**（`BlockPool.get_new_blocks`，`block_pool.py:542`）：
`free_block_queue.popleft_n(n)` 从 LRU 头部弹 n 个；每个被弹出的 block 若还带着 hash，先
`_maybe_evict_cached_block()` 剥掉旧 hash（防止陈旧的 prefix 条目指向被复用的存储），再
`ref_cnt += 1`。**没有“部分 block”特殊处理**——最后一个不满的 block 照常分配一个完整 block，
只是它没写满就不入 hash 表（见 `cache_full_blocks`）。

#### 3.3.2 prefix caching：hash、命中、LRU

**block hash 怎么算**（`kv_cache_utils.py:577` `hash_block_tokens`）：

```
hash = hash_function( (parent_block_hash, curr_block_token_ids_tuple, extra_keys) )
```

- `parent_block_hash` 沿前缀链式哈希——所以每个 block 的 hash 指纹的是**截至该 block 边界的整条前缀**。
- `extra_keys`（`generate_block_hash_extra_keys`，`:539`）包含：多模态 feature 的
  `(identifier, offset)`（`:431`）、LoRA 名（`:498`）、`cache_salt`（仅第一个 block，`:560`）、
  prompt embeds 的 sha256（`:513`）。
- hash 与 group id 组合成 `BlockHashWithGroupId`（`:57`），保证同样内容在不同 KV cache group 里是不同 key。

每请求的 block hash 列表由 `get_request_block_hasher`（`:673`）按 `hash_block_size` 对齐窗口生成，
只哈希**写满**的 block（`:709`）。

**hash 表**（`BlockHashToBlockMap`，`block_pool.py:34`）：`dict[BlockHashWithGroupId, KVCacheBlock|dict]`。
单 block 直接存 `KVCacheBlock`；同 hash 不同 block_id 的碰撞升级成 `dict[int, KVCacheBlock]`。
**不做去重**——block 写满入表时不查重，block table 是 append-only（`block_pool.py:48-52` 注释）。

**命中查找**（`KVCacheManager.get_computed_blocks`，`kv_cache_manager.py:202`）：

```
get_computed_blocks(request)
  ├─ if not enable_caching or request.skip_reading_prefix_cache: return (empty, 0)   :218-219
  ├─ max_cache_hit_length = request.num_tokens - 1   :227
  │     ↑ 整个 prompt 都命中时，最后一个 token 必须重算（要 logits）；且 num_computed 需 block 对齐
  ├─ computed = coordinator.find_longest_cache_hit(request.block_hashes, max_cache_hit_length)  :228
  └─ return (create_kv_cache_blocks(computed), num_new_computed_tokens)
```

`find_longest_cache_hit` 三种实现（`kv_cache_coordinator.py`）：
- `FullAttentionManager`（`single_type_kv_cache_manager.py:541`）：从左到右扫 `block_hashes`，逐个
  `block_pool.get_cached_block()`，命中就 append，**第一个 miss 就 break**——这就是“最长前缀命中”。
- `SlidingWindowManager`（`:619`）：从右往左，要求滑动窗口内有连续命中。
- `HybridKVCacheCoordinator`（`:622`）：**定点迭代**——每种 attention 类型要么接受当前候选长度、要么
  缩短它，反复迭代到稳定；full attention 最先处理且向下封闭。

**命中如何免重算**：命中的 block 经 `allocate_slots` ⑦步 `touch`（`ref_cnt++`，移出空闲链表）挂回请求。
因为它们已写满且已哈希，scheduler 把它们的 token 计入 `num_computed_tokens`（`scheduler.py:960`），
model runner 直接跳过不重算。

**写满入表**（`BlockPool.cache_full_blocks`，`block_pool.py:226`）：遍历
`blocks[num_cached_blocks:num_full_blocks]`，对每个写满的 block 取 `block_hashes[i]`，
`_insert_block_hash(...)` 写入 hash 表（已带 hash 的走别名索引 `cached_block_hashes_by_block`）。

**LRU 与淘汰**（`BlockPool`，`block_pool.py`）：
- 空闲链表是 `FreeKVCacheBlockQueue`（`kv_cache_utils.py:179`）双向链表，**头部最该淘汰**。
- `touch()`（`:597`）：`ref_cnt==0` 的 block 还在链表里，先 `remove`，再 `ref_cnt++`——这是“使用”。
- **淘汰是隐式的**：`get_new_blocks` 弹出一个还带 hash 的 block 时，`_maybe_evict_cached_block`
  （`:574`）把它的所有 hash key 从表里 `pop` 掉、`reset_hash()`。所以“淘汰”= 把可淘汰的缓存 block
  复用给新内容。
- 链表顺序不变量（`kv_cache_utils.py:186-195`）：LRU 在前；同优先级时**链尾先淘汰**（靠释放时逆序 free 实现）。

#### 3.3.3 free / 回收：ref_cnt 决定去留

**`KVCacheManager.free`**（`kv_cache_manager.py:460`）→ `coordinator.free`（`kv_cache_coordinator.py:285`）
→ 每个 `manager.free`（`single_type_kv_cache_manager.py:401`）→ `block_pool.free_blocks(reversed(...))`
（**逆序** free，让链尾先回空闲，符合 LRU tie-breaking）。

**`BlockPool.free_blocks`**（`block_pool.py:614`）——核心：

```
for block in blocks:
    block.ref_cnt -= 1
    if ref_cnt == 0 and not null:
        if block.block_hash is None:
            blocks_without_hash.append(block)     ← 纯内存，最该先复用
        else:
            blocks_with_hash.append(block)        ← 还在 prefix 表里，留作 LRU 候选
free_block_queue.prepend_n(blocks_without_hash)   ← 放到淘汰最前
free_block_queue.append_n(blocks_with_hash)       ← 放到淘汰靠后，但仍可被命中
```

这就是“**free vs free + 留在 prefix cache**”的区别：带 hash 的 block 回到空闲链表，但**仍留在
`cached_block_hash_to_block`**——它现在是 LRU 候选，还能服务后续的 prefix 命中，直到被复用时才剥 hash。
`ref_cnt==0` = “在空闲链表里、可淘汰”。这也是被抢占的请求恢复时常常能重新命中自己前缀的原因。

#### 3.3.4 准入：返回 None 即“放不下”

v1 **没有** `can_fit`/`can_allocate`/MAYBE-NO-YES 枚举。准入就是 `allocate_slots` 的返回值约定：

- 返回 `KVCacheBlocks` → 放得下，调度。
- 返回 `None` → 放不下；RUNNING 路径触发抢占（`scheduler.py:570-572`），WAITING 路径停止准入
  （`scheduler.py:888-895`）。

两个数值门槛在 `allocate_slots` 内（③ 和 ⑥），都是 `required_blocks > available_blocks → None`。

#### 3.3.5 抢占：只有 recompute，没有 swap

**v1 没有 SWAP**——grep `PreemptionMode`/`swap_out`/`swap_in` 在 `vllm/v1/core/` 下为零。抢占永远是
**recompute**：释放该请求全部 block，`num_computed_tokens` 归零，回 waiting 队列头，重跑时靠幸存的
prefix 命中省一部分。

抢占触发在 RUNNING 路径的 `allocate_slots` 重试循环（`scheduler.py:523-572`）：

```
while True:
    new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens, ...)   :525
    if new_blocks is not None: break              :531   放得下
    # 放不下 → 抢占一个低优先级请求
    if policy == PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))   :537-541
        self.running.remove(preempted_req)
        # 若它本步已调度，回滚本步给它分配的预算/block/spec/encoder                        :543-560
    else:
        preempted_req = self.running.pop()        :562   FCFS：抢最新的
    self._preempt_request(preempted_req, scheduled_timestamp)                          :564
    if preempted_req == request: break            :566   抢到自己了，没得抢
if new_blocks is None: break                      :570   放不下就跳出 RUNNING 循环
```

**`_preempt_request`**（`scheduler.py:1107`）：

```
_preempt_request(request, scheduled_timestamp)
  ├─ assert request.status == RUNNING
  ├─ _free_request_blocks(request)        ← kv_cache_manager.free()（或延迟 free）  scheduler.py:1116
  ├─ encoder_cache_manager.free(request)                                                   :1117
  ├─ _inflight_prefills.discard(request)                                                   :1118
  ├─ request.status = PREEMPTED                                                            :1119
  ├─ request.num_computed_tokens = 0       ← 关键：recompute，进度清零                  :1120
  ├─ request.spec_token_ids = []                                                            :1121
  ├─ request.num_preemptions += 1                                                          :1123
  └─ self.waiting.prepend_request(request)  ← 回 waiting 队列头                          :1128
```

延迟 free（`defer_block_free`，PP/async 配置）走 `_free_request_blocks`（`scheduler.py:2082`）：
若该请求还有在飞 GPU 写未完成（`last_sched_seq > processed_step_seq`），只 `pop_blocks_for_free`
把 block 摘下、暂存 `deferred_frees`，等 `update_from_output` 里 `processed_step_seq` 追上后由
`_drain_deferred_frees`（`scheduler.py:2097`）真正还回 pool——避免释放还在被 GPU 写的 block。

#### 3.3.6 进阶：KV cache 的种类全景：Spec → Manager

前面讲的 `allocate_slots` / prefix cache / 抢占都是**机制**，跟"缓存里到底存什么"无关。vLLM 用
**`KVCacheSpec`（每层一份）**描述"这一层缓存什么、一个 block 多大"，用 **`SingleTypeKVCacheManager`**
描述"这类缓存怎么分配/命中/释放"。spec 决定**内容与尺寸**，manager 决定**调度行为**，两者通过注册表
绑定（`single_type_kv_cache_manager.py:1380-1415` `get_manager_for_kv_cache_spec`）。

spec 层级（`vllm/v1/kv_cache_interface.py`）：

```
KVCacheSpec (:96)                           # block_size = 每 block 的 token 数
├── AttentionSpec (:160)                    # num_kv_heads, head_size, dtype
│   ├── FullAttentionSpec (:205)            # 基本全注意力；hybrid 关闭时 SWA 也归此
│   │   ├── TQFullAttentionSpec (:341)      # 三值量化
│   │   ├── MLAAttentionSpec (:367)         # ★ MLA（latent 压缩）
│   │   │   └── HiddenStateCacheSpec (:434)
│   │   └── SinkFullAttentionSpec (:690)    # 带 sink block 的全注意力
│   ├── ChunkedLocalAttentionSpec (:441)    # 分块局部注意力
│   ├── SlidingWindowSpec (:478)            # 滑动窗口
│   │   └── SlidingWindowMLASpec (:550)     # ★ 滑动窗口 + MLA 缓存格式
│   ├── EncoderOnlyAttentionSpec (:670)     # 不需要 KV cache
│   └── CrossAttentionSpec (:677)           # 交叉注意力（encoder-decoder）
└── MambaSpec (:629)                        # ★ 状态缓存（Mamba1/2/Linear/GDN/ShortConv）
```

spec → manager 绑定（注册在 `single_type_kv_cache_manager.py:1420-1467`）：

| spec | manager | 缓存内容 | 随序列增长 |
|------|---------|----------|------------|
| `FullAttentionSpec` | `FullAttentionManager` (:540) | 每 token 的 K 和 V | O(seq_len) |
| `MLAAttentionSpec` | `FullAttentionManager` | latent `kv_c`+`k_pe`（无 V） | O(seq_len)，~128× 小 |
| `SlidingWindowSpec` | `SlidingWindowManager` (:601) | K/V，窗口外回收 | O(window) 真实保留 |
| `SlidingWindowMLASpec` | `SlidingWindowManager` | MLA latent + 滑窗 | O(window)，latent 尺寸 |
| `ChunkedLocalAttentionSpec` | `ChunkedLocalAttentionManager` (:808) | K/V，分块局部 | O(chunk) 真实保留 |
| `MambaSpec` | `MambaManager` (:958) | 固定 state（conv+ssm） | O(1)/request |
| `CrossAttentionSpec` | `CrossAttentionManager` (:1293) | encoder K/V | O(encoder_len) |
| `EncoderOnlyAttentionSpec` | — | 无 | 0 |

**关键**：MLA 用 `FullAttentionManager`（没有专门的 MLA manager）——因为分配机制相同
（per-token block），只是 block 里装的是 latent 而非 K/V，尺寸差异完全封装在
`MLAAttentionSpec.real_page_size_bytes`。Mamba 用独立的 `MambaManager`，因为它是**状态**而非
per-token，分配语义不同。

#### 3.3.7 基本情况：Full Attention（per-token K/V）

每一层一个 KV 张量，shape（flash-attn 后端，`v1/attention/backends/flash_attn.py:149`）：

```
(num_blocks, 2, block_size, num_kv_heads, head_size)
     ↑          ↑    ↑           ↑              ↑
  物理block数  K和V  slot数=token数  KV头数      每头维度
```

- **`block_size` = slot 数 = 每 block 的 token 数**（默认 16，须为 16 倍数，`flash_attn.py:147-148`）。
- 那个 **`2` 是 K 和 V 拼出的维度**；等价地，一个 block（一层）的内存 =
  `block_size × num_kv_heads × (head_size + head_size_v) × dtype`
  （`FullAttentionSpec.real_page_size_bytes`，`kv_cache_interface.py:323-328`，`head_size_v` 默认 = `head_size`）。
- **block_table 跨层共享**：一个请求的 block_table = `[物理block_id_0, …]` 一维列表，对所有层都用
  同一个。token 在位置 p、第 L 层的 K/V 寻址 =
  `kv_cache[L][ block_table[p//block_size] ][p%block_size]`。
- 内存 **O(seq_len)**：`max_memory_usage_bytes = cdiv(max_model_len, block_size) × page_size_bytes`（`:236-244`）。

> 一个 token 一次 forward 会为**每一层**都产生一对 K/V——这并不矛盾：层是张量的独立维度（每层一份
> 张量），slot 只承担"位置"语义。block_table 是它在 block 这一维的 1D 投影，跨层复用。所以
> "一个 token 一个 slot"和"每层都有一对 K/V"正交，不冲突。

#### 3.3.8 MLA：latent 压缩缓存（DeepSeek-V2/V3）

MLA 不缓存完整 K/V，而是缓存一个**低秩 latent 向量**：

- 每层每 token 缓存 = `kv_c_normed`（压缩 latent，dim `kv_lora_rank`=512）拼接 `k_pe`（解耦 RoPE 位置
  key，dim `qk_rope_head_dim`=64）= **576 元素/token**（DSv3）。
- **V 根本不缓存**——attention 时用 `kv_b_proj` 上投影从 latent 重建 V。
- cache tensor shape = `(num_blocks, block_size, head_size=576)`，`num_kv_heads=1`
  （`mla_attention.py:1216-1223`；`get_kv_cache_spec` 在 `:1004-1010` 传 `num_kv_heads=1`）。
- `real_page_size_bytes = storage_block_size × num_kv_heads(=1) × head_size × dtype`——**没有那个 `2`**
  （`kv_cache_interface.py:393-398`），因为只存 latent 不存 V。
- 写入：`concat_and_cache_mla(kv_c_normed, k_pe, kv_cache, slot_mapping, ...)`
  （`v1/attention/backend.py:963-983`，C++ op `_custom_ops.py:2620-2630`）。

**省多少**：DSv3 标准注意力本要存 `128 头 × (128+64) 的 K + 128×128 的 V`；MLA 只存一个 576 维
latent，约 **~128× 缩减**（per token）。

**absorption（矩阵吸收）优化**（`mla_attention.py:696-832`）——让 decode 不展开 latent：

- **decode** 走 `forward_mqa`（吸收路径）：把上投影 `W_UK` 吸进 query——`q_nope @ W_UK_T` 把 query
  变到 latent 空间，attention 直接对缓存的 latent 做（QK head dim = 512+64，V head dim = 512），
  attention 输出后再用 `W_UV` 投回 V 维（`_v_up_proj`，`:1012-1034`）。decode **不展开 latent 成完整
  K/V**，省显存省带宽。
- **prefill** 走 `forward_mha`（展开路径）：`kv_b_proj(kv_c_normed)` 展开成完整 K/V 跑标准 MHA
  （`:2317-2321`）——prefill 算力富裕，展开换 kernel 效率。
- 吸收权重在 `process_weights_after_loading` 预备：`kv_b_proj` 拆成 `W_UK`/`W_UV` 并转置
  （`:899-962`）。

**分配**：MLA 用 `FullAttentionManager`，分配逻辑与全注意力完全相同（per-token block），只是
`page_size_bytes` 是 latent 尺寸。prefix cache 的 hash 也照常（latent 内容哈希）。

#### 3.3.9 SparseMLA：两种"稀疏"，别混淆

"稀疏 MLA"在 vLLM 里有两类：

**① SlidingWindowMLASpec（窗口稀疏 + MLA 缓存）** —— `kv_cache_interface.py:550`，docstring
"Sliding window attention with MLA cache format"。一层同时是滑动窗口注意力 + 用 MLA latent 格式
存缓存。窗口外的 block 经 `remove_skipped_blocks` 回收（`SlidingWindowManager`）。→ `SlidingWindowManager`。

**② Top-k 选择稀疏 MLA（DeepSeek V3.2 / V4 的 `index_topk`）** —— 这才是社区常说的 "sparse MLA"：

- 一个**独立的轻量 indexer 注意力**先算每个 query token 对**所有已缓存 token** 的相关度，选
  top-`index_topk`（通常 2048）个 token 位置（`vllm/model_executor/models/deepseek_v2.py:603-674` 的
  `Indexer`；top-k kernel 在 `vllm/model_executor/layers/sparse_attn_indexer.py` + `csrc/libtorch_stable/topk.cu`）。
- 主 MLA attention **只对这 top-k 个 token** 做注意力，而不是全序列。
- **缓存内容不变**——还是同一个 MLA latent 张量；只是 kernel 用 top-k 的 block table 去 gather 那
  2048 个 latent，而非全序列（`v1/attention/backends/mla/*_sparse.py`：`flashmla_sparse.py`、
  `flashinfer_mla_sparse.py`、`xpu_mla_sparse.py`、`rocm_aiter_mla_sparse.py`，`is_sparse()=True`，
  只有 `forward_mqa` 没有 `forward_mha`，`backend.py:986-1041`）。
- FP8 打包布局：DSv3.2 用 656 字节/token（512 fp8 NoPE + 16 scale + 128 bf16 RoPE），DSv4 用 584
  字节/token（`flashmla_sparse.py:65-87,132-144`）。
- **启用条件**：模型 config 带 `index_topk` 即 DeepSeek V3.2（`deepseek_v2.py:1243-1253` 自动探测
  `is_v32 = hasattr(config, "index_topk")`）；V4 用 `model_version="deepseek_v4"`。普通
  DeepSeek-V2/V3（无 `index_topk`）走 dense MLA。没有独立 `--sparse-mla` 开关。

> 一句话：**SparseMLA 的"稀疏"在 attention kernel（只看 top-k token），不在缓存布局**——缓存还是
> per-token 的 MLA latent，该长还是长（O(seq_len)）。省的是 attention 的算力/带宽，不是 KV 显存。

#### 3.3.10 Linear / 状态缓存：Mamba / Linear / GDN（"LinearHybrid"）

与前述都不同：Mamba/线性注意力是**状态空间/递归**模型，它的"缓存"是一个**固定大小 state**，
**不随 token 增长**。

- 一个 Mamba 层的 state = `conv_state`（卷积暂存）+ `ssm_state`/`temporal_state`（递归状态），shape
  与序列长度无关（`vllm/model_executor/layers/mamba/mamba_utils.py:162-187`，如 Mamba2
  `temporal_state=(num_heads/tp, head_dim, state_size)`，例 `(128,64,128)`）。
- `MambaSpec`（`kv_cache_interface.py:629`）：`shapes`（各 state 的 shape）、`dtypes`、`mamba_type`
  （`MAMBA1/MAMBA2/SHORT_CONV/LINEAR/GDN_ATTN`，`v1/attention/backends/registry.py:167-171`）、
  `mamba_cache_mode`。
- **一个 block = 一份完整 state 快照**（不是 block_size 个 token 的 K/V）：
  `page_size_bytes = Σ prod(shape)×dtype_size`（`:638-646`）。
- **线性注意力（`linear_attn.py`）和 GDN（`gdn_attn.py`）复用 `MambaSpec`/`MambaManager`**，靠
  `mamba_type` 区分（线性注意力只有 recurrent_state 无 conv_state，`mamba_utils.py:130-137`）。

**`mamba_cache_mode`**（`vllm/model_executor/models/config.py:417-456` 决定）决定状态怎么进 block 体系：

| 模式 | 每 request block 数 | 含义 | prefix cache |
|------|---------------------|------|--------------|
| `none` | `1 + num_spec` | 只存当前运行 state，1 个 block | 无 |
| `align` | `2 + num_spec` | 当前 state + 1 个前驱 block 做 copy 暂存 | 有（靠最近 state block） |
| `all` | `cdiv(max_model_len, block_size) + num_spec` | 每个 block 边界存一份 state 快照 | 全（像 attention，但每 block 是 state） |

`MambaManager`（`single_type_kv_cache_manager.py:958`）的关键差异：

- `find_longest_cache_hit` **从右往左**找**单个最近命中的 state block**就停（`:973-1019`，Mamba 只需最新
  状态，不像 full attention 要最长连续前缀）。
- `get_num_blocks_to_allocate` 有 force-defer 逻辑（`:1121-1129`）：若本步该请求的最新 state block 是
  **别的请求刚在本步缓存**的，返回 `num_gpu_blocks+1` 强制推迟到下一步——因为 Mamba 的递归状态是请求
  私有的，不能复用别请求本步的产物。
- align 模式每步每请求**至多分配 1 个新 state block**（`:1246-1248`）。
- `new_step_starts` 清 `cached_blocks_this_step`（`:1289-1290`）——只有 MambaManager 有实质的 per-step
  清理。

**内存**：Mamba 层 O(1)/request（`none`/`align`）；全注意力层 O(seq_len)。混合模型里 attention 层主导
KV 预算，Mamba 层近乎免费（长上下文尤其明显）。

#### 3.3.11 混合模型：HybridKVCacheCoordinator

一个模型可以同时有多种层（如 Jamba = full attention + mamba；Gemma3 = full + sliding 5:1；LLaMA4 =
full + chunked-local）。这些层的 KV cache 由 `HybridKVCacheCoordinator`（`kv_cache_coordinator.py:506`）
统一管：

- **分组**：相同 spec 的层合并成一个 group（`KVCacheGroupSpec`，`kv_cache_interface.py:864`），每组一个
  `SingleTypeKVCacheManager`。分组逻辑 `get_kv_cache_groups`（`v1/core/kv_cache_utils.py:1697`），通用
  混合路径 `_get_kv_cache_groups_uniform_page_size`（`:1108`）按 spec 等价分组并凑成等大 group。
- **一个共享 BlockPool**：所有 group 共用同一个空闲链表和 hash 表（`kv_cache_coordinator.py:91-97,
  107-120`）。block 身份 = `(block_hash, group_id)`（`BlockHashWithGroupId`，`kv_cache_utils.py:57-66`）
  ——**同样内容在不同 group 是不同的缓存条目**，所以一次"命中"要求**所有 group 各自都命中**。
- **分配 fan-out**：`get_num_blocks_to_allocate` 把各 group 需求**相加**（`:130-185`）；
  `allocate_new_blocks` 给每组各返回一份 block 列表（`:233-266`），一个 `request_id` 因此在每组各有一份
  `req_to_blocks`。model runner 据此为每组各建一张 block_table。
- **命中：定点迭代**（`find_longest_cache_hit`，`:622-732`）：各 spec 类型要么接受当前候选命中长度、
  要么缩短它，反复迭代到收敛。**为什么要定点**：sliding-window/chunked-local 的命中依赖绝对位置（窗口
  边界取决于前面有多少 token 命中），一个 group 缩短会让另一个 group 的窗口边界变化、原本的命中可能
  失效，必须重查。**full attention 最先处理且向下封闭**（前缀命中恒成立，不会主动缩短，只被裁剪），
  给它一个紧的初始上界（`:584-586` 先排序 full attention 在前）。
- **准入**：`apply_admission_cap` 只对 SWA/chunked-local 生效（`_max_admission_blocks_per_request`，
  `single_type_kv_cache_manager.py:133-145`），full/mamba 不裁剪。一个混合请求的预留 =
  full(不裁) + swa(裁) + mamba(自有逻辑) 之和。
- **per-step 重置**：只有 `MambaManager.new_step_starts` 有实质工作（清 `cached_blocks_this_step`），
  其余 manager 继承 no-op（`:535-537`）。

实例：`Jamba`（`models/jamba.py:118` mamba 层 / `:188` attention 层）、`Gemma3`
（`models/gemma3.py:156-159` 按 `layer_types` 交替 sliding/full）、`LLaMA4`（chunked-local + full）。

#### 3.3.12 各类 KV cache 内存与增长对比

| 类型 | 每 token 缓存内容 | 每 block 含 | 随序列增长 | 命中查找 |
|------|-------------------|-------------|-----------|----------|
| Full Attention | K + V（num_kv_heads×head_size 各一） | block_size token | O(seq_len) | 左→右最长前缀 |
| MLA | latent kv_c(512)+k_pe(64)=576（无 V） | block_size token | O(seq_len)，~128× 小 | 左→右最长前缀 |
| SparseMLA (top-k) | 同 MLA latent | block_size token | O(seq_len)（缓存仍全存） | 左→右；attention 只看 top-k |
| Sliding Window | K + V | block_size token，窗口外回收 | O(window) 真实保留 | 右→左连续窗口 |
| Chunked Local | K + V | block_size token，块外 null | O(chunk) 真实保留 | 块内左→右 |
| Mamba / Linear / GDN | 固定 state（conv+ssm） | 1 block = 1 state 快照 | O(1)/request（none/align） | 右→左单个最新 |
| Cross Attention | encoder K/V | block_size token | O(encoder_len) | 左→右 |

#### 3.3.13 把这些对象串起来：一个 E2E 例子

前面零散提了 `KVCacheSpec` / `BlockPool` / `SingleTypeKVCacheManager` / `FullAttentionManager` /
`KVCacheCoordinator` / `KVCacheManager`，这里用一个端到端例子把它们串清楚。

**角色（一句话职责 + file:line）**

| 对象 | 是什么 | 一句话 | 位置 |
|------|--------|--------|------|
| `KVCacheSpec` | **每层**一份的不可变描述 | "这层缓存什么、一个 block 多大字节" | `kv_cache_interface.py:96` |
| `BlockPool` | block **仓库** | 空闲链表（LRU）+ 内容哈希表；所有组共享 | `block_pool.py:144` |
| `SingleTypeKVCacheManager` | **每组**一个 | 跟踪"这组里每个请求持有哪些 block、哪些已缓存"，实现命中/分配/释放 | `single_type_kv_cache_manager.py:32` |
| `FullAttentionManager` | 上面这个的具体实现 | 全注意力的命中（左→右最长前缀）与分配 | `:540` |
| `KVCacheCoordinator` | **策略**对象 | 拥有 `BlockPool` + 一组 manager；跨组协调命中（定点迭代）和分配（fan-out） | `kv_cache_coordinator.py:61` |
| `KVCacheManager` | scheduler 的**唯一接口** | 门面，几乎不存逻辑，全转发给 coordinator | `kv_cache_manager.py:110` |

**所有权 / 委托链**

```
Scheduler
  └─ KVCacheManager (门面)                       kv_cache_manager.py:110
       └─ KVCacheCoordinator (策略)              kv_cache_coordinator.py:61
            │   1 group → UnitaryKVCacheCoordinator (:419)
            │   >1 group → HybridKVCacheCoordinator (:506)
            ├─ BlockPool                          block_pool.py:144
            │    ├─ FreeKVCacheBlockQueue  空闲链表(LRU)   kv_cache_utils.py:179
            │    └─ BlockHashToBlockMap    内容哈希表        block_pool.py:34
            └─ (SingleTypeKVCacheManager, ...)   每组一个
                 └─ FullAttentionManager         single_type_kv_cache_manager.py:540
                      └─ req_to_blocks[req_id]   本组里该请求持有的 block 列表
                      └─ num_cached_block[req_id] 其中已写满入表的 block 数
```

> 记住一条线：**scheduler 只认 `KVCacheManager`；manager 转发给 `coordinator`；coordinator 对每个 group
> 调对应的 `SingleTypeKVCacheManager`；manager 最终向共享的 `BlockPool` 要/还 block。** `KVCacheSpec`
> 不参与运行时，它在**加载期**决定了 group 怎么分、manager 用哪个、block 多大。

**生命周期三阶段**

1. **Setup（引擎初始化，一次）**：每个 `Attention` 层的 `get_kv_cache_spec` 声明它的 `KVCacheSpec`
   （`attention.py:581`）→ engine core 收集所有层 spec，`get_kv_cache_groups` 把相同 spec 的层合并成
   group（`kv_cache_utils.py:1697`）→ 得到 `KVCacheConfig`（`num_blocks` + groups）→ `Scheduler.__init__`
   构造 `KVCacheManager`（`scheduler.py:254`）→ 按 group 数选 `Unitary`/`Hybrid` coordinator
   （`get_kv_cache_coordinator` `kv_cache_coordinator.py:774`），为每个 group 建一个 manager，所有
   manager 共享一个 `BlockPool`（`kv_cache_coordinator.py:91-120`）。`BlockPool` 把 `num_blocks` 个
   `KVCacheBlock` 全塞进空闲链表，哈希表为空。
2. **每步（schedule）**：`new_step_starts` → 对 WAITING 请求 `get_computed_blocks`（查哈希表找命中）
   → `allocate_slots`（分配 + 写满的入哈希表）。
3. **结束 / 抢占**：`free` → `BlockPool.free_blocks` → `ref_cnt--`，归零的进 LRU 尾部（带 hash 的仍可被命中）。

---

**E2E 例子**：纯 full-attention 模型，32 层，`block_size=16`，共 1000 个 block。1 个 group → `Unitary`
coordinator + 1 个 `FullAttentionManager`。初始 `BlockPool`：`#0..#999` 全空闲，哈希表空。

> 说明：每个 `#N` 是一个**物理 block id**，跨所有 32 层复用（同一 id 对每层都指该层的第 N 块存储）。
> 下面的 token 编号 0-based：token `t` 落在逻辑 block `t//16`、slot `t%16`。

**BlockPool 状态演进图**（每格 = 1 个物理 block，block_size=16）

图例：`·` 空闲　`R1` R1 持有　`R2` R2 持有　`⊕` 共享(ref_cnt=2)　`∗` 写满入哈希表　`≈` 已 free 但留哈希(LRU)　(无 ∗ = 半满未入表)

```
初始     #0·     #1·     #2·     #3·     #4·    ··· #999·

Beat1    #0[R1∗] #1[R1∗] #2[R1 ] #3·     #4·    ···   R1 prefill 40 tok → get_new_blocks(3) 弹 #0,#1,#2
                                                         #0,#1 写满→入表; #2 半满不入

Beat3    #0[R1∗] #1[R1∗] #2[R1∗] #3[R1 ] #4·    ···   decode 到 tok48: #2 写满→入表; tok48 越界→新 #3
                                                         (Beat2 decode tok40 无变化: cdiv(41,16)−3=0)

Beat4    #0[⊕∗]  #1[⊕∗]  #2[R1∗] #3[R1 ] #4[R2 ] ···  R2 prefix 命中 #0,#1 → touch 使 ref_cnt 1→2(⊕)
                                                         R2 只算 tok32-39 写进新 #4; tok0-31 免重算

Beat5    #0[R2∗] #1[R2∗] #2[≈∗]  #3·     #4[R2 ] ···  R1 free: #0,#1 ref_cnt 2→1 归 R2; #2 free 留 hash(≈)
                                                         #3 半满无 hash 直接 free

Beat6    #0[≈∗]  #1[≈∗]  #2[≈∗]  #3·     #4·     ···  KV 满 → 抢占 R2(recompute): free #0,#1(留hash),#4
                                                         R2 num_computed=0 回 waiting; 重调度可再命中 #0,#1
```

**要点速记**

- **Beat 1**（prefill）：`get_computed_blocks` 0 命中 → `allocate_slots` 要 `cdiv(40,16)=3` block →
  `BlockPool.get_new_blocks` 弹 `#0,#1,#2`；`#0,#1` 写满入哈希表，`#2` 半满不入。
- **Beat 2-3**（decode 增长）：`cdiv(n,16) − 已持有` 多数为 0 → **不分配**，token 写进现有 block 的空 slot；
  每跨 16 边界才 +1 block；**block 写满才入哈希表**。
- **Beat 4**（prefix 命中）：`find_longest_cache_hit` 命中 `#0,#1`（与 R1 同前缀）→ `BlockPool.touch` →
  `ref_cnt` 1→2（**⊕ 共享**）→ R2 的 token0-31 **免重算**，只算 token32-39 写进新 `#4`。
- **Beat 5**（free）：`ref_cnt` 归零才进空闲；带 hash 的 `#2` 进 LRU 尾但**仍在哈希表**（≈，可再命中），
  半满的 `#3` 无 hash 直接 free。
- **Beat 6**（抢占）：`allocate_slots` 返回 None → `_preempt_request`（`scheduler.py:1107`，
  **recompute 无 swap**）→ free `#0,#1,#4`；R2 `num_computed_tokens=0` 回 waiting，重调度可再命中 `#0,#1`。

---

**侧栏：hybrid 模型（多 group）哪里不同**

以 Gemma3（full + sliding 5:1）为例 —— 1 个共享 `BlockPool`，2 个 group：

```
          ┌───────────── 共享 BlockPool  (#0 #1 #2 #3 #4 #5 ...) ─────────────┐
          │                                                                     │
   ┌──────┴──────────────────┐                           ┌──────────────────────┴──────────┐
   │ group0  (full-attn 层)   │                           │ group1  (sliding-window 层)      │
   │ FullAttentionManager     │                           │ SlidingWindowManager             │
   │ R1.req_to_blocks=[#0,#1] │                           │ R1.req_to_blocks=[#5,#6]         │
   └──────────────────────────┘                           └──────────────────────────────────┘
        hash = H(tokens, group_id=0)                            hash = H(tokens, group_id=1)
```

- **hash 带 `group_id`**（`BlockHashWithGroupId`，`kv_cache_utils.py:57-66`）：同样 token 在两 group 是
  **不同** key → 一次"命中"要求**所有 group 都命中**。
- **命中定点迭代**（`find_longest_cache_hit` `:622-732`）：full-attn 左→右给紧上界，sliding 右→左查窗口，
  任一组缩短就重查，收敛到公共前缀。
- **分配 fan-out**：`get_num_blocks_to_allocate` 两 group 相加；`allocate_new_blocks` 每组各返一份。
  同一 R1：group0 持 `[#0,#1]`、group1 持 `[#5,#6]`——两份独立 `req_to_blocks`，但都从同一 `BlockPool` 取。
- 每层有自己的 KV 张量，但 **block_table 跨层共享**（同串 id 对所有层用）——这点和单 group 一样。

**一句话总括**：`KVCacheSpec`（加载期定 group/manager/block 大小）→ `KVCacheManager`（门面）→
`KVCacheCoordinator`（跨组策略）→ `SingleTypeKVCacheManager`/`FullAttentionManager`（组内 per-request
记账 + 命中/分配）→ `BlockPool`（物理 block 的 LRU + 哈希）。六者各管一层，靠委托链串起来。

#### 3.3.14 加载期规划：从 KVCacheSpec 到 group 与 num_blocks

3.3.1–3.3.13 讲的是**运行时**怎么分配/命中/释放 block。但有两个上游问题一直没回答：① 这些 group 是
怎么从每层的 `KVCacheSpec` 分出来的？② 池子里到底有多少个 block（`num_blocks`）是怎么从可用显存算
出来的？这一节补这个洞——它发生在**引擎初始化时一次**，产出的 `KVCacheConfig`（groups + num_blocks）
是后面所有运行时记账的基准。

> **版本说明**：本节 `file:line` 基于 `glm5_next` 工作树（含本地 commit `08f98cd5c` "refine kv cache
> group"），与全文的 v0.25.1 基准不同。**分组决策树、num_blocks、max_concurrency 这套机制跨版本稳定**；
> 带 ★ 的是本地改动（mamba 均衡分组 + 多层组内存核算）。

**数据流总览（单页）** —— 从每层 spec 一路到 num_blocks 与 max_concurrency：

```
┌─ 输入（加载期，每模型一次）─────────────────────────────────────────────────────┐
│  kv_cache_spec   : dict[layer_name → KVCacheSpec]   每层一份（spec 层级见 3.3.6）│
│  available_memory: int                             profiling 出的 KV 可用显存     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
   get_kv_cache_groups(vllm_config, kv_cache_spec)              kv_cache_utils.py:1809
      分组决策树（早返回级联，从特殊→一般，详见 (1)）：
        uniform_spec  → uniform_type  → grouped(DSv4)  → 通用混合 ★
                                                       (抽 mamba → attention 凑等大
                                                        → mamba 按位置均衡分桶加回)
                                      │
                                      ▼
                     list[KVCacheGroupSpec]            每 group = (layer_names, kv_cache_spec)
                                      │
                                      ▼
   get_kv_cache_config_from_groups(groups, available_memory)     kv_cache_utils.py:1340
      total_page = Σ _group_bytes_per_block(g)                    :1395   ← (4) 每 group 字节核算
                     ├─ UniformType : page_size          (已含多层)
                     └─ 普通        : page_size × len(layer_names)   ★
      num_blocks  = available_memory // total_page
                   → may_override_num_blocks  (应用 --num-blocks 覆盖 / watermark)
                                      │
                                      ▼
                KVCacheConfig(num_blocks, kv_cache_groups, kv_cache_tensors)
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
   ┌─ 运行时消费者 ──────────┐  get_max_concurrency_for_kv_cache_config   其它
   │ Scheduler.__init__ →    │     kv_cache_utils.py:920                   │
   │   KVCacheManager +      │   max_mem/req = Σ max_memory × len(layers)   │ - /metrics 上报
   │   BlockPool(num_blocks) │       (_max_memory_usage_bytes_from_groups  │   KV token 容量
   │ gpu_model_runner →      │        :1950)   ★                          │
   │   按 group 分配每层 KV  │   mem/block   = _pool_bytes_per_block(...)  │
   │   tensor（block_table   │       (4 分支：单UniformType/packed/混合★/通用)│
   │   跨层共享 num_blocks   │   max_concurrency = num_blocks              │
   │   个 id）               │       / cdiv(max_mem/req, mem/block)        │
   │ NiXL PD 握手 →          └────────────────────────────────────────────┘
   │   D 侧按 group/page
   │   核算对齐 block_len
   │   （这里算错 PD region
   │    跟着错）
   └─────────────────────────┘
```

一句话：**spec → `get_kv_cache_groups` 分组 → `Σ _group_bytes_per_block` 算 total_page →
`available // total_page` 得 num_blocks → `num_blocks / (max_mem÷mem_per_block)` 得 max_concurrency**。
其中 `_group_bytes_per_block` 的 `× len(layer_names)`（★）是多层组不漏算内存的关键。

**(1) 分组决策树** —— `get_kv_cache_groups`（`kv_cache_utils.py:1809`）

输入 `{layer_name: KVCacheSpec}`（每层一份），输出 `list[KVCacheGroupSpec]`。是一条**早返回级联**，
从最特殊到最一般：

```
get_kv_cache_groups(vllm_config, kv_cache_spec)                 kv_cache_utils.py:1809
  ├─ disable_hybrid_kv_cache_manager? → unify_hybrid_kv_cache_specs   强制把所有 spec 同化成一种
  ├─ attention-free 模型?        → return []                          :1825
  ├─ 所有层 spec 完全相同?        → _get_kv_cache_groups_uniform_spec  (1 个大 group)   :1830  def :1012
  ├─ 所有层"等价类型"(等 token 槽数)? → UniformTypeKVCacheSpecs.from_specs
  │      └─ _get_kv_cache_groups_uniform_type  (1 个 group)            :1835  def :1029
  │      例：全是 full-attn，或全是等窗 SWA
  ├─ 可分组+统一 (DeepseekV4)?   → group_and_unify_kv_cache_specs
  │      └─ _get_kv_cache_groups_uniform_groups (多个 UniformType group) :1840  def :1684
  │      例：等槽数但类型/尺寸不同（full + 多种 SWA）
  └─ 【通用混合路径】★                                                 :1848+
         抽出 HiddenStateCacheSpec / MambaSpec → 剩余 attention 凑等大 page
         → 把抽出的 mamba 按位置均衡加回（见 (2)）
```

| 场景 | 命中分支 | 典型模型 |
|------|----------|----------|
| 纯 full-attn | uniform_spec | Llama / Qwen 等绝大多数 |
| MLA + indexer（等槽） | uniform_type | DeepSeek-V3.2 sparse MLA |
| full + 多种 SWA（等槽） | grouped（DSv4） | DeepSeek-V4 |
| full/MLA + Mamba/线性（不等槽） | 通用混合 ★ | Jamba / Nemotron-H / KimiLinear / **GLM-5Next** |

**(2) 通用混合路径：抽出 + 凑等大 + 均衡加回** ★（本地改动）

hybrid 模型（attention + Mamba/线性）走的路，`get_kv_cache_groups` 末尾（`:1848`）：

```
① 抽出 hidden_specs (HiddenStateCacheSpec) 和 mamba_specs (MambaSpec)
   → filtered_spec 只剩 attention 层
② filtered_spec 再走 uniform_type / uniform_page_size 分组（attention 层之间凑等大 page）
③ hidden 层按 common_page 对齐后逐层成组
④ ★ mamba 层不再"每层一组"，而是按"在每段连续 mamba run 中的位置"分桶成少数几个均衡组   :1898-1909
```

④ 的分桶：遍历层序，遇到非 mamba（attention）就把 `run_pos` 归零；mamba 层按 `run_pos` 进对应桶。
注释例子 `(M,M,M,A)×11 + M` → 3 个组，分别 12/11/11 层。

**为什么均衡而不是每层一组**：原来每个 mamba 层各自一组（N 组 = N 个 block table + N 份 per-group
记账）；均衡后 ~3 个组，**组数/block-table 数大幅减少**，池子更紧凑。注意 mamba 是 O(1)/request 的固定
state（见 3.3.10），**无论怎么分组，总字节数不变**——均衡只改"怎么打包"，不改"占多少"。

**(3) num_blocks 核算** —— `get_kv_cache_config_from_groups`（`:1340`）

```
total_page_size = Σ _group_bytes_per_block(group)            :1395   一个 block 跨所有 group 的总字节
num_blocks      = available_memory // total_page_size
                 （再经 may_override_num_blocks 应用 --num-blocks 覆盖 / watermark）
```

每层一个物理 tensor，`num_blocks` 是它们**共享的块数**（同一物理 block id 被所有层复用，见 3.3.7）。
`available_memory` 是 profiling 出的 KV 可用显存。

**(4) 每 group 字节核算** —— `_group_bytes_per_block`（`:947`）+ `_pool_bytes_per_block`（`:954`）

`_group_bytes_per_block`（`:947`）—— **多层组的关键**：

```python
if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
    return group.kv_cache_spec.page_size_bytes          # UniformType 已含多层，直接用
return group.kv_cache_spec.page_size_bytes * len(group.layer_names)  # ★ 普通 spec：× 层数
```

`_pool_bytes_per_block`（`:954`）四个分支，对应不同分配策略：

| 分支 | 条件 | 公式 |
|------|------|------|
| 单 UniformType 组 | 1 group 且是 UniformType | `spec.page_size_bytes` |
| packed 分配 | `_use_packed_kv_cache_config` | `Σ page_size × 该桶层数` |
| 混合（UniformType + 其它）★ | 有 UniformType 又有 Mamba 等 | `Σ _group_bytes_per_block(g)` |
| 通用 | 其余（纯 full+SWA 等） | `uniform_page_size × max(group_size)` |

> **★ 这个 `× len(layer_names)` 是本 commit 修的 bug**：`page_size_bytes` 是**单层**的，多层组（如均衡
> 后的 12 层 mamba 组）必须乘层数，否则 `_pool_bytes_per_block` / `_max_memory_usage` 会**少算** →
> `num_blocks` 偏大、`max_concurrency` 偏高 → 实际跑起来**超出显存 / OOM**。原来 mamba 每层一组（×1）
> 所以"碰巧"对；一旦改成均衡多层组就必须乘。**两者是耦合的改动**——分组（②）创造多层组，核算（④）才
> 需要乘层数；缺一不可。

**(5) 最大并发** —— `get_max_concurrency_for_kv_cache_config`（`:920`）

```
max_memory_per_request = _max_memory_usage_bytes_from_groups(...)   :1950   单请求最大 KV 占用
memory_per_block       = _pool_bytes_per_block(...)                          每 block 字节
num_block_per_request  = cdiv(max_memory_per_request, memory_per_block)
max_concurrency        = num_blocks / num_block_per_request
```

`_max_memory_usage_bytes_from_groups`（`:1950`）同样要 `× len(layer_names)`（`:2011`）：每个物理层都有
状态，单组多层要按层累加。旧版用 `max(group_size) × 单一 per-group 内存` 的近似，对异构 hybrid 不准——
本 commit 改成逐组精确累加。

**(6) 实例：GLM-5Next**（commit 的测试 `test_get_kv_cache_config_balanced_mamba_hybrid`）

45 层：每 4 层一段，第 4 层是 `MLA + indexer`（11 段 attention = 22 个层名），其余 34 层是 mamba/KDA。
`get_kv_cache_groups` 产出 **4 个 group**：

```
group 0 : UniformTypeKVCacheSpecs (MLA + indexer)   22 层名 / 11 物理层
group 1 : MambaSpec               12 层     ┐
group 2 : MambaSpec               11 层     ├ 均衡三组 (12+11+11 = 34)
group 3 : MambaSpec               11 层     ┘
```

测试断言（验证内存不漏算）：

- `max_memory == Σ 每层 max_memory_usage_bytes`（逐层不漏）
- `bytes_per_block == Σ 每层 page_size_bytes`（池核算不漏）
- `num_blocks == available // bytes_per_block == 100`
- 45 个 tensor：11 个 `shared_by==2`（MLA/indexer 共享一块）+ 34 个 `shared_by==1`（mamba）
- `max_concurrency == 100 / blocks_per_request`

**要点速记**

- **加载期 vs 运行时**：group 分法 + num_blocks 是**一次**定的（本节）；allocate_slots/命中/抢占是**每步**
  的（3.3.1–3.3.5）。后者以前者为基准。
- **多层组必须 `× len(layer_names)`**，否则少算内存 → OOM。`UniformTypeKVCacheSpecs` 已含多层，别重复乘。
- **mamba 均衡分组改的是打包方式**（组数↓、block-table↓），不改总字节数。
- **和 PD disagg 的衔接**：NiXL 握手时 D 侧按这里的 group/page 核算来对齐 `block_len`；这里算错，PD 的
  region 大小也跟着错。
- **上游查重**：已有 open PR（#40384 排除 O(1) mamba 组 / #41124 mamba `block_size=None`）在修同一片
  区域，独立提交前需先对齐。

#### 3.3.15 DSv4 Packed KV cache：多层共享一个物理 tensor

3.3.7 说"每层一个 KV tensor"。DSv4 打破了这个默认——它把所有层（同 page size）的 per-block KV 数据
**打包进同一个物理 tensor**，每层只是这块内存的一个 strided view（按 `storage_offset` 切出自己的区域）。
这是 DeepSeek-V4 的标志性内存布局优化。本节基于 v0.25.1。

**(1) 为什么 packed**

DSv4 全是 `UniformTypeKVCacheSpecs`（等 token 槽数、同 page size），层数多。默认"每层一个 tensor"会得到
N 个独立分配 + N 个碎片化区域；packed 则**一个物理 tensor，所有层 view 共享、per-block 数据连续**——内存
局部性好、分配开销低、下游（如 NiXL）只需注册一个 region。

**(2) 何时启用** —— `_use_packed_kv_cache_config`（`kv_cache_utils.py:1255`）

```
is_dsv4 = 所有 group 都是 UniformTypeKVCacheSpecs
return is_dsv4 or (enable_cross_layers_blocks=True and 多 group)   # 后者是实验 API（issue #42082）
```

即：DSv4 默认走 packed；其它多 group 模型可用 `enable_cross_layers_blocks` 显式 opt in。

**(3) 规划 packed 布局** —— `_get_kv_cache_config_packed`（`kv_cache_utils.py:1277`，别名
`_get_kv_cache_config_deepseek_v4`）

```
① _bucket_layers_by_page_size(groups)            :1230
     buckets = {page_size: [[layer@slot0], [layer@slot1], ...]}
     规则：同一 (page_size, slot_idx) 的不同 group 的层共享一个 tensor
           （它们各有独立 block_table，block-id 命名空间不冲突）
② total_num_bytes_per_block = Σ page_size × len(slots)        一个 block 跨所有 slot 的总字节
③ num_blocks = available_memory // total_num_bytes_per_block
④ ONE 物理分配 total_size = total_num_bytes_per_block × num_blocks
⑤ 每个 KVCacheTensor(
        size=total_size,
        shared_by=slot,                         # 该 slot 上的所有层
        offset=byte_offset,                     # ★ 块内字节偏移，逐 slot 累加 ps
        block_stride=total_num_bytes_per_block) # ★ packed 时每 block 总字节
   byte_offset += ps    # 逐 slot
```

**(4) `KVCacheTensor` 的两个新字段**（`kv_cache_interface.py:893`）

| 字段 | 含义 |
|------|------|
| `offset` | 该层在**一个连续 block 内**的字节偏移（packed 才用） |
| `block_stride` | packed 布局下**每 block 总字节**；`0` = 非 packed（每层独占 tensor） |

**(5) runner 侧物化** —— `_allocate_kv_cache_raw_tensors`（`gpu_model_runner.py:7095`）+
`_reshape_kv_cache_tensors`（`:7133`）

```
分配：if block_stride > 0:
          packed_backing = torch.zeros(total_size, int8)   # 只 new 一次
          所有 packed KVCacheTensor 的 raw_tensor = packed_backing   # alias 同一块内存
      else: 每层独立 torch.zeros(...)
reshape：layer_packing[layer] = (offset, block_stride)        :7154
         _reshape_attention_kv_cache(raw, shape, ..., packing)
           └─ torch.as_strided(raw.view(dtype), shape, stride,
                               storage_offset = offset // dtype_size)  # ★ 每层按 offset 切 view
```

所以"共享一个 tensor"= 多层是**同一块 int8 内存的 strided view**，靠 `storage_offset` 区分各自的数据。

**(6) 内存布局对比**

```
默认（per-layer，非 packed）：每层一个独立 tensor
  layer0: [blk0][blk1][blk2]...   ← tensor A  ┐
  layer1: [blk0][blk1][blk2]...   ← tensor B  ├ 各自独立分配
  layer2: [blk0][blk1][blk2]...   ← tensor C  ┘

DSv4 packed：所有层 view 进同一个物理 tensor（block_stride = Σ ps）
  packed_backing (int8, total_size):
  byte   0      ps0    ps0+ps1              block_stride
        ├ slot0 ┤ slot1 ┤ slot2 ┤ ... ┤
  blk0  [layer0 ][layer1 ][layer2 ] ... [ ]     ← storage_offset: 0 / ps0 / ps0+ps1
  blk1  [layer0 ][layer1 ][layer2 ] ... [ ]     ← +1×block_stride
  blk2  [layer0 ][layer1 ][layer2 ] ... [ ]     ← +2×block_stride

  layer_i = as_strided(packed_backing, shape, stride,
                       storage_offset = offset_i)   # 同一块内存的不同视图
```

block 寻址：layer L（在 slot s、offset o_s、page_size ps_s）的 block b 起始字节 =
`b × block_stride + o_s`，长度 `ps_s`。

**要点速记**

- **packed 只改物理分配**（一个 tensor vs N 个），**不改 block_table 语义**——每 group/层仍各有自己的
  block_table，只是它们最终都索引进同一个物理 backing 的不同 offset。
- **NiXL 受益**：`register_kv_caches` 有 packed fast-path，所有层 strided view 同一 storage 时只注册**一个**
  NIXL region（而不是 N 个），减少描述符开销。
- **packed ≠ co-location**：packed 是 DSv4（全 UniformType）的"全层共享一个 backing"；GLM-5Next 的
  MLA+indexer 那种"成对共享"（`shared_by=[idx, mla]`，3.3.14 提到）是 co-location，走的是 glm5_next 本地的
  mixed 分支，**不是** packed——GLM-5Next 含 MambaSpec，不满足 `_use_packed_kv_cache_config` 的 is_dsv4。
- **不漏算内存**：packed 下 `total_num_bytes_per_block` 已把所有 slot 的 ps 求和，`num_blocks` 据此算，
  不会像多层组那样需要额外 `× len`（见 3.3.14 的对比）。

### 3.4 KV cache 规划与分配流程图

```
                          Scheduler.schedule()
                                  │
            ┌─────────────────────┼─────────────────────┐
            ▼                     ▼                     ▼
      [RUNNING req]         [WAITING req]          (new_step_starts)
            │                     │
            │          get_computed_blocks(request)─────► find_longest_cache_hit
            │                     │  (prefix 命中 blocks + num_new_computed_tokens)
            │                     │
            ▼                     ▼
   allocate_slots(req,         allocate_slots(req,
       num_new_tokens)             num_new_tokens,
            │                       new_computed_blocks=…,
            │                       full_sequence_must_fit=…)
            │                     │
            └──────────┬──────────┘
                       ▼
        KVCacheManager.allocate_slots   (kv_cache_manager.py:244)
          ① 算 token 总数
          ② watermark 门槛
          ③ full-sequence 准入 (WAITING)
          ④ remove_skipped_blocks
          ⑤ get_num_blocks_to_allocate  ──► cdiv(num_tokens, block_size)
          ⑥ 空闲容量检查  ─── 不足 ──► return None ──► 抢占 / 停止准入
          ⑦ allocate_new_computed_blocks ──► BlockPool.touch() (命中 block ref_cnt++)
          ⑧ allocate_new_blocks ──► BlockPool.get_new_blocks(n)
          │        └─ popleft_n(LRU 头) + _maybe_evict_cached_block + ref_cnt++
          ⑨ cache_blocks ──► BlockPool.cache_full_blocks (写满的入 hash 表)
          ⑩ return KVCacheBlocks
                       │
                       ▼
            req_to_new_blocks[req_id] = blocks
            num_scheduled_tokens[req_id] = num_new_tokens
            (spec: scheduled_spec_decode_tokens[req_id] = draft[:n])
                       │
                       ▼
                  SchedulerOutput   (v1/core/sched/output.py:180)
```

**prefix cache 命中与淘汰的生命周期**：

```
   新 block 写满                入 hash 表 (cached_block_hash_to_block)
        │                              │
        ▼                              │
   请求结束 / 抢占 free ──► ref_cnt=0 ──┘  仍在 hash 表 + 进空闲链表尾部 (LRU 候选)
        │                              │
        │                              │ (后续请求) get_cached_block 命中 → touch → ref_cnt++
        │                              │        复用，免重算
        │                              │
        ▼                              ▼
   get_new_blocks 弹出该 block ──► _maybe_evict_cached_block ──► pop hash + reset_hash
        │                                                       (此时才真正“淘汰”)
        ▼
   复用给新内容
```

### 3.5 SchedulerOutput

**文件**: `vllm/v1/core/sched/output.py:180`

关键字段：

| 字段 | 含义 |
|------|------|
| `scheduled_new_reqs` / `scheduled_resumed_reqs` / `scheduled_running_reqs` | 本步各类请求 |
| `num_scheduled_tokens: dict[str,int]` (`:193`) | 每请求本步算多少 token（含 draft） |
| `scheduled_spec_decode_tokens: dict[str,list[int]]` (`:200`) | 每请求本步要验证的 draft token |
| `num_spec_tokens_to_schedule: int` (`:245`) | 动态 SD 选定的 K |
| `num_invalid_spec_tokens` (`:230`) | ngram_gpu 按接受率裁剪用 |
| `req_to_block_ids` / `block_ids` | 每请求的 KV block id 映射（GPU block_table 来源） |
| `finished_req_ids` | 本步结束的请求 |
| `total_num_scheduled_tokens` | step() 用它判断是否“真跑了模型” |

---

## 4. 执行本轮工作：model executor 与 GPU

> **本章主线**：把 `SchedulerOutput` 当作 Worker 的“执行说明书”。先读 4.1 → 4.5：它如何跨进程
> 下发、如何构造输入、何时走 CUDA Graph、为什么 logits 暂存在 GPU。4.6 多模态是同一输入准备流程的扩展。

### 4.1 EngineCore → model_executor：非阻塞 dispatch

**文件**: `vllm/v1/engine/core.py:491`

`model_executor` 在 `EngineCore.__init__` 构造（`core.py:123`），类型由 `Executor.get_class`
（`v1/executor/abstract.py:47`）按 `distributed_executor_backend` 选：

| backend | 类 | 文件:行 |
|---------|-----|---------|
| `uni` | `UniProcExecutor` | `uniproc_executor.py:45` |
| `mp` | `MultiprocExecutor` | `multiproc_executor.py:103` |
| `ray` | `RayDistributedExecutor` / `RayExecutorV2` | `ray_executor.py:64` / `ray_executor_v2.py:218` |

`execute_model`/`sample_tokens` 在基类 `Executor`（`abstract.py:221` / `:241`）都委托给
`collective_rpc`：

```
Executor.execute_model(scheduler_output, non_block)             abstract.py:221
  └─ collective_rpc("execute_model", args=(scheduler_output,), non_block=non_block)
       └─ 返回 [output]，取 output[0]
```

- `non_block=False` → 直接返回 `ModelRunnerOutput | None`
- `non_block=True` → 返回 `Future[ModelRunnerOutput | None]`

**UniProc（同进程）** 路径（`uniproc_executor.py:79` `collective_rpc`）：直接
`run_method(self.driver_worker, method, ...)`。若结果是 `AsyncModelRunnerOutput`（异步调度），
non_block 时包成 `AsyncOutputFuture`（`uniproc_executor.py:26`），其 `result()` 才惰性调
`get_output()` 触发 D2H 拷贝。

**Multiproc（跨进程）** 路径（`multiproc_executor.py:340` `collective_rpc`）：把
`(method, args, kwargs, output_rank)` 4-tuple 写入 `rpc_broadcast_mq`（共享 `MessageQueue`）；
子进程 `WorkerProc.worker_busy_loop`（`multiproc_executor.py:979`）`dequeue` 后
`getattr(worker, method)(*args)` 执行，仅 `output_rank` 匹配的 rank 把结果写回 `response_mq`。
返回 `FutureWrapper`，non_block 时不等。

完整调用链（UniProc）：

```
EngineCore.step()                                          core.py:479
  └─ model_executor.execute_model(sched_out, non_block=True)   core.py:491
       └─ UniProcExecutor.execute_model                     uniproc_executor.py:108
            └─ collective_rpc(non_block=True)               uniproc_executor.py:111 → :79
                 └─ run_method(driver_worker, "execute_model", ...)   :98
                      └─ WorkerWrapperBase.execute_model    worker_base.py:346
                           └─ Worker.execute_model          gpu_worker.py:836
                                └─ GPUModelRunner.execute_model(sched_out, intermediate_tensors)  gpu_model_runner.py:4056
```

`Worker.execute_model`（`gpu_worker.py:836`，`@torch.inference_mode`）先处理 PP 的
`_pp_send_work`/`irecv_tensor_dict`（`:839-893`），再 `self.model_runner.execute_model(...)`
（`:895`）。`Worker.sample_tokens`（`gpu_worker.py:829`）直接 `return self.model_runner.sample_tokens(grammar_output)`。

### 4.2 GPUModelRunner.execute_model 骨架

**文件**: `vllm/v1/worker/gpu_model_runner.py:4056`

```
execute_model(scheduler_output, intermediate_tensors=None)        gpu_model_runner.py:4056
  │  → ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors | None
  │
  ├─ 守卫：execute_model_state 必须为 None（上次 sample_tokens 已消费）   :4061-4065
  ├─ 清 routed-experts buffer                                           :4067
  ├─ ngram_gpu scheduler_output 拷贝                                    :4073-4085
  ├─ KV-transfer 抢占处理                                               :4087-4090
  │
  ├─ _update_states(scheduler_output)          :4098   (def :1127)
  │    ├─ 移除 finished 请求的 InputBatch 条目 / cached state
  │    ├─ 释放 encoder cache、处理需清零的新 KV block
  │    ├─ 写入 new / resumed / running 请求状态
  │    └─ 更新 block_table、采样元数据、LoRA 状态
  │    → 返回 deferred_state_corrections_fn
  │
  ├─ EC-transfer consumer 短路 / 零 token 早返回                         :4100-4124
  │
  ├─ num_scheduled_tokens_np, max_num_scheduled_tokens, num_tokens_unpadded   :4133-4138
  │
  ├─ _prepare_inputs(scheduler_output, num_scheduled_tokens_np)  :4140  (def :1889)
  │    → (logits_indices, spec_decode_metadata)
  │
  ├─ _determine_batch_execution_and_padding(...)        :4155  (def :3822)
  │    → (cudagraph_mode, batch_desc, should_ubatch, ...)
  │    ★ eager / FULL cuda graph / PIECEWISE 决策点
  │
  ├─ _build_attention_metadata(...)                     :4267  (def :2216)
  │    → (attn_metadata, spec_decode_common_attn_metadata)
  │
  ├─ _preprocess(scheduler_output, num_tokens_padded, intermediate_tensors)  :4283  (def :3439)
  │    → (input_ids, inputs_embeds, positions, intermediate_tensors, model_kwargs, ec_connector_output)
  │    └─ [有 MM] _execute_mm_encoder()  → vision/audio encoder forward → encoder_cache
  │    └─ [有 MM] _gather_mm_embeddings() + embed_input_ids() → inputs_embeds
  │
  ├─ _model_forward(input_ids, positions, ..., inputs_embeds, **model_kwargs)  :4332  (def :3769)
  │    └─ self.model(...)
  │         ├─ Embedding lookup（input_ids → hidden_states）
  │         ├─ [有 MM] inputs_embeds 已含 MM embedding，跳过 token embedding
  │         ├─ N × TransformerLayer（RMSNorm → PagedAttention 读写 KV cache → RMSNorm → FFN/MoE）
  │         └─ Final RMSNorm → Linear → logits
  │
  ├─ 后处理：split hidden/aux；PP 非末级返回 IntermediateTensors；
  │           pooling 返回 _pool()；否则 logits = compute_logits(hidden[logits_indices])   :4340-4396
  │
  ├─ ★ self.execute_model_state = ExecuteModelState(             :4398-4409
  │       scheduler_output, logits, spec_decode_metadata,
  │       spec_decode_common_attn_metadata, hidden_states,
  │       sample_hidden_states, aux_hidden_states, ec_connector_output,
  │       cudagraph_stats, slot_mappings)
  │
  └─ return None     :4417   ← 不采样！logits 暂存在 GPU，等 sample_tokens 取
```

**`ExecuteModelState`** 是 `NamedTuple`（`gpu_model_runner.py:405`），是把 forward 产物从
`execute_model` 递到 `sample_tokens` 的“暂存箱”。`execute_model_state` 初始为 `None`
（`:899`），`sample_tokens` 取出后立即清空（`:4462`），保证一次 forward 对应一次采样。

### 4.3 _prepare_inputs：构建 GPU 输入 + spec metadata

**文件**: `vllm/v1/worker/gpu_model_runner.py:1889`

核心是把 scheduler 给的“每请求 token 数”变成扁平的 GPU tensor，并算出 attention 需要的索引：

```
_prepare_inputs(scheduler_output, num_scheduled_tokens_np)   :1889
  ├─ 排序请求（decode 在前，prefill 在后），建 idx_mapping（request → batch index）
  ├─ query_start_loc = cumsum(num_scheduled_tokens)  ← attention 的每请求边界
  ├─ [Prefill] Triton kernel 填 input_ids
  ├─ [Decode] 合并上次采样 token + draft tokens  ← draft 在这里被并入输入
  │     ★ spec decode：input = [last_committed, draft_1, …, draft_k]
  ├─ Triton kernel 算 positions / seq_lens
  ├─ torch.index_select 把每请求 token gather 进 self.input_ids.cpu   :1948-1953
  │
  └─ spec metadata 构建（若 scheduled_spec_decode_tokens 非空）        :2161-2199
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0   :2161
        每请求 num_draft_tokens = len(scheduled_spec_decode_tokens[req_id])        :2179-2190
        _calc_spec_decode_metadata(...)                                          :2191 (def :2750)
          → SpecDecodeMetadata(
              draft_token_ids, num_draft_tokens, cu_num_draft_tokens,
              cu_num_sampled_tokens, target_logits_indices, bonus_logits_indices,
              logits_indices)
        ★ logits_indices 指向 [last_committed, draft_1, …, draft_k, bonus] 这些位置
          的 hidden state，后续 compute_logits 只算这些位置
```

**`_build_attention_metadata`**（`:2216`）拿到 `use_spec_decode` 和 `max_query_len = max_num_scheduled_tokens`
（= `1 + max(num_draft_tokens)`，`:4137`/`:4273`），于是 attention 把所有 draft 位置都当 query 算——
这就是“一次 forward 验证所有 draft”。

### 4.4 attention metadata 与 block_table

scheduler 给的 `req_to_block_ids` 在 `_update_states` 里被写进 `InputBatch` 的 GPU block_table。
PagedAttention kernel 用 block_table 把逻辑 block 序列映射到物理 KV cache block。spec decode 时
query 覆盖 `[last_committed, draft_1…k]`，但 KV 仍是该请求已有的全部 block（draft token 的 KV
在本次 forward 中被写入对应 block 的新槽位）。

### 4.5 CUDA Graph 路径

决策在 `_determine_batch_execution_and_padding`（`:3822`）内的 `dispatch_cudagraph(num_tokens, disable_full, valid_modes)`
闭包（`:3867`），委托 `CudagraphDispatcher.dispatch`（`v1/cudagraph_dispatcher.py:235`），返回
`(CUDAGraphMode, BatchDescriptor)`，mode ∈ `{NONE, PIECEWISE, FULL}`（`cudagraph_dispatcher.py:40`）。

规则：`force_eager`→NONE；cascade attention 或有 encoder output→禁 FULL（`:3878`）；KV scale 计算强制
NONE（`:4297-4300`）。模型在 warmup 时被包成 `CUDAGraphWrapper`/`BreakableCUDAGraphWrapper`（`:5299-5328`），
forward 时按 `cudagraph_runtime_mode` + `batch_descriptor` 选已捕获的 graph replay（`:4314-4325`）。
`capture_model()` 在 `gpu_worker.py:658` 调用，捕获的 batch size 在 `cudagraph_batch_sizes`（`:702`）。

### 4.6 多模态：MM embedding 如何进入 token 序列与 KV cache

多模态在 API 侧（renderer / `BaseMultiModalProcessor`）的处理见 [`request.md`](./request.md) Stage 3。
本节只讲**引擎侧**：worker 怎么跑 vision/audio encoder、怎么把 MM embedding 塞进 token 序列、怎么和
KV cache / prefix cache 衔接。三个要点先记下：

1. **MM embedding 占的是真实 token 位置 → 真实 KV slot**（没有单独的"vision KV"池）。
2. **encoder cache ≠ KV cache**（一个存 vision tower 输出，一个存 attention K/V）。
3. **`mm_hash` 把 encoder cache 和 prefix cache 串起来**（同图免重跑 + 不误共享 KV）。

#### 4.6.1 数据对象：mm_features / mm_position / mm_hash

- `Request.mm_features: list[MultiModalFeatureSpec]`（`v1/request.py:157`，ctor 参数 `:70`）。每个
  `MultiModalFeatureSpec`（`multimodal/inputs.py:302`）含：`data`（编码后的 MM tensor）、`modality`
  （`image`/`audio`/`video`/`prompt_embeds`）、`identifier`（= `mm_hash`，可能带 LoRA 前缀）、
  `mm_position`（`PlaceholderRange`）、`mm_hash`（原始 hash）。
- `PlaceholderRange(offset, length)`（`multimodal/inputs.py:119`）：`offset` = 该 MM 项在 prompt token
  序列里的起始位置，`length` = 占多少个 placeholder token（如一张图占 1000 个 image-pad token）。
- 跨进程：`EngineCoreRequest.mm_features`（`v1/engine/__init__.py:96`）经 ZMQ 传到 EngineCore，
  `Request.from_engine_core_request`（`request.py:197`，`:209` 拷贝）建内部 `Request`。
- bridge（API→引擎）：`input_processor.py:354-368` 构造每个 `MultiModalFeatureSpec`；`identifier` 由
  `_get_mm_identifier`（`:165`）对 `mm_hash` 加 LoRA 前缀；`mm_hash` 由 `MultiModalHasher.hash_kwargs`
  （`multimodal/processing/inputs.py:25`）算。

#### 4.6.2 调度侧：encoder cache（≠ KV cache）与 encoder 预算

先区分两套缓存：

| | KV cache | encoder cache |
|---|----------|---------------|
| 存什么 | attention 的 K/V block | vision/audio tower 的输出 embedding |
| 存哪 | `self.kv_caches`（`gpu_model_runner.py:527`） | scheduler 侧 `EncoderCacheManager`（`encoder_cache_manager.py:17`，只记账）+ worker 侧 `self.encoder_cache: dict[str, torch.Tensor]`（`:536`，真存 tensor） |
| key | `(block_hash, group_id)` | `mm_hash`/`identifier` |
| 随什么长 | O(seq_len) | O(MM item 数)，每图一份 |

**encoder 预算**：`encoder_compute_budget = max_num_encoder_input_tokens`（`scheduler.py:415`，
`max_num_encoder_input_tokens` 在 `:218-220` 由 `mm_budget` 设；预算算法
`compute_mm_encoder_budget`，`encoder_cache_manager.py:269`）。

**`_try_schedule_encoder_inputs`**（`scheduler.py:1280`）决定本步 encode 哪些 MM item：

```
for 每个被调度的请求:
  get_mm_features_in_window(mm_features, num_computed_tokens, +num_new_tokens)  # 找窗口内 MM feature
  for 每个重叠的 MM feature i:
    if identifier 已在本步 scheduled: skip                       # 同图不重复 encode
    if check_and_update_cache(mm_hash) 命中: skip                # encoder cache 命中 → 跳过 vision tower  :1362
    if can_allocate(req, i, budget) 失败:                        # 预算/slot 不够                         :1383
        clamp num_new_tokens 到 MM item 之前 / 设 0, break       # 不超额 encode
    else:
        num_embeds_to_schedule += num_encoder_embeds
        encoder_compute_budget  -= num_encoder_embeds            # 扣预算                                 :1429
        encoder_inputs_to_schedule.append(i)
```

- `scheduled_encoder_inputs: dict[str, list[int]]`（`output.py:204`）：本步要 encode 的 MM item 索引。
- `free_encoder_mm_hashes`（`scheduler.py:1073`，字段 `output.py:215`）：scheduler 淘汰的 `mm_hash`，
  worker 据此 `encoder_cache.pop`（`gpu_model_runner.py:1159-1160`）。
- **chunked prefill + 图**：图像只在 decoder token 窗口到达其 placeholder 范围时才 encode；跨 chunk 时
  已 encode 的经 `check_and_update_cache` 跳过，**vision tower 每图只跑一次**。

#### 4.6.3 worker 侧：`_preprocess` 跑 encoder + 合并 embedding

`_preprocess`（`gpu_model_runner.py:3439`），MM 门控在 `:3460`（`supports_mm_inputs and is_first_rank
and not is_encoder_decoder`）：

```
_preprocess(scheduler_output, ...)                              gpu_model_runner.py:3439
  ├─ _execute_mm_encoder(scheduler_output)                      :2892 / 调用 :3466
  │    ├─ _batch_mm_inputs_from_scheduler()                     :2849
  │    │    遍历 scheduled_encoder_inputs → 收集 (modality, data) → mm_kwargs, identifier → mm_hashes
  │    ├─ group_and_batch_mm_kwargs(mm_kwargs) 按 modality 分组
  │    ├─ [可选] encoder_cudagraph_manager.execute(mm_kwargs_batch)  :3079   ← encoder 自己的 cuda graph
  │    │    否则 model.embed_multimodal(**mm_kwargs_batch)            :3086   ← vision/audio tower forward
  │    │                                                              (interface interfaces.py:155)
  │    └─ self.encoder_cache[mm_hash] = output                  :3094   ← 存 vision embedding
  │
  ├─ _gather_mm_embeddings(scheduler_output)                    :3101 / 调用 :3467
  │    ├─ is_mm_embed = zeros(total_num_scheduled_tokens, bool) :3109  ← 标记哪些 token 位置放 MM embedding
  │    ├─ get_mm_features_in_window(...) 找本步窗口内 MM feature
  │    ├─ encoder_cache.get(mm_hash) 取 embedding                :3154  (miss 抛 RuntimeError :3164)
  │    └─ is_mm_embed[切片] = True                                :3175
  │
  └─ model.embed_input_ids(input_ids, mm_embeds, is_multimodal=is_mm_embed)  :3485/3497
       (interface interfaces.py:375)
       ├─ _embed_text_input_ids(input_ids, ...) 算文本 embedding   # MM placeholder 若 OOV 先 mask 成 id 0 (:363-371)
       └─ _merge_multimodal_embeddings(inputs_embeds, mm_embeds, is_multimodal)  models/utils.py:479
            └─ inputs_embeds[is_multimodal] = mm_embeds_flat       :500  ★ scatter: placeholder 位置覆盖成 vision embedding
```

#### 4.6.4 MM embedding 占的是真实 KV slot

`PlaceholderRange(offset, length)` 的 `length` 个位置在 `input_ids` 里是 placeholder token（如
image-pad）。`_merge_multimodal_embeddings`（`models/utils.py:500`）把这些位置的 `inputs_embeds`
覆盖成 vision embedding 行。随后 decoder forward（`_model_forward`，`:4332`）对**所有位置**（含 MM
位置）算 K/V，写进 `kv_cache_manager.allocate_slots` 分配的 KV slot（`scheduler.py:525`/`:874`）。

> **decoder-attention 的 VL 模型没有"vision KV"独立池**——MM 位置和文本位置一样占 KV slot，只是
> embedding 来源不同。只有 encoder-decoder 模型（如 Whisper）才把 encoder 输出单独放（cross-attention，
> `_get_encoder_seq_lens` `:1844`）。

#### 4.6.5 `mm_hash` 串起 encoder cache 与 prefix cache

`_gen_mm_extra_hash_keys`（`v1/core/kv_cache_utils.py:431`）：对每个 block，找 token 范围
`[start, end)` 内重叠的 MM feature，append `(mm_feature.identifier, offset - start_token_idx)`（`:482`）
作为 block hash 的 extra key（最终在 `generate_block_hash_extra_keys` `:539` 与 LoRA/cache_salt/prompt_embeds
合并进 block hash）。

- **为什么**：同样文本但不同图（或同图在不同 block 相对位置）的 KV **不能共享**——`identifier` 区分图，
  `offset - start_token_idx` 区分同图在不同 block 的相对位置。
- `identifier` = `mm_hash`（可能 LoRA 前缀），**同时是 encoder cache 的 key**
  （`encoder_cache_manager.py:109`、`gpu_model_runner.py:3094`/`:3154`）。
- ★ **同图** → encoder cache 命中（跳过 vision tower）**且** block hash 贡献相同（可共享前缀 KV）；
  **异图** → 重跑 encoder **且** block hash 不同（不误共享 prefix KV）。一个 `mm_hash` 把两个缓存
  的命中语义对齐。

#### 4.6.6 CUDA graph 与 MM

- **encoder forward 在 decoder cuda graph 之外**：`_execute_mm_encoder` 在 `_preprocess`（`:3466`），
  早于 `_model_forward`（`:4332`），不在 `set_forward_context`/`CUDAGraphWrapper` 范围（`:4314-4338`）。
  encoder 形状可变，不适合进 decoder graph。
- **encoder 自己有可选的 cuda graph**：`encoder_cudagraph_manager`（`:540`，`v1/worker/encoder_cudagraph.py:53`），
  按 budget 捕获，仅支持部分 modality（`supports_modality`，`encoder_cudagraph.py:203`），不支持则 eager
  跑 `model.embed_multimodal`。
- **decoder forward 看到的是合并好的 `inputs_embeds`**（统一张量，MM 位置已被覆盖），对 MM 位置一视同仁
  算 K/V——所以 decoder cuda graph 不需要为 MM 特化。
- encoder-decoder 模型：有 encoder input 时 `skip_compiled=has_encoder_output`（`:4324`），强制不走
  compiled/cuda graph。

---

## 5. 把 logits 变成 token：sampling

> **本章主线**：`execute_model()` 的职责到 logits 为止；本章负责约束、采样和把结果交回 CPU scheduler。
> 先记住一条性能边界：除 `_bookkeeping_sync()` 外，正常采样路径都尽量不让 GPU 等 CPU。

### 5.1 sample_tokens 全流程

**文件**: `vllm/v1/worker/gpu_model_runner.py:4435`

```
sample_tokens(grammar_output)                                 gpu_model_runner.py:4435
  │  @torch.inference_mode
  │
  ├─ if execute_model_state is None:                          :4438-4446
  │     return ModelRunnerOutput.with_kv_conn_output_only(...)   ← 本 rank 无 forward（PP 非末级 / 空步）
  │
  ├─ 解包 execute_model_state → (sched_out, logits, spec_decode_metadata,          :4449-4460
  │     spec_decode_common_attn_metadata, hidden_states, ...)
  ├─ execute_model_state = None          ← 消费掉，防止重复采样              :4462
  │
  ├─ [结构化输出] apply_grammar_bitmask(sched_out, grammar_output, input_batch, logits)  :4465-4468
  │     ★ 在 GPU 上原地改 logits：不允许的 token → -inf
  │
  ├─ sampler_output = self._sample(logits, spec_decode_metadata)   :4471  (def :3582)
  │     ├─ [无 spec] self.sampler(logits, sampling_metadata)
  │     └─ [有 spec] self.rejection_sampler(spec_decode_metadata, draft_probs, logits, sampling_metadata)
  │
  ├─ _update_states_after_model_execute(sampler_output.sampled_token_ids, sched_out)   :4473 (def :1497)
  ├─ [PP async] _pp_broadcast_prev_sampled_token_ids(...)           :4476-4484
  ├─ 清 draft / prev-sample 缓存                                   :4486-4491
  │
  ├─ [spec decode] propose_draft_token_ids(...)                     :4493-4507 / 4606
  │     ★ GPU-token 路径(eagle/draft/ngram_gpu)：bookkeeping 前调
  │     ★ CPU-token 路径(ngram/suffix/medusa)：bookkeeping 后调
  │
  ├─ _bookkeeping_sync(...)                                         :4586-4601  (def :3613)
  │     → (num_nans, logprobs_lists, valid_sampled_token_ids,
  │        prompt_logprobs_dict, ...)
  │     ★ 唯一的 GPU→CPU 同步在这里
  │
  ├─ 构造 ModelRunnerOutput                                         :4621-4635
  └─ return ModelRunnerOutput（同步）或 AsyncGPUModelRunnerOutput（异步）  :4637-4694
```

### 5.2 grammar bitmask：与 forward 并行的设计

**为什么拆分**：`EngineCore.step()` 在 `execute_model(non_block=True)` 之后立刻调
`get_grammar_bitmask()`（`core.py:491-492`）。xgrammar 的 FSM 遍历 / bitmask 填充是纯 CPU 工作，
被藏到 GPU forward 后面；等 `future.result()`（`:497`）回来，bitmask 已算好，`sample_tokens`
直接拿来用。

**生成**（`Scheduler.get_grammar_bitmask`，`scheduler.py:1440`）→
`StructuredOutputManager.grammar_bitmask`（`structured_output/__init__.py:204`）：

```
grammar_bitmask(requests, struct_req_ids, scheduled_spec_decode_tokens)
  ├─ allocate_token_bitmask(max_batch_size * (1 + max_num_spec_tokens))   :224-226
  │     shape = [B*(1+K), ceil(vocab/32)]，int32，每 int32 压 32 个 token bit
  ├─ [并行] 请求数 > 128(默认) 且无 spec：分 16 一组丢 ThreadPoolExecutor   :236-262
  │     每组 _fill_bitmasks() → grammar.fill_bitmask(...) 或填 -1（全允许）
  └─ [串行] 有 spec：对每请求沿 draft 位置逐个填
        token_iter = chain(req_tokens, (-1,))      :282
        for token: grammar.fill_bitmask(...); grammar.accept_tokens(...)   :289
        grammar.rollback(state_advancements)        :294   ← FSM 只是临时推进，算完回滚
  → numpy 数组（跨进程序列化便宜）
```

**应用**（`apply_grammar_bitmask`，`structured_output/utils.py:85`）：

```
apply_grammar_bitmask(sched_out, grammar_output, input_batch, logits)
  ├─ bitmask 是“压缩 + scheduler 顺序”的，要重排成 gpu runner 的 batch 顺序    :112-140
  │     每请求占 (1 + num_spec_tokens) 行 logits（K 个 draft + 1 bonus），
  │     累积偏移 cumulative_offset
  ├─ sorted_bitmask = full(-1) 默认全允许，再按重排拷贝行                    :125-140
  ├─ grammar_bitmask = sorted_bitmask.to(logits.device, non_blocking=True)  :143  异步 H2D
  ├─ index_tensor = async_tensor_h2d(out_indices, int32, device)            :156-158
  │     ★ 预先 H2D 拷 index，避免 xgrammar 内部 CPU sync
  └─ xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)  :160
        ★ GPU 原地：bit=0 的 logit → -inf
```

### 5.3 Sampler.forward：采样流水线

**文件**: `vllm/v1/sample/sampler.py:72`

```
Sampler.forward(logits, sampling_metadata)                    sampler.py:72
  ├─ [需要 logprobs] 算 raw_logprobs / clone raw logits       :84-93
  │     ★ top-k logprobs 用“原始 logits”（penalty/temperature 之前），与 v0 不同   :81-83
  ├─ logits = logits.to(float32)                              :96
  ├─ apply_logits_processors(logits, sampling_metadata)       :98   (def :371)
  ├─ sampled, processed_logprobs = self.sample(logits, ...)   :102  (def :243)
  ├─ sampled = sampled.to(int64)                              :109   FlashInfer 返 int32，要转
  ├─ [需要] gather logprob_token_ids / top-k logprobs         :113-136
  ├─ sampled = sampled.to(int32)                              :139   减小 tensor
  └─ return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors)
```

**`apply_logits_processors` 顺序**（`sampler.py:371`）——全部 GPU：

| # | 处理 | 行 | 说明 |
|---|------|-----|------|
| 1 | `allowed_token_ids` 白名单 | `:396` | `masked_fill_(mask, -inf)`，mask True=禁用 |
| 2 | `bad_words` | `:400` | `apply_bad_words`（`ops/bad_words.py:30`），前缀匹配才禁末 token |
| 3 | 非 argmax 不变处理器 | `:404` | `MinTokensLogitsProcessor`(`builtin.py:165`)、`LogitBiasLogitsProcessor`(`:119`) |
| 4 | penalties | `:408` | `apply_all_penalties`（`ops/penalties.py:10`）= repetition+frequency+presence 一起 |
| 5 | thinking-budget | `:409` | 若启用 |

**`sample`**（`sampler.py:243`）继续：

| # | 处理 | 行 | 说明 |
|---|------|-----|------|
| 6 | `greedy_sample = argmax(dim=-1)` | `:260` / `:239` | 提前算好，供 `torch.where` 按行混用 |
| 7 | `apply_temperature` | `:276` | `logits.div_(temp)`，greedy 的 temp 钳到 1.0 |
| 8 | argmax 不变处理器 | `:282` | 默认是 `MinPLogitsProcessor`(`builtin.py:23`)：softmax→max_prob×min_p 阈值 |
| 9 | `topk_topp_sampler` | `:286` | top-k / top-p 选 token |
| 10 | 混用 | `:296` | `torch.where(temp<1e-5, greedy, random)` 按行选 greedy/采样 |

### 5.4 避免 multinomial：Gumbel-max / FlashInfer

**`TopKTopPSampler`**（`v1/sample/ops/topk_topp_sampler.py:70`）按平台绑实现：CUDA+FlashInfer→
`forward_cuda`(`:147`)；纯 CUDA→`forward_native`(`:123`)；CPU→`forward_cpu`；ROCm+aiter→`forward_hip`。

**为什么不用 `torch.multinomial`**（`random_sample` docstring，`:446-468`）：multinomial 会触发
CPU-GPU 同步。v1 改用 **Gumbel-max trick**（`sample_with_exponential_noise`，`:437`）：

```
q ~ Exponential(1)                       ← 每行每 vocab 一个噪声
sampled = (probs / q).argmax(dim=-1)     ← 全程 GPU
```

每请求 generator 用 `q[i].exponential_(generator=generator)` 注入（`:466`）。`forward_cuda` 走
FlashInfer 的 `top_p_sampling_from_probs` / `top_k_sampling_from_probs`（内部用 rejection sampling，
免排序，`:479-506`），返回 int32（故 `Sampler.forward` 在 `:109` 转 int64）。

greedy 路径就是 `logits.argmax(dim=-1)`（`sampler.py:239`），纯 GPU 无同步。

### 5.5 _bookkeeping_sync：唯一的 GPU→CPU 同步

**文件**: `gpu_model_runner.py:3613`

```
_bookkeeping_sync(...)
  ├─ [VLLM_COMPUTE_NANS_IN_LOGITS] _get_nans_in_logits(logits)        :3629
  ├─ 丢弃请求：回退 RNG generator offset 4 字节，保种子确定性        :3633-3640
  ├─ 拷贝 req_ids / req_id_to_index（防返回后被改）                   :3644-3645
  │
  ├─ [同步路径 not use_async_scheduling]                              :3652-3686
  │     ├─ routed-experts D2H 进 pinned buffer
  │     ├─ max_gen_len==1（无 spec）: valid_sampled_token_ids = self._to_list(sampled_token_ids)  :3672
  │     │     ★ _to_list 就是那个同步点
  │     ├─ 有 logprobs: logprobs_tensors.tolists()  (已同步，.cpu() 几乎免费)
  │     └─ max_gen_len>1（spec）: RejectionSampler.parse_output(sampled_token_ids, ...)   :3681-3686
  │
  └─ [异步路径] 不在此同步，prev_sampled_token_ids 留 GPU，D2H 推迟到 AsyncGPUModelRunnerOutput.get_output
```

**`_to_list`**（`gpu_model_runner.py:7504`）：把 `sampled_token_ids`（GPU int32）非阻塞拷进
pinned CPU buffer，在 copy stream 上记 `transfer_event`，`event.synchronize()`，再 `tolist()`。
注释（`:7505-7512`）解释：这避免 `.tolist()` 触发的整 CUDA stream 同步（issue #22754）。

**为什么只有这一个同步**：argmax / softmax / topk / masked_fill / gumbel-max / FlashInfer 全是
GPU op、返回 GPU tensor。sampled id 必须变 Python int 才能（a）更新 CPU 侧 `token_ids_cpu`/
`output_token_ids`、（b）序列化进 `ModelRunnerOutput.sampled_token_ids: list[list[int]]` 跨进程序列化
给 scheduler。这个不可避免的物化被收口在 `_to_list` 的一次 event 同步里；logprobs 的 `.cpu()`
搭便车。异步调度连这个同步都推迟到 async copy stream。

### 5.6 ModelRunnerOutput

**文件**: `vllm/v1/outputs.py:233`

`ModelRunnerOutput`（跨进程序列化给 scheduler，故字段是 Python list 而非 tensor）：

- `req_ids: list[str]` (`:236`)、`req_id_to_index: dict[str,int]` (`:238`)
- `sampled_token_ids: list[list[int]]` (`:244`) —— spec 时是 `[接受draft + bonus, -1 填充]`
- `logprobs: LogprobsLists | None` (`:249`)
- `prompt_logprobs_dict` (`:255`)、`pooler_output` (`:260`)、`kv_connector_output` (`:262`)、
  `ec_connector_output` (`:264`)、`cudagraph_stats` (`:270`)、`routed_experts` (`:281`)

`SamplerOutput`（`outputs.py:185`）：`sampled_token_ids: torch.Tensor`（GPU，`[num_reqs, max_gen_len]`，`-1` 填充）+ `logprobs_tensors`。

`SamplingParams → sampler`：每请求参数在 `InputBatch.add_request/_update_request`（`v1/worker/gpu_input_batch.py:383+`）写进 CPU numpy 镜像 + GPU tensor，由 `_make_sampling_metadata`（`:832`）在
`_update_states` 后按类别（`all_greedy`/`no_top_p`/`no_penalties` 等）只拷需要的切片上 GPU。

---

## 6. 同一闭环的变体：spec decode

spec decode 是 propose → verify → accept/reject 的跨步循环，**复用前文所有机制**：
draft token 计入 `num_tokens_with_spec`、走 `allocate_slots` 分配、走一次 `execute_model` 验证、
走 `RejectionSampler` 判定、走 `update_from_output` 回滚。

> **读法**：不要把 spec decode 当成第二条执行管线。它只改变两件事：`schedule()` 一次多放入
> `K` 个候选 token；`update_from_output()` 根据接受数回退没被接受的位置。其余节点仍是普通 decode
> 的 `schedule → forward → sample → update`。

### 6.1 总体架构

```
                  step N-1 末尾                       step N
              ┌──────────────────────┐         ┌─────────────────────────────┐
              │ sample_tokens()      │         │ schedule()                  │
              │  └ propose_draft_    │         │  draft 计入 num_tokens_     │
              │     token_ids()      │ draft   │  with_spec → num_new_tokens │
              │     → draft_token_ids│────────►│  = 1 + num_draft            │
              └──────────┬───────────┘         │  scheduled_spec_decode_     │
                         │                     │  tokens[req] = draft[:K]    │
                  post_step()                  └──────────────┬──────────────┘
                  update_draft_token_ids                      │
                  → request.spec_token_ids                    ▼
                                             ┌─────────────────────────────┐
                                             │ execute_model()  [verify]   │
                                             │  input = [last_committed,    │
                                             │     draft_1…K]              │
                                             │  forward 一次算所有位置 logits│
                                             └──────────────┬──────────────┘
                                                            │
                                                            ▼
                                             ┌─────────────────────────────┐
                                             │ sample_tokens()             │
                                             │  RejectionSampler           │
                                             │   find-first-mismatch       │
                                             │   + bonus token             │
                                             └──────────────┬──────────────┘
                                                            │
                                                            ▼
                                             ┌─────────────────────────────┐
                                             │ update_from_output()        │
                                             │  num_accepted / num_rejected│
                                             │  num_computed_tokens -=     │
                                             │     num_rejected   ← 回滚   │
                                             │  append 接受 draft + bonus  │
                                             └─────────────────────────────┘
```

**启用**：`--speculative-config` / `-sc`（`engine/arg_utils.py:1493`）→ `create_speculative_config`
（`:1698`）→ `SpeculativeConfig`（`config/speculative.py:75`），method 在 `__post_init__`
（`:564`）自动探测。proposer 持有在 `GPUModelRunner.drafter`，构造分支在 `gpu_model_runner.py:547-618`。

### 6.2 propose 阶段

**文件**: `vllm/v1/worker/gpu_model_runner.py:4864`

`sample_tokens` 末尾的闭包 `propose_draft_token_ids`（`:4493-4507`）按 proposer 类型决定在
bookkeeping 前/后调用：

- **GPU-token 路径**（EAGLE/draft/extract_hidden/ngram_gpu，`:4534`/`:4555`）：直接吃 GPU
  `sampled_token_ids`，在 `_bookkeeping_sync` **之前**调，省一次 D2H。
- **CPU-token 路径**（ngram/suffix/medusa/custom，`:4606`）：吃 CPU list，在 bookkeeping **之后**调。

`propose_draft_token_ids` 方法（`:4882-5140`）按 method 分发：

| method | 调用 | 行 |
|--------|------|-----|
| ngram | `NgramProposer.propose(sampled_token_ids_cpu, ...)` | `:4882` |
| ngram_gpu | `NgramProposerGPU.update_token_ids_ngram` + `propose` | `:4902` |
| suffix | `SuffixDecodingProposer.propose` | `:4940` |
| medusa | gather target hidden states → `MedusaProposer.propose` | `:4949` |
| eagle/eagle3/draft/dflash | `prepare_next_token_ids_*` + `drafter.prepare_inputs[_padded]` + `drafter.propose(...)` | `:5007-5134` |

**model-based 路径的 draft forward** 不在单独 runner，而在 proposer 内：`SpecDecodeBaseProposer.propose`
（`v1/spec_decode/llm_base_proposer.py:443`）跑 `self.model(**model_kwargs)`（首 pass `:524`，
多步循环 `:613-681` 的第二次起 `:667`）。`num_speculative_tokens>1` 时是自回归循环；
`==1` 或并行 drafting 单次 forward 后采样（`:550`）。

返回后 `_copy_draft_token_ids_to_cpu`（`:4749`）异步 D2H 到 `self.draft_token_ids_cpu`。
`post_step`（`core.py:510-517`，同步路径）经 `take_draft_token_ids()` 把 draft 拉回 EngineCore，
`Scheduler.update_draft_token_ids`（`scheduler.py:1900`）写到 `request.spec_token_ids`（`:1920`）。

### 6.3 调度 draft token

见 3.2：`num_tokens_with_spec` 含 `spec_token_ids`，于是
`num_new_tokens = num_tokens_with_spec + num_output_placeholders - num_computed_tokens`
自然 = `1 + num_draft`（`scheduler.py:463-467`）。committed 与 spec 的切分在 `scheduler.py:582-594`：

```
num_scheduled_spec_tokens = num_new_tokens + request.num_computed_tokens
                            - request.num_tokens - request.num_output_placeholders
if > 0:
    scheduled_spec_decode_tokens[req_id] = request.spec_token_ids[:num_scheduled_spec_tokens]
    request.spec_token_ids 清空（本轮已消费）
```

写入 `SchedulerOutput.scheduled_spec_decode_tokens`（`output.py:200`）。动态 SD 的 K 由
`num_spec_tokens_to_schedule`（`output.py:245`，算于 `scheduler.py:1052-1057`）按 batch size 选。

### 6.4 verify 阶段：一次 forward 验证所有 draft

见 4.3：`_prepare_inputs` 把 committed+draft gather 进 `input_ids`（`:1948`），构建
`SpecDecodeMetadata`（`_calc_spec_decode_metadata`，`:2750`）；`_build_attention_metadata` 设
`max_query_len` 覆盖所有 draft 位置（`:4273`）；forward 后 `compute_logits(hidden[logits_indices])`
只算 `[last_committed, draft_1…K]` 这些位置的 logits（`:4366`）。

**`SpecDecodeMetadata`**（`v1/spec_decode/metadata.py:9`）字段：`draft_token_ids`、`num_draft_tokens`、
`cu_num_draft_tokens`、`cu_num_sampled_tokens`、`target_logits_indices`、`bonus_logits_indices`、
`logits_indices`。

### 6.5 accept/reject：RejectionSampler

**文件**: `vllm/v1/sample/rejection_sampler.py:37`

`_sample` 检测到 `spec_decode_metadata is not None` 就走 `rejection_sampler(...)`（`gpu_model_runner.py:3605-3610`）。

```
RejectionSampler.forward(spec_decode_metadata, draft_probs, logits, sampling_metadata)  rejection_sampler.py:88
  ├─ ① 从 bonus_logits_indices 采 bonus token（target 在最后一个接受 draft 之后会发的 token） :129-143
  ├─ ② 取 target_logits[target_logits_indices]，apply processors + 约束                   :148-163
  └─ ③ rejection_sample(draft_token_ids, draft_probs, target_probs, ...)                  :169-181
        ├─ greedy: rejection_greedy_sample_kernel   (rejection_sampler.py:715)
        └─ random: rejection_random_sample_kernel   (rejection_sampler.py:773)
```

**find-first-mismatch**（greedy kernel，`:744-760`）：

```
for pos in range(num_draft_tokens):
    rejected = (draft_token_id != target_argmax_id)      :756
    output[pos] = target_argmax   # 接受则写 target 的 token
    if rejected: break           # 第一个不匹配后，后续全留 PLACEHOLDER_TOKEN_ID(-1)
# 全接受则把 bonus token 写到 output[num_accepted]                       :762-768
```

输出 `sampled_token_ids` 形状 `[batch, max_spec_len+1]`：前 `num_accepted` 个是接受的 draft（取
target 的 argmax 值，等价于 draft 被接受），第 `num_accepted` 个是 bonus，之后 `-1` 填充。

### 6.6 update_from_output：接受计数与回滚

**文件**: `vllm/v1/core/sched/scheduler.py:1464`

```
update_from_output(scheduler_output, model_runner_output)        scheduler.py:1464
  └─ for req_id in scheduled_req_ids:                             :1527+
        generated_token_ids = sampled_token_ids[req_index]        :1544-1546
        scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)  :1548
        if scheduled_spec_token_ids and (generated_token_ids or num_sampled==0):
            num_draft_tokens = len(scheduled_spec_token_ids)      :1554
            num_sampled    = self.num_sampled_tokens_per_step      # LLM=1
            num_accepted   = max(len(generated_token_ids) - num_sampled, 0)   :1556
                            # generated = 接受draft + bonus；减 1 bonus = 接受的 draft 数
            num_rejected   = num_draft_tokens - num_accepted       :1557
            # ★ 回滚：被拒的 draft 视为“没算过”
            if request.num_computed_tokens > 0:
                request.num_computed_tokens -= num_rejected        :1564
            if request.num_output_placeholders > 0:
                request.num_output_placeholders -= num_rejected    :1568
            make_spec_decoding_stats(... num_draft_tokens, num_accepted ...)  :1569-1575
        # grammar 推进
        if new_token_ids and structured_output_manager.should_advance(request):
            grammar.accept_tokens(req_id, new_token_ids)           :1603
        # _update_request_with_output: append 接受draft+bonus, check stop
        new_token_ids, stopped = _update_request_with_output(request, new_token_ids)  :1591
```

`_update_request_with_output`（`scheduler.py:1849`）把非 `-1` 的 token（接受 draft + bonus）经
`request.append_output_token_ids(...)`（`:1857`）追加进 `_output_token_ids`。被拒的 draft 因
`num_computed_tokens` 回退，下轮 `schedule` 会重新计入 `num_new_tokens` 重算。

### 6.7 proposer 后端一览

| 后端 | 类 (file:line) | 类型 | 生成 draft 的方式 |
|------|----------------|------|-------------------|
| ngram | `NgramProposer` (`ngram_proposer.py:12`) | 查表，无模型 | CPU/numba 最长后缀 n-gram 匹配，取匹配后的 k 个 token |
| ngram_gpu | `NgramProposerGPU` (`ngram_proposer_gpu.py:28`) | 查表，无模型 | 同上，但 GPU 向量化（unfold+argmax） |
| suffix | `SuffixDecodingProposer` (`suffix_decoding.py:9`) | 查表，无模型 | suffix-tree 匹配 prompt + 已缓存输出 |
| eagle / eagle3 | `EagleProposer` (`eagle.py:10`) | 模型 | 自回归 draft 模型，吃 target hidden states；eagle3 拼 aux hidden |
| draft_model | `DraftModelProposer` (`draft_model.py:17`) | 模型 | 独立 draft LM，不共享 embedding/lm_head |
| medusa | `MedusaProposer` (`medusa.py:18`) | 模型(并行头) | 单次 forward 多头 argmax |
| mtp(通用) | `EagleProposer`/`DraftModelProposer` 视 hf_config | 模型 | MTP head 共享 target embedding/lm_head |
| gemma4_mtp | `Gemma4Proposer` (`gemma4.py:31`) | 模型 | MTP + 跨模型 KV 共享 |
| step3p5_mtp | `Step3p5MTPProposer` (`step3p5.py:24`) | 模型 | Eagle 子类，逐层 draft-step |
| dflash | `DFlashProposer` (`dflash.py:21`) | 模型(并行) | 一次出全部 spec token |
| extract_hidden_states | `ExtractHiddenStatesProposer` (`extract_hidden_states.py:28`) | 非推测(KV 转移) | 把 target hidden 存进 KV，返回已采 token 当“draft”（恒接受） |
| custom_class | `create_custom_proposer` (`custom_class_proposer.py:12`) | 用户定义 | 导入 `speculative_config.model` 指向的类 |

**查表（无 draft 模型）**：ngram、ngram_gpu、suffix、custom_class。**模型类**：eagle/eagle3、
draft_model、medusa、mtp 系列、dflash。

### 6.8 spec decode 跨步时序

```
 step N-1 (propose)         step N (verify + accept)
─────────────────────       ─────────────────────────────────────
sample_tokens()             schedule()
 └ propose_draft_token_ids  ├ draft → num_tokens_with_spec → num_new_tokens=1+K
    (draft model forward)   ├ scheduled_spec_decode_tokens[req]=draft[:K]
 └ draft_token_ids          └ SchedulerOutput
post_step()                            │
 └ update_draft_token_ids              ▼
    → request.spec_token_ids          execute_model()  [verify]
                                          input=[last_committed, draft_1…K]
                                          forward 一次 → logits[所有 draft 位置]
                                                    │
                                                    ▼
                                      sample_tokens()
                                       └ RejectionSampler
                                            find-first-mismatch + bonus
                                                    │
                                                    ▼
                                      update_from_output()
                                       ├ num_accepted = len(gen)-1
                                       ├ num_rejected = K - num_accepted
                                       ├ num_computed_tokens -= num_rejected  ← 回滚
                                       └ append 接受draft + bonus
                                                    │
                                                    ▼
                                      (若未结束) 下一步再 propose…
```

**指标**：`SpecDecodingStats`（`v1/spec_decode/metrics.py:17`）每步由
`Scheduler.make_spec_decoding_stats`（`scheduler.py:2266`）聚合 `num_drafts`/`num_draft_tokens`/
`num_accepted_tokens` 及逐位置接受率；Prometheus 计数器 `vllm:spec_decode_num_*`
（`metrics.py:228-264`）。

---

## 7. 回查：完整时序与数据对象流转

### 7.1 一次完整 step 的端到端时序

```
EngineCoreProc          Scheduler          KVCacheManager        model_executor/Worker         GPU
  │                        │                    │                        │                      │
  │ run_busy_loop           │                    │                        │                      │
  │ _process_input_queue    │                    │                        │                      │
  │  add_request → waiting  │                    │                        │                      │
  │ _process_engine_step    │                    │                        │                      │
  │  step()                 │                    │                        │                      │
  │  ├ schedule()──────────►│                    │                        │                      │
  │  │                      │ new_step_starts()─►│                        │                      │
  │  │                      │ 调度 RUNNING        │                        │                      │
  │  │                      │  allocate_slots()─►│ ①…⑩                    │                      │
  │  │                      │  ← KVCacheBlocks   │ touch/get_new_blocks/  │                      │
  │  │                      │                    │ cache_full_blocks      │                      │
  │  │                      │ 调度 WAITING        │                        │                      │
  │  │                      │  get_computed_blocks() (prefix 命中)        │                      │
  │  │                      │  allocate_slots()─►│                        │                      │
  │  │                      │  ← SchedulerOutput  │                        │                      │
  │  │◄─────────────────────│                    │                        │                      │
  │  │                      │                    │                        │                      │
  │  ├ execute_model(non_block=True)─────────────────────────────────────►│                      │
  │  │                      │                    │  _update_states         │                      │
  │  │                      │                    │  _prepare_inputs        │                      │
  │  │                      │                    │  _build_attn_metadata   │                      │
  │  │                      │                    │  _preprocess (MM enc)   │                      │
  │  │                      │                    │  _model_forward ─────────────────────────────►│ forward
  │  │ get_grammar_bitmask()─►│                  │  stash execute_model_state, return None       │
  │  │  [CPU, 与 forward 并行]│ grammar_bitmask   │                        │                  │
  │  │◄─────────────────────│                    │                        │                      │
  │  │                      │                    │                        │                      │
  │  ├ future.result() ◄─────────────────────────────────────────────── logits (GPU) 暂存        │
  │  │                      │                    │                        │                      │
  │  ├ sample_tokens(grammar_output)────────────────────────────────────►│                      │
  │  │                      │                    │  apply_grammar_bitmask [GPU]                   │
  │  │                      │                    │  Sampler.forward / RejectionSampler [GPU]     │
  │  │                      │                    │  _bookkeeping_sync [唯一 GPU→CPU 同步]         │
  │  │                      │                    │  propose_draft_token_ids (spec)               │
  │  │◄──────────────────────────────────────────── ModelRunnerOutput                            │
  │  │                      │                    │                        │                      │
  │  ├ _process_aborts_queue│                    │                        │                      │
  │  ├ update_from_output()─►│                   │                        │                      │
  │  │                      │ append tokens / check stop / spec accept+回滚  │                      │
  │  │                      │ free KV if done ──►│ free_blocks (ref_cnt--) │                      │
  │  │                      │ ← EngineCoreOutputs │                        │                      │
  │  │◄─────────────────────│                    │                        │                      │
  │  output_queue.put(...)  │                    │                        │                      │
  │  post_step() (拉 draft) │                    │                        │                      │
  │  loop again…            │                    │                        │                      │
```

### 7.2 关键数据对象流转

```
Request (v1/request.py)                  num_computed_tokens, spec_token_ids, block_hashes
    │  (Scheduler 内部对象)
    ▼
SchedulerOutput (v1/core/sched/output.py:180)
    │  num_scheduled_tokens, scheduled_spec_decode_tokens, req_to_block_ids,
    │  num_spec_tokens_to_schedule
    │  ═══ model_executor (Future, 跨进程) ═══
    ▼
GPUModelRunner.execute_model             input_ids, positions, attn_metadata (block_table),
    │  SpecDecodeMetadata, ExecuteModelState.logits (GPU)
    ▼
Hidden States → Logits (GPU)             compute_logits(hidden[logits_indices])
    │  (暂存 execute_model_state)
    ▼
sample_tokens → sampled_token_ids (GPU)  apply_grammar_bitmask → Sampler/RejectionSampler
    │  _bookkeeping_sync → CPU list
    ▼
ModelRunnerOutput (v1/outputs.py:233)    sampled_token_ids: list[list[int]], logprobs
    │  (含接受的 draft + bonus)
    │  ═══ 回 EngineCore ═══
    ▼
Scheduler.update_from_output             num_accepted/num_rejected, num_computed_tokens 回滚,
    │  append_output_token_ids, check_stop
    ▼
EngineCoreOutput                         new_token_ids, finish_reason, scheduler_stats(含 spec stats)
    │  ═══ ZMQ PUSH ═══
    ▼
(前端 OutputProcessor → RequestOutput → SSE/JSON)
```

### 7.3 三大设计决策再回顾（对照代码）

| 决策 | 代码位置 | 体现 |
|------|----------|------|
| execute_model / sample_tokens 拆分 | `gpu_model_runner.py:4417` return None；`:4435` sample_tokens；`core.py:491-499` | grammar bitmask 藏在 GPU forward 后面 |
| 采样全程 GPU，单一同步 | `sampler.py` 全 GPU op；`gpu_model_runner.py:3613` `_bookkeeping_sync`；`:7504` `_to_list` | 回避 `torch.multinomial`，用 Gumbel-max/FlashInfer |
| spec decode 复用 token 记账 | `scheduler.py:390-399` num_tokens_with_spec；`:463-467` num_new_tokens；`:582-594` 切 draft；`:1564` 回滚 | chunked prefill/prefix cache/spec 同一套逻辑 |

---

## 8. 回查：代码阅读顺序

建议按“主干 → 分支”读，每步都先建立数据对象，再追调用栈：

1. **busy loop 主干**：`core.py:1259 run_busy_loop` → `:1269 _process_input_queue` →
   `:1300 _process_engine_step` → `:479 step`。建立“输入线程 / 主线程 / 输出线程”三线程模型。
2. **socket 解耦**：`core.py:1484 process_input_sockets`、`:1589 process_output_sockets`，理解
   ZMQ IO 与调度循环如何解耦、ABORT 为何入两个队列。
3. **schedule 主干**：`scheduler.py:388 schedule`，先读 `:390-399` 的记账注释，再读 RUNNING
   循环（`:430-623`）和 WAITING 循环（`:625-988`），最后看 SchedulerOutput 构建（`:1004-1075`）。
4. **KV cache 三层**：`kv_cache_manager.py:244 allocate_slots`（十步）→
   `single_type_kv_cache_manager.py:101 get_num_blocks_to_allocate`（block 数怎么算）→
   `block_pool.py:542 get_new_blocks` / `:597 touch` / `:614 free_blocks`（LRU 与淘汰）→
   `kv_cache_utils.py:577 hash_block_tokens` / `:673 get_request_block_hasher`（prefix hash）。
5. **抢占**：`scheduler.py:523-572` 抢占循环 → `:1107 _preempt_request`（recompute，无 swap）→
   `:2082 _free_request_blocks`（延迟 free）。
6. **executor dispatch**：`abstract.py:221 execute_model` → `uniproc_executor.py:79 collective_rpc`
   → `gpu_worker.py:835 execute_model` → `gpu_model_runner.py:4056 execute_model`。对比
   `multiproc_executor.py:340 collective_rpc` 和 `:979 worker_busy_loop` 看跨进程。
7. **execute_model 内部**：`:1127 _update_states` → `:1889 _prepare_inputs`（含 spec metadata
   `:2161`/`:2750`）→ `:2216 _build_attention_metadata` → `:3439 _preprocess` → `:3822
   _determine_batch_execution_and_padding`（cuda graph 决策）→ `:4332 _model_forward` →
   `:4398` 暂存 `execute_model_state`。
8. **sampling**：`:4435 sample_tokens` → `structured_output/utils.py:85 apply_grammar_bitmask` →
   `sampler.py:72 forward` / `:371 apply_logits_processors` / `:243 sample` →
   `topk_topp_sampler.py:70 TopKTopPSampler`（Gumbel-max/FlashInfer）→ `:3613 _bookkeeping_sync` /
   `:7504 _to_list`。
9. **spec decode**：`gpu_model_runner.py:4864 propose_draft_token_ids` → `llm_base_proposer.py:443
   propose`（draft forward）→ `scheduler.py:1900 update_draft_token_ids`（写 spec_token_ids）→
   回 schedule（`:582-594` 切 draft）→ verify forward → `rejection_sampler.py:88 forward` /
   `:394 rejection_sample` / `:715` greedy kernel（find-first-mismatch + bonus）→
   `scheduler.py:1464 update_from_output`（`:1556-1564` 接受计数与回滚）。
10. **指标**：`scheduler.py:2266 make_spec_decoding_stats` → `v1/spec_decode/metrics.py:17 SpecDecodingStats`。

> 一条记忆线索：**所有“特殊”机制（chunked prefill、prefix cache、spec decode、jump decoding）
> 都不是独立分支，而是被 `num_tokens_with_spec` / `num_computed_tokens` 这套统一记账吸收进
> 同一个 `schedule → allocate_slots → execute_model → sample → update_from_output` 循环**。
> 抓住这条主线和三个设计决策，其余都是在这条主线上挂的分支。
