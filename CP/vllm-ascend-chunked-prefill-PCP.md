# PCP 与 Chunked Prefill 的兼容

> 对应原始问题：*How does PCP integrate with chunked prefill?*

## 1. 问题背景

Chunked prefill 把一条很长的 prefill 拆成多个 chunk，跨 iteration 逐步算，以控制单步显存峰值。而 PCP 又把 **同一条序列** 沿序列维度切到多个 rank。二者叠加时会出现“双重切分”：

- PCP：query 和 KV cache 都被 shard；
- chunked prefill：本 iteration 只算某一段 query（current chunk），context 部分（已算过的 chunk）的 KV 在 cache 里。

设计文档（`context_parallel.md`）指出有三种可行思路，vLLM-Ascend 实现了其中两种：

| 方案 | 含义 | 用在哪 |
|---|---|---|
| **AllGatherQ** | 把本 rank 的 query allgather 成完整 query，再按 decode 流程算 | **GQA 后端**（`attention_cp.py`） |
| **AllGatherKV** | 把 context 的 KV allgather 完整，按标准 prefill 算 | **MLA 后端**（`mla_cp.py`） |
| **Ring-Attn** | 环形传递 KV，计算通信 overlap | **未实现**（开发复杂度高、收益有限） |

核心矛盾：PCP 同时 shard 了 query 和 KV，chunked prefill 又只能看到部分 context，所以 **必须让 query 或 KV 至少有一侧信息完整**，否则 attention 算不对。

---

## 2. GQA 后端：AllGatherQ 流程

**源码：`vllm_ascend/attention/context_parallel/attention_cp.py`**

入口 `forward_impl`（行 951）里 `has_chunked_context` 分支（行 990-1049）。多流 overlap 的注释（行 978-982）把流程画得很清楚：

```
current_stream: init -- pre -- head attn ----------------- tail attn -- post -- update
context part                                                                  -/
current_stream: -----                    -- context attn --                   -/
COMM_STREAM:     \-- all_gather Q --/                  \-- a2a ag output --/
```

### 2.1 步骤

1. **allgather Q（在 COMM_STREAM 上，与 head attn overlap）**
   `_prefill_query_all_gather`（行 716）：
   ```python
   if pcp_size > 1:
       prefill_query = get_pcp_group().all_gather(prefill_query, 0)
       prefill_query = index_select(prefill_query, 0, cp_kv_recover_idx_for_chunk)  # 还原顺序
   if dcp_size > 1:
       prefill_query = get_dcp_group().all_gather(prefill_query, 1)  # head 维
   ```
   `cp_kv_recover_idx_for_chunk` = `argsort(pcp_allgather_restore_idx[...])`（builder 里 `attention_cp.py:181`），用来把 allgather 后交错的 query 还原。

2. **head/tail attention（current_stream）**
   `_forward_prefill_cp_pre` + `_forward_prefill_cp_attn`（行 500/541）：本 rank 的 current chunk 仍然按 head-tail 切，做 nomask+mask 两次 attention。注意：此时用的是 **本地 KV cache（current chunk 部分）**，还没算 context。

3. **context attention（current_stream，等 allgather Q 完成）**
   `_compute_prefill_context`（行 726）：
   - `_load_kv_for_chunk`（行 774）：从 KV cache 里按 `local_chunked_kv_lens_rank` 把 context 的 KV load 出来（用 `DeviceOperator.kv_cache_load` + block_table + chunk starts）；
   - 用 allgather 后的完整 query 与 context KV 做 attention（`atten_mask=None`，`sparse_mode=0`，全可见），得到 `(prefix_chunk_output, prefix_chunk_lse)`。

4. **合并 context 与 current chunk 的结果（COMM_STREAM 上 a2a+ag，与 tail attn overlap）**
   `_gather_global_context_output`（行 919）：DCP 组 all_to_all + PCP 组 allgather（沿最后一维）；
   `_update_global_context_output`（行 934）：用 `_update_out_and_lse` 做 online-softmax 合并得到全局 context 的 `(output, lse)`；
   `_update_chunk_attn_out_lse_with_current_attn_out_lse`（行 680）：把全局 context 结果与本 rank 的 current chunk 结果（head/tail）再做一次 `_npu_attn_out_lse_update` 合并，**只更新真正属于 chunked prefill 的那些 request**（用 `chunk_seq_mask_filtered_indices` 过滤）。

> 设计文档原话：*“AllGatherQ 之后的流程与 decode 阶段一致；AllGatherKV 之后的流程与标准 prefill 一致。”*

### 2.2 关键 metadata

`AscendMetadataForPrefill.ChunkedContextMetadata`（`common_cp.py:77`）：
- `actual_chunk_seq_lengths` / `actual_seq_lengths_kv`：喂给 attention 的 seq 长度；
- `local_context_lens_allranks`：`[num_reqs, pcp_size, dcp_size]`，每个 CP rank 上的 context 长度（`attention_cp.py:163`）；
- `cp_kv_recover_idx_for_chunk` / `kv_inverse_idx_for_chunk`：allgather query 还原 / 反向索引（仅 pcp>1 时，行 175-184）；
- `chunked_req_mask` / `chunk_seq_mask_filtered_indices`：哪些 req 是 chunked prefill、哪些 token 要参与合并（`_get_chunked_req_mask` 行 93、`filter_chunked_req_indices`）；
- `starts`：每个 chunk 在 KV cache 里的起始 slot 偏移。

builder 在 `attention_cp.py:160` 检测 `chunked_prefill_enabled and max_context_len_cpu > 0` 才构建这套 metadata；否则走纯 prefill PCP 路径。

---

## 3. MLA 后端：AllGatherKV 流程

**源码：`vllm_ascend/attention/context_parallel/mla_cp.py`**

MLA 的 chunked prefill 走 **AllGatherKV**，即把 context 的 KV cache gather 成完整后按标准 prefill 算。核心是 `_reorg_kvcache`（行 767）。

### 3.1 `build_chunked_metadata`（`mla_cp.py:146`）

- 把 context_len 按 `cp_virtual_block_size`（= `cp_local_block_size * dcp_size * pcp_size`）向上 pad，得到每个 rank 的 `padded_local_chunk_seq_lens`；
- `local_chunk_starts` / `local_chunk_ends`：每个 chunk 在本地 cache 的起止；
- `local_context_lens_allranks`：`[num_prefills, dcp_size*pcp_size]`，每个 CP rank 的 context 长度；
- `chunk_size` = `padded_local_max_context_chunk_across_ranks`：单轮处理的本地 context 上限（用来分段，控峰值显存）。

> 注意：MLA 的 block_size 会被 `cp_virtual_block_size` 重新计算（`mla_cp.py:77`），因为 MLA 的 KV 是压缩的 `kv_lora_rank`，块语义不同。

### 3.2 `_reorg_kvcache`（行 767）——把 gather 来的交错 KV 整理连续

这是 MLA chunked prefill 最微妙的一步。注释（行 778-794）给了清晰例子：

```
kv_c_normed in rank0 = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...]
kv_c_normed in rank1 = [T0_4, T0_5, pad, pad, T1_2, pad, ...]
allgatered_kv_c     = [T0_0..T0_3, T1_0,T1_1,  T0_4,T0_5,pad,pad, T1_2,pad, ...]
-> reorganized_kv_c  = [T0_0,T0_1,T0_2,T0_3,T0_4,T0_5,  T1_0,T1_1,T1_2, ...]   # 同一 request 连续
```

实现：
```python
cache_kv_c_k_pe = cat([kv_c_normed, k_pe], dim=-1)
if dcp_size > 1: cache_kv_c_k_pe = get_dcp_group().all_gather(cache_kv_c_k_pe, 0)
if pcp_size > 1: cache_kv_c_k_pe = get_pcp_group().all_gather(cache_kv_c_k_pe, 0)
# 然后按 (rank, request, chunk) 把每个 rank 的有效段 cat 成连续布局
```
之后按 `chunk_idx`、`chunk_size` **分段处理**（行 827-833 的注释）：因为一个 chunk 的 context 在不同 rank 上长度不同（DCP0 长、DCP1 短），需要按 workspace 大小把 context 切成多轮，逐轮 attention + online-softmax 累积更新。这正是设计文档说的 *“AllGatherKV 在 context 过长时峰值显存大，采用分段处理策略”*。

### 3.3 context attention

`get_context_seq_len_npu`（行 483）从 `chunked_context.padded_chunk_seq_lens_npu[index]` 取本轮 context 长度；逐轮算、用 out/lse 合并。

---

## 4. SFA 后端的 chunked prefill

SFA（稀疏 MLA，`sfa_cp.py`）chunked prefill 与 prefill 共用一条路径（`_execute_sparse_flash_attention_process`，行 235）。它用 **compact block view**（`build_prefill_compact_block_metadata`，行 157）：
- `valid_block_ids`：本 batch 真实用到的 block id（去重）；
- `block_table_cp`：重映射后的 block table，指向 CP allgather 后的 compact KV buffer；
- `gather_kv_cross_cp_compact`（行 386）：只 allgather 这些 **真实 block**，而不是整条 request-scoped KV，省通信量。

prefill 部分同样 head/tail 拆 Q（行 292-323），用 `head_attn_nomask_seqlens` / `tail_attn_nomask_seqlens` 作为 KV 长度。

---

## 5. 约束与注意点

1. **触发条件**：只有当 `chunked_prefill_enabled` 且存在已算过的 context（`max_context_len_cpu > 0`）时才走 chunked context 路径；首次 prefill（context=0）走纯 PCP prefill。
2. **`cp_kv_cache_interleave_size`**：在 KV 需要“搬运”的场景（KV pool / PD disagg），必须设为 `block_size`（默认 128），否则 chunked prefill 的 block 对齐会出错（见用户指南 Constraints）。
3. **多流 overlap**：GQA 后端用 `cp_chunkedprefill_comm_stream()`（`vllm_ascend/utils.py`）把 allgather Q 和 a2a/ag output 与本地 attention overlap，隐藏通信。RFC 若移植到 NVIDIA，可用对应 的 CUDA stream / 异步通信。
4. **`kv_inverse_idx_for_chunk` / `cp_kv_recover_idx_for_chunk`** 只在 pcp>1 时计算，DCP-only 时为 None（`attention_cp.py:175-184`）。

---

## 6. 一图总结（GQA AllGatherQ）

```
本 rank current chunk Q ──allgather(PCP)+recover──► 完整 Q
        │                                                  │
        ├──(current_stream) head attn + tail attn          │
        │        (本地 KV, current chunk)                  │
        │                                                  │
        └──(等 ag Q) context attn◄──load context KV────────┘
                  │
        DCP all2all + PCP allgather (COMM_STREAM)
                  ▼
        _update_global_context_output (online softmax)
                  ▼
        合并到 current chunk 的 head/tail 输出 (_npu_attn_out_lse_update)
                  ▼
        q_full_idx 还原 → 写回 output
```

对 RFC 的启示：**chunked prefill + CP 的三种方案应作为可配置策略**，GQA 偏向 AllGatherQ（与 decode 复用），MLA 偏向 AllGatherKV（与 prefill 复用）；Ring 作为高阶可选。分段处理（控峰值）和 compact-block gather（省通信）是两个值得保留的工程优化。
