# DCP（Decode Context Parallel）与 PCP 的协作

> 对应原始问题（来自 intro 衍生）：*DCP 是什么？它如何与 PCP 叠加？通信域如何复用？*
>
> 设计文档：`vllm-ascend/docs/source/developer_guide/Design_Documents/context_parallel.md` §“Decode Context Parallel (DCP)”。

## 1. DCP 的定位

DCP（Decode Context Parallel）**复用 TP 通信域**，不额外占卡。它的目的是：在 decode 阶段，把原本在 TP 组内 **冗余存储** 的 KV cache 沿序列维度再分片一份，腾出显存装更多 KV cache → 更大 batch → 更高吞吐。

- 开关：`--decode-context-parallel-size`（`decode_context_parallel_size`）；
- 不增加 world size（不像 PCP）；
- 主要影响 **decode**、**chunked prefill**、**cached prefill** 的逻辑。

### 与 PCP 的统一关系

```
cp_size = pcp_size * dcp_size
cp_rank = pcp_rank * dcp_size + dcp_rank
```

PCP 和 DCP 共享同一套 KV cache 分片布局（见下节 block table），二者可单独或组合开启。组合时通信顺序为：**先 DCP 组 all-to-all（head 维），再 PCP 组 all-gather（seq 维）**。

## 2. KV cache 分片布局（PCP/DCP 共用）

**源码：`vllm_ascend/worker/block_table.py:170` `compute_slot_mapping`**

```python
virtual_block_size = block_size * dcp_world_size * pcp_world_size   # “虚拟块”

logical_block_idx = positions // virtual_block_size
block_table_indices = req_indices * max_num_blocks_per_req * blocks_per_phys_block + logical_block_idx
block_numbers = block_table[...]                                    # 物理块号

virtual_block_offsets = positions % virtual_block_size
current_rank = dcp_world_size * pcp_rank + dcp_rank
# 判断该 token 归哪个 rank（按 interleave 粒度）
mask = (virtual_block_offsets // cp_kv_cache_interleave_size
        % (dcp_world_size * pcp_world_size)) == current_rank

block_offsets = (virtual_block_offsets
                 // (cp_world * cp_kv_cache_interleave_size) * cp_kv_cache_interleave_size
                 + virtual_block_offsets % cp_kv_cache_interleave_size)
slot_mapping = block_numbers * block_size + block_offsets
# 只有落在本 rank 的 token 写真实 slot，其余填 -1
slot_mapping = where(mask, slot_mapping, -1)
```

要点：
- 定义 **虚拟块** = `block_size * cp_size`；同一虚拟块里的 token 按 `cp_kv_cache_interleave_size`（默认 1，token 粒度交错）轮流分给各 rank；
- 每个 rank 只对自己负责的那些 token 写真实 slot，其他为 `-1`；
- 约束：`block_size % cp_kv_cache_interleave_size == 0`；
- KV 搬运场景（PD disagg / KV pool）必须 `cp_kv_cache_interleave_size = block_size`（128），否则传输对齐出错。

## 3. DCP 的通信方式（按后端）

### 3.1 MLA 后端（`mla_cp.py`）

**Decode**（`_forward_decode`, `mla_cp.py:613`）：
1. `reorg_decode_q`：DCP>1 时把 `cat([q_nope, q_pe])` 在 **head 维** allgather，使 DCP 组内 Q head 一致（行 493）；
2. `num_heads = num_heads * dcp_size`；
3. 本地 KV cache（已分片）做 attention，`actual_seq_lengths_kv = decode_meta.cp_seq_len`（本 rank 本地长度）；
4. `_process_attn_out_lse`（`common_cp.py:108`）：
   - DCP>1：head 维 **all_to_all_single**（交换 out/lse）；
   - PCP>1：seq 维 **all_gather**；
5. `_npu_attention_update` 合并（online softmax）。

**Chunked Prefill**（`_reorg_kvcache`, `mla_cp.py:767`）：
- `get_dcp_group().all_gather(kv_c_k_pe, 0)` 把 context 的压缩 KV 在 DCP 组 gather 完整；
- 再 PCP gather；
- 按 `(rank, request, chunk)` 整理成连续布局（见 chunked-prefill 笔记第 3 节例子）。

### 3.2 GQA 后端（`attention_cp.py`）

**Decode**（`_forward_decode_pcp_dcp`, 行 566）：
1. `query = get_dcp_group().all_gather(query, 1)`（head 维）；
2. `num_heads = num_heads * dcp_size`；
3. 本地 KV attention，`actual_seq_lengths_kv = num_computed_tokens_of_pcp_dcp[:, pcp_rank, dcp_rank]`；
4. `_process_attn_out_lse`（DCP all2all + PCP ag）→ `_npu_attention_update`。

设计文档对 GQA chunked prefill / decode 的描述：
> “GQA 后端先对 Q 做 head 维 allgather 保证 DCP 组内一致，本地算完后用 `cp_lse_ag_out_rs` 风格聚合 out/lse 并 reduce-scatter；也可用 all-to-all 交换 out/lse 后本地 update（与 PCP 兼容的写法一致）。”

### 3.3 DSA / SFA 后端

- DSA：CP 复用 TP 域的线性切片，output 还原走 TP all_to_all（见 DSA 笔记第 4 节）；
- SFA：`gather_kv_cross_cp` 里 `dcp_size>1` 先 `get_dcp_group().all_gather(req_kv_cache, 0)`，再 pcp gather（`sfa_cp.py:380`）。

## 4. `num_computed_tokens_of_pcp_dcp`：DCP 的核心 metadata

每个 req 在每个 CP rank 上的 **本地 KV 长度**，shape `[num_reqs, pcp_size, dcp_size]`，由 `PCPManager._get_cp_local_seq_lens`（`pcp_utils.py:1042`）算：

```python
total_world_size = pcp_world_size * dcp_world_size
base = seq_lens // cp_kv_cache_interleave_size // total_world_size * cp_kv_cache_interleave_size
remainder = clip(seq_lens - base*total_world_size - rank_offsets*cp_kv_cache_interleave_size,
                 0, cp_kv_cache_interleave_size)
dcp_local_seq_lens = (base + remainder).reshape([-1, pcp_world_size, dcp_world_size])
```

即把 context_len 按 interleave 粒度尽量均分到各 rank，余数按 rank 顺序分配。decode 时各 rank 用 `[:, pcp_rank, dcp_rank]` 取自己的本地长度。

`generate_pcp_metadata`（`pcp_utils.py:1098`）计算并存进 `AscendPrefillContextParallelMetadata.num_computed_tokens_of_pcp_dcp`，再传到 `AscendMetadataForDecode`（`common_cp.py:103`）。

## 5. 约束（用户指南 §Constraints）

- MLA 模型：`tp_size >= dcp_size` 且 `tp_size % dcp_size == 0`；
- GQA 模型：`(tp_size // num_kv_heads) >= dcp_size` 且 `(tp_size // num_kv_heads) % dcp_size == 0`；
- KV 搬运场景：`cp_kv_cache_interleave_size = block_size`。

原因：DCP 复用 TP 域，KV head 在 TP 内分摊，DCP 子组必须能从 TP head 里干净切出来。

## 6. MTP / spec decode 下的 DCP

`generate_mtp_attention_mask_for_decode`（`pcp_utils.py:1371`）：spec decode 时 decode request 有多个 token（history + MTP），需要按 `(history_len + mtp_idx) % cp_size` 把 MTP token 分配到各 rank，并为每个 rank 生成 attention mask（`dcp_mtp_attn_mask`，shape `[num_decode_reqs, decode_threshold, max_model_len]`）。注释里有完整例子（行 1387-1401）。

slot_mapping 在 MTP 下也要做 varlen compact（`sfa_cp.py:61 _compact_varlen_decode_slot_mapping`），因为同 batch 的 decode request 可能 query 长度不同。

## 7. 通信域拓扑（PCP2, DCP2, TP4 示意）

```
8 张卡，TP4 + DCP2 + PCP2：
- TP4: 卡 0-3 一组（DCP 复用）
- DCP2: TP 组内再分 2（KV seq 分片）
- PCP2: 跨两个 TP 组（卡 0-3 与 4-7），独立通信域

decode 一次 attention 的通信：
  本地 KV attn → DCP 组 all2all(head) → PCP 组 allgather(seq) → update
```

## 8. 小结

- **DCP = 复用 TP 域的 KV seq 分片**，省冗余 KV、提吞吐；不增卡；
- 与 PCP 共享 `cp_size = pcp*dcp` 模型与 block-table 分片布局；
- 通信：head 维 allgather Q → 本地 KV attn → out/lse **DCP all2all + PCP allgather** → online-softmax 合并；
- `num_computed_tokens_of_pcp_dcp` 是连接调度器与各 rank 本地 KV 长度的桥梁。

对 RFC 的启示：DCP 是一个 **低成本、高收益的 TP 域内优化**，与 PCP 正交。上游 CP RFC 应把 PCP/DCP 设计为可独立开启、共享同一分片布局与 out/lse 合并原语的两个特性，而不是耦合在一起。
