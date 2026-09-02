# vLLM-Ascend PCP 的序列切分：Head-Tail（DualChunkSwap / Zigzag）

> 对应原始问题：*How does vllm-Ascend deal with PCP? zigzag? how it works*

## 1. 为什么需要“非朴素”切分

如果把一条长度为 L 的 prefill 序列朴素地切成 `pcp_size` 段（rank0 拿 `[0, L/N)`，rank1 拿 `[L/N, 2L/N)`…），会有两个严重问题：

1. **负载不均衡**：因果注意力里，靠后的 Q 能看到更多 KV（attention 计算量随位置单调递增）。朴素切分会让靠后的 rank 计算量远大于靠前的 rank，导致长尾。
2. **allgather 后顺序错乱**：PCP 经常需要 allgather Q 或 KV 的结果（每层 gather 完整 KV），不同 rank 拿到的段拼接后顺序与原始序列不一致，需要还原索引。

vLLM-Ascend 的解法叫 **DualChunkSwap / Head-Tail 风格**（设计图 `docs/.../assets/cp/head-tail-style.png`），本质就是 zigzag 配对：

> 把序列（pad 到 `2*pcp_size` 的倍数后）切成 `2*pcp_size` 等份，第 0 份配第 `2N-1` 份，第 1 份配第 `2N-2` 份……每对分配给一个 rank。这样每个 rank 拿到的都是“一段靠前（计算量小）+ 一段靠后（计算量大）”，前后配平，**计算量近似相等**。

---

## 2. 核心实现：`PCPManager.update_tokens_for_pcp`

**源码：`vllm_ascend/worker/pcp_utils.py:502`**（`update_tokens_for_pcp`，由 `model_runner_v1.py:839` 调用）。

### 2.1 步骤拆解

```python
# 1. pad 每条 prefill 序列长度到 2*pcp_size 的倍数（DualChunkSwap 要求对齐）
num_padded_scheduled_tokens =
    ceil(num_scheduled_tokens / (2*pcp_world_size)) * (2*pcp_world_size)

# decode 请求不切，而是在各 rank 上复制（duplication）
num_padded_scheduled_tokens[:num_decode_reqs] *= pcp_world_size

# 记录 padding 数量
self.num_pcp_pads_cpu = num_padded_scheduled_tokens - num_scheduled_tokens

# 每个 rank 实际处理的 token 数 = padded_len / pcp_world_size
pcp_tokens = num_padded_scheduled_tokens // pcp_world_size

# 每条 prefill 进一步切成 head/tail 两个 chunk（chunk_size = pcp_tokens/2）
pcp_chunk_sizes = (pcp_tokens // 2).clip(min=1)
```

### 2.2 每个位置的绝对位置怎么算（`get_current_rank_positions`, 行 604）

对 rank `r`：
- **head chunk**：从 `positions_start_loc + r * chunk_size` 开始；
- **tail chunk**：从 `positions_start_loc + (2*pcp_world_size - r - 1) * chunk_size` 开始。

也就是说 rank0 的 head 取第 0 段、tail 取第 `2N-1` 段；rank1 取第 1 段 + 第 `2N-2` 段……这正是 **zigzag 首尾配对**。

函数内带了一个完整 Example（行 543-554），值得直接看：

```
tokens = [1, 5, 8], pcp_world_size = 2
pcp_rank=0 -> pcp_tokens=[1,4,4], positions=[0, 0,1,6,7, 0,1,6,7]
pcp_rank=1 -> pcp_tokens=[1,4,4], positions=[0, 2,3,4,5, 2,3,4,5]
num_pcp_pads      = [1,3,0]
pcp_unpad_mask    = [T,F, T,T,T,T,T,  F,F,  F,T,T,T,T,T,T,T]
pcp_allgather_restore_idx = [0,9,1,2,10,11,12,13, 3,4,5,6,14,15,16,17, 7,8]
```

可以验证：rank0 拿到的是 `[seg0, seg3]`（0 号段 + 最后一段），rank1 拿到 `[seg1, seg2]`。

### 2.3 还原索引 `pcp_allgather_restore_idx`

allgather 后，buffer 里是 `[rank0 的 tokens..., rank1 的 tokens..., ...]`，每个 rank 内部又是 head-tail 交错的段。`pcp_allgather_restore_idx` 就是把这些交错的 token 重新排成 **原始顺序** 的下标（行 640-645）：

```python
all_positions = concat([get_current_rank_positions(padded_pos_start_loc, rank_i)
                        for rank_i in range(pcp_world_size)])
self.pcp_allgather_restore_idx = all_positions.argsort()
```

这个索引在三个地方用：
- `reshape_and_cache` / `mla_preprocess_prefill`：allgather KV 后用 `index_select` 还原顺序（`attention_cp.py:834`、`mla_cp.py:444`）；
- `get_restore_hidden_states`：最后一层后 allgather hidden states 还原（`pcp_utils.py:844`）；
- chunked prefill 的 KV 恢复（`attention_cp.py:176` `kv_inverse_idx_for_chunk`）。

---

## 3. attention 后端的 head/tail 拆分索引

切分不仅发生在“哪些 token 归哪个 rank”，还要为 attention 计算准备 **Q 的 head/tail 下标** 和 **KV 的 nomask/mask 下标**。这部分在 `PCPManager.generate_pcp_metadata()`（`pcp_utils.py:1066`，行 1115-1218）里生成，存进 `AscendPCPMetadata`。

对每条 prefill 请求（`chunk_len = seq_len // 2`）：

```python
q_head_chunk_id = pcp_world_rank                          # head 段 id
q_tail_chunk_id = pcp_world_size * 2 - 1 - pcp_world_rank # tail 段 id（首尾配对）

q_head_idx = 本 rank 的 head 部分 Q 下标
q_tail_idx = 本 rank 的 tail 部分 Q 下标

# 对 head 段做 attention 时：
kv_with_q_head_nomask_idx = rank 之前所有段的 KV（完全可见，不用 mask）
kv_with_q_head_mask_idx   = head 段对应的 KV（需要 causal mask）
# tail 段同理
```

> 含义：当算 head 段的 Q 时，它**之前**的 KV（更靠前的段）是完整的、不需要 mask 的，可以走 nomask 路径（更快，`sparse_mode=0`，`atten_mask=None`）；它**自己**那一段需要 causal mask（`sparse_mode=3`）。这样把一次 attention 拆成 **nomask + mask 两次** 调用 `npu_fused_infer_attention_score`，再用 `_npu_attn_out_lse_update` 用 online-softmax 合并（见 `attention_cp.py:_attention_with_nomask_and_mask`，行 421）。

最后用 `q_full_idx`（行 1186，对 `[head, tail]` 拼接做 `argsort`）把两段 attention 输出还原成 Q 的本 rank 原顺序（`_forward_prefill_cp_post`，`attention_cp.py:558`）。

---

## 4. MLA 的特殊点

MLA 后端（`mla_cp.py`）用的是 **allgather KV** 而不是 head-tail 各自独立算。它在 `generate_pcp_metadata` 里额外生成了：
- `kv_tail_proj_idx`：tail 段需要把 gather 来的 `kv_c` 投影成 nope/value（`kv_b_proj`）的索引，**只投影 tail 需要的那部分**（行 1167-1169），避免对整条 KV 都做昂贵的 `kv_b_proj`；
- `kv_with_q_head_attn_idx_in_tail` / `kv_with_q_tail_attn_idx_in_tail`：tail 段 attention 的 KV 选择索引。

MLA prefill 流程（`mla_cp.py:416 mla_preprocess_prefill`）：
1. 本 rank 算 q；
2. allgather `kv_c + k_pe` 并用 `pcp_allgather_restore_idx` 还原顺序；
3. `reshape_and_cache` 写本地分片；
4. 用 `kv_tail_proj_idx` 选出 tail 段的 `kv_c`，过 `kv_b_proj` 得到 nope/value；
5. head/tail 分别 attention，`q_full_idx` 还原。

---

## 5. DSA 不是 zigzag（重要区别）

DSA 后端（DeepSeek-V4 / GLM-5，`dsa_cp.py`）**不走 head-tail**，而是沿 flatten 后的 token 流 **按 TP/CP rank 线性切片**。见 `AscendDSACPMetadataBuilder._build_local_token_metadata`（`dsa_cp.py:645`）：

```python
num_tokens_pad = ((num_input_tokens + tp_size - 1) // tp_size) * tp_size
tokens_per_rank = num_tokens_pad // tp_size
local_start = tp_rank * tokens_per_rank
local_end   = local_start + tokens_per_rank
# 每个 request 的全局 token 区间与本 rank 的 [local_start, local_end) 求交
```

它用 `DSACPMetadata`（`dsa_cp.py:65`）保存 `local_query_start_loc` / `local_seq_lens` / `local_start` / `local_end` / `tokens_per_rank` / `num_tokens_pad` 以及本 rank 切片的 `local_cos` / `local_sin`。这是因为 DSA 的稀疏注意力 + compressor/indexer 结构与普通 attention 差异大，线性切片更自然。**写 RFC 时要注意：zigzag 不是唯一切法，稀疏/混合架构可能需要不同策略。**

---

## 6. 小结：zigzag 的工作流

```
scheduler 给出整条序列 (num_scheduled_tokens per req)
        │
        ▼  update_tokens_for_pcp
pad 到 2N 倍数 → 切 2N 段 → 首尾配对分配到 N 个 rank
        │  产出: pcp_positions(本 rank 绝对位置), num_pcp_pads, pcp_unpad_mask,
        │        pcp_allgather_restore_idx(allgather 后还原顺序)
        ▼  generate_pcp_metadata
生成 per-layer 索引: q_head/tail_idx, kv_nomask/mask_idx, q_full_idx,
                    kv_tail_proj_idx(MLA), num_computed_tokens_of_pcp_dcp(decode)
        ▼  各 attention 后端 forward
head 段(nomask+mask) 与 tail 段(nomask+mask) 各自 attention
        → q_full_idx 还原 → (chunked prefill 还要合并 context 部分)
```

zigzag 的核心价值：**负载均衡**（首尾配对）+ **allgather 友好**（每 rank 等量，配 `restore_idx` 还原）。这是 vLLM 上游 CP 实现可以直接借鉴的标准做法。
