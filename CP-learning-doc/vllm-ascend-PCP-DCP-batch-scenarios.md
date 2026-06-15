# PCP/DCP 开启后的三种 Batch 场景：实现与通信

> 对应问题：*PCP/DCP 开启后，纯 prefill、纯 decode、prefill+decode 混合，分别是怎么实现的？通信怎么走？*
>
> 主线后端：**GQA**（`vllm_ascend/attention/context_parallel/attention_cp.py`，`AscendAttentionCPImpl`），MLA 差异见第 6 节。
>
> 前提：`cp_size = pcp_size * dcp_size`，`cp_rank = pcp_rank * dcp_size + dcp_rank`。本文统一用「PCP 域 = 跨卡独占通信域（额外卡）」「DCP 域 = 复用 TP 域」这两个概念。

---

## 0. 前置：PCP/DCP 下 token 在一个 batch 里的统一布局

无论哪种 batch，`forward_impl`（`attention_cp.py:951`）都把输入 token **排成 `[decode_tokens | prefill_tokens]`**：

```python
has_decode  = num_decodes > 0
has_prefill = num_prefills > 0
num_decode_tokens = attn_metadata.num_decode_tokens

if has_decode:
    decode_query = query[:num_decode_tokens]                      # decode 在前
    output_decode = self._forward_decode_pcp_dcp(decode_query, ...)
    output[:num_decode_tokens] = output_decode
if has_prefill:
    num_actual_tokens_pcp_padded = num_actual_tokens_pcp_padded // pcp_size
    prefill_query = query[num_decode_tokens : num_actual_tokens_pcp_padded]   # prefill 在后
    key = key[pcp_size * num_decode_tokens : num_actual_tokens_pcp_padded]    # decode 在 KV 里复制了 pcp_size 份
```

两条关键规则：
1. **decode token 在 batch 最前面**，prefill token 紧随其后；
2. **decode token 在 PCP 组里被复制 `pcp_size` 份**（每个 rank 都拿到同一批 decode token），所以 KV 投影缓冲里 decode 部分的长度是 `num_decode_tokens * pcp_size`，prefill 的 key/value 起点要偏移 `pcp_size * num_decode_tokens`。

> 三种 batch 场景的差别，本质上就是 `has_decode` / `has_prefill` 哪个为真、以及各自走哪条通信路径。下面分别拆。

---

## 场景 A：纯 Prefill（`has_prefill=True, has_decode=False`）

典型场景：长序列首 token / 长 prompt 一次性 prefill。这是 **PCP 收益最大**的场景（拆序列降 TTFT），DCP 在纯 prefill 里作用小（DCP 主要省 decode 的冗余 KV）。

### A.1 reshape_and_cache：写 KV cache（`attention_cp.py:829`）

普通 GQA（非 hybrid）走 **AllGather-KV**（Mode 1）：

```python
kv = cat([key, value], dim=-1)                                    # 本 rank 算出的那段 K/V
all_kv = get_pcp_group().all_gather(kv[...local...].contiguous(), dim=0)   # ① PCP allgather → 完整序列 KV
all_kv = index_select(all_kv, 0, pcp_allgather_restore_idx)      # ② 用 restore_idx 还原成原始顺序
key, value = all_kv.split([head_size, head_size], dim=-1)
# ③ 按本 rank 的分片 slot_mapping 写本地 cache
```

- **通信①**：PCP 域 allgather KV（每层一次，算完即丢，控制峰值显存）。
- restore_idx 把各 rank 拼出的 `[rank0段 | rank1段 | …]`（DualChunkSwap head/tail 交错）还原成全局顺序，再按本 rank 的 slot 写 cache。

### A.2 prefill forward：head/tail + nomask/mask（`_forward_prefill_cp_*`）

KV 已经是完整序列后，本 rank 的 Q（head 段 + tail 段）对完整 KV 做两次 attention（`attention_cp.py:485`）：

```python
# pre: 从完整 KV 里选「我之前的段(nomask)」+「我自己(mask)」
data_head = _forward_prefill_cp_pre(prefill_query, key, value, ...)        # attention_cp.py:500
# attn: head 段做 nomask(无mask快路径) + mask(causal) 两次
output_head, lse_head = _forward_prefill_cp_attn(data_head, is_head=True)   # 行 421/1000
output_tail, lse_tail = _forward_prefill_cp_attn(data_tail, is_head=False)  # 行 1034
# post: head/tail 输出用 q_full_idx(argsort) 还原本 rank Q 的顺序
attn_output_prefill = _forward_prefill_cp_post([out_head, out_tail], ...)   # 行 558
```

- **通信**：纯 prefill 这一步 **没有跨 rank 通信**（KV 已在 A.1 gather 完整，本 rank 的 Q 对完整 KV 算出的结果就是最终结果）。
- 这里 `_npu_attn_out_lse_update` 合并的是 **nomask 段 vs mask 段**（同一 rank 内 KV 的两半），**不是跨 rank 合并**。
- 性能点：把 causal attention 拆成「完全可见 KV 走无 mask 快路径」+「自己那段走 causal」，省掉一半 mask 计算。

### A.3 纯 prefill 通信小结

| 步骤 | 计算 | 通信 |
|---|---|---|
| reshape_and_cache | 还原 KV 顺序 + 写 cache | **PCP allgather KV**（每层，full KV） |
| prefill attn | head/tail 各 nomask+mask attn | 无（本地即最终） |

> 结论：纯 prefill 是教科书 **Mode 1（Partial Q, Full KV）**，唯一通信是每层一次 PCP allgather KV。DCP 在纯 prefill 下基本不参与。

---

## 场景 B：纯 Decode（`has_decode=True, has_prefill=False`）

典型场景：已 prefill 完、纯生成。这是 **PCP + DCP 同时发力**的典型路径（PCP 让 decode token 跨 pcp rank 复制；DCP 把 KV cache 沿 seq 分片省显存、提 batch）。

### B.1 reshape_and_cache：decode 只写一份 KV（`attention_cp.py:816`）

decode token 每个 rank 都有副本，但只写真实 slot：

```python
if has_decode:
    slot_mapping = attn_metadata.slot_mapping[: num_decode_tokens * self.pcp_size : self.pcp_size]  # 取 [::pcp_size]
    reshape_and_cache(key[:num_decode_tokens], value[:num_decode_tokens], ..., slot_mapping=slot_mapping)
```

- **通信**：无。
- `[::pcp_size]` 表示只取 rank0 的副本对应的真实 slot，其余副本 slot 为 `-1`（不写），保证 KV 只落一份到该 token 应属的 CP rank 分片。配合 `block_table.py:170` 的 round-robin 分片布局。

### B.2 decode forward：DCP head-allgather Q → 本地 attn → out/lse 合并（`_forward_decode_pcp_dcp`, `attention_cp.py:566`）

```python
# ① DCP 域：head 维 allgather Q，让 DCP 组内 Q head 一致
if dcp_size > 1:
    query = get_dcp_group().all_gather(query.contiguous(), 1)
    num_heads = self.num_heads * dcp_size
# ② 本地（已分片）KV cache 做 attention，actual_seq_lengths_kv = 本 rank 本地 KV 长度
attn_out, attn_lse = npu_fused_infer_attention_score(
    query, k_nope, value,
    actual_seq_lengths_kv=decode_meta.num_computed_tokens_of_pcp_dcp[:num_decodes, pcp_rank, dcp_rank],
    actual_seq_lengths=actual_seq_lengths_q[:num_decodes],
    block_table=decode_meta.block_tables, softmax_lse_flag=True, ...)
# ③ 合并 out/lse：先 DCP all2all，再 PCP allgather，最后 online-softmax
attn_out_lse = _process_attn_out_lse(attn_out, attn_lse)            # common_cp.py:108
attn_out = _npu_attention_update(self.head_size, attn_out_lse)       # common_cp.py:130
```

### B.3 `_process_attn_out_lse`（`common_cp.py:108`）—— 合并枢纽

```python
attn_out_lse = cat([attn_output, softmax_lse], dim=-1)          # [bs, h, d+1]
if dcp_size > 1:
    attn_out_lse = attn_out_lse.permute([1,2,0]).contiguous()
    dist.all_to_all_single(..., group=dcp_group)                # ① DCP head 维 all2all
    attn_out_lse = attn_out_lse.permute([2,0,1])
if pcp_size > 1:
    attn_out_lse = get_pcp_group().all_gather(attn_out_lse.contiguous(), dim=0)  # ② PCP seq 维 allgather
# ③ _npu_attention_update：硬件 online-softmax 归并各 rank 的 partial (out, lse)
```

合并顺序固定：**DCP all2all（head） → PCP allgather（seq） → `npu_attention_update`（logsumexp 合并）**。

### B.4 纯 decode 通信小结

| 步骤 | 计算 | 通信 |
|---|---|---|
| reshape_and_cache | 写本地 KV（只一份，`[::pcp_size]`） | 无 |
| decode attn 准备 | 本地 KV attention | **DCP allgather Q(head)** |
| decode attn 合并 | online-softmax 归并 partial out/lse | **DCP all2all(head)** → **PCP allgather(seq)** → update |

> 结论：纯 decode 是教科书 **Mode 2（Partial Q, Partial KV）**，但**不用 Ring Attention**——用「DCP head-allgather Q + 本地 partial-KV attn + out/lse 在线 softmax 合并」实现。每 rank 持有 partial KV，Q 被 head-allgather，结果靠 logsumexp 合并。`num_computed_tokens_of_pcp_dcp[:, pcp_rank, dcp_rank]` 是连接调度器与各 rank 本地 KV 长度的桥梁。

---

## 场景 C：Prefill + Decode 混合（`has_decode=True, has_prefill=True`）

典型场景：**continuous batching 实际运行的最常见形态**——同一 step 里既有新 prefill 的长 prompt，又有正在生成的 decode。这是三种场景里最复杂的，因为两条完全不同的通信路径要在**同一层 forward**里共存。

### C.1 一次 forward 同时处理两段（`forward_impl`, `attention_cp.py:951`）

```python
if has_decode:
    decode_query = query[:num_decode_tokens]                          # decode 段
    output_decode = self._forward_decode_pcp_dcp(decode_query, ...)   # 走场景 B 的路径
    output[:num_decode_tokens] = output_decode
if has_prefill:
    prefill_query = query[num_decode_tokens : ...]                    # prefill 段
    key = key[pcp_size * num_decode_tokens : ...]                     # KV 起点偏移 pcp_size*decode
    # prefill 段走场景 A 的路径（zigzag allgather-KV / 或 chunked prefill）
    ...
    output[num_decode_tokens : ...] = attn_output_prefill
```

关键点：**两条路径是串行拼接，不是合并**——decode 先算完（含它自己的 DCP a2a + PCP ag），prefill 再算（含它自己的 PCP allgather KV）。它们共享同一份 `attn_metadata`，但 metadata 内部把 decode（`AscendMetadataForDecode`）和 prefill（`AscendMetadataForPrefill`）的索引/长度分开（`common_cp.py:99/73`）。

### C.2 slot_mapping 如何同时覆盖两段

混合 batch 的 `slot_mapping` 是连续的一维 buffer，前半段是 decode（复制 pcp_size 份，取 `[::pcp_size]` 写一份），后半段是 prefill（按本 rank 分片写）：

```python
# decode 部分（行 816）
slot_mapping_decode = slot_mapping[: num_decode_tokens * pcp_size : pcp_size]
# prefill 部分（行 845-856）：用 prefill 段的 pcp_padded_slot_mapping 写
reshape_and_cache(key, value, ..., slot_mapping=slot_mapping)
```

`reshape_and_cache` 一次性把 decode + prefill 的 K/V 都写进 cache，decode 部分靠 stride 跳过副本，prefill 部分靠分片 slot。

### C.3 prefill 段还可能叠 chunked prefill（多流 overlap）

如果 prefill 段带 `chunked_context`（`has_chunked_context=True`，行 975），prefill 那段会进一步切出 context 部分，用 **AllGatherQ**（Mode 2 变体）+ 多流 overlap：

```
current_stream : init -- pre -- head attn ------------------ tail attn -- post -- update
context part                                                                     -/
current_stream : -----                    -- context attn --                     -/
COMM_STREAM    :         \-- all_gather Q --/                  \-- a2a ag output --/
```

- **通信**：PCP allgather Q（`_prefill_query_all_gather`, 行 716）+ context output 的 DCP all2all + PCP allgather（`_gather_global_context_output`, 行 919）+ online-softmax 合并。
- decode 段的通信（场景 B 的 a2a+ag）与 prefill context 的通信在不同 stream 上，可以 overlap。

### C.4 混合 batch 通信小结

| 阶段 | decode 段通信 | prefill 段通信 |
|---|---|---|
| reshape_and_cache | 无（`[::pcp_size]`） | **PCP allgather KV**（普通）/ 或无（hybrid: allgather QKV） |
| attn 计算 | 本地 KV attn | head/tail nomask+mask（普通）/ chunked context attn |
| 合并 | **DCP a2a(head) + PCP ag(seq) + update** | 无（纯 prefill）/ **PCP ag Q + DCP a2a + PCP ag(output)**（chunked） |

> 结论：混合 batch = **场景 A + 场景 B 串行拼接**，两段各自的通信路径不变，只是共享 metadata 和一次 reshape_and_cache。复杂度来自 prefill 段可能叠 chunked prefill 的多流 overlap，以及 slot_mapping 要同时正确表达两段。

---

## 5. 三场景通信汇总表（GQA 后端，PCP+DCP 全开）

| 场景 | 主要 Mode | reshape_and_cache 通信 | attn 计算通信 | out 合并通信 |
|---|---|---|---|---|
| **纯 Prefill** | Mode 1 | PCP allgather **KV**（full，每层） | 无（本地 head/tail vs full KV） | 无 |
| **纯 Decode** | Mode 2 | 无（`[::pcp_size]` 写一份） | DCP allgather **Q**(head) | DCP **all2all**(head) + PCP **allgather**(seq) + npu_update |
| **混合** | Mode 1 + Mode 2 串行 | decode 无 + prefill PCP ag KV | decode: DCP ag Q；prefill: head/tail | decode: a2a+ag+update；prefill: 无（或 chunked 的 ag Q + a2a+ag） |

观察：
- **通信密度：纯 decode > 混合 > 纯 prefill**（decode 每层都要 a2a+ag 合并，prefill 每层只 ag KV 一次）；
- **PCP allgather 用在**：prefill 的 KV、decode/混合的 out(lse) seq 维；
- **DCP all2all/allgather 用在**：decode/混合的 Q head 维 与 out(lse) head 维；
- DCP 在纯 prefill 里几乎不出现（KV cache 分片对 prefill 计算无意义，省的是 decode 的冗余）。

---

## 6. MLA 后端的差异（`mla_cp.py`）

MLA（DSV2/V3、Kimi-K25）结构不同（压缩 kv_c + k_pe），但三场景的通信骨架与 GQA 一致，差异在：

| 场景 | GQA | MLA |
|---|---|---|
| 纯 Prefill reshape_and_cache | allgather 完整 KV | allgather **kv_c + k_pe**（`mla_cp.py:442`），只对 tail 段做昂贵的 `kv_b_proj`（`kv_tail_proj_idx`） |
| 纯 Decode | DCP ag Q(head) + 本地 attn + a2a/ag/update | 同骨架，额外 `_v_up_proj` 解压缩，`actual_seq_lengths_kv = cp_seq_len`（`mla_cp.py:613`） |
| 混合 / chunked prefill | AllGatherQ | **AllGatherKV** + `_reorg_kvcache`（`mla_cp.py:767`）—— MLA 在 chunked prefill 选 AllGatherKV（与 GQA 相反） |

> MLA 的 chunked prefill 用 AllGatherKV（而非 GQA 的 AllGatherQ）的原因：MLA 的压缩 KV 很小，gather 完整 kv_c 比 gather Q 更省通信量。

DSA / SFA 后端的切分与通信又不同（DSA 走 TP 域线性切片，SFA 走 compact block gather），见 `vllm-ascend-PCP-DSA.md` / `vllm-ascend-chucked-prefill-PCP.md`。

---

## 7. 对 vLLM 上游 RFC 的启示

1. **三种 batch 场景应作为 CP 实现的回归基线**：纯 prefill（验证 PCP allgather-KV 正确性）、纯 decode（验证 out/lse 在线 softmax 合并 + DCP a2a/ag 顺序）、混合（验证 metadata 双段切分 + slot_mapping 共存）。这是最容易暴露 bug 的三组 case。
2. **混合 batch 的串行拼接**说明 CP 的 decode 路径与 prefill 路径应解耦设计，共用 metadata 但各自独立通信，避免把两套合并逻辑耦合在一个 kernel 里。
3. **out/lse 合并原语**（`_process_attn_out_lse` + `_npu_attention_update`）是 decode/混合场景的核心，应抽成框架级 API，统一「DCP a2a → PCP ag → update」顺序供所有后端复用。
4. **decode token 跨 PCP rank 复制 + `[::pcp_size]` 只写一份**是 KV 不冗余的关键，上游需在 block-table/slot_mapping 层统一支持，而非每个 backend 各写一遍。
5. **通信量排序**（decode > 混合 > prefill）提示：CP 优化的重点应放在 decode 的 out/lse 合并（频次最高、batch 最大），prefill 的 allgather-KV 借助「算完即丢」控制峰值显存即可。
