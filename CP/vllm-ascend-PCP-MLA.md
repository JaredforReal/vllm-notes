# MLA 模型的 PCP 支持（DeepSeek-V2/V3、Kimi-K25）

> 对应原始问题：*How does vllm-Ascend support MLA model like DSV32, kimi-k25 with PCP?*

适用对象：使用 **MLA（Multi-head Latent Attention）** 的模型 —— DeepSeek-V2/V3/R1、Kimi-K25 等。
对应后端：`vllm_ascend/attention/context_parallel/mla_cp.py`（`AscendMlaCPImpl` + `AscendMlaCPMetadataBuilder`）。
选择入口：`vllm_ascend/attention/mla_v1.py:106` `get_impl_cls()` 在 `enable_cp()` 时返回 `AscendMlaCPImpl`。

## 1. MLA 的特殊性回顾

MLA 把 KV 压缩成低秩的 `kv_c`（`kv_lora_rank`，512 维），只有 `k_pe`（`qk_rope_head_dim`）单独存。decode 时 q 也拆成 `q_nope`（`kv_lora_rank`）+ `q_pe`。attention 用的是 *“先在压缩空间算 q_nope·kv_c，再用 W_UV 投影到 v_head_dim”* 的方式。这导致：
- **KV cache 维度特殊**：`kv_c_normed`（kv_lora_rank）+ `k_pe`（rope）两块 cache；
- **q/k 投影昂贵**：`kv_b_proj`、`q_proj`、`W_UV` 不想在 allgather 的整条 KV 上重复算。

因此 MLA 的 PCP **没有**像 GQA 那样对 head/tail 分别独立 attention，而是 **allgather 完整 KV，但只对必要部分做投影**，再 head/tail 拆 Q 算 attention。

## 2. KV cache 布局与 block_size 调整

`AscendMlaCPMetadataBuilder.__init__`（`mla_cp.py:60`）：

```python
self.cp_local_block_size = cp_kv_cache_interleave_size
self.cp_virtual_block_size = cp_local_block_size * dcp_size * pcp_size
self.block_size = (block_size * cp_virtual_block_size) // gcd(block_size, cp_virtual_block_size)
```

MLA 把 block_size 重新定义为本地 block 与 virtual block 的最小公倍数维度，保证 CP 分片与 MLA 块语义一致。

slot_mapping 也在 builder 里做了 CP 改写（`mla_cp.py:88 build`）：decode 部分 `slot_mapping[:num_decode_tokens]` 只取每隔 `pcp_size` 的（`[::pcp_size]`），其余填 `-1`，因为 decode token 在 PCP 组里是被复制分布的。

## 3. Prefill 流程：`mla_preprocess_prefill`（`mla_cp.py:416`）

```
1. 算本 rank 的 prefill_q（q_proj + rope）
2. prefill_kv_c_k_pe = cat([kv_c_normed, k_pe])
   allgather (PCP 组, dim=0) → index_select(pcp_allgather_restore_idx) 还原顺序
   → 取掉 decode 段后 split 回 (kv_c_normed, k_pe)
3. reshape_and_cache: 把完整 kv_c/k_pe 写到本地 cache（slot_mapping 已按 CP 分片）
4. tail 投影: 用 kv_tail_proj_idx 选出 tail 段的 kv_c_normed
   → kv_b_proj 得到 (prefill_k_nope, prefill_value)
   k_pe 用 kv_tail_proj_idx 选出并 expand 到 head 维
5. 返回 PrefillMLAPreprocessResult(q_nope, q_pe, k_nope, k_pe, value)
```

**关键优化**：`kv_b_proj` 只对 **tail 段** 的 KV 做（行 460-466），而不是整条 gather 来的 KV。因为 head 段的 Q 只看 tail 之前的 KV（已在 `kv_with_q_head_nomask_idx` 里），而 tail 段才需要 nope/value 投影参与 attention。这避免了在对 allgather 出来的整条 KV 上做昂贵的 `kv_b_proj`。

## 4. Prefill attention：`_forward_prefill`（`mla_cp.py:500`）

head / tail 两段分别调 `_attention_with_optional_kv_select`（行 567）：

- **head 段**：`kv_attn_idx = kv_with_q_head_attn_idx_in_tail`，即从 tail 投影结果里 **选出** head 需要的那部分 nope/value/k_pe（行 533-534）；
- **tail 段**：`kv_attn_idx = None`，直接用全部 tail 投影结果（行 543）。

两段都用 `npu_fused_infer_attention_score`，`query_rope=q_pe`，`key_rope=k_pe`，`sparse_mode=3`（causal），`actual_seq_lengths_kv` 来自 `head_actual_seq_lengths_kv` / `tail_actual_seq_lengths_kv`（在 `generate_pcp_metadata` 里算）。

最后 `attn_output = index_select(cat([out_head, out_tail]), q_full_idx)` 还原顺序（行 554）。若有 chunked context，再过 `_compute_prefill_context` 合并（详见 chunked-prefill 笔记第 3 节）。

## 5. Decode 流程：`_forward_decode`（`mla_cp.py:613`）

MLA decode 用 **BNSD 布局**（block table + 压缩 cache）：

```python
if dcp_size > 1:
    num_heads = self.num_heads * self.dcp_size   # DCP 组 allgather q 的 head 维
# q_nope/q_pe reshape 成 [num_tokens, num_heads, 1, dim] 或 spec decode 的 BSND
common_kwargs = dict(
    block_table=decode_meta.block_table,
    block_size=block_size,
    actual_seq_lengths_kv=decode_meta.cp_seq_len,   # 本 rank 的 KV 长度
    ...
)
attn_output, softmax_lse = npu_fused_infer_attention_score(q_nope, k_nope, k_nope, **common_kwargs)
# 合并 out/lse
attn_out_lse = _process_attn_out_lse(attn_output, softmax_lse)   # DCP all2all + PCP allgather
attn_output = _npu_attention_update(self.kv_lora_rank, attn_out_lse)
return self._v_up_proj(attn_output)   # W_UV 投影到 v_head_dim
```

`cp_seq_len`（`build_decode_metadata`, `mla_cp.py:247`）= `num_computed_tokens_of_pcp_dcp[:, pcp_rank, dcp_rank]`，即本 CP rank 上的本地 KV 长度。spec decode 时用 `dcp_mtp_attn_mask` 做 MTP 的 attention mask。

`reorg_decode_q`（行 493）：DCP>1 时把 `cat([q_nope, q_pe])` 在 head 维 allgather 再 split，使 DCP 组内 Q head 一致。

`_v_up_proj`（行 407）：把合并后的 `[B, num_heads, kv_lora_rank]` 用 `W_UV`（bmm）投影成 `[B, num_heads*v_head_dim]`，这是 MLA 特有的最后一步。

## 6. `kv_tail_proj_idx` 等索引从哪来

在 `PCPManager.generate_pcp_metadata`（`pcp_utils.py:1115-1218`）里，对每条 prefill 请求（`chunk_len = seq_len//2`）：

```python
tail_proj_offset = len(kv_tail_proj_idx)
tail_proj_len = chunk_len * (q_tail_chunk_id + 1)
kv_tail_proj_idx.extend(range(kv_req_offset, kv_req_offset + tail_proj_len))
kv_with_q_head_attn_idx_in_tail.extend(range(tail_proj_offset, tail_proj_offset + chunk_len*(q_head_chunk_id+1)))
kv_with_q_tail_attn_idx_in_tail.extend(range(tail_proj_offset, tail_proj_offset + tail_proj_len))
```

这些索引编码了 “tail 段需要投影哪些 KV” + “head/tail attention 从 tail 投影结果里取哪些”。

## 7. 约束（来自用户指南）

MLA 模型用 DCP 时：
```
tensor_parallel_size >= decode_context_parallel_size
tensor_parallel_size % decode_context_parallel_size == 0
```
因为 MLA 的 attention head 在 TP 域内分摊，DCP 复用 TP 域，必须能整除。PCP 不受此约束（独立通信域）。

## 8. 小结：MLA-PCP 的设计精髓

1. **allgather 完整 kv_c+k_pe（压缩表示），但只对 tail 段做昂贵的 kv_b_proj** —— 兼顾正确性与算力；
2. **head/tail 拆 Q，用预计算索引从 tail 投影结果里选 KV** —— 避免重复投影；
3. **decode 用 BNSD + block_table + cp_seq_len**，out/lse 经 DCP all2all + PCP allgather 合并，最后 `_v_up_proj`；
4. **block_size 按 cp_virtual_block_size 调整**，slot_mapping 按 CP 分片改写。

对 RFC 的启示：MLA 这类“KV 有压缩、投影昂贵”的结构，CP 不能简单 allgather 整条高维 KV，而要 **在压缩空间 gather + 选择性投影**。上游需要一个能表达 “gather 哪些、投影哪些” 的接口（对应这里的 `kv_tail_proj_idx` / `kv_with_q_*_attn_idx_in_tail`）。
