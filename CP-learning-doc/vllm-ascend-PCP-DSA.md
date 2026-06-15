# DSA 模型的 PCP 支持（DeepSeek-V4、GLM-5）

> 对应原始问题：*How does vllm-Ascend support MLA model like DSV32, GLM-5 with PCP?*
>
> 注：DSV4 / GLM-5 用的是 **DSA（DeepSeek Sparse Attention）**，与 DeepSeek-V2/V3 的 MLA、Kimi-K25 的 MLA 不同。DSA = 稀疏注意力 + compressor（压缩 KV）+ indexer（lightning indexer 选 top-k block）。对应后端 `dsa_cp.py`（`AscendDSACPImpl` + `AscendDSACPMetadataBuilder`），非 `mla_cp.py`。

选择入口：`vllm_ascend/attention/dsa_v1.py:209` `get_impl_cls()` 在 `enable_cp()` 时返回 `AscendDSACPImpl`。

## 1. DSA 结构回顾

DSA attention（见 `dsa_cp.py:870 AscendDSACPImpl.__init__`）由几部分组成：
- **indexer**（行 932-949）：`npu_quant_lightning_indexer` 选出每个 query 要 attend 的 top-k block（`compress_topk_idxs`）；
- **compressor**（行 951-961）：`compressor_ratio`（4 或 128）把 KV 压缩成 `cmp_kv` cache；
- **SWA（sliding window）KV**：原始（未压缩）KV cache，带 sliding window mask；
- 最终用 `npu_sparse_attn_sharedkv`（行 1200-1233）做稀疏 + 滑窗混合注意力。

KV cache 是多元组：`(compress_kv_cache, swa_kv_cache, state_cache, indexer_state_cache, indexer_k_cache, indexer_scale_cache)`。

## 2. DSA-CP 的切分方式：线性切片（非 zigzag）

**这是 DSA 与 GQA/MLA/SFA 最大的不同**。DSA 不走 head-tail，而是沿 flatten 的 token 流 **按 TP rank 线性切片**（`dsa_cp.py` 里 CP 复用了 TP 的切分逻辑）。

核心函数 `_build_local_token_metadata`（`dsa_cp.py:645`），注释里有完整例子（行 656-669）：

```
TP size 3, num_input_tokens=45, query_start_loc=[0,1,3,6,10,15,21,28,36,45]
=> 9 个 request, seq lens [1..9]
对 tp_rank 1: local_start=15, local_end=30, tokens_per_rank=15
local_query_lens = [0,0,0,0,0,6,7,2,0]   # 只有跨越本 rank 切片的 request 才有 token
local_seq_lens   = [0,0,0,0,0,6,7,2,0]
```

逻辑：
```python
num_tokens_pad = ((num_input_tokens + tp_size - 1)//tp_size) * tp_size
tokens_per_rank = num_tokens_pad // tp_size
local_start = tp_rank * tokens_per_rank
local_end   = local_start + tokens_per_rank
# 每个 request 的全局区间 [query_start_loc[i], query_start_loc[i+1]) 与 [local_start, local_end) 求交
local_query_start = clamp(query_start_loc[:-1], local_start, local_end)
local_query_end   = clamp(query_start_loc[1:],  local_start, local_end)
local_query_lens  = local_query_end - local_query_start
# 跨边界的 request 要减去落在后续 rank 的 token（offset）
offset = query_start_loc[1:] - local_query_end
local_seq_lens = (local_query_lens > 0) * (seq_lens - offset)
```

这些切片信息存进 `DSACPMetadata`（`dsa_cp.py:65`）：
```python
@dataclass
class DSACPMetadata:
    local_query_start_loc: Tensor
    local_seq_lens: Tensor
    local_start: int
    local_end: int
    tokens_per_rank: int
    num_tokens_pad: int
    local_sin / local_cos: Tensor   # 本 rank 切片对应的 RoPE 表
```

`pad`（`num_tokens_pad`）让每个 rank 切片等长，简化稀疏 kernel 的 tiling。

## 3. Prefill 流程：`AscendDSACPImpl._forward`（`dsa_cp.py:1010`）

```
1. (可选 overlap) allgather hidden_states（need_gather_q_kv 时，多流）
2. q = wq_a -> q_norm -> wq_b -> reshape(num_heads, head_dim) -> q_rms
   q 用 local_cos/local_sin 做部分 RoPE（inplace_partial_rotary_mul）
3. kv = wkv(hidden_states) -> kv_norm -> reshape(1, nope+rope) -> 部分 RoPE
   npu_scatter_nd_update_v2 写 swa_kv_cache（slot_mapping 是 2D [block, offset]）
4. 若 compress_ratio > 1:
   - _update_indexer_cache: 更新 indexer 的压缩 KV cache（FP8/int8 量化）
   - _indexer_select_topk: npu_quant_lightning_indexer 选 top-k block（用 local 切片）
   - compressor 算压缩 KV，写 compress_kv_cache
5. npu_sparse_attn_sharedkv(q, ori_kv=swa_kv_cache, cmp_kv=compress_kv_cache, ...)
   => attn_output
```

所有 `actual_seq_lengths_query` / `seqused_kv` 都用 **local** 版本（`cp_metadata.local_query_start_loc` / `local_seq_lens`），即本 rank 的切片长度。

### indexer/compressor 的 CP

- indexer select（`_indexer_select_topk`, 行 1324）用 `local_seq_lengths_query` / `local_seq_lengths_key`，在全 KV cache 上做稀疏选择，但因为 query 是 local 切片，top-k 自然只针对本 rank 的 query；
- compressor（行 1157）用 `actual_seq_lengths_query`（global，因为是更新 cache），`start_pos` 控制压缩块的位置；
- SAS / QLI metadata（`_build_sas_metadata` 行 756、`_build_qli_metadata` 行 814）用 `npu_sparse_attn_sharedkv_metadata` / `npu_quant_lightning_indexer_metadata` 算子预计算稀疏 mask 元数据，CP 下用 local 的 query_start_loc / seq_lens。

## 4. output 的 TP head 还原：`_restore_tp_head_layout`（`dsa_cp.py:1236`）

DSA 的 attention head 按 TP 分组（`n_local_groups`），算完后要做 head 维 all_to_all 还原：

```python
# 先用 local_cos/-local_sin 做反向 RoPE（解旋转）
# 再 permute 成 [tp_size, num_tokens, n_local_heads, head_dim] -> all_to_all_single(TP 组)
send = local_attn_output.view(num_tokens, tp_size, n_local_heads, head_dim).permute(1,0,2,3)...
recv = empty_like(send)
dist.all_to_all_single(recv, send, group=tp_group.device_group)
```

DSA-CP 的 CP 维度其实是借用了 TP 的切分（线性切片按 `tp_rank`），所以 output 还原走 TP 的 all_to_all。

## 5. slot_mapping：2D 布局

DSA 用 **2 维 slot_mapping** `[block_idx, block_offset]`（`dsa_cp.py:213-217, 319-321`）：

```python
self.slot_mapping[:num_input_tokens] = torch.stack(
    [slot_mapping // block_size, slot_mapping % block_size], dim=-1)
```

因为 DSA 的 KV cache 形状是 `[block_nums, block_size, head_num, head_dim]`，scatter 用 2D 索引更高效。block_size 默认 128（`dsa_cp.py:152`）。

## 6. spec decode / graph 兼容

- `build_for_drafting`（行 347）：每个 draft step 单独构建 metadata，RoPE 不缓存（`use_cache=False`，避免 draft step 间互相覆盖）；
- `build_for_graph_capture`（行 848）：只支持 DecodeOnly / SpecDecoding 状态做 graph capture（prefill 不进图）；
- `decode_threshold = 1 + num_speculative_tokens`（行 220-238），且 `<= 16`（npu_fused_infer_attention_score TND 布局限制）。

## 7. GLM-5 vs DeepSeek-V4

两者都用 DSA 后端，差异在 `hf_config`：
- `compressor_ratio`（DSV4 可能为 4 或 128，GLM-5 可能不同）决定走哪条 `npu_sparse_attn_sharedkv` 分支（行 1199/1207/1221）；
- `deepseek_v4` model_type 会额外构造 Hadamard 矩阵用于 indexer（行 189-202，`hadamard_transform_ref` / `rotate_activation`）；
- `sliding_window`、`index_topk`、`index_head_dim` 等超参从 `hf_config` 读。

## 8. 小结：DSA-PCP 的特点

1. **线性切片（TP-rank 风格）而非 zigzag** —— 稀疏 + 压缩结构下，线性切分更自然，且能复用 TP 的 all_to_all 还原；
2. **local metadata 全套**：local_query_start_loc / local_seq_lens / local_cos / local_sin / start_pos；
3. **2D slot_mapping** 适配 `[block, offset, head, dim]` cache；
4. **多流预处理 overlap**（`multistream_dsa_preprocess`）把 hidden_states allgather 与 q/kv 投影 overlap；
5. CP 维度借用 TP 域，所以没有独立的 PCP allgather（与 GQA/MLA 不同）。

对 RFC 的启示：**稀疏/压缩注意力模型需要一套不同于 zigzag 的 CP 接口** —— 基于线性 token 切片 + local metadata + (block,offset) 索引。上游若统一 CP 抽象，必须允许 backend 自定义切分策略，而不是把 head-tail 硬编码进框架。
