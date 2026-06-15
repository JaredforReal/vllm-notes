# FullAttention / GQA PCP-DCP

本文补充普通 FullAttention/GQA 在 PCP/DCP 下的实现细节。这里的 FullAttention 指标准 Q/K/V attention，不包括 MLA sparse/indexer、Mamba/KDA 或线性 attention。

## 结论

FullAttention/GQA 的 CP 支持可以拆成三种路径：

1. DCP decode/context: partial KV + out/LSE merge，vLLM core 已有 FlashAttention 支持。
2. PCP current prefill: partial Q + full current KV，vLLM-Ascend 已实现。
3. PCP + DCP chunked context: current chunk 可 full-KV，history/context KV 更适合 partial KV + out/LSE merge。

GQA 不是单独的 CP backend。它通过 `num_heads` 和 `num_kv_heads` 参数进入同一个 attention kernel。CP 的关键约束是：DCP 合并后必须能按 query heads reduce-scatter 回本 rank，因此每个 KV head 对应的 query group 必须能被 DCP size 整除。

## Ascend FullAttention PCP

vLLM-Ascend 的普通 attention CP backend 是 `AscendAttentionCPImpl`。

### Current Prefill: Partial Q + Full KV

对 current prefill chunk，Ascend 先在 runner/metadata 阶段把 query token 做 DualChunkSwap，然后 backend 内恢复 full current KV。

关键路径：

1. `PCPManager.update_tokens_for_pcp()` 让当前 rank 只拿 head/tail query chunk。
2. `PCPManager.generate_pcp_metadata()` 生成：
   - `q_head_idx`
   - `q_tail_idx`
   - `kv_with_q_head_nomask_idx`
   - `kv_with_q_head_mask_idx`
   - `kv_with_q_tail_nomask_idx`
   - `kv_with_q_tail_mask_idx`
   - `q_full_idx`
3. `AscendAttentionCPImpl.reshape_and_cache()` 对 current prefill K/V 做 PCP all-gather。
4. 用 `pcp_allgather_restore_idx` 恢复原始 token 顺序。
5. 用 restored full current K/V reshape/cache。
6. `_forward_prefill_cp_pre()` 从 full K/V 中按 index 选出 head/tail query 对应的 nomask/mask KV。
7. `_attention_with_nomask_and_mask()` 分别计算无 mask prefix attention 和 causal mask attention。
8. `_forward_prefill_cp_post()` 用 `q_full_idx` 把 head/tail output 恢复成本 rank local query 顺序。

这条路径本质是 Mode 1：每个 PCP rank 只处理 partial Q，但 current chunk KV 被恢复成 full KV。

### 为什么要拆 nomask/mask

DualChunkSwap 让一个 rank 同时处理靠前 query 和靠后 query。对每段 query：

- 某些 KV 是完整可见 prefix，不需要 causal mask。
- 当前 chunk 对应的局部 KV 需要 causal mask。

Ascend 用两次 NPU attention 分别处理：

```text
nomask attention: sparse_mode=0, atten_mask=None
mask attention:   sparse_mode=3, atten_mask=attn_mask
```

然后用 LSE merge 把两段结果合并。即使 current chunk 是 full KV，拆 nomask/mask 后仍然需要局部 LSE 合并，因为两次 kernel 覆盖的是同一 query 的不同 KV range。

## Ascend FullAttention DCP

Decode path 是 `_forward_decode_pcp_dcp()`，明显是 Mode 2。

流程：

1. 如果 `dcp_size > 1`，先把 query 在 DCP group 内按 head 维 all-gather。
2. kernel 输入 `num_heads = self.num_heads * dcp_size`。
3. `num_key_value_heads = self.num_kv_heads` 保持 local KV head 数。
4. KV cache 只读取本 rank 的 local context shard。
5. `actual_seq_lengths_kv` 来自：

```text
num_computed_tokens_of_pcp_dcp[:, pcp_rank, dcp_rank]
```

6. attention kernel 返回 `attn_out` 和 `attn_lse`。
7. `_process_attn_out_lse()` 跨 DCP/PCP 收集 out+LSE。
8. `_npu_attention_update()` 用 LSE 做全局 softmax 合并。

这和 vLLM core DCP 的目标一致：KV 被切开时，必须拿 LSE 才能正确恢复全局 attention。

## Ascend Chunked Prefill

chunked prefill 同时有 current chunk KV 和 context/history KV。

Ascend 的策略：

- current chunk: 走 `_forward_prefill_cp_pre/attn/post`，更接近 Partial Q + Full current KV。
- context/history: 走 `_compute_prefill_context()`，只读本地 context KV shard，然后跨 DCP/PCP 收集 out+LSE。
- 最后 `_update_chunk_attn_out_lse_with_current_attn_out_lse()` 把 context output 和 current chunk output 再用 LSE 合并。

因此 FullAttention 的 chunked prefill 是混合模式，不应该用单一的 “PCP 是否 active” 判断覆盖所有逻辑。

## GQA 在 Ascend 中如何表达

Ascend 的 NPU attention op 同时接收：

```text
num_heads
num_key_value_heads
```

GQA/MQA 只是 `num_heads > num_key_value_heads`。PCP/DCP 逻辑不需要为 GQA 写一个独立 kernel，但 metadata 和 head 维通信必须保持两个不变量：

1. Query heads 可以被 DCP 切分和恢复。
2. KV heads 不随 DCP all-gather query head 数一起扩大。

所以 DCP decode 中会看到：

```text
num_heads = local_num_heads * dcp_size
num_key_value_heads = local_num_kv_heads
```

这表示每个 rank 为完整 query-head group 计算本地 KV shard 的局部结果，然后再按 LSE merge 并返回本 rank 负责的 query heads。

## vLLM Core FullAttention DCP

vLLM core 的 FullAttention/GQA DCP 主要在 `vllm/v1/attention/backends/flash_attn.py`。

### Metadata Build

`FlashAttentionMetadataBuilder.build()` 在 `dcp_world_size > 1` 时：

1. 计算 `query_lens = query_start_loc[1:] - query_start_loc[:-1]`。
2. 计算 context KV length：

```text
context_kv_lens = seq_lens - query_lens
```

3. 调 `get_dcp_local_seq_lens()` 得到当前 DCP rank 的 local context KV length。
4. 设置 `max_dcp_context_kv_len`，避免 GPU->CPU sync，并为 workspace 预留上界。
5. 构造 context attention 的 scheduler metadata。

`get_dcp_local_seq_lens()` 按 `cp_kv_cache_interleave_size` 处理余数，等价于 block/interleave 粒度的 KV 分片。

### Forward

`FlashAttentionImpl._forward_with_dcp()`：

1. Query 在 DCP group 内 all-gather 到 head 维：

```text
query_across_dcp = get_dcp_group().all_gather(query, dim=1)
```

2. 对 context KV cache 做 non-causal attention，返回 `context_attn_out/context_lse`。
3. 用 `cp_lse_ag_out_rs()` 或 `dcp_a2a_lse_reduce()` 跨 DCP ranks 合并 context output，并 reduce-scatter 回本 rank heads。
4. 对当前 query 的 K/V 做 causal attention，返回 `query_attn_out/query_lse`。
5. 用 `merge_attn_states()` 合并 context 和 current query 两段 attention。

这和 Ascend chunked/context path 的语义一致，只是实现细节不同。

## vLLM Core 的 GQA DCP 约束

非 MLA 模型开启 DCP 时，`ModelConfig` 会检查：

```text
tensor_parallel_size > total_num_kv_heads
dcp_size <= tensor_parallel_size // total_num_kv_heads
num_q_per_kv % dcp_size == 0
```

这些约束说明 vLLM core 的 GQA DCP 是在一个 KV head 对应的 query group 内切 query heads。否则 DCP all-gather 和 reduce-scatter 后无法保持每个 rank 的 query head/KV head 对齐。

这也是为什么文档或 RFC 里不能只写 “DCP supports GQA”；更准确的说法是：

```text
DCP supports GQA/MQA when each KV head's query group can be evenly partitioned across DCP ranks.
```

## vLLM Core 的 PCP 现状和缺口

当前 vLLM core 的 DCP 已经有成熟路径：

- FullAttention FlashAttention: `can_return_lse_for_decode=True`
- DCP combine: `cp_lse_ag_out_rs` 或 A2A `dcp_a2a_lse_reduce`
- GQA constraints: model config validation

但标准 FullAttention backend 还没有打开 `supports_pcp`。`check_attention_cp_compatibility()` 会要求 PCP backend 显式支持 PCP。

因此 FullAttention PCP 仍需要补：

1. Runner-level PCP metadata：local query lens、restore index、unpad mask、head/tail index。
2. Current prefill full-KV path：类似 Ascend `reshape_and_cache()` 的 K/V all-gather + restore。
3. Head/tail query path：类似 Ascend `_forward_prefill_cp_pre/attn/post`。
4. Chunked context path：复用 DCP out+LSE merge。
5. GQA head constraints：沿用 DCP 的 query-group divisibility，并确认 PCP 是否也需要 head 或 token layout 限制。

## 对 RFC / Roadmap 的建议

建议把 FullAttention/GQA 放在 MLA/KimiLinear 之前或并行推进，因为它是最标准的 CP attention surface：

1. PR 1: `PrefillContextParallelMetadata` + runner zigzag/padding metadata。
2. PR 2: FullAttention current prefill PCP，Partial Q + Full current KV。
3. PR 3: FullAttention chunked prefill context PCP+DCP，复用 LSE merge。
4. PR 4: GQA/MQA validation 和 tests，覆盖 `num_q_per_kv % cp_size/dcp_size` 相关组合。
5. PR 5: 把相同 metadata/merge helper 复用到 dense MLA 和 sparse/indexer MLA。

FullAttention 是更适合 upstream review 的第一块：语义简单，已有 DCP/LSE merge 可复用，也能先验证 PCP metadata 是否合理。

## Source Map

vLLM-Ascend:

- `vllm_ascend/attention/context_parallel/attention_cp.py`
  - `AscendAttentionCPImpl`
  - `_forward_prefill_cp_pre`
  - `_attention_with_nomask_and_mask`
  - `_forward_decode_pcp_dcp`
  - `reshape_and_cache`
  - `_compute_prefill_context`
- `vllm_ascend/worker/pcp_utils.py`
  - `update_tokens_for_pcp`
  - `generate_pcp_metadata`
- `vllm_ascend/attention/context_parallel/common_cp.py`
  - `_process_attn_out_lse`
  - `_npu_attention_update`
  - `_update_out_and_lse`

vLLM core:

- `vllm/v1/attention/backends/flash_attn.py`
  - `FlashAttentionMetadataBuilder.build`
  - `FlashAttentionImpl._forward_with_dcp`
  - `use_cascade_attention`
- `vllm/v1/attention/backends/utils.py`
  - `get_dcp_local_seq_lens`
- `vllm/v1/attention/ops/common.py`
  - `cp_lse_ag_out_rs`
- `vllm/v1/attention/ops/dcp_alltoall.py`
  - `dcp_a2a_lse_reduce`
- `vllm/v1/attention/ops/merge_attn_states.py`
  - `merge_attn_states`
- `vllm/config/model.py`
  - GQA/MQA DCP validation
