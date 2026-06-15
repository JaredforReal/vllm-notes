# DSA / Sparse PCP in vLLM-Ascend

本文总结 vLLM-Ascend 的 DSA/SFA PCP 路径，并对比 SGLang GLM5Next 的 DSA indexer 方案。

## Ascend DSA CP 不是普通 MLA PCP 的简单复用

Ascend 有单独的 DSA CP backend：

- `vllm_ascend/attention/context_parallel/dsa_cp.py`
- `vllm_ascend/attention/context_parallel/sfa_cp.py`

DSA CP metadata 会记录 local token metadata，例如：

- local query start/end
- local seq lens
- local token range
- tokens per rank
- padding token 数

它的 split 更像 DSA 专用 token shard，而不是所有模型通用的 PCPManager zigzag current chunk path。

## Ascend SFA / Indexer 的 Full KV 思路

`sfa_cp.py` 中有两个重要 helper：

- `gather_kv_cross_cp()`
- `gather_kv_cross_cp_compact()`

它们会跨 CP ranks 收集 KV cache blocks，供 sparse/indexer 路径使用。decode 和 prefill 都会把相关 KV block gather 出来，再做 index selection。

这和 sparse attention 的 correctness 诉求一致：topk/indexer 必须基于全局 K，否则不同 rank 会选择局部 topk，最终 attention 语义错误。

## SGLang DSA 的关键点

SGLang GLM5Next DSA indexer 也明确走 global-K：

- `_get_q_k_bf16()` 中 CP active 时对 projected key 做 `cp_all_gather_rerange_output()`。
- `_get_topk_ragged_with_cp()` 用 `kv_len_prev/kv_len_next` 读取全局 K cache 范围。
- `get_index_k_continuous(layer_id, end_seq_position, block_tables[batch_idx])` 的 end position 是全局长度，不是 local slice。
- FlashMLA Sparse kernel 接收的 page table/KV pool 已经是上层处理后的全局视图。

因此 SGLang 的 sparse MLA/DSA 属于 Partial Q + Full KV/indexer 的 Mode 1 baseline。

## Round-Robin vs Zigzag

SGLang DSA 还支持 round-robin split：

- Q token 按 `token_idx % cp_size` 分给 rank。
- 辅助 metadata 如 `ks/ke/token_to_batch_idx` 也按同样规则 split。
- round-robin 对整除更敏感，SGLang 在主路径上要求整除或依赖 scheduler padding。

Zigzag 则把每个序列切成 `2 * cp_size` 个 block，rank 拿 head/tail 两段，适合 dense/sparse attention 的 load balance。

## 对 vLLM Core 的建议

对于 sparse MLA with indexer，第一阶段应该坚持 Mode 1：

1. Q split。
2. Indexer sees full K。
3. Sparse attention sees full KV page table/pool。
4. Kernel 不需要 CP-aware LSE merge。
5. 层输出再按 PCP metadata gather/restore。

这是最容易保证 correctness 的路径。Partial KV sparse attention 可以作为后续优化，但需要：

- indexer 支持 global topk 或 distributed topk merge。
- sparse attention 返回 LSE。
- CP ranks 基于 LSE 合并 output。

这些都明显比 Mode 1 更大。

## Source Map

- vLLM-Ascend:
  - `vllm_ascend/attention/context_parallel/dsa_cp.py`
  - `vllm_ascend/attention/context_parallel/sfa_cp.py`
- SGLang:
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
  - `python/sglang/srt/layers/attention/dsa/dsa_backend.py`
  - `python/sglang/srt/layers/attention/dsa/utils.py`
