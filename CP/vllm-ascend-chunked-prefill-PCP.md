# Chunked Prefill PCP in vLLM-Ascend

## 为什么 Chunked Prefill 特殊

普通 prefill 只有当前 chunk，所有 query 和 KV 都来自同一段新 token。Chunked prefill 有两类 KV：

- context/prefix KV: 已经存在于 cache 的历史 token。
- current chunk KV: 当前 scheduled token 产生的新 KV。

PCP 下这两部分不能简单用同一个策略：

- current chunk 可以通过 PCP all-gather 恢复 full KV，走 Mode 1。
- context KV 已经按 PCP/DCP 分布在 cache 中，直接 all-gather full context KV 成本高，更适合 Mode 2 out+LSE merge。

## Ascend 的 Context Path

普通 attention backend 的关键函数：

- `_prefill_query_all_gather()`
- `_compute_prefill_context()`
- `_gather_global_context_output()`
- `_update_global_context_output()`
- `_update_chunk_attn_out_lse_with_current_attn_out_lse()`

流程：

1. Query all-gather：为了让每个 rank 都能为完整 query 计算自己 KV shard 的局部 attention。
2. 本地 context KV load：`_load_kv_for_chunk()` 根据 local context lens 从 KV cache 取当前 rank 的 context KV。
3. 局部 attention：kernel 返回 `prefix_chunk_output` 和 `prefix_chunk_lse`。
4. 跨 rank 收集：DCP all-to-all，PCP all-gather。
5. LSE 合并：把各 rank 对 partial KV 的局部 output 合并成全局 context output。
6. 与 current chunk output 合并：如果 current chunk attention 也返回 LSE，则再做一次 LSE merge。

这套流程是典型 Mode 2。

## Current Chunk Path

在 `reshape_and_cache()` 中，非 hybrid attention 的 current prefill K/V 会：

1. 拼接 `key,value`。
2. 对 PCP rank 做 all-gather。
3. 用 `pcp_allgather_restore_idx` 恢复原始 token 顺序。
4. 用 full current chunk KV reshape/cache。

这更接近 Mode 1。也就是说，Ascend 在同一个 chunked prefill request 内混合了 Mode 1 和 Mode 2。

## 为什么需要 LSE

当 KV 被切分时，每个 rank 只能计算：

```text
softmax(Q K_local^T) V_local
```

这不是全局 attention。正确合并需要每个 shard 的 log-sum-exp：

```text
lse_global = logsumexp(lse_0, lse_1, ...)
out_global = sum(exp(lse_i - lse_global) * out_i)
```

Ascend 通过 NPU 的 attention update helper 完成这个操作。

## 对 vLLM Core 的建议

RFC 里应该把 chunked prefill 单独列为后续阶段，而不是和第一版 PCP 混在一起。

推荐 roadmap：

1. 第一阶段：非 chunked / current chunk 的 Partial Q + Full KV。
2. 第二阶段：chunked prefill context 的 Partial KV + LSE merge。
3. 第三阶段：把 current chunk output 和 context output 的 LSE merge 做成通用 helper。

这样 reviewer 可以先审核较小的 correctness surface。

## Source Map

- `vllm_ascend/attention/context_parallel/attention_cp.py`
  - `_prefill_query_all_gather`
  - `_compute_prefill_context`
  - `_load_kv_for_chunk`
  - `_gather_global_context_output`
  - `_update_global_context_output`
  - `_update_chunk_attn_out_lse_with_current_attn_out_lse`
- `vllm_ascend/attention/context_parallel/common_cp.py`
  - `_update_out_and_lse`
  - `_npu_attn_out_lse_update`
