# PCP and DCP in vLLM-Ascend

本文解释 vLLM-Ascend 中 PCP 与 DCP 的关系。核心结论：Ascend 不是简单把 PCP 当作 DCP 的别名，而是在 current prefill chunk、decode、chunked context 中分别使用不同的数据分布和合并方式。

## 术语

- PCP: Prefill Context Parallelism，主要切 prefill query。
- DCP: Decode Context Parallelism，主要切 decode/context KV。
- Mode 1: partial Q + full KV，不需要跨 rank softmax 合并。
- Mode 2: partial Q + partial KV，需要 out+LSE 合并。

## Ascend 的 PCP+DCP Metadata

`PCPManager.generate_pcp_metadata()` 会计算：

```text
num_computed_tokens_of_pcp_dcp[req, pcp_rank, dcp_rank]
```

它表示每个 request 的历史/context token 在 PCP+DCP 网格上的 local KV 长度。计算时还考虑 `cp_kv_cache_interleave_size`。

这个字段被 decode 和 chunked context path 用来告诉 attention kernel 当前 rank 能看到多少 KV。

## Decode Path: Mode 2

Decode path 不是 full-KV current chunk，而是明显的 Mode 2。

流程：

1. 每个 PCP/DCP rank 根据 `num_computed_tokens_of_pcp_dcp` 读取本地 KV shard。
2. attention kernel 计算局部 `attn_output` 和 `softmax_lse`。
3. `_process_attn_out_lse()` 把 output 和 LSE 拼在一起：
   - DCP 维度做 `all_to_all_single`
   - PCP 维度做 `all_gather`
4. `_npu_attention_update()` 把所有 rank 的局部 softmax 结果用 LSE 合并成全局结果。

这和 vLLM DCP 的核心思想一致：KV 可以分片，但必须有 LSE 才能正确合并。

## Current Prefill Chunk: Mostly Mode 1

对当前正在处理的 prefill chunk，Ascend 经常把 KV all-gather 回全量，然后再让本 rank 的 Q 做 attention。

普通 attention 的 `reshape_and_cache()` 中：

- 非 hybrid attention: gather current prefill K/V across PCP ranks。
- 用 `pcp_allgather_restore_idx` 恢复原始顺序。
- 之后 cache 的是恢复顺序后的 full current chunk KV。

MLA 的 `mla_preprocess_prefill()` 中：

- gather `prefill_kv_c_k_pe` across PCP ranks。
- restore 顺序。
- 再 reshape/cache latent KV。

所以 current prefill chunk 更接近 Mode 1。

## Chunked Prefill Context: Mode 2

Chunked prefill 有两部分 KV：

- current chunk KV: 当前正在算的 token，可 all-gather 成 full KV。
- context/prefix KV: 之前 chunk 或 cache hit 的历史 KV，分布在 PCP/DCP shards 上。

Ascend 对 context 部分走 Mode 2：

1. `_prefill_query_all_gather()` 先收集 full query。
2. `_compute_prefill_context()` 用本地 context KV shard 计算局部 context attention。
3. `_gather_global_context_output()` 跨 DCP/PCP 收集 out+LSE。
4. `_update_global_context_output()` 用 LSE 合并。
5. `_update_chunk_attn_out_lse_with_current_attn_out_lse()` 把 context output 和 current chunk output 再合并。

这说明 chunked prefill 是 PCP/DCP 最复杂的场景：同一个 request 内，current chunk 和 context KV 可以走不同模式。

## 对 vLLM Core 的启发

建议 RFC 明确分阶段：

1. 先支持 Mode 1 current prefill chunk。
2. 复用 DCP out+LSE merge 设计支持 decode/context Mode 2。
3. 在 metadata 中显式区分：
   - current query local lens
   - current chunk full/restore indices
   - context KV local lens per PCP/DCP rank
   - output LSE merge requirements
4. 不要把 `pcp_size > 1` 等价为“所有 attention 都需要 CP kernel”。

## Source Map

- `vllm_ascend/worker/pcp_utils.py`: `generate_pcp_metadata`, `_get_cp_local_seq_lens`
- `vllm_ascend/attention/context_parallel/common_cp.py`: `_process_attn_out_lse`, `_npu_attention_update`
- `vllm_ascend/attention/context_parallel/attention_cp.py`: `_compute_prefill_context`, `_gather_global_context_output`
- `vllm_ascend/attention/context_parallel/mla_cp.py`: MLA decode/current prefill paths
