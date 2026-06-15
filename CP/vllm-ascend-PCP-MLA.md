# MLA PCP in vLLM-Ascend

本文总结 vLLM-Ascend 中 MLA 的 PCP 实现，并和 SGLang GLM5Next 的 MLA CP 做对比。

## Ascend MLA 的 Current Prefill

入口是 `AscendMlaCPImpl.mla_preprocess_prefill()`。

当 `pcp_size > 1` 时：

1. 当前 rank 先用 local prefill Q latent 生成 Q。
2. 对 `kv_no_split` 做 MLA latent KV 预处理，包括 `kv_a_layernorm` 和 RoPE。
3. 把 `prefill_kv_c_k_pe` 在 PCP group 内 all-gather。
4. 用 `pcp_allgather_restore_idx` 恢复原始 token 顺序。
5. 把 restored full current chunk latent KV reshape/cache。
6. 根据 metadata 的 `kv_tail_proj_idx` 选择需要做 `kv_b_proj` 的 tail KV。
7. 后续 attention 用 q head/tail index 和 KV index 分段计算。

这说明 Ascend MLA current prefill 不是 Partial KV + LSE merge 的主路径，而是先恢复 full current chunk KV，再做本 rank query 的 attention，接近 Mode 1。

## Head/Tail Attention

DualChunkSwap 后，每个 rank 拿一段 head Q 和一段 tail Q。metadata 会提供：

- `q_head_idx`
- `q_tail_idx`
- `kv_with_q_head_attn_idx_in_tail`
- `kv_with_q_tail_attn_idx_in_tail`
- `head_actual_seq_lengths_kv`
- `tail_actual_seq_lengths_kv`

这些 index 让 backend 能分别计算 head 和 tail query 对应的可见 KV range。因为 causal attention 中不同 query 位置能看到的 KV 前缀不同，不能只按 shape 做简单切分。

## Decode Path

MLA decode path 使用 `actual_seq_lengths_kv = decode_meta.cp_seq_len`，并要求 attention kernel 返回 LSE。随后通过 common CP helper 跨 DCP/PCP 合并 output。这是 Mode 2，与 DCP 思路一致。

## 与 SGLang GLM5Next 的差异

SGLang GLM5Next 的 sparse MLA CP 更像模型内 layout contract：

- 模型入口 residual stream 用 plain layout。
- MLA 层边界做 `cp_plain_to_scattered()`。
- MLA attention 在 scattered layout 上跑。
- MLA 输出后做 `cp_scattered_to_plain()`。
- Indexer 对 K 做 all-gather，确保 topk 基于全局 K。
- FlashMLA Sparse kernel 本身不需要 CP 感知。

Ascend 则把很多 token layout 和恢复逻辑提前放到 runner metadata。两者都支持“Partial Q + Full current KV”这个 correctness baseline，但抽象位置不同：

```text
SGLang: model/layer communicator 管 layout contract。
Ascend: runner PCPManager + attention metadata 管 layout contract。
```

## 对 KimiLinear Dense MLA 的启发

如果 vLLM core 先做 KimiLinear dense MLA PCP，建议从 Mode 1 开始：

1. Q 按 PCP rank 切分。
2. current chunk K/V 在每个 PCP rank 上恢复为 full KV。
3. dense MLA backend 只为 local Q 计算 output。
4. 层出口按 metadata all-gather/restore。

这样 kernel 侧不需要先支持 CP-aware LSE merge。等 dense MLA correctness 稳定后，再把 DCP 的 out+LSE 合并扩展到 Partial KV。

## Source Map

- vLLM-Ascend:
  - `vllm_ascend/attention/context_parallel/mla_cp.py`: `AscendMlaCPImpl`
  - `vllm_ascend/attention/context_parallel/mla_cp.py`: `mla_preprocess_prefill`
  - `vllm_ascend/attention/context_parallel/mla_cp.py`: `_forward_prefill`, `_forward_decode`
- SGLang:
  - `python/sglang/srt/models/glm5_next.py`: MLA layer `mla_cp_wrap`
  - `python/sglang/srt/layers/attention/dsa/utils.py`: plain/scattered helpers
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`: CP key all-gather and topk
