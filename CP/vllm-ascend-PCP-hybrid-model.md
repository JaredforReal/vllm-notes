# PCP for Hybrid Models

本文讨论 KDA/MLA、linear attention/attention 混合模型中的 PCP 问题。代表模型包括 GLM5Next、KimiLinear、Qwen3-Next 类模型。

## Hybrid Model 的核心难点

普通 dense transformer 只有 attention + MLP，层间 residual stream 的 layout 可以比较自由。但 hybrid 模型有额外约束：

- linear attention / KDA / Mamba 需要自然 token 顺序。
- causal conv 或 recurrent state 不能随便按 zigzag layout 直接执行。
- dense attention/MLA 为了 load balance 又希望 query 走 zigzag/scattered layout。
- 如果层间 layout 不稳定，相邻层会看到错误 token 顺序。

所以必须定义一个跨层合约。

## SGLang 的 Plain Layout 合约

SGLang GLM5Next 的设计非常清晰：

- residual stream 始终是 plain layout：rank `i` 持有连续 token slice。
- KDA 层内部：
  - projection 前 `cp_plain_all_gather()` 得到完整自然序列。
  - output 后 `cp_plain_reduce_scatter()` 回到 plain slice。
- MLA 层边界：
  - 进入 MLA 前 `cp_plain_to_scattered()`。
  - MLA 输出后 `cp_scattered_to_plain()`。
- 模型出口 `cp_plain_all_gather()` 恢复完整序列。

这使 KDA 层无需理解 zigzag，MLA 层仍然可以用 scattered layout。

## Ascend 的 Hybrid Attention Metadata

Ascend 在 `PCPManager` 中有 `pcp_use_hybrid_attn`，目前针对：

- `qwen3_next`
- `qwen3_5`
- `qwen3_5_moe`

hybrid path 额外维护：

- `pcp_fa_query_idx`
- `pcp_enter_fa_restore_idx`
- `pcp_exit_fa_scatter_idx`
- `pcp_padded_tokens_fla`

这些字段用于在 linear-attention 和 full-attention 之间恢复/重排 query 和 hidden states。

这与 SGLang 的 plain layout 合约目的相同：让不同类型层在各自需要的 layout 下执行，同时保证层间数据顺序正确。

## mHC / Cross-Layer KV Cache 问题

对于有 mHC 或跨层状态共享的模型，需要额外确认：

- 每层写入 KV cache 的 token 顺序是什么。
- 下一层读取 KV cache 时看到的是 full sequence 还是 local shard。
- 如果 current chunk KV all-gather 后再 cache，slot mapping 必须对应恢复后的全局顺序。
- 如果 decode/context 使用 partial KV + LSE，metadata 必须明确每个 PCP/DCP rank 的 local context length。

这类问题不适合藏在 model forward 里，应该由 runner metadata + backend contract 统一表达。

## 对 KimiLinear 的建议

KimiLinear 的 first PCP PR 可以走 conservative path：

1. 层间 residual stream 使用 plain contiguous layout。
2. KDA/Mamba/linear attention 暂时使用 full sequence 或 local slice baseline，先保证 correctness。
3. Dense MLA 用 Partial Q + Full current KV。
4. 后续再实现真正的 KDA CP：
   - head/state shard 明确；
   - projection 前 full sequence gather；
   - output reduce-scatter/all-reduce；
   - state cache 更新顺序可验证。

这会比一次性把 KDA、MLA、DCP、chunked prefill 都塞进一个 PR 更容易 review。

## Source Map

- vLLM-Ascend:
  - `vllm_ascend/worker/pcp_utils.py`: `pcp_use_hybrid_attn`, hybrid indices
  - `vllm_ascend/attention/context_parallel/attention_cp.py`: `_gather_and_restore_pcp_qkv`
- SGLang:
  - `python/sglang/srt/models/glm5_next.py`: `Glm5NextLinearAttention`, `Glm5NextDecoderLayer`
  - `python/sglang/srt/layers/communicator_dsa_cp.py`
  - `python/sglang/srt/layers/attention/dsa/utils.py`: plain layout helpers
