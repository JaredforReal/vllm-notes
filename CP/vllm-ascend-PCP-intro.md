# vLLM-Ascend PCP Intro

本文总结 vLLM-Ascend 当前 Prefill Context Parallelism (PCP) 的总体设计，并提炼对 vLLM core 的参考价值。

## 核心结论

vLLM-Ascend 的 PCP 不是一个单纯的模型层工具函数实现，而是从 runner 阶段开始重排 token、构造 metadata，再交给 attention backend 消费。它最值得 vLLM core 借鉴的不是某一个 kernel 调用，而是这几个抽象边界：

- `PCPManager` 在 model runner 里决定每个 request 在当前 PCP rank 上看到哪些 query token。
- `AscendPrefillContextParallelMetadata` 显式携带恢复顺序、padding mask、head/tail query/KV index、PCP+DCP local KV 长度等信息。
- attention backend 根据 metadata 决定是 full-KV current chunk、partial-KV context，还是 out+LSE 合并。
- `pcp_size > 1` 只表示 PCP 被启用；实际是否使用 PCP 由当前 batch/request/mode 决定。

这比在模型代码里临时判断 `get_pcp_group().world_size > 1` 更可靠，因为 decode、短 prefill、chunked prefill、hybrid attention 都需要不同策略。

## Enabled vs Active

需要明确区分：

- `pcp_enabled`: 配置层面启用，通常是 `prefill_context_parallel_size > 1`。
- `pcp_active`: 当前 forward 确实对 prefill query 做了 PCP split。

Decode 通常属于 `enabled=True, active=False` 或者使用 DCP/PCP 的 decode context 合并路径，而不是 prefill split 路径。短 prefill 也可能 fallback 到非 PCP。长 prefill 若不整除，不应该简单 fallback；更合理的是 runner/scheduler 生成 padding/ragged metadata，把 shape 问题消化掉。

vLLM-Ascend 的 `PCPManager.update_tokens_for_pcp()` 体现了这个设计：decode request 不 split，而是复制到 PCP ranks；prefill request 则按 DualChunkSwap/zigzag 切分。

## 两种 CP 模式

为了讨论清楚，可以把 CP 实现分为两类。

### Mode 1: Partial Q + Full KV

每个 PCP rank 只处理自己负责的 query token，但当前 chunk 的 KV 在各 rank 上恢复为全量。attention kernel 不需要跨 rank 合并 softmax，只需要对本 rank 的 Q 和完整 KV 做普通 attention。

特点：

- 正确性直观，kernel 对 CP 感知少。
- current prefill chunk 常用。
- sparse/indexer 路径更容易先做正确，因为 indexer 能看到全局 K。
- 通信主要是 prefill KV all-gather 和输出 all-gather/恢复顺序。

SGLang GLM5Next 的 sparse MLA/DSA 基本属于这个方向：Q 被 CP split，indexer 通过 all-gather 看到 full K，FlashMLA Sparse kernel 本身不需要知道 CP。

vLLM-Ascend 的普通 attention/MLA current prefill chunk 也有明显 Mode 1 成分：`reshape_and_cache()` / `mla_preprocess_prefill()` 会跨 PCP all-gather K/V 或 MLA latent KV，然后用 restore index 恢复原顺序。

### Mode 2: Partial Q + Partial KV + LSE Merge

每个 rank 只持有一部分 KV cache，attention kernel 输出局部 `attn_out` 和 `softmax_lse`，跨 rank 收集后用 LSE 做 numerically correct softmax 合并。

特点：

- 更接近 decode context parallel。
- KV cache 不需要在 prefill/decode 时全量复制。
- kernel/backend 需要返回 LSE，且合并逻辑必须正确。
- 与 DCP 组合更自然。

vLLM-Ascend 的 decode 和 chunked prefill context 明显属于这个方向：`_process_attn_out_lse()` 会先 DCP all-to-all，再 PCP all-gather out+LSE，最后 `_npu_attention_update()` 做合并。

## vLLM-Ascend 的总体数据流

1. Model runner 读取 scheduler 输出。
2. `PCPManager.init_batch_info()` 记录 decode/prefill request 数量。
3. `PCPManager.update_tokens_for_pcp()` 修改当前 rank 的 token count 和 positions。
4. runner 用 PCP 后的 `input_ids/positions/slot_mapping/query_lens` 构造 attention metadata。
5. `PCPManager.generate_pcp_metadata()` 生成 `AscendPrefillContextParallelMetadata`。
6. attention backend 消费 metadata：
   - ordinary attention: `attention/context_parallel/attention_cp.py`
   - MLA: `attention/context_parallel/mla_cp.py`
   - DSA/SFA: `attention/context_parallel/dsa_cp.py`, `sfa_cp.py`

这个设计说明 PCP 的核心状态应该进入 vLLM core 的 attention metadata，而不是散落在 model implementation 中。

## 对 vLLM Core 的建议

适合拆成几层做：

1. Core metadata: 引入 `PrefillContextParallelMetadata`，表达 active/layout/padding/restore/local lens。
2. Runner plumbing: 在 model runner 阶段完成 token split/padding/restore index 构造。
3. Mode 1 baseline: 先支持 Partial Q + Full current KV，覆盖 dense MLA 和 sparse MLA/indexer 的 correctness。
4. Zigzag layout: 支持不整除 query length，不把整除条件暴露给用户。
5. Mode 2 extension: 基于 DCP 的 out+LSE merge，支持 partial KV decode/chunked context。
6. Hybrid model contract: 明确 KDA/Mamba/linear attention 层之间 residual stream 的 layout。

## Source Map

- vLLM-Ascend:
  - `vllm_ascend/worker/pcp_utils.py`: `PCPManager`, `update_tokens_for_pcp`, `generate_pcp_metadata`
  - `vllm_ascend/attention/utils.py`: `AscendPrefillContextParallelMetadata`
  - `vllm_ascend/attention/context_parallel/common_cp.py`: out+LSE gather/update helpers
  - `vllm_ascend/attention/context_parallel/attention_cp.py`: dense attention PCP/DCP backend
  - `vllm_ascend/attention/context_parallel/mla_cp.py`: MLA PCP backend
  - `vllm_ascend/attention/context_parallel/dsa_cp.py`, `sfa_cp.py`: sparse/DSA paths
- SGLang:
  - `python/sglang/srt/models/glm5_next.py`
  - `python/sglang/srt/layers/attention/dsa/utils.py`
  - `python/sglang/srt/layers/communicator_dsa_cp.py`
  - `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
