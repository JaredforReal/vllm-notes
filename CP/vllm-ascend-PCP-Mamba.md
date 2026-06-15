# Mamba / KDA / Linear Attention under PCP

本文说明 Mamba/KDA/linear attention 在 PCP 下需要什么，以及 vLLM-Ascend 与 SGLang 各自能提供什么参考。

## 关键结论

vLLM-Ascend 对 hybrid attention 有 PCP metadata 和 Qwen3-Next 相关路径，但它不是一个完整、通用的 Mamba/KDA PCP 参考。对 KDA 类层，SGLang GLM5Next 的实现更直接：它把层间 residual stream 固定为 plain layout，然后 KDA 层内部自行 all-gather full sequence，输出 reduce-scatter。

## 为什么 KDA/Mamba 不能直接用 Zigzag Layout

KDA/Mamba/linear attention 往往依赖自然 token 顺序：

- causal conv1d 需要连续时间顺序。
- recurrent state/chunk scan 需要按序推进。
- state cache 的更新位置必须与 token 的真实顺序一致。

如果把 zigzag 后的 token 直接喂给 KDA，语义通常是错的。zigzag 适合 attention query load balance，不适合作为 KDA 的执行顺序。

## SGLang GLM5Next 的 KDA CP

SGLang 的做法：

1. 构造 KDA 层时，如果 DSA prefill CP enabled，head shard 用 CP rank/size，而不是 TP rank/size。
2. `forward_qkvbfg()` / fused path 开头，如果 CP active，先 `cp_plain_all_gather()`。
3. KDA 在完整自然序列上做 qkv projection、causal conv/chunk/recurrent attention。
4. output projection 后：
   - CP active: `cp_plain_reduce_scatter()`。
   - CP enabled but inactive: `all_reduce()`，保证 head-sharded result 合并正确。

这个设计把 correctness 和层间 layout 分开：层间只传 plain slice；KDA 自己决定什么时候需要 full sequence。

## Ascend 的相关但不完整参考

Ascend runner 中有 GDN/Mamba 相关处理，例如：

- `gdn_query_start_loc` 需要 unpadded query start。
- Mamba cache mode 需要 block/page 对齐。
- hybrid attention path 有 `pcp_use_hybrid_attn` 和 enter/exit restore/scatter index。

但这些更多是平台和具体模型适配，不等价于一个通用的 KDA PCP contract。它们能提醒 vLLM core：linear attention 需要 unpadded natural-order metadata，不能只给 padded zigzag positions。

## 真正支持 KDA PCP 需要什么

至少需要明确：

1. 层间 layout：建议 plain contiguous。
2. KDA 执行输入：是否 all-gather full sequence，还是实现 sequence-parallel scan。
3. head/state shard：按 TP 还是 PCP shard。
4. state cache 更新：哪个 rank 写哪些 state，decode 如何恢复。
5. output 合并：reduce-scatter、all-reduce，还是 explicit gather。
6. enabled-but-inactive fallback：短 prefill/decode 时不能因为 head shard 改了就漏掉通信。

第一版不建议直接做 sequence-parallel KDA scan。更稳的路线是复制 SGLang：

```text
plain local slice -> all-gather full natural sequence -> KDA -> reduce-scatter plain slice
```

## 对 vLLM KimiLinear PCP 的建议

短期建议：

- Dense MLA 先实现 Partial Q + Full KV。
- KDA 保持 correctness baseline，不强行 zigzag。
- 如果 KDA 权重/head 已经按 PCP shard，需要 enabled-but-inactive all-reduce path。
- metadata 里保留 unpadded query start/lens，供后续 KDA CP 使用。

长期可以考虑：

- KDA head-sharded CP；
- state cache CP sharding；
- sequence-parallel scan；
- 与 decode context parallel 的 state handoff。

## Source Map

- SGLang:
  - `python/sglang/srt/models/glm5_next.py`: `Glm5NextLinearAttention`
  - `python/sglang/srt/layers/attention/dsa/utils.py`: `cp_plain_all_gather`, `cp_plain_reduce_scatter`
- vLLM-Ascend:
  - `vllm_ascend/worker/model_runner_v1.py`: GDN/Mamba runner metadata comments
  - `vllm_ascend/worker/pcp_utils.py`: hybrid PCP metadata
  - `vllm_ascend/patch/platform/patch_mamba_config*.py`: Mamba cache/page alignment
