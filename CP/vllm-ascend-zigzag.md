# vLLM-Ascend Zigzag / DualChunkSwap

本文解释 vLLM-Ascend PCP 中的 zigzag token layout，也就是代码注释里的 DualChunkSwap。它解决的问题是：prefill attention 的计算量随 query 位置增长，直接连续切分会让靠后的 rank 更重；zigzag 把序列头和序列尾配对给同一个 rank，尽量平衡每个 rank 的 KV span。

## Ascend 的做法

入口是 `PCPManager.update_tokens_for_pcp()`。

对每个 prefill request，Ascend 会先把 scheduled token 数 padding 到 `2 * pcp_world_size` 的倍数：

```text
padded_len = ceil(len / (2 * pcp_size)) * (2 * pcp_size)
```

然后切成 `2 * pcp_size` 个等长 chunk。rank `r` 拿：

```text
head chunk id = r
tail chunk id = 2 * pcp_size - 1 - r
```

以 `pcp_size = 4` 为例：

```text
chunks: 0 1 2 3 4 5 6 7
rank0: 0 + 7
rank1: 1 + 6
rank2: 2 + 5
rank3: 3 + 4
```

这就是 zigzag/head-tail 分配。

## Padding 与 Restore

Ascend 没有让 backend 处理变长 chunk，而是在 runner 阶段统一 padding：

- `num_pcp_pads_cpu`: 每个 request padding 了多少 token。
- `pcp_unpad_mask`: all-gather 后哪些 token 是真实 token。
- `pcp_allgather_restore_idx`: all-gather 后恢复原始 token 顺序的 index。
- `pcp_tokens`: 当前 rank 对每个 request 实际处理的 token 数。

这个设计的好处是 kernel 可以看到规整 shape，复杂度集中在 metadata。

Decode request 特殊处理：decode 不按 zigzag split，而是把 scheduled decode token 复制到 PCP ranks。`pcp_unpad_mask` 只保留 rank0 的 decode token，其他复制位置标为 padding。

## 与 SGLang Zigzag 的差异

SGLang GLM5Next 的 zigzag 思路类似，但余数处理不同。

SGLang 在 `prepare_input_dp_with_cp_dsa()` 中把序列分成 `2 * cp_size` 个 block：

```text
base = L // (2 * cp_size)
rem = L % (2 * cp_size)
block[i] = base + 1 if i < rem else base
```

也就是说，余数分配给前几个 block。rank 仍然拿 `r` 和 `2*cp_size-1-r` 两块。不同 rank 的 token 数可以差 1 到 2 个，然后通过 `per_rank_actual_token` 和 `max_rank_len` 做 all-gather padding。

对比：

```text
Ascend: 先 per-request padding 到 2*pcp_size 倍数，block 等长。
SGLang: block 可以变长，metadata 记录每个 rank 的真实 token 数。
```

Ascend 更适合直接落到 vLLM runner，因为它把 shape 对齐问题提前解决；SGLang 更节省 padding token，但 backend/metadata 要支持 ragged block。

## 为什么不应该要求 Query Length 整除 PCP Size

用户不应该因为 prompt 长度不是 `pcp_size` 或 `2*pcp_size` 的倍数而看到 correctness fallback 或 assert。合理路径是：

- 对短 prefill：`pcp_enabled=True, pcp_active=False`，走非 PCP。
- 对长 prefill：通过 padding 或 ragged metadata 让 `pcp_active=True`。

Ascend 的 padding-to-`2*pcp_size` 做法是一个可迁移到 vLLM core 的实现方案。

## 对 vLLM Core 的建议

建议先支持 Ascend 风格的 padded zigzag：

1. 在 runner 里根据 original query lens 生成 padded lens。
2. 生成 local query positions、restore index、unpad mask。
3. attention metadata 携带这些字段。
4. 模型和 backend 不再自己猜测是否整除。
5. 后续如果需要减少 padding，再扩展到 SGLang 风格 ragged block。

## Source Map

- vLLM-Ascend:
  - `vllm_ascend/worker/pcp_utils.py`: `PCPManager.update_tokens_for_pcp`
  - `vllm_ascend/worker/pcp_utils.py`: `pcp_allgather_restore_idx`, `pcp_unpad_mask`
- SGLang:
  - `python/sglang/srt/layers/attention/dsa/utils.py`: `prepare_input_dp_with_cp_dsa`
  - `python/sglang/srt/layers/attention/dsa/utils.py`: `cp_plain_to_scattered`, `cp_scattered_to_plain`
