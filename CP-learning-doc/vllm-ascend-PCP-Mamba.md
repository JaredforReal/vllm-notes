# Mamba / 线性注意力模型的 PCP 支持（Qwen3.5 GDN、Kimi-Linear KDA、Mamba2）

> 对应原始问题：*How does vllm-Ascend support Mamba model like Qwen3.5(GDN), kimilinear(KDA), Mamba2 with PCP?*

## 1. 核心难点：线性/递归层是“有状态”的

Mamba / GDN（Gated Delta Net）/ KDA（Kimi Delta Attention）这类线性注意力层 = **causal conv1d + 递归状态（ssm state）**。它们沿时间维递推，token i 的输出依赖 token 0..i-1 累积的状态。这与标准 attention 的“任意 token 间可并行”完全不同。

PCP 把序列切到多个 rank 后，**每个 rank 只看到自己那一段**，但递归状态必须从“上一段”延续过来，否则结果错误。因此这类层的 PCP 关键是 **跨 rank 的状态传播（state propagation）**，而不是 attention 的 out/lse 合并。

> 总原则：conv1d 的 last-state（最后 width-1 个输入）和 ssm 的最终递归状态，需要按 PCP rank 顺序串行传播：rank0 算完 → 把终态交给 rank1 当初态 → rank1 算完 → … 实现上用 **allgather 把各 rank 的边界状态广播**，每个 rank 取“前一个 rank”的状态当初态。

## 2. causal conv1d 的跨 rank 状态传播

**源码：`vllm_ascend/ops/triton/mamba/causal_conv1d.py:90`**（`causal_conv1d_fn`，GDN/Mamba2 共用路径）

```python
# 抽取每个 prefill request 最后 width-1 个输入（作为 conv 的“边界状态”）
last_width_prefill_x = extract_last_width(x, query_start_loc[num_decodes:], state_len)  # [width-1, num_prefills, dim]

if get_pcp_group().world_size > 1:
    all_last_width_prefill_x = get_pcp_group().all_gather(
        last_width_prefill_x.unsqueeze(0).contiguous(), 0)   # [pcp_size, width-1, num_prefills, dim]
    pcp_rank = get_pcp_group().rank_in_group
    if pcp_rank > 0:
        # 非 rank0：把前一个 rank 的边界输出当作本 rank conv 的初始 state
        conv_states[cache_indices[num_decodes:], :, :state_len] = all_last_width_prefill_x[pcp_rank - 1, ...]

# 逐 request 跑 causal_conv1d_ref（return_final_states=True，更新 conv_states）
for i in range(len(seqlens)):
    causal_conv1d_ref(x_s, weight, bias, ..., final_states_out=conv_states[...], initial_states=conv_states[...])

if get_pcp_group().world_size > 1:
    # 最后统一用最后一个 rank 的终态写回（保证 conv_states 一致）
    conv_states[cache_indices[num_decodes:], :, :state_len] = all_last_width_prefill_x[-1, ...]
```

要点：
- `extract_last_width`（行 164）取每个 seq 末尾 `width-1` 个 token，作为 conv 状态；
- allgather 把所有 rank 的边界状态拼起来（dim=0）；
- rank r 用 rank r-1 的边界状态作为自己的 **initial_state**，从而把因果卷积的边界正确接续；
- 注意 GDN 在 PCP>1 时走 `causal_conv1d_fn`（triton ref 路径），而 PCP=1 时走 `npu_causal_conv1d_custom`（CANN 算子，`gdn.py:529`）—— 因为 CANN 算子不支持跨 rank 初态注入。

## 3. GDN（Qwen3.5）的完整流程

**源码：`vllm_ascend/ops/gdn.py`**（`AscendGatedDeltaNetAttention` 类）。GDN = causal conv1d + chunked delta rule 递归 + 门控。

PCP 相关分支在 `gdn.py:504`（prefill 部分）：
```python
if get_pcp_group().world_size > 1:
    # prefill 走 triton causal_conv1d_fn（支持跨 rank 初态），见上节
    mixed_qkv_non_spec = causal_conv1d_fn(mixed_qkv_non_spec_T, conv_weights, ...,
                                          conv_states=conv_state,
                                          has_initial_state=has_initial_state,
                                          cache_indices=non_spec_state_indices_tensor,
                                          query_start_loc=non_spec_query_start_loc, ...)
```

递归核心（delta rule）在 `gdn.py:665`：
```python
# prefill
initial_state = ssm_state[non_spec_state_indices_tensor].transpose(-1,-2).contiguous()
clear_ssm_states(initial_state, has_initial_state)
core_attn_out_non_spec, last_recurrent_state = chunk_gated_delta_rule(
    q, k, v, g, beta, initial_state=initial_state, output_final_state=True,
    cu_seqlens=non_spec_query_start_loc, ...)
ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.transpose(-1,-2).contiguous()
# decode 走 npu_recurrent_gated_delta_rule（自定义 AscendC 算子，逐 token 递推）
```

> SSM 递归状态（delta rule 的 state）的跨 rank 传播，本质上由 `ssm_state` 持久化 + conv1d 的边界传播保证正确性：每 rank 的 prefill 用自己的 initial_state 起步，算出的 `last_recurrent_state` 写回 cache。conv 边界的 allgather 保证“前序 rank 的信息”能流到后续 rank 的 conv 初态，从而 ssm 初态也间接连续。（具体 ssm state 是否做额外 allgather 取决于模型；conv1d 是主要显式传播点。）

Qwen3.5 的 patch 在 `vllm_ascend/patch/worker/patch_qwen3_5.py`（含 `_split_ba_for_tp` 等 TP/CP 相关 weight 切分）。

## 4. Kimi-Linear（KDA）与 Mamba2

- **KDA（Kimi Delta Attention）**：结构与 GDN 同族（delta rule + conv），走相同的 `causal_conv1d_fn` PCP 路径。相关 patch 在 `patch/worker/patch_minimax_m2_linear_attn.py`、`ops/bailing_moe_linear_attn.py`（bailing/MoE + 线性注意力复用同一套 conv/ssm 基础设施）。
- **Mamba2（selective scan）**：标准 Mamba2 用 selective_scan + conv1d。conv1d 部分同样走 `causal_conv1d.py` 的跨 rank allgather（行 135-158）。selective_scan 的 ssm state 持久化在 `ssm_states`，每 rank prefill 从自己 cache 起步。

> 三者（GDN / KDA / Mamba2）的 PCP 公约数 = **conv1d 边界状态跨 rank allgather + ssm state 持久化**。差异在递归核心算子：GDN/KDA 用 `chunk_gated_delta_rule` / `npu_recurrent_gated_delta_rule`，Mamba2 用 selective_scan。

## 5. hybrid-attention 模型（Mamba + Attention 混合）

Qwen3-Next / Qwen3.5 这类 **混合模型** 同时有线性注意力层（GDN/Mamba）和标准 attention 层。PCP 对这两类层要 **分别** 处理：

- **线性层**：用本笔记第 2-4 节的状态传播；
- **attention 层**：走 hybrid 专用布局（`pcp_use_hybrid_attn=True`）。

`PCPManager.__init__`（`pcp_utils.py:131`）：
```python
self.pcp_use_hybrid_attn = model_type in ("qwen3_next", "qwen3_5", "qwen3_5_moe")
```

hybrid attention 层在 `update_tokens_for_pcp` 走 **另一条分支**（行 650-784）：
- 不是 head-tail zigzag，而是 **按 rank 线性切分** prefill token（`_get_cp_local_seq_lens`），每 rank 拿等量连续段；
- 额外计算 `pcp_enter_fa_restore_idx`（进入 FA 前的还原索引）和 `pcp_exit_fa_scatter_idx`（FA 输出的 scatter 索引）；
- 在 `attention_cp.py` 的 `_gather_and_restore_pcp_qkv`（行 861）里：allgather qkv → 用 `pcp_enter_fa_restore_idx` 还原 → 喂给普通 FA → 输出用 `pcp_exit_fa_scatter_idx` scatter 回各 rank。

详细见 `vllm-ascend-PCP-hybrid-model.md`。

## 6. 为什么线性层的 PCP 难做、要小心

1. **状态串行依赖**：rank 间的状态传递是串行的（rank r-1 → rank r），无法完全并行，是 PCP 在线性层上的固有开销。conv1d 用 allgather 一次性广播所有边界状态，把“串行”降到一次集合通信。
2. **初态注入要 triton/CANN 算子支持**：PCP>1 时只能走支持 `initial_state` 的 `causal_conv1d_fn`（triton ref），CANN `npu_causal_conv1d_custom` 不支持跨 rank 初态，故 PCP=1 才用它。
3. **decode 与 prefill 状态不同**：decode 走 `npu_recurrent_gated_delta_rule`（逐 token），prefill 走 `chunk_gated_delta_rule`（分块并行）。PCP 主要影响 prefill（decode 单 token 无需切分）。
4. **`num_computed_tokens_of_pcp_dcp` 同样适用**：linear 层的 cu_seqlens 也要按 CP rank 本地化。

## 7. 小结

```
线性注意力层 (Mamba2 / GDN / KDA) 的 PCP：
 ┌─ causal conv1d: 抽 last-(width-1) 输入 → PCP allgather → rank r 用 rank r-1 当 initial_state
 ├─ ssm/delta state: 持久化在 cache，prefill 从 initial_state 起步，算完写回 last_recurrent_state
 └─ 核心算子: GDN/KDA=chunk_gated_delta_rule, Mamba2=selective_scan; decode=npu_recurrent_gated_delta_rule

hybrid 模型 (qwen3_next/3.5):
 ├─ 线性层: 上述状态传播
 └─ attention 层: pcp_use_hybrid_attn=True, 线性切分 + enter/exit FA restore/scatter 索引
```

对 RFC 的启示：**CP 对线性/递归层的支持与 attention 层本质不同**——前者是“状态串行传播”，后者是“out/lse 并行合并”。上游 CP 抽象必须为线性层提供 **跨 rank 的 state propagation 原语**（如 conv1d 边界 allgather + initial_state 注入接口），而不能套用 attention 的 allgather-KV 模式。这是 vLLM 当前 CP 最欠缺的一块。
