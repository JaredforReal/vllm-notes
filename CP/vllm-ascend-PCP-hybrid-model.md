# Hybrid 模型（Mamba + Attention 混合）的 PCP 支持

> 对应原始问题：*How does vllm-Ascend support hybrid model PCP?*

适用对象：**混合架构模型** —— 同一个模型里既有线性注意力层（Mamba/GDN/KDA），又有标准 attention 层。代表：**Qwen3-Next、Qwen3.5、Qwen3.5-MoE**。

这是 CP 最复杂的场景，因为两类层需要 **完全不同的 PCP 处理**（线性层走状态传播，attention 层走 hybrid 专用布局），且二者要在同一批次、同一套 token 布局下协同。

## 1. 触发条件

`PCPManager.__init__`（`vllm_ascend/worker/pcp_utils.py:131`）：

```python
self.pcp_use_hybrid_attn = vllm_config.model_config.hf_config.model_type in (
    "qwen3_next", "qwen3_5", "qwen3_5_moe",
)
```

这个 flag 一旦为 True，`update_tokens_for_pcp`（`pcp_utils.py:650`）走完全不同的分支，且 attention 后端的 `reshape_and_cache`（`attention_cp.py:836`）也走 `_gather_and_restore_pcp_qkv` 而非普通 allgather-KV。

## 2. token 切分：线性切分（非 zigzag）

hybrid 模型 **不用 head-tail zigzag**。`update_tokens_for_pcp` 的 hybrid 分支（行 650-784）把 prefill token **按 rank 线性等分**：

```python
# 用 _get_cp_local_seq_lens 算每 rank 的 prefill token 数
num_prefill_tokens_allranks = _get_cp_local_seq_lens(prefill_tokens, pcp_world_size, 1, 1)
# shape [num_prefill_reqs, pcp_world_size, 1]
num_prefill_scheduled_tokens_linear = num_prefill_tokens_allranks[:, pcp_world_rank, 0]
# 算每 rank 的线性 positions（连续段）
```

- decode token 不切，复制到各 rank（与普通 PCP 一致）；
- prefill token 按 rank 顺序切成 `pcp_world_size` 段连续区间；
- 因为线性注意力层（Mamba）必须 **按顺序处理连续 token**（状态依赖），所以 hybrid 模型整体放弃 zigzag，改用线性切分，保证线性层的递归正确性。

## 3. 关键索引：`enter_fa_restore_idx` / `exit_fa_scatter_idx`

hybrid 下，attention 层用的是 **完整的、顺序正确的 token 布局**（因为线性层需要全序），所以 PCP 要做“本 rank 线性段 → 全局 FA 布局 → 本 rank 线性段”的来回转换。这两个索引完成转换：

在 `update_tokens_for_pcp`（行 691-777）里计算：

- **`pcp_enter_fa_restore_idx`**：把 allgather 后的 qkv（各 rank 线性段拼接）**还原成全局顺序**，喂给普通 FlashAttention。decode 部分和 prefill 部分分别构造（行 723-745）。
  - prefill：`prefill_all_offset = rank_offset * max_scheduled_tokens + local_offset`，再 `+ prefill_arange_allranks`；
  - decode：按 `decode_threshold` 复制。

- **`pcp_exit_fa_scatter_idx`**：FA 输出后，把全局顺序的结果 **scatter 回各 rank 的线性段布局**（行 762-771）：
  ```python
  exit_fa_scatter_indices = positions_linear[...] + np.repeat(ori_tokens_start_loc, ...)
  exit_fa_scatter_idx = index_select(all_exit_fa_restore_idx[unpad_mask_prefill], 0, exit_fa_scatter_indices)
  ```

- **`pcp_fa_query_idx`**：attention 的 prefill query 索引（行 773-777）。

## 4. attention 层的 PCP：`_gather_and_restore_pcp_qkv`

**源码：`attention_cp.py:861`**（`AscendAttentionCPImpl` 在 hybrid 模式下的 reshape_and_cache 分支）

```python
def _gather_and_restore_pcp_qkv(self, query, key, value, attn_metadata):
    # 1. cat qkv，必要时 pad（pcp_padded_tokens_fla）
    qkv_fla = cat([query.reshape(N,-1), key.reshape(N,-1), value.reshape(N,-1)], dim=-1)
    # 2. allgather (PCP 组, dim=0)，取 max_num_tokens_across_pcp
    all_qkv = get_pcp_group().all_gather(qkv_fla[:max_num_tokens_across_pcp].contiguous(), dim=0)
    # 3. 用 pcp_enter_fa_restore_idx 还原成全局顺序
    actual_qkv = index_select(all_qkv, 0, pcp_enter_fa_restore_idx)
    # 4. 填入 workspace，padding 位填 0
    qkv_fa_padding_workspace[...][pcp_unpad_mask] = actual_qkv[...]
    qkv_fa_padding_workspace[...][~pcp_unpad_mask].fill_(0)
    # 5. split 回 q/k/v，q 用 pcp_fa_query_idx 选出 attention 要算的 query
    return q_fa, k, v
```

attention 计算完后（`forward_impl` 行 1051-1056）：
```python
if pcp_size > 1 and pcp_use_hybrid_attn:
    pcp_exit_fa_scatter_idx = attn_metadata.prefill.pcp_exit_fa_scatter_idx
    attn_output_prefill = get_pcp_group().all_gather(attn_output_prefill.contiguous(), dim=0)
    attn_output_prefill = index_select(attn_output_prefill, 0, pcp_exit_fa_scatter_idx)
```
即 FA 输出 allgather 后用 `exit_fa_scatter_idx` scatter 回各 rank 的线性布局。

> 一句话：hybrid 模型的 attention 层 = **allgather qkv → enter 还原成全序 → 普通 FA → exit scatter 回线性段**。这样 attention 层在“看到完整序列”的前提下正常算，又能在 rank 间恢复线性切分布局，与相邻的线性层衔接。

## 5. 线性层（Mamba/GDN）的 PCP

线性层走的是 `vllm-ascend-PCP-Mamba.md` 描述的状态传播：
- causal conv1d 边界 allgather（`ops/triton/mamba/causal_conv1d.py:135`）；
- ssm/delta state 持久化。

hybrid 模型的线性层和 attention 层交替出现，二者共享同一套 **线性切分 + 本地 positions**（`positions_linear`），所以状态能在层间正确衔接。

## 6. logits / 采样

`get_logits_indices`（`pcp_utils.py:787`）在 hybrid 模式下走专门分支（行 799-806）：
```python
# decode 用 pcp_pads_logits_hybrid_attn（= pcp_size-1 的 pad），
# prefill 用原始 token 数，cumsum 算每个 req 最后一个 token 的索引
tokens_logits = tokens_original + F.pad(decode_pads, (0, pad_len), value=0)
logits_indices = cumsum(tokens_logits) - 1
```

## 7. slot_mapping：多个 kv_cache_group

`initialize_slot_mapping`（`pcp_utils.py:488`）注释说明：hybrid-attention 模型有 **多个 kv_cache_groups**（线性层和 attention 层的 KV cache 结构不同），需要为每个 group 维护独立的 `pcp_padded_slot_mapping`，否则会被后一个 group 覆盖（共享同一地址）。所以 `pcp_padded_slot_mapping_list` 是个 list。

`get_padded_slot_mapping`（行 809）按 `kv_cache_group_id` 取对应的 slot mapping。

## 8. 数据流总览

```
                 scheduler (整条序列)
                        │
                        ▼  update_tokens_for_pcp (hybrid 分支)
        按 rank 线性切分 prefill token (decode 复制)
        产出: positions_linear, pcp_enter_fa_restore_idx,
              pcp_exit_fa_scatter_idx, pcp_fa_query_idx,
              num_scheduled_tokens_padded, pcp_padded_tokens_fla
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   线性注意力层 (Mamba/GDN)         标准 attention 层
   - conv1d 边界 allgather         - allgather qkv → enter 还原
   - ssm state 传播                 - 普通 FA (全序)
   - 按 positions_linear 递推       - exit scatter 回线性段
        │                               │
        └───────────────┬───────────────┘
                        ▼
              下一层 (交替) ... 直到最后一层
                        ▼
        get_restore_hidden_states (hybrid 分支, pcp_utils.py:850)
        allgather + pcp_enter_fa_restore_idx 还原 → lm_head
```

## 9. 对 vLLM 上游 RFC 的启示

hybrid 模型是 CP 最难也最有价值的场景，因为它同时考验：
1. **线性层的 state propagation** 与 **attention 层的 allgather 合并** 必须在同一框架内共存；
2. **切分策略不能一刀切** —— hybrid 模型为迁就线性层必须用线性切分，而非负载更均衡的 zigzag（attention 层因此牺牲一点均衡）；
3. **多 kv_cache_group** 的 slot_mapping 管理；
4. **enter/exit 索引** 这类“布局转换原语”是 hybrid 模型必需的接口。

上游 CP RFC 若要支持 Qwen3-Next / 3.5 这类 SOTA 混合架构，必须：
- 提供 **per-layer-type 的 CP 策略注册**（线性层 vs attention 层不同）；
- 提供 **state propagation 原语**（conv1d 边界 + ssm state）；
- 支持 **多 kv_cache_group** 与 **layout 转换索引**。

否则 vLLM 在 hybrid 长上下文场景会持续落后于 vllm-ascend 与 sglang。
