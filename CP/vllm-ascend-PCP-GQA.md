# FullAttention / GQA 后端的 PCP+DCP 支持

> 对应问题：*FullAttention 和 GQA 后端如何支持 PCP/DCP？*
>
> 适用对象：所有走 **通用 GQA/MHA attention** 的模型（Qwen3-235B、Llama 系等），以及 **混合模型里的 FullAttention 层**（Qwen3-Next / Qwen3.5 中与线性注意力层交替的标准 attention 层）。
>
> 对应后端：`vllm_ascend/attention/context_parallel/attention_cp.py` —— `AscendAttentionCPImpl`（计算）+ `AscendAttentionCPMetadataBuilder`（建 metadata）。
>
> 选择入口：`vllm_ascend/attention/attention_v1.py:85` `get_impl_cls()`，`enable_cp()` 为真时返回 `AscendAttentionCPImpl`（builder 同理，行 93）。

> 说明：vllm-ascend 里没有单独叫 "FullAttention" 的 backend。混合模型（`qwen3_next`/`qwen3_5`/`qwen3_5_moe`）的标准 attention 层 **复用同一个 `AscendAttentionCPImpl`**，只是把 `pcp_use_hybrid_attn=True` 打开后走另一条 qkv-gather 分支（见第 6 节）。所以本文件同时覆盖「普通 GQA」和「hybrid FullAttention」。

---

## 1. 数据结构与 metadata 构建

### 1.1 builder：`AscendAttentionCPMetadataBuilder.build`（`attention_cp.py:104`）

输入是 `common_attn_metadata`（含 `prefill_context_parallel_metadata`，即 PCPManager 产出的共享 metadata）。builder 把它整理成 `AscendMetadata`：

```python
num_actual_tokens_pcp_padded = long_seq_metadata.num_actual_tokens_pcp_padded   # pcp pad 后的 token 数
slot_mapping = common_attn_metadata.slot_mapping[:num_actual_tokens_pcp_padded]
# 区分 decode / prefill
num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(...)
```

- **Prefill metadata**（`AscendMetadataForPrefill`，`common_cp.py:73`）：把 `AscendPCPMetadata`（q_head/tail_idx、kv_nomask/mask_idx、q_full_idx、attn_mask_seqlens、restore_idx…）包进去；chunked prefill 时额外带 `ChunkedContextMetadata`。
- **Decode metadata**（`AscendMetadataForDecode`，`common_cp.py:99`）：`num_computed_tokens_of_pcp_dcp`（shape `[num_reqs, pcp_size, dcp_size]`，本 rank 本地 KV 长度）、`dcp_mtp_attn_mask`（MTP 用）。

`actual_seq_lengths_q`（行 246）按 `decode_threshold`（= 1 + num_speculative_tokens）构造，喂给 attention 的 `actual_seq_lengths`。

### 1.2 graph 支持

`get_cudagraph_support` 返回 `AttentionCGSupport.ALWAYS`（行 91）—— GQA-CP 支持 ACL Graph 全场景。decode 路径在 capturing 时把 attention 参数（含 `dcp_size/pcp_rank/dcp_rank/actual_seq_lengths_kv`）存进 `graph_params.attn_params`，回放时用 `update_graph_params`（行 313）更新。

---

## 2. reshape_and_cache：PCP/DCP 下的 KV 写入（`attention_cp.py:795`）

这是 PCP/DCP 正确写入分片 KV cache 的关键。

### 2.1 decode 部分（行 816）

```python
if has_decode:
    # decode token 在 PCP 组里是被复制分布的，只取每隔 pcp_size 的真实 slot
    slot_mapping = attn_metadata.slot_mapping[: num_decode_tokens * self.pcp_size : self.pcp_size]
    DeviceOperator.reshape_and_cache(key=key[:num_decode_tokens],
                                     value=value[:num_decode_tokens],
                                     key_cache=self.key_cache, value_cache=self.value_cache,
                                     slot_mapping=slot_mapping)
```

decode token 每个 rank 都有副本，但只有 `[::pcp_size]`（即 rank0 的副本）对应真实 slot，其余是 `-1`（不写）。这样 KV 只写一份，落在该 token 应属的 CP rank 的分片里。

### 2.2 prefill 部分（行 826）

分两种情况：

**(A) 普通 GQA（`pcp_use_hybrid_attn=False`，行 829-835）—— allgather KV：**
```python
kv = cat([key, value], dim=-1)
num_actual_tokens_pcp_padded_local = num_actual_tokens_pcp_padded // pcp_size
all_kv = get_pcp_group().all_gather(kv[:num_actual_tokens_pcp_padded_local].contiguous(), dim=0)
all_kv = index_select(all_kv, 0, pcp_allgather_restore_idx)   # 还原成原始顺序
key, value = all_kv.split([head_size, head_size], dim=-1)
```
即：每个 rank 先算自己那段的 K/V，allgather 拼成完整序列，用 `pcp_allgather_restore_idx` 还原顺序，再按 CP 分片 slot_mapping 写本地 cache。**每层都 allgather 一次完整 KV，算完即丢，控制峰值显存。**

**(B) Hybrid FullAttention（`pcp_use_hybrid_attn=True`，行 836-843）—— gather+restore QKV：**
```python
query, key, value = self._gather_and_restore_pcp_qkv(query, key, value, attn_metadata)
output_local_padded_tokens_fa = num_actual_tokens_pcp_padded // pcp_size - num_tokens
if output_local_padded_tokens_fa > 0:
    output_padded = F.pad(output, ..., value=0)   # 给 output 也 pad
```
走 `_gather_and_restore_pcp_qkv`（第 6 节详述），把 qkv 还原成全序后交给普通 FA。output 也按需 pad。

随后统一用 prefill 段的 slot_mapping 写 cache（行 845-856）。

---

## 3. Prefill forward（纯 prefill，无 chunked context）

入口 `forward_impl`（`attention_cp.py:951`）→ `_forward_prefill_cp`（行 485）。分三步：

### 3.1 pre：拆 head/tail，选 KV（`_forward_prefill_cp_pre`, 行 500）

```python
q_head = index_select(query, 0, q_head_idx)
q_tail = index_select(query, 0, q_tail_idx)
# KV 拆成 nomask（完全可见，rank 之前的段）和 mask（自己的段，要 causal）
k_head_nomask = index_select(key, 0, kv_with_q_head_nomask_idx) if pcp_rank > 0 else None
k_head_mask   = index_select(key, 0, kv_with_q_head_mask_idx)
# tail 同理
```

注意：KV 这里是 **reshape_and_cache 之前** 的、本 rank 算出来的 K/V（已在第 2 节 allgather 还原成完整序列）。所以 head 段的 Q 可以从完整 KV 里选「我之前的段（nomask）+ 我自己（mask）」。

### 3.2 attn：nomask + mask 两次 attention（`_attention_with_nomask_and_mask`, 行 421）

```python
# nomask 路径：atten_mask=None, sparse_mode=0（更快）
attn_out_nomask, attn_lse_nomask = npu_fused_infer_attention_score(
    q, k_nomask, v_nomask, atten_mask=None, sparse_mode=0, softmax_lse_flag=True, ...)
# mask 路径：atten_mask=causal, sparse_mode=3
attn_out_mask, attn_lse_mask = npu_fused_infer_attention_score(
    q, k_mask, v_mask, atten_mask=mask, sparse_mode=3, softmax_lse_flag=True, ...)
# 合并两段（online softmax）
output = _npu_attn_out_lse_update(attn_lse_mask, attn_lse_nomask, attn_out_mask, attn_out_nomask)
```

把一次 attention 拆成「完全可见的 KV（nomask，无需 mask 计算，省算力）」+「需要 causal mask 的 KV」两次调用，用硬件算子 `npu_attention_update` 做归并。head 段、tail 段各做一遍。

### 3.3 post：还原顺序（`_forward_prefill_cp_post`, 行 558）

```python
output = index_select(cat([output_head, output_tail], dim=0), 0, q_full_idx)
```
`q_full_idx`（对 `[head, tail]` 做 argsort）把两段输出还原成本 rank Q 的原始顺序。

---

## 4. Decode forward：PCP+DCP 叠加（`_forward_decode_pcp_dcp`, 行 566）

这是 PCP 和 DCP **同时** 作用的典型路径。

```python
# 1. DCP 组：head 维 allgather Q（让 DCP 组内 Q head 一致）
if dcp_size > 1:
    query = get_dcp_group().all_gather(query.contiguous(), 1)
    num_heads = self.num_heads * dcp_size
# 2. 本地 KV cache（已按 CP 分片）做 attention
common_kwargs = dict(
    num_heads=num_heads,
    block_table=decode_meta.block_tables,
    block_size=key_cache.shape[1],
    actual_seq_lengths_kv=decode_meta.num_computed_tokens_of_pcp_dcp[:num_decodes, pcp_rank, dcp_rank],  # 本地 KV 长度
    actual_seq_lengths=actual_seq_lengths_q[:num_decodes],
    softmax_lse_flag=True, ...)
attn_out, attn_lse = npu_fused_infer_attention_score(query, k_nope, value, **common_kwargs)
# 3. 合并 out/lse：先 DCP all2all，再 PCP allgather，最后 online-softmax
attn_out_lse = _process_attn_out_lse(attn_out, attn_lse)
attn_out = _npu_attention_update(self.head_size, attn_out_lse)
```

### 4.1 `_process_attn_out_lse`（`common_cp.py:108`）—— PCP/DCP 的合并枢纽

```python
attn_out_lse = cat([attn_output, softmax_lse], dim=-1)   # [bs, h, d+1]
if dcp_size > 1:
    # head 维 all_to_all（DCP 组内 Q head 不同，要交换 out/lse）
    attn_out_lse = attn_out_lse.permute([1,2,0]).contiguous()
    dist.all_to_all_single(..., group=dcp_group)
    attn_out_lse = attn_out_lse.permute([2,0,1])
if pcp_size > 1:
    # seq 维 all_gather（PCP 组内）
    attn_out_lse = get_pcp_group().all_gather(attn_out_lse.contiguous(), dim=0)
```

合并顺序固定：**DCP all2all（head） → PCP allgather（seq） → `_npu_attention_update`（online softmax 归并）**。

### 4.2 MTP / spec decode（行 582-590）

spec decode 时 `input_layout="BSND"`，每个 decode request 有多个 token（history + draft）。此时：
- `attn_mask = decode_meta.dcp_mtp_attn_mask`（PCPManager 用 `generate_mtp_attention_mask_for_decode` 按 `(history_len + mtp_idx) % cp_size` 生成）；
- `actual_seq_lengths_q` 复制成每 request 相同值。

graph 捕获时把 `pcp_rank/dcp_rank/dcp_size` 存进参数（行 659-662），回放时更新（`update_graph_params` 行 359-389）。

---

## 5. Chunked prefill：AllGatherQ 路径

GQA 后端对 chunked prefill 用 **AllGatherQ**（与 decode 流程一致），而不是 MLA 的 AllGatherKV。详见 `vllm-ascend-chunked-prefill-PCP.md` 第 2 节。要点：

- `_prefill_query_all_gather`（行 716）：PCP allgather Q + `cp_kv_recover_idx_for_chunk` 还原 + DCP allgather Q(head)；
- 多流 overlap：`cp_chunkedprefill_comm_stream()` 把 allgather Q / a2a-ag output 与本地 head/tail attention overlap（行 978-982 的流程图）；
- context 部分 `_compute_prefill_context`（行 726）+ `_gather_global_context_output`（行 919，DCP all2all + PCP allgather）+ `_update_chunk_attn_out_lse_with_current_attn_out_lse`（行 680，合并到 current chunk）。

---

## 6. Hybrid FullAttention：`_gather_and_restore_pcp_qkv`（行 861）

混合模型（`pcp_use_hybrid_attn=True`）的标准 attention 层。因为相邻的线性注意力层要求 token **按全序、线性切分** 处理，所以 FullAttention 层不能像普通 GQA 那样 head-tail 独立算，而要：

```python
# 1. cat qkv，必要时 pad（pcp_padded_tokens_fla）
qkv_fla = cat([query.reshape(N,-1), key.reshape(N,-1), value.reshape(N,-1)], dim=-1)
if pcp_padded_tokens_fla > 0:
    qkv_fla = F.pad(qkv_fla, ..., value=0)
# 2. allgather (PCP 组, dim=0)，取 max_num_tokens_across_pcp
all_qkv = get_pcp_group().all_gather(qkv_fla[:max_num_tokens_across_pcp].contiguous(), dim=0)
# 3. 用 pcp_enter_fa_restore_idx 还原成全局顺序
actual_qkv = index_select(all_qkv, 0, pcp_enter_fa_restore_idx)
# 4. 填进 workspace：真实 token 用 actual_qkv，padding 位填 0
qkv_fa_padding_workspace[decode_offset:][pcp_unpad_mask] = actual_qkv[decode_offset:]
qkv_fa_padding_workspace[decode_offset:][~pcp_unpad_mask].fill_(0)
# 5. split 回 q/k/v；q 用 pcp_fa_query_idx 选出 FA 要算的 query
q_fa[num_decode_tokens:] = index_select(q[decode_offset:], 0, pcp_fa_query_idx)
```

attention 算完后（`forward_impl` 行 1051-1056）：
```python
if pcp_size > 1 and pcp_use_hybrid_attn:
    attn_output_prefill = get_pcp_group().all_gather(attn_output_prefill.contiguous(), dim=0)
    attn_output_prefill = index_select(attn_output_prefill, 0, pcp_exit_fa_scatter_idx)
```

> 一句话：hybrid FullAttention = **allgather qkv → enter 还原成全序 → 普通 FA → allgather output → exit scatter 回线性切分布局**。`enter_fa_restore_idx` / `exit_fa_scatter_idx` 由 PCPManager 在 `update_tokens_for_pcp` 的 hybrid 分支预算（详见 `vllm-ascend-PCP-hybrid-model.md`）。

---

## 7. 通信与计算汇总表

| 阶段 | 计算 | 通信 | 说明 |
|---|---|---|---|
| reshape_and_cache (decode) | 写本地 KV | — | slot_mapping 取 `[::pcp_size]`，只写一份 |
| reshape_and_cache (prefill, 普通) | — | **PCP allgather KV**（每层）+ restore_idx 还原 | allgather 完整 KV，算完即丢 |
| reshape_and_cache (prefill, hybrid) | — | **PCP allgather QKV** + enter_idx 还原 | 给 FA 全序输入 |
| prefill (纯) | head/tail 各 nomask+mask attn | — | KV 已在前一步 gather |
| chunked prefill | head/tail attn + context attn | **PCP allgather Q**（recover idx）+ **DCP all2all + PCP allgather**（context output） | 多流 overlap |
| decode | 本地 KV attn | **DCP allgather Q(head)** → 本地 attn → **DCP all2all + PCP allgather**（out/lse）→ update | PCP+DCP 同时 |
| hybrid FA 输出 | FA | **PCP allgather output** + exit_idx scatter | 还原线性布局 |

---

## 8. forward_impl 顶层分发（`attention_cp.py:951`）

```python
def forward_impl(self, query, key, value, kv_cache, attn_metadata, output):
    if has_decode:
        output_decode = self._forward_decode_pcp_dcp(query[:num_decode_tokens], attn_metadata)
        output[:num_decode_tokens] = output_decode
    if has_prefill:
        has_chunked_context = prefill.chunked_context is not None
        # qkv 切出 prefill 段（考虑 pcp padding）
        prefill_query = query[num_decode_tokens : num_actual_tokens_pcp_padded//pcp_size]
        ...
        if has_chunked_context:   # AllGatherQ 多流 overlap 流程
            ... (见第 5 节)
        if pcp_size > 1:          # 纯 PCP prefill
            head/tail attn + post 还原
        if has_chunked_context:   # 合并 context
            ...
        if pcp_size > 1 and pcp_use_hybrid_attn:   # hybrid FA 输出 scatter
            allgather output + exit_idx
        output[num_decode_tokens:...] = attn_output_prefill
```

一次 forward 可同时处理 decode + prefill（mixed batch），decode 在前、prefill 在后，与 PCPManager 的 `num_decode_reqs`/`num_prefill_reqs` 划分一致。

---

## 9. 对 vLLM 上游 RFC 的启示（GQA/FullAttention 部分）

1. **allgather-KV 是通用 GQA 的默认 PCP 策略**：实现简单、正确性易保证，代价是峰值显存（每层一份完整 KV）。Ring-attention 可作为高阶可选。
2. **nomask/mask 拆分**是降低 causal attention 算力的有效手段，跨 rank 的「完全可见 KV」可走无 mask 快路径 —— 上游 FA kernel 应支持这种分段调用 + online-softmax 合并。
3. **out/lse 合并原语**（`_process_attn_out_lse` + `_npu_attention_update`）应抽成框架级 API，所有后端复用，统一 PCP/DCP 的合并顺序（DCP all2all → PCP allgather → update）。
4. **hybrid FullAttention 的 enter/exit 布局转换**表明：同一个 backend 要支持「独立 head-tail」和「全序 FA」两种模式，上游需要 **per-model-type 的布局转换钩子**，而不是把布局硬编码。
5. **decode slot_mapping `[::pcp_size]`** 与 **block-table 虚拟块分片**配合，是 KV 写入只写一份的关键 —— 这套逻辑应在框架层统一，避免每个 backend 重复实现。
