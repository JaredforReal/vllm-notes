# vLLM-Ascend Context Parallel (CP) 入门总览

> 对应原始问题：*Intro to vLLM-Ascend PCP support, how does it integrate with DCP? How is it supported under different attn backends? PCPManager? PCPMetadata?*

本文件是 CP（Context Parallel）系列学习笔记的总纲。读完这篇后，可以再按需阅读同目录下的：
`vllm-ascend-zigzag.md`（序列切分）、`vllm-ascend-chucked-prefill-PCP.md`（chunked prefill 兼容）、
`vllm-ascend-PCP-DCP.md`（DCP）、`vllm-ascend-PCP-MLA.md`、`vllm-ascend-PCP-DSA.md`、
`vllm-ascend-PCP-Mamba.md`、`vllm-ascend-PCP-hybrid-model.md`。

---

## 1. 概念：CP = PCP + DCP

vLLM-Ascend 把 Context Parallel 拆成两个互不冲突、可以单独/组合开启的特性：

| | **PCP** (Prefill Context Parallel) | **DCP** (Decode Context Parallel) |
|---|---|---|
| 解决的问题 | 长 prefill 的 TTFT（首个 token 时间） | 长 decode 的 KV cache 冗余、batch 上限 |
| 切分维度 | **序列维度**，把一条 prefill 序列切给多张卡 | **序列维度**，把 KV cache 按 seq 分片 |
| 通信域 | 新建独立的 PCP 通信域（占用额外显存/卡） | **复用 TP 通信域**，不额外占卡 |
| 影响阶段 | Prefill（及 decode 的收尾 allgather） | Decode、Chunked Prefill、Cached Prefill |
| 开关 | `--prefill-context-parallel-size` (`prefill_context_parallel_size`) | `--decode-context-parallel-size` (`decode_context_parallel_size`) |

关键关系式（贯穿整个实现）：

```
cp_size = pcp_size * dcp_size
cp_rank = pcp_rank * dcp_size + dcp_rank
```

总 world size = `tensor_parallel_size * prefill_context_parallel_size`（注意 DCP 复用 TP 域，所以不算进总卡数，但要求 `tp_size` 能被 `dcp_size` 整除/约束）。

参考文档：`vllm-ascend/docs/source/developer_guide/Design_Documents/context_parallel.md`、
`vllm-ascend/docs/source/user_guide/feature_guide/context_parallel.md`。

设备拓扑示意（PCP2, DCP2, TP4）：

```
TP4 group 内部还有 DCP2 维度（复用 TP 卡）；
PCP2 则跨越两组 TP 域，是独立的通信域。
```

---

## 2. 在 vLLM-Ascend 中的总体架构

CP 在 vLLM-Ascend 里是“一层是调度/布局改写，另一层是 attention backend 替换”的组合。

### 2.1 两层结构

```
┌──────────────────────────────────────────────────────────────────┐
│ Scheduler / ModelRunner 层（CPU 侧改写 token 布局）                │
│   - worker/pcp_utils.py          PCPManager                       │
│   - worker/model_runner_v1.py    调用 PCPManager、生成 metadata    │
│   - worker/block_table.py        slot_mapping / block table 重算  │
└──────────────────────────────────────────────────────────────────┘
                            ↓ common_attn_metadata.prefill_context_parallel_metadata
┌──────────────────────────────────────────────────────────────────┐
│ Attention backend 层（GPU/NPU 侧真正的通信+计算）                  │
│   attention/context_parallel/                                    │
│     common_cp.py        公共 metadata + out/lse 合并工具          │
│     attention_cp.py     GQA 后端   (AscendAttentionCPImpl)        │
│     mla_cp.py           MLA 后端   (AscendMlaCPImpl)              │
│     dsa_cp.py           DSA 后端   (AscendDSACPImpl, DSv4/GLM-5)  │
│     sfa_cp.py           SFA 后端   (AscendSFACPImpl, 稀疏MLA)     │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 backend 的选择：`enable_cp()` 开关

每个 attention backend 的工厂方法都用 `enable_cp()` 做分支，CP 开启时返回 `*CPImpl` / `*CPMetadataBuilder`，否则返回普通实现。这是一个很干净的“条件替换”式接入点：

- GQA：`vllm_ascend/attention/attention_v1.py:85` `get_impl_cls()` → `AscendAttentionCPImpl`
- MLA：`vllm_ascend/attention/mla_v1.py:106` → `AscendMlaCPImpl`
- DSA：`vllm_ascend/attention/dsa_v1.py:209` → `AscendDSACPImpl`
- SFA：`vllm_ascend/attention/sfa_v1.py:100` → `AscendSFACPImpl`

`enable_cp()` 定义在 `vllm_ascend/attention/utils.py:61`：

```python
return prefill_config.prefill_context_parallel_size > 1 \
    or prefill_config.decode_context_parallel_size > 1
```

即：**只要 pcp_size>1 或 dcp_size>1，所有 attention 后端都会切换到 CP 版本**。不同模型按 KV cache spec 走对应后端（GQA/MLA/DSA/SFA），各自的 CP 版本都从同一套 `common_attn_metadata.prefill_context_parallel_metadata` 取数据。

---

## 3. PCPManager（PCP 的“大脑”）

**源码：`vllm_ascend/worker/pcp_utils.py`，类 `PCPManager`**，在 ModelRunner 中以 `self.pcp_manager` 持有（`model_runner_v1.py:422` 创建，仅在 `pcp_size*dcp_size>1` 时创建）。

PCPManager 负责把“调度器给出的整条序列”改写成“本 PCP rank 上要处理的那一段”，并准备好 attention 后端需要的索引张量。核心职责：

1. **批次信息**：`init_batch_info()`（`pcp_utils.py:181`）——统计本批次有多少 decode req / prefill req，多少 decode token。`decode_threshold = 1 + num_speculative_tokens` 用来区分 decode 与 prefill。

2. **token / position 改写（最核心）**：`update_tokens_for_pcp()`（`pcp_utils.py:502`）——
   - 把每条 prefill 序列 pad 到 `2*pcp_size` 的倍数；
   - 用 **DualChunkSwap / head-tail**（即 zigzag）方式把序列切成 `2*pcp_size` 段，head 段与 tail 段配对，分配到各 rank，保证 **负载均衡**（前后部分计算量不同，首尾配平）；
   - 计算本 rank 的 `pcp_positions`（每个 token 在原序列里的绝对位置）；
   - 计算 `pcp_allgather_restore_idx`：因为之后对 KV/Q 做 allgather 后不同 rank 的块是交错的，需要这个索引把 allgather 结果 **还原成原始顺序**；
   - 计算 `num_pcp_pads_cpu` / `pcp_unpad_mask_cpu`：标记哪些是真实 token、哪些是 padding。
   - 对 **hybrid-attention 模型**（`qwen3_next` / `qwen3_5` / `qwen3_5_moe`，`pcp_utils.py:131`）走另一条 `pcp_use_hybrid_attn` 分支，输出的是“按 rank 线性切分”的布局，并额外准备 `pcp_enter_fa_restore_idx` / `pcp_exit_fa_scatter_idx`（详见 hybrid-model 笔记）。

3. **slot_mapping 对齐**：`get_padded_slot_mapping()`（`pcp_utils.py:809`）——allgather 后 KV 带了 padding，需要把 slot_mapping 也 pad 对齐，非真实 token 的 slot 填 `-1`。

4. **隐藏态还原**：`get_restore_hidden_states()`（`pcp_utils.py:831`）——在 **最后一层** 之后，把本 rank 的 hidden states allgather 回来并按 `pcp_allgather_restore_idx` 还原成原始顺序，供 lm_head / sampler 使用。

5. **logits 索引**：`get_logits_indices()`（`pcp_utils.py:787`）——因为 token 数被 pad/复制了，采样时要映射回每个 request 最后一个 token。

6. **本地化 MM/scheduler**：`maybe_localize_scheduler_output_for_mm_preprocess()`、`gather_mm_embeddings_for_pcp()` 等——多模态场景下每个 PCP rank 只处理序列的一段，需要把 scheduler_output 和 mm embedding 也“本地化”到本 rank 实际处理的 token。

7. **spec decode / MTP 兼容**：`generate_pcp_mtp_input()`、`generate_mtp_attention_mask_for_decode()`——MTP 的 input_ids 要在 PCP 切分 **之前** 做偏移，所以要单独保存一份“未切分”的 input_ids/positions（`input_ids_pcp_full` 等 buffer）。

PCPManager 持有大量 `CpuGpuBuffer`（CPU/GPU 双 buffer，避免热路径 D2H 同步阻塞 AsyncScheduler），如 `pcp_allgather_restore_idx`、`pcp_exit_fa_scatter_idx`、`dcp_mtp_attn_mask` 等。

> 一句话：**PCPManager 在 CPU 侧把全局 token 布局改写成 PCP-local 布局，并产出所有后端共享的索引张量。**

---

## 4. PCPMetadata（传递给后端的张量集合）

CP 共享 metadata 通过 `common_attn_metadata.prefill_context_parallel_metadata`（类型 `AscendPrefillContextParallelMetadata`）传递，由 `PCPManager.generate_pcp_metadata()`（`pcp_utils.py:1066`）生成。

后端层把它进一步整理成几个 dataclass，定义在 `vllm_ascend/attention/context_parallel/common_cp.py`：

- **`AscendPCPMetadata`**（`common_cp.py:9`）：per-layer 的 PCP 索引。最核心的几个：
  - `q_head_idx` / `q_tail_idx`：把本 rank 的 Q 切成 head / tail 两段（head-tail 风格）的下标；
  - `kv_with_q_head_nomask_idx` / `kv_with_q_head_mask_idx`（以及 tail 版本）：对一个 Q 段做 attention 时，KV 里 **不需要 mask（已完全可见）** 的部分 vs **需要 causal mask** 的部分；
  - `kv_tail_proj_idx` / `kv_with_q_head_attn_idx_in_tail`：MLA tail 段把 KV 投影到 nope/value 用的索引；
  - `q_full_idx`：head/tail 两段 attention 输出 cat 之后，还原成 Q 原始顺序的索引（`argsort` 得到）；
  - `attn_mask_seqlens` / `head_attn_nomask_seqlens` / `tail_attn_nomask_seqlens` / `head_actual_seq_lengths_kv` / `tail_actual_seq_lengths_kv`：喂给 `npu_fused_infer_attention_score` 的 `actual_seq_lengths*`；
  - `pcp_allgather_restore_idx`：allgather 后还原顺序；
  - `pcp_use_hybrid_attn` / `pcp_unpad_mask` / `pcp_enter_fa_restore_idx` / `pcp_exit_fa_scatter_idx` / `pcp_fa_query_idx` / `max_num_tokens_across_pcp`：hybrid-attention（mamba+attn 混合）专用。

- **`AscendMetadataForPrefill`**（`common_cp.py:73`）+ 内嵌 `ChunkedContextMetadata`：把 `AscendPCPMetadata` 包进去，再加上 chunked prefill 相关字段。
- **`AscendMetadataForDecode`**（`common_cp.py:99`）：decode 阶段用的 `num_computed_tokens_of_pcp_dcp`（shape `[num_reqs, pcp_size, dcp_size]`，表示每个 CP rank 上该 req 的本地 KV 长度）、`dcp_mtp_attn_mask`。

`generate_pcp_metadata()` 还会算出 `num_computed_tokens_of_pcp_dcp`（`_get_cp_local_seq_lens`, `pcp_utils.py:1042`）——把每个 req 的 context_len 按 `cp_kv_cache_interleave_size` 均分到各 CP rank，这是 **decode/chunked prefill 时本地 KV 长度** 的来源。

---

## 5. out & lse 的合并：online softmax

CP 下，一条序列的 attention 被 shard 到多个 rank 各自算了一部分，每个 rank 产出 `(attn_output, softmax_lse)`。要合并这些部分结果，必须用 **online softmax / log-sum-exp** 合并，而不是简单求和。核心工具函数在 `common_cp.py`：

- `_process_attn_out_lse()`（`common_cp.py:108`）：把 `[bs,h,d]` 的 out 和 `[bs,h,1]` 的 lse cat 成 `[bs,h,d+1]`；
  - DCP>1 时做一次 head 维的 **all_to_all**（DCP 组内 Q head 不同，要交换）；
  - PCP>1 时做一次 seq 维的 **all_gather**（PCP 组内）。
- `_npu_attention_update()`（`common_cp.py:130`）：把 allgather/all2all 后的 `[pcp*dcp, s, h, d+1]` 拆回 out/lse，调 `torch_npu.npu_attention_update` 做归并（硬件算子）。
- `_update_out_and_lse()`（`common_cp.py:183`）：纯 PyTorch 版的 `logsumexp` 合并，`LSE_final=log(sum(exp(LSE_i)))`，`O_final=sum(exp(LSE_i-LSE_final))*O_i`。
- `_npu_attn_out_lse_update()`：nomask 段与 mask 段结果合并（chunked prefill 的 context 部分更新）。

这套 out/lse 合并机制是 **所有后端共用的**，是 CP 正确性的基石。

---

## 6. PCP 与 DCP 如何协作

PCP 和 DCP 作用于 **同一份 KV cache 分片布局**，二者叠加：

- **KV cache 存储布局**（`block_table.py:170` `compute_slot_mapping`）：
  - 定义“虚拟块” `virtual_block_size = block_size * cp_size`（cp_size = pcp_size*dcp_size）；
  - token `x` 的虚拟块号 = `x // virtual_block_size`，块内偏移 = `x % virtual_block_size`；
  - 本地块号 = `offset // cp_kv_cache_interleave_size`，目标 rank = `local_block_index % cp_size`；
  - 落到本 rank 的 token 才写真实 slot，否则填 `-1`。
  - `cp_kv_cache_interleave_size`（默认 1=token 粒度交错；KV 搬运场景如 PD disagg/KV pool 需设为 `block_size`=128）决定交错粒度。

- **Prefill 阶段**：PCP 用 **allgather KV**（每层临时 gather 完整 KV，算完即丢，控制峰值显存）。DCP（MLA 后端）则是 allgather KV 后用 `reorg_kvcache` 把多 request 的交错结果整理连续；GQA 后端是 allgather Q（head 维），再用 `cp_lse_ag_out_rs` 风格合并。

- **Decode 阶段**：先在 DCP 组做 head 维 allgather Q，本地算，再 all-to-all 交换 out/lse；PCP>1 时在 DCP 收尾的 allgather 之后再加一次 **PCP 组内 seq 维 allgather**（`_process_attn_out_lse` 里 `pcp_size>1` 分支）。顺序：DCP all2all → PCP allgather → `_npu_attention_update` 合并。

- **Chunked Prefill**：有 AllGatherQ / AllGatherKV / Ring 三种思路。GQA 后端实现 AllGatherQ（与 decode 流程一致），MLA 后端实现 AllGatherKV（与标准 prefill 一致），并按 workspace 分段处理以控峰值显存。详见 `vllm-ascend-chucked-prefill-PCP.md`。

> 设计取舍（来自设计文档）：vLLM-Ascend **优先选 allgather-KV 实现**，而没用 Ring-Attention。理由是 Ring 开发复杂度高、且 overlap 收益有限。这也是写 vLLM RFC 时一个值得讨论的点。

---

## 7. 不同 attention backend 的差异速览

| 后端 | 文件 | 适用模型 | Prefill 通信 | Decode 通信 | 特殊点 |
|---|---|---|---|---|---|
| **GQA** | `attention_cp.py` | Qwen3-235B 等通用 GQA | head-tail 拆 Q，nomask/mask 两段 attn；chunked prefill 用 AllGatherQ | DCP allgather Q(head) → 本地 attn → DCP all2all + PCP allgather → update | 支持普通 + hybrid(qwen3_next) 两种布局 |
| **MLA** | `mla_cp.py` | DeepSeek-V2/V3、Kimi-K25 | head-tail 拆 Q，KV 用 `kv_tail_proj_idx` 做投影；chunked prefill 用 AllGatherKV + `_reorg_kvcache` | DCP allgather q_nope+q_pe → attn → out/lse 合并 → `_v_up_proj` | MLA 有 kv_lora 压缩、W_UV 投影；block_size 会因 cp_virtual_block_size 调整 |
| **DSA** | `dsa_cp.py` | DeepSeek-V4、GLM-5 | 按 TP rank 线性切 token（`_build_local_token_metadata`），不走 head-tail | 同 prefill 切法 + sparse_attn_sharedkv | 稀疏注意力 + compressor/indexer，`DSACPMetadata` 保存 local 切片；multi-stream 预处理 overlap |
| **SFA** | `sfa_cp.py` | 稀疏 MLA（带 lightning indexer） | head-tail 拆 Q；KV 用 **compact block view** allgather（只 gather 真实 block） | `gather_kv_cross_cp` 按 request-scoped allgather KV | 拓扑要求 `cp_kv_cache_interleave_size = block_size` |

> DSA 与 GQA/MLA/SFA 的切法不同：DSA 是 **沿 flatten token 流按 TP/CP rank 线性切片**（见 `_build_local_token_metadata` 注释里的例子），而不是 head-tail zigzag。

---

## 8. 推荐的阅读顺序与代码入口

1. **调度改写**：`pcp_utils.py:update_tokens_for_pcp` → `generate_pcp_metadata` → `block_table.py:compute_slot_mapping`
2. **GQA 后端**：`attention_cp.py:forward_impl`（看 `forward_impl` 里 current_stream / COMM_STREAM 的多流 overlap 注释）
3. **out/lse 合并**：`common_cp.py:_process_attn_out_lse` + `_npu_attention_update`
4. **MLA/DSA/SFA**：分别 `mla_cp.py` / `dsa_cp.py` / `sfa_cp.py`
5. **Mamba/线性注意力**：`ops/triton/mamba/causal_conv1d.py`（conv state 跨 rank 传播）、`ops/gdn.py`
6. **设计图**：`docs/source/developer_guide/Design_Documents/context_parallel.md` 下的多张 png（overview/device_world/blocktable/pcp-prefill/pcp-decode/dcp-prefill/dcp-decode/chunkedprefill/head-tail-style）

---

## 9. 对 vLLM 上游 CP RFC 的启示（待补充）

读完这套实现，起草 vLLM CP RFC 时值得提炼的设计点：
- **PCP/DCP 解耦** + `cp_size = pcp_size*dcp_size`、`cp_rank = pcp_rank*dcp_size+dcp_rank` 的统一模型；
- **head-tail(zigzag) 负载均衡** + `pcp_allgather_restore_idx` 还原顺序的通用方案；
- **统一的 out/lse online-softmax 合并原语**（`_process_attn_out_lse`/`_npu_attention_update`）跨后端复用；
- **allgather-KV 优先**（开发成本低），Ring-attention 作为可选高阶路径；
- **backend 工厂 `enable_cp()` 条件替换**的接入点设计；
- 不同模型族（GQA/MLA/稀疏MLA/线性注意力/混合）在 KV 结构、切分粒度上的差异，需要 RFC 提供一个 **可扩展的 per-backend 接口** 而非一套硬编码逻辑。
