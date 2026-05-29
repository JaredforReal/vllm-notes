# KDA 状态管理与 vLLM Speculative Decoding 要点分析

> 代码基准路径: `/Users/jared/vllm-project/vllm/`
>
> 本文档梳理 vLLM speculative decoding 的完整流程，以及 KDA (Kimi Delta Attention) 层的状态管理机制，分析二者对接的关键点。

---

## 目录

- [1. Speculative Decoding 全流程](#1-speculative-decoding-全流程)
  - [1.1 整体架构](#11-整体架构)
  - [1.2 Mamba Cache 三种模式](#12-mamba-cache-三种模式)
  - [1.3 Block 分配策略](#13-block-分配策略)
  - [1.4 一次推理循环的完整过程](#14-一次推理循环的完整过程)
- [2. Metadata 构建与 Spec/Non-Spec 分离](#2-metadata-构建与-specnon-spec-分离)
  - [2.1 MambaHybridModelState: 数据来源](#21-mambahybridmodelstate-数据来源)
  - [2.2 GDNAttentionMetadataBuilder: 分发逻辑](#22-gdnattentionmetadatabuilder-分发逻辑)
  - [2.3 Block Table 的 2D 结构](#23-block-table-的-2d-结构)
- [3. KDA 状态管理详解](#3-kda-状态管理详解)
  - [3.1 四种 State 的物理布局](#31-四种-state-的物理布局)
  - [3.2 Decode 路径: fused_recurrent_kda](#32-decode-路径-fused_recurrent_kda)
  - [3.3 Prefill 路径: chunk_kda](#33-prefill-路径-chunk_kda)
  - [3.4 当前 KDA 层的问题](#34-当前-kda-层的问题)
- [4. Mamba2 参考实现](#4-mamba2-参考实现)
  - [4.1 状态索引传递](#41-状态索引传递)
  - [4.2 Conv State 的 num_spec 扩展](#42-conv-state-的-num_spec-扩展)
  - [4.3 selective_state_update 的 Spec 处理](#43-selective_state_update-的-spec-处理)
- [5. 关键差异: KDA vs Mamba2](#5-关键差异-kda-vs-mamba2)

---

## 1. Speculative Decoding 全流程

### 1.1 整体架构

```
Scheduler Step
    │
    ├── 1. preprocess_mamba()        ─── 复制 state 到 running block
    │
    ├── 2. build_attn_metadata()     ─── 构建 spec/non-spec 分离的 metadata
    │
    ├── 3. 主模型 forward()          ─── target model 处理 accepted + new tokens
    │
    ├── 4. sample()                  ─── 采样 next token
    │
    ├── 5. postprocess_state()       ─── 记录 num_accepted_tokens 到 GPU
    │
    ├── 6. propose()                 ─── MTP draft model 生成 spec tokens
    │   │                                (循环 num_spec 次，每次一个 draft token)
    │   │
    │   └── 对于每次 draft:
    │       ├── update positions (+1)
    │       ├── rebuild attn metadata (draft_index++)
    │       └── draft model forward
    │
    ├── 7. postprocess_mamba()       ─── 复制 state 到 completed blocks
    │
    └── 下一步: Scheduler 验证 spec tokens，接受/拒绝
```

**关键时序：** 验证（verification）发生在**下一个** scheduler step 的主模型 forward 中。当验证结果到达时，`num_accepted_tokens` 已经通过 `postprocess_state` 写入了 GPU tensor，供 metadata builder 使用。

### 1.2 Mamba Cache 三种模式

> 文件: `vllm/config/cache.py:36`

```python
MambaCacheMode = Literal["all", "align", "none"]
```

| 模式 | 触发条件 | 每请求 Block 数 | State 保存位置 |
|------|---------|----------------|---------------|
| `"none"` | prefix caching 关闭 | `1 + num_spec_blocks` | 仅 running block |
| `"align"` | prefix caching 开启 + spec decode | `2 + num_spec_blocks` | running block + 上一个对齐 block |
| `"all"` | prefix caching 开启，无 spec decode | `cdiv(max_len, block_size)` | 每个 block boundary |

**Spec decode 通常使用 `"none"` 或 `"align"`。**

- `"none"` 模式下，每请求只有 1 个 running state block + `num_spec` 个 speculative blocks
- `"align"` 模式下额外多 1 个 block 用于 prefix caching 对齐

### 1.3 Block 分配策略

> 文件: `vllm/v1/core/single_type_kv_cache_manager.py:893-958` (`MambaManager.get_num_blocks_to_allocate`)

**`"none"` 模式的 Block 布局：**

```
Block 0: running state (conv_state + recurrent_state)
Block 1: speculative slot 1
Block 2: speculative slot 2
...
Block num_spec: speculative slot num_spec
```

Block Table 形状: `[num_requests, 1 + num_spec]`

- `block_table[i, 0]` → 请求 i 的 running state block
- `block_table[i, 1..num_spec]` → 请求 i 的 speculative state slots

### 1.4 一次推理循环的完整过程

以 `num_spec=3`, `block_size=1` 为例，一次完整循环：

**Step N: 主模型 forward（验证上一步的 spec tokens + 处理新 token）**

```
请求 A（上一步 spec 了 3 个 draft，验证后接受了 2 个）:
  input: [accepted_token_0, accepted_token_1, new_token]
         ├── verified spec tokens ──┤── new token ──┤

  num_accepted_tokens = 2 (从上一步 postprocess_state 写入)
  num_decode_draft_tokens = 3 (spec tokens)

  Metadata:
    spec_state_indices_tensor = [[block_A_0, block_A_1, block_A_2, block_A_3]]  (2D)
    spec_query_start_loc = [0, 3]
    non_spec_* = None (全是 spec tokens)

  主模型 forward 对这 3 个 token 处理:
    - token 0: 从 block_A_0 读 state → 写回 block_A_0
    - token 1: 从 block_A_0 读 state → 写回 block_A_1
    - token 2: 从 block_A_0 读 state → 写回 block_A_2
    (具体写哪个 block 由 2D ssm_state_indices 决定)
```

**Step N: MTP propose（生成新的 draft tokens）**

```
proposer.propose():
  第一轮: 用主模型输出 + 上一步 hidden_states → draft_token_1
    positions += 1, rebuild metadata
    draft model forward (只有 1 token per request)

  第二轮: 用 draft_token_1 的 hidden_states → draft_token_2
    positions += 1, rebuild metadata
    draft model forward

  第三轮: 用 draft_token_2 的 hidden_states → draft_token_3
    positions += 1, rebuild metadata
    draft model forward

  返回: [draft_1, draft_2, draft_3]
```

**Step N+1: Scheduler 验证**

```
验证 draft tokens:
  假设接受了 1 个，拒绝了 2 个

  postprocess_state(): num_accepted_tokens = 1
  下一步的 num_decode_draft_tokens = 3 (新的 spec tokens)
```

---

## 2. Metadata 构建与 Spec/Non-Spec 分离

### 2.1 MambaHybridModelState: 数据来源

> 文件: `vllm/v1/worker/gpu/model_states/mamba_hybrid.py`

两个关键 tensor，在每步之间传递：

| Tensor | 类型 | 含义 |
|--------|------|------|
| `num_accepted_tokens_gpu` | GPU, `[max_num_reqs]` | 上一步验证接受了多少 spec tokens |
| `num_decode_draft_tokens_cpu` | CPU, `[max_num_reqs]` | 当前步有多少 draft tokens（-1 表示非 spec） |

**生命周期：**

```python
# 步骤结束:
postprocess_state(num_sampled):
    num_accepted_tokens_gpu[req_idx] = clamp(num_sampled, min=1)

# 下一步开始:
prepare_attn():
    num_accepted_tokens = num_accepted_tokens_gpu[idx_mapping]  # 传给 metadata builder
    num_decode_draft_tokens_cpu = [-1, -1, 3, -1, 3, ...]       # spec 请求 >= 0
```

### 2.2 GDNAttentionMetadataBuilder: 分发逻辑

> 文件: `vllm/v1/attention/backends/gdn_attn.py:156-450`

**核心分发流程：**

```python
def build(self, ..., num_accepted_tokens, num_decode_draft_tokens_cpu):
    # Phase 1: 分类
    spec_sequence_masks = (num_decode_draft_tokens_cpu >= 0)  # bool mask
    # 值 >= 0 的是 spec decode 请求，值为 -1 的是普通请求

    # Phase 2: 分离
    if 无 spec:
        non_spec_state_indices_tensor = block_table[:, 0]   # 1D: 每个 req 的 running block
        non_spec_query_start_loc = query_start_loc          # 正常 cu_seqlens
    else:
        # Spec 请求
        spec_state_indices_tensor = block_table[spec_masks, :num_spec+1]  # 2D! [N_spec, num_spec+1]
        spec_query_start_loc = filtered_query_start_loc                    # [N_spec+1]

        # Non-spec 请求（可能为空）
        non_spec_state_indices_tensor = block_table[~spec_masks, 0]       # 1D 或 None
        non_spec_query_start_loc = filtered_query_start_loc               # 或 None

    # Phase 3: 互斥规则
    if num_decodes > 0 and num_spec_decodes > 0:
        # Non-spec decodes 重新归类为 prefills
        num_prefills += num_decodes
        num_decodes = 0
        # (prefill kernel 能处理 1-token 序列)
```

**三种场景：**

| 场景 | spec_indices | non_spec_indices | 含义 |
|------|-------------|-----------------|------|
| 纯普通 decode | None | 1D `[N]` | 每请求一个 block |
| 纯 spec decode | 2D `[N_spec, num_spec+1]` | None | 只有 spec tokens |
| 混合 batch | 2D `[N_spec, num_spec+1]` | 1D `[N_non_spec]` | 两者都有 |

### 2.3 Block Table 的 2D 结构

`spec_state_indices_tensor` 是 2D tensor，形状 `[num_spec_decodes, num_spec + 1]`：

```
spec_state_indices_tensor[i, j]:
  i = spec decode 请求索引
  j = token 位置索引
    j=0: running state block (accepted token)
    j=1: speculative slot 1
    j=2: speculative slot 2
    ...
    j=num_spec: speculative slot num_spec
```

**验证回滚机制：** 当 spec tokens 被部分接受时：

- `num_accepted_tokens[i] = k` 表示接受了 k 个 spec tokens
- Triton kernel 从 `spec_state_indices_tensor[i, k-1]` 读取初始 state
- 这就是 "回滚" 的实现：从最后一个被接受的 token 的 state 重新开始
- 被拒绝的 spec tokens 的 state slots 在下一步会被覆盖重用

---

## 3. KDA 状态管理详解

### 3.1 四种 State 的物理布局

> 文件: `vllm/model_executor/layers/mamba/mamba_utils.py:236-267`

KDA 有 **4 个独立的 state tensor**（对比 Mamba2 只有 2 个）：

| State | 形状 (每 block, 单 TP rank) | dtype | 说明 |
|-------|------------------------------|-------|------|
| `conv_state_q` | `(local_proj, conv_kernel-1)` | 模型精度 | Q 的 1D causal conv 滑动窗口 |
| `conv_state_k` | `(local_proj, conv_kernel-1)` | 模型精度 | K 的 1D causal conv 滑动窗口 |
| `conv_state_v` | `(local_proj, conv_kernel-1)` | 模型精度 | V 的 1D causal conv 滑动窗口 |
| `recurrent_state` | `(local_num_heads, head_dim, head_dim)` | **float32** | 线性注意力 hidden state |

**关键差异: Mamba2 的 conv_state 包含 `+ num_spec` 的宽度扩展，KDA 目前没有。**

```python
# Mamba2: conv_dim 包含 spec slots
conv_state_shape = (conv_dim / tp, conv_kernel - 1 + num_spec)

# KDA: 不使用 num_spec
conv_state_shape = (proj_size / tp, conv_kernel - 1)  # 没有 + num_spec!
```

完整 state pool 形状: `(num_blocks, *state_shape)`

### 3.2 Decode 路径: fused_recurrent_kda

> 文件: `vllm/model_executor/layers/fla/ops/kda.py:32-145`
>
> 底层 kernel: `vllm/model_executor/layers/fla/ops/fused_recurrent.py:27-176`

#### Kernel 参数

```python
fused_recurrent_gated_delta_rule_fwd_kernel[
    grid = (NK, NV, N * HV)
](
    q, k, v, g, beta, o, h0, ht, cu_seqlens,
    ssm_state_indices,        # 1D 或 2D state index tensor
    num_accepted_tokens,      # [N] 每个 seq 接受的 token 数
    ...
    IS_CONTINUOUS_BATCHING,   # ssm_state_indices is not None
    IS_SPEC_DECODING,         # num_accepted_tokens is not None
    IS_KDA,                   # per-element gating
)
```

#### 初始 State 读取 (Spec Decode 关键)

```triton
# fused_recurrent.py:103-120
if USE_INITIAL_STATE:
    if IS_CONTINUOUS_BATCHING:
        if IS_SPEC_DECODING:
            # 关键: 从 num_accepted_tokens 确定读哪一列
            i_t = load(num_accepted_tokens + i_n) - 1
        else:
            i_t = 0  # 非 spec: 读第 0 列 (running state)

        # 从 2D index tensor 读取对应的 block index
        state_idx = load(ssm_state_indices + i_n * stride_seq + i_t * stride_tok)

        # 从 state pool 加载
        p_h0 = h0 + state_idx * stride_init_state_token
        b_h = load(p_h0, ...)   # [BV, BK]
```

#### 逐 Token 状态写回

```triton
# fused_recurrent.py:152-165
for i_t in range(T):
    # ... 递推计算 ...
    h = exp(g) * h + beta * (v - h @ k) ⊗ k
    o = h @ q

    # 写回 state: 每个 token 都保存中间状态
    if INPLACE_FINAL_STATE:
        final_state_idx = load(ssm_state_indices + i_n * stride_seq + i_t * stride_tok)
        if final_state_idx > 0:  # NULL_BLOCK_ID 检查
            store(ht + final_state_idx * stride + ..., b_h)
```

**这意味着 kernel 已经支持 spec decode 的全部语义：**
1. 从 `num_accepted_tokens` 列读取初始 state（回滚到上一步的接受位置）
2. 逐 token 将中间 state 写入 2D index tensor 指定的 slot
3. 如果后续 token 被拒绝，下次直接从对应的 slot 读取即可

#### KDA 层当前调用方式

> 文件: `vllm/model_executor/layers/kda.py:445-457`

```python
# 当前代码 (仅 non-spec decode):
fused_recurrent_kda(
    q=q, k=k, v=v, g=g1, beta=beta,
    initial_state=recurrent_state,     # 整个 state pool (in-place)
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=non_spec_query_start_loc[:num_decodes + 1],
    ssm_state_indices=non_spec_state_indices_tensor,  # 1D only!
    # num_accepted_tokens 未传递! (默认 None)
)
```

**问题:**
- 只使用 `non_spec_state_indices_tensor`（1D）
- 当全是 spec tokens 时，`non_spec_state_indices_tensor = None`，直接 early return
- 没有传递 `num_accepted_tokens`
- 没有处理 `spec_state_indices_tensor`（2D）

### 3.3 Prefill 路径: chunk_kda

> 文件: `vllm/model_executor/layers/fla/ops/kda.py:1169-1260`

Prefill 路径不使用 `ssm_state_indices`，而是 Python 侧 gather/scatter：

```python
# kda.py:420-439 (prefill 路径)
# 1. 初始化没有 prefix state 的序列
zero_idx = non_spec_state_indices_tensor[~has_initial_state]
recurrent_state[zero_idx] = 0

# 2. Gather: 从 state pool 中提取每个 seq 的 state
initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()

# 3. 计算
core_attn_out, last_state = chunk_kda(q, k, v, g, beta,
                                       initial_state=initial_state, ...)

# 4. Scatter: 写回 state pool
recurrent_state[non_spec_state_indices_tensor] = last_state
```

**Spec decode 的特殊情况：** 当 spec tokens 存在时，GDN metadata builder 可能将 non-spec decodes 重新归类为 prefills（1-token prefill）。此时 `num_prefills > 0` 走 chunk_kda 路径，使用 `non_spec_query_start_loc` 和 `non_spec_state_indices_tensor`。

### 3.4 当前 KDA 层的问题

> 文件: `vllm/model_executor/layers/kda.py:343-344`

```python
if non_spec_state_indices_tensor is None:
    return  # early return — core_attn_out 保持全零
```

**问题链：**

```
全是 spec tokens
  → non_spec_state_indices_tensor = None
  → early return (跳过所有 conv1d + recurrent 处理)
  → core_attn_out 全零
  → 但 recurrent_state 和 conv_state 没有被更新!
  → 下一步（验证后）从 stale state 开始
  → 输出 garbage
```

---

## 4. Mamba2 参考实现

Mamba2 是 vLLM 中唯一完整支持 spec decode 的 SSM 模型，是 KDA 的主要参考。

### 4.1 状态索引传递

> 文件: `vllm/model_executor/layers/mamba/mamba_mixer2.py:603, 838-856`

```python
# Mamba2 decode 路径:
state_indices_tensor_d = attn_metadata.state_indices_tensor_d  # 2D [N, num_spec+1]

# 按 cache mode 分离输入/输出索引
if is_mamba_cache_all:
    state_indices_tensor_d_input = gather(1, block_idx_last_computed)
    state_indices_tensor_d_output = gather(1, block_idx_last_scheduled)
else:
    state_indices_tensor_d_input = state_indices_tensor_d       # 读写同一组 indices
    state_indices_tensor_d_output = state_indices_tensor_d

selective_state_update(
    ...,
    state_batch_indices=state_indices_tensor_d_input,    # 2D: 读哪些 block
    dst_state_batch_indices=state_indices_tensor_d_output, # 2D: 写哪些 block
    num_accepted_tokens=num_accepted_tokens,              # [N] 每个 seq 接受数
    cu_seqlens=query_start_loc_d,
)
```

### 4.2 Conv State 的 num_spec 扩展

> 文件: `vllm/model_executor/layers/mamba/mamba_utils.py:161-187`

```python
# Mamba2: conv_state 预留 spec slots
conv_state_shape = (conv_dim / tp, conv_kernel - 1 + num_spec)
#                                               ^^^^^^^^^^^^^^^^
# 为每个 spec token 预留额外的 ring-buffer 空间
```

**作用：** 在 propose 循环中，每一步 draft token 都需要更新 conv_state。预留的空间让每个 spec token 有自己的 conv_state slot，避免相互覆盖。

### 4.3 selective_state_update 的 Spec 处理

> 文件: `vllm/model_executor/layers/mamba/ops/mamba_ssm.py:63-317`

```python
# 初始 state 读取 (lines 155-176)
if HAS_STATE_BATCH_INDICES:
    if IS_SPEC_DECODING:
        # 从 num_accepted_tokens 确定读取列
        num_accepted = load(num_accepted_tokens_ptr + pid_b)
        init_token_idx = max(num_accepted - 1, 0)
    else:
        init_token_idx = 0

    # 2D index: [batch, token_position]
    state_batch_idx = load(state_batch_indices_ptr + pid_b * stride_batch
                          + init_token_idx * stride_T)
    state_ptr += state_batch_idx * stride_state_batch

# 逐 token 写回 (lines 257-270)
if IS_SPEC_DECODING:
    # 每个 token 都写入对应的 slot
    dst_idx = load(dst_state_batch_indices_ptr + i_t * stride_T)
    if dst_idx != null_block_id:
        store(state_ptr_base + dst_idx * stride_state_batch, state)
```

---

## 5. 关键差异: KDA vs Mamba2

| 方面 | Mamba2 | KDA (当前) | KDA (需要) |
|------|--------|-----------|-----------|
| **State 种类** | 2 (conv + ssm) | 4 (conv_q + conv_k + conv_v + recurrent) | 4 |
| **Decode kernel** | `selective_state_update` | `fused_recurrent_kda` | 同 |
| **2D state_indices** | 已支持 | kernel 支持, layer 未传递 | layer 需传递 |
| **num_accepted_tokens** | 已传递 | 未传递 | 需传递 |
| **Conv state spec 扩展** | `+ num_spec` | 无 | 需要评估 |
| **Spec token 处理** | 完整 | early return (全零) | 需完整实现 |
| **causal_conv1d_update** | 支持 batch indices | 仅 1D indices | 需要 2D 或 flatten |
| **Layer metadata** | 使用 `state_indices_tensor_d` | 仅 `non_spec_*` | 需处理 `spec_*` |

### 需要修改的文件清单

| 文件 | 修改内容 |
|------|---------|
| `vllm/model_executor/layers/kda.py` | `_forward` 方法: 处理 spec tokens, 传递 2D indices 和 num_accepted_tokens |
| `vllm/model_executor/layers/mamba/mamba_utils.py` | `kda_state_shape`: 评估是否需要 `+ num_spec` 扩展 conv_state |
| `vllm/model_executor/layers/fla/ops/kda.py` | `fused_recurrent_kda`: 透传 `num_accepted_tokens` 参数 |
| `vllm/model_executor/models/glm5_next_mtp.py` | `get_mamba_state_shape_from_config`: 确保 num_spec 正确传递 |
