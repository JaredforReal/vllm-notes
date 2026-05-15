# Kimi Linear 在 vLLM 中的实现分析

> 代码基准路径: `/Users/jared/vllm-project/vllm/`
>
> 涉及核心模块: Modeling → FLA Triton Kernels → KV Cache/State 管理 → Attention Backend

---

## 目录

- [1. 架构总览](#1-架构总览)
- [2. 配置与模型注册](#2-配置与模型注册)
- [3. Modeling 层详解](#3-modeling-层详解)
  - [3.1 DecoderLayer 分发](#31-decoderlayer-分发)
  - [3.2 KimiMLAAttention（标准 MLA 层）](#32-kimimlaattention标准-mla-层)
  - [3.3 KimiDeltaAttention（KDA 线性注意力层）](#33-kimideltaattentionkda-线性注意力层)
- [4. FLA Triton Kernel 详解](#4-fla-triton-kernel-详解)
  - [4.1 Prefill 路径: chunk_kda](#41-prefill-路径-chunk_kda)
  - [4.2 Decode 路径: fused_recurrent_kda](#42-decode-路径-fused_recurrent_kda)
  - [4.3 辅助 Kernel](#43-辅助-kernel)
- [5. State / KV Cache 管理](#5-state--kv-cache-管理)
  - [5.1 四种 State 的定义](#51-四种-state-的定义)
  - [5.2 State 的数据类型](#52-state-的数据类型)
  - [5.3 State 的分页管理](#53-state-的分页管理)
  - [5.4 State Copy（Prefix Caching）](#54-state-copyprefix-caching)
- [6. Attention Backend 分发](#6-attention-backend-分发)
  - [6.1 GDNAttentionBackend（KDA 层）](#61-gdnattentionbackendkda-层)
  - [6.2 MLA Backends（MLA 层）](#62-mla-backendsmla-层)
  - [6.3 Hybrid 模型的 Backend 选择机制](#63-hybrid-模型的-backend-选择机制)
- [7. 推理全流程（End-to-End）](#7-推理全流程end-to-end)
  - [7.1 Prefill 阶段](#71-prefill-阶段)
  - [7.2 Decode 阶段](#72-decode-阶段)
  - [7.3 Speculative Decoding 支持](#73-speculative-decoding-支持)
- [8. 文件导航索引](#8-文件导航索引)

---

## 1. 架构总览

Kimi Linear 是 Moonshot AI 的**混合线性注意力模型**。其核心设计是在同一个 Transformer 中交替使用两种注意力机制：

| 层类型 | 注意力机制 | KV Cache | 计算复杂度 |
|--------|-----------|----------|-----------|
| **KDA 层** | Kimi Delta Attention (Gated DeltaNet) | 固定大小 recurrent state | O(1) per token |
| **MLA 层** | Multi-head Latent Attention (DeepSeekV2 风格) | 压缩 KV cache | O(n) per token |

同时 MLP 部分支持 MoE（Mixture of Experts）。

**关键设计思想：**

- KDA 层用**线性注意力**替代 softmax attention，将 KV cache 转化为固定大小的 hidden state（类似 SSM），内存开销与序列长度无关
- MLA 层通过**低秩压缩**大幅减少 KV cache 占用
- 两者结合使得模型在超长序列上的推理效率极高

---

## 2. 配置与模型注册

### 模型配置

> 文件: `vllm/transformers_utils/configs/kimi_linear.py`

```python
class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"

    def __init__(self, ..., linear_attn_config: dict | None = None, **kwargs):
        # linear_attn_config 决定了每层的类型:
        #   kda_layers: [1, 3, 5, ...]      -> 线性注意力层
        #   full_attn_layers: [0, 2, 4, ...] -> 标准 MLA 层
        #   num_heads, head_dim, short_conv_kernel_size -> KDA 超参
        self.linear_attn_config = linear_attn_config

    def is_kda_layer(self, layer_idx: int):
        return (self.linear_attn_config is not None
                and (layer_idx + 1) in self.linear_attn_config["kda_layers"])

    @property
    def is_mla(self):
        return (self.q_lora_rank is not None
                or self.kv_lora_rank is not None
                or self.mla_use_nope)

    @property
    def is_linear_attn(self) -> bool:
        return not (
            self.linear_attn_config is None
            or len(self.linear_attn_config["kda_layers"]) == 0
        )
```

### 模型注册

> 文件: `vllm/model_executor/models/kimi_linear.py`

KimiLinearForCausalLM 实现了四个关键接口:

```python
class KimiLinearForCausalLM(nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid):
    # HasInnerState   -> 告知 vLLM 该模型有内部状态 (SSM state)
    # SupportsPP      -> 支持流水线并行
    # MixtureOfExperts -> 包含 MoE 层
    # IsHybrid        -> 包含多种注意力类型
```

`IsHybrid` 接口使得 vLLM 的 scheduler 会为不同类型的层分配不同的 attention backend。

---

## 3. Modeling 层详解

### 3.1 DecoderLayer 分发

> 文件: `vllm/model_executor/models/kimi_linear.py:285-379`

`KimiDecoderLayer.__init__` 根据配置选择注意力层:

```python
class KimiDecoderLayer(nn.Module):
    def __init__(self, config: KimiLinearConfig, layer_idx: int, ...):
        # 关键分发逻辑
        if config.is_kda_layer(layer_idx):
            self.self_attn = KimiDeltaAttention(   # 线性注意力
                layer_idx=layer_idx,
                hidden_size=config.hidden_size,
                model_config=config,              # 传入 config 以读取 linear_attn_config
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = KimiMLAAttention(     # 标准 MLA
                config=config,
                qk_nope_head_dim=config.qk_nope_head_dim,
                qk_rope_head_dim=config.qk_rope_head_dim,
                kv_lora_rank=config.kv_lora_rank,
                use_nope=config.mla_use_nope,      # True: Q 端无 LoRA
            )

        # MLP / MoE 分发
        if (self.is_moe and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = KimiMoE(config=config)      # MoE 层
        else:
            self.mlp = KimiMLP(hidden_size=...)     # Dense MLP
```

Decoder 层的 forward 采用 pre-norm + residual 连接:

```python
def forward(self, positions, hidden_states, residual):
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    attn_output = torch.empty_like(hidden_states)
    self.self_attn(hidden_states=hidden_states, positions=positions, output=attn_output)

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

注意 `attn_output` 使用 **预分配的输出 tensor** (`torch.empty_like`)，输出通过 `output[:] = ...` 写入，这是 vLLM 中 `torch.compile` 友好的模式。

### 3.2 KimiMLAAttention（标准 MLA 层）

> 文件: `vllm/model_executor/models/kimi_linear.py:177-283`
>
> 参考: DeepSeekV2 论文, `vllm/model_executor/layers/mla.py`

MLA 的核心思想是**将 KV 压缩到低秩空间**，减少 KV cache 存储。Kimi Linear 的 MLA 有一个特殊点: `use_nope=True` 且 `q_lora_rank=None`，即 Q 端不做 LoRA 压缩。

#### 结构

```
                    hidden_states
                         │
            ┌────────────┼────────────┐
            │                         │
     kv_a_proj_with_mqa           q_proj
   (hidden_size →                 (hidden_size →
    kv_lora_rank +                   num_heads ×
    qk_rope_head_dim)                qk_head_dim)
            │                         │
     ┌──────┴──────┐                  │
     │             │                  │
  kv_a_norm    (split rope)           │
     │             │                  │
  kv_b_proj   + RoPE                 │
  (kv_lora_rank →                     │
   num_heads ×                        │
   (qk_nope + v_head_dim))            │
     │                                │
     └──────────┬─────────────────────┘
                │
         MLA Attention
         (MLA Backends)
                │
            o_proj
```

#### 代码关键路径

```python
class KimiMLAAttention(nn.Module):
    def __init__(self, ...):
        # 注意: use_nope=True, q_lora_rank=None
        assert self.use_nope is True
        assert self.q_lora_rank is None

        # KV 端: 压缩投影
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,  # 压缩维度
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)

        # KV 端: 从低秩恢复到多头空间
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        )

        # Q 端: 直接投影（无 LoRA）
        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads * self.qk_head_dim,
        )

        # 所有子模块打包到 MLAModules，交给 PluggableLayer
        mla_modules = MLAModules(
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa,
            q_proj=self.q_proj,
            rotary_emb=None,  # MLA 使用自己的 RoPE 逻辑
            indexer=None,
            is_sparse=False,
            ...
        )
        self.mla_attn = MultiHeadLatentAttentionWrapper(..., mla_modules)

    def forward(self, positions, hidden_states, output):
        output[:] = self.mla_attn(positions, hidden_states)
```

**KV Cache 效果**: 只存储 `kv_lora_rank + qk_rope_head_dim` 维度的向量，而非完整的 `num_heads × head_dim`。

> MultiHeadLatentAttentionWrapper 的实现在 `vllm/model_executor/layers/mla.py:33-100`，是一个 `PluggableLayer`，允许不同硬件后端注册自定义实现。

### 3.3 KimiDeltaAttention（KDA 线性注意力层）

> 文件: `vllm/model_executor/layers/kda.py:85-455`

KDA 是 Kimi Linear 的核心创新，基于 **Gated DeltaNet** 的线性注意力变体。与传统 softmax attention 不同，KDA 维护一个**固定大小的 recurrent hidden state**，使推理的内存和计算开销与序列长度无关。

#### 3.3.1 子模块

```python
class KimiDeltaAttention(nn.Module, MambaBase):
    def __init__(self, layer_idx, hidden_size, model_config, ...):
        kda_config = model_config.linear_attn_config
        self.head_dim = kda_config["head_dim"]          # e.g. 128
        self.num_heads = kda_config["num_heads"]         # e.g. 32
        self.conv_size = kda_config["short_conv_kernel_size"]  # e.g. 4

        # ── 标准投影 ──
        projection_size = self.head_dim * self.num_heads
        self.q_proj = ColumnParallelLinear(hidden_size, projection_size)
        self.k_proj = ColumnParallelLinear(hidden_size, projection_size)
        self.v_proj = ColumnParallelLinear(hidden_size, projection_size)

        # ── Gating g1: 衰减门（控制遗忘率）──
        self.f_a_proj = ReplicatedLinear(hidden_size, self.head_dim)
        self.f_b_proj = ColumnParallelLinear(self.head_dim, projection_size)
        self.A_log = nn.Parameter(...)  # 可学习的衰减参数，per-head
        self.dt_bias = nn.Parameter(...)  # 偏置

        # ── Gating g2: 输出门 ──
        self.g_a_proj = ReplicatedLinear(hidden_size, self.head_dim)
        self.g_b_proj = ColumnParallelLinear(self.head_dim, projection_size)

        # ── Beta: 遗忘门 ──
        self.b_proj = ColumnParallelLinear(hidden_size, self.num_heads)

        # ── 短卷积: 提供局部上下文 ──
        self.q_conv1d = ColumnParallelLinear(self.conv_size, projection_size)
        self.k_conv1d = ColumnParallelLinear(self.conv_size, projection_size)
        self.v_conv1d = ColumnParallelLinear(self.conv_size, projection_size)

        # ── 输出 ──
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(projection_size, hidden_size)
```

#### 3.3.2 Forward 流程

> 文件: `vllm/model_executor/layers/kda.py:251-287`（forward）和 `289-455`（_forward）

`forward` 负责计算所有投影，然后通过 `torch.ops.vllm.kda_attention` 调用 `_forward`（注册为 custom op 以支持 `torch.compile`）。

```python
def forward(self, hidden_states, positions, output):
    num_tokens = hidden_states.size(0)

    # 1. Q/K/V 投影
    q = self.q_proj(hidden_states)[0]
    k = self.k_proj(hidden_states)[0]
    v = self.v_proj(hidden_states)[0]

    # 2. Beta (遗忘门): sigmoid 激活
    beta = self.b_proj(hidden_states)[0].float().sigmoid()  # [num_tokens, num_heads]

    # 3. Gating g1 (衰减门): 两级投影 + fused_kda_gate
    #    g1 = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)
    g1 = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
    g1 = fused_kda_gate(g1, self.A_log, self.head_dim, g_bias=self.dt_bias)
    # g1 shape: [num_tokens, num_heads, head_dim]

    # 4. Gating g2 (输出门)
    g2 = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
    g2 = rearrange(g2, "... (h d) -> ... h d", d=self.head_dim)

    # 5. 核心线性注意力计算（通过 custom op 进入 _forward）
    core_attn_out = torch.zeros((1, num_tokens, local_num_heads, head_dim), ...)
    torch.ops.vllm.kda_attention(q, k, v, g1, beta, core_attn_out, self.prefix)

    # 6. 输出门控 + RMSNorm + 输出投影
    core_attn_out = self.o_norm(core_attn_out, g2)  # RMSNorm(x) * sigmoid(g2)
    core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
    output[:] = self.o_proj(core_attn_out)[0]
```

#### 3.3.3 _forward 内部逻辑

> 文件: `vllm/model_executor/layers/kda.py:289-455`

`_forward` 是实际执行逻辑，分为 **prefill** 和 **decode** 两条路径:

```python
def _forward(self, q_proj_states, k_proj_states, v_proj_states, g1, beta, core_attn_out):
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata[self.prefix]
    assert isinstance(attn_metadata, GDNAttentionMetadata)

    # 获取 4 种 state cache
    (conv_state_q, conv_state_k, conv_state_v, recurrent_state) = self.kv_cache

    # ── 短卷积处理 ──
    if attn_metadata.num_prefills > 0:
        # PREFILL: 使用完整的 causal_conv1d_fn（批量处理整个序列）
        q = causal_conv1d_fn(q_proj_states, q_conv_weights, ...,
                             conv_states=conv_state_q,
                             cache_indices=non_spec_state_indices_tensor)
        k = causal_conv1d_fn(...)
        v = causal_conv1d_fn(...)
    else:
        # DECODE: 使用增量 causal_conv1d_update（每次处理 1 token）
        q = causal_conv1d_update(q_proj_states, conv_state_q, q_conv_weights, ...)
        k = causal_conv1d_update(...)
        v = causal_conv1d_update(...)

    # reshape: [num_tokens, num_heads*head_dim] -> [1, num_tokens, num_heads, head_dim]
    q, k, v = map(lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim), (q, k, v))

    # ── 核心线性注意力 ──
    if attn_metadata.num_prefills > 0:
        # PREFILL 路径: chunk-wise parallel
        initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()
        core_attn_out, last_state = chunk_kda(
            q=q, k=k, v=v, g=g1, beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=non_spec_query_start_loc,
        )
        # 更新 state cache
        recurrent_state[non_spec_state_indices_tensor] = last_state
    else:
        # DECODE 路径: fused recurrent (直接读写 state cache)
        core_attn_out, last_state = fused_recurrent_kda(
            q=q, k=k, v=v, g=g1, beta=beta,
            initial_state=recurrent_state,      # 直接传入 state cache
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=non_spec_query_start_loc,
            ssm_state_indices=non_spec_state_indices_tensor,  # block table index
        )

    core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[0, :num_actual_tokens]
```

---

## 4. FLA Triton Kernel 详解

> 文件目录: `vllm/model_executor/layers/fla/ops/`
>
> 基于 [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) 项目 (MIT License, Songlin Yang & Yu Zhang)

所有 kernel 都用 **Triton** 编写，分 prefill 和 decode 两条路径。

### 4.1 Prefill 路径: `chunk_kda`

> 文件: `vllm/model_executor/layers/fla/ops/kda.py:1169-1260`
>
> Chunk size: `FLA_CHUNK_SIZE = 64`（定义在 `vllm/model_executor/layers/fla/ops/utils.py:31`）

Prefill 使用 **chunk-wise parallel** 算法。将长序列按 chunk_size=64 切分，在 chunk 间做 sequential scan，在 chunk 内做 parallel matmul。

#### 算法流程（6 步）

```python
def chunk_kda(q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens):
    # Step 1: 对 gating g 做 chunk 内 cumsum
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
    # -> g[t] 现在是 chunk 内的 cumsum，用于指数衰减

    # Step 2: 计算 chunk 内 β·K·K^T 矩阵 (inter + intra)
    A, Aqk = chunk_kda_scaled_dot_kkt_fwd(
        q=q, k=k, gk=g, beta=beta, scale=scale, cu_seqlens=cu_seqlens
    )
    # A:    β·K·K^T (用于 delta rule 的 WY 表示)
    # Aqk:  Q·K^T  (用于 intra-chunk attention)

    # Step 3: 对 A 做下三角求解
    A = solve_tril(A=A, cu_seqlens=cu_seqlens)

    # Step 4: 重计算 w 和 u
    # w = β·k - A·(β·k·exp(gk))  (修正后的 K)
    # u = β·v - A·(β·v)            (修正后的 V)
    w, u, _, kg = recompute_w_u_fwd(k=k, v=v, beta=beta, A=A, gk=g, cu_seqlens=cu_seqlens)

    # Step 5: 计算 chunk 间 hidden state 递推
    # h[t] = exp(g[t]) * h[t-1] + w[t] ⊗ u[t]
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # Step 6: 计算最终输出
    # o = Q·g · H (inter-chunk) + Aqk · v_new (intra-chunk)
    o = chunk_gla_fwd_o_gk(q=q, v=v_new, g=g, A=Aqk, h=h, scale=scale, cu_seqlens=cu_seqlens)

    return o, final_state
```

#### Step 1: `chunk_local_cumsum`

> 文件: `vllm/model_executor/layers/fla/ops/cumsum.py`

对 gating g 在每个 chunk 内做 cumulative sum。这个 cumsum 会被用作指数衰减因子: `exp(g[i] - g[j])` 实现衰减权重。

#### Step 2: `chunk_kda_scaled_dot_kkt_fwd`

> 文件: `vllm/model_executor/layers/fla/ops/kda.py:510-798`

两个 Triton kernel:
- `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter`: 处理不同 sub-chunk 之间的 K·K^T
- `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra`: 处理同一 sub-chunk 内部的 K·K^T

核心计算:
```python
# 带指数衰减的 K·K^T
b_k_gated = b_k * exp(b_g - b_gn)          # K 经过 gating 衰减
b_ktg = b_kt * exp(b_gn - b_gk)            # K^T 经过反向衰减
b_A = tl.dot(b_k_gated, b_ktg)             # 加权 K·K^T
b_A *= b_b[:, None]                         # 乘以 beta

# 同时计算 Q·K^T (用于 intra-chunk attention)
b_Aqk = tl.dot(b_qg, b_ktg)               # Q·K^T
```

Autotune 配置:
```python
BK_list = [32, 64]     # K 维度的 block size
num_warps = [1, 2, 4, 8]
num_stages = [2, 3, 4]
```

#### Step 3: `solve_tril`

> 文件: `vllm/model_executor/layers/fla/ops/solve_tril.py`

对 A 矩阵做下三角求解。这是 WY (Wang-Yan) 表示的一部分:
```
A_new = A * (I + tril(A))^(-1)
```

#### Step 4: `recompute_w_u_fwd`

> 文件: `vllm/model_executor/layers/fla/ops/kda.py:817-1004`

Delta rule 的核心思想: 不直接用 K、V，而是用修正后的 w、u:
```python
w = β·k·exp(gk) - A·(β·k·exp(gk))    # 修正后的 key
u = β·v - A·(β·v)                       # 修正后的 value
```

同时计算 `kg = k·exp(g_last - gk)`（归一化到 chunk 末尾的 key）。

#### Step 5: `chunk_gated_delta_rule_fwd_h`

> 文件: `vllm/model_executor/layers/fla/ops/chunk_delta_h.py`

计算 chunk 间 hidden state 的递推关系:
```
h[0] = initial_state
h[t+1] = exp(g_chunk_end) * h[t] + sum(w[t_i] ⊗ u[t_i])  for t_i in chunk_t
```

输出 `h` 为每个 chunk 起始位置的 hidden state（用于 inter-chunk attention），`final_state` 为序列末尾的 state（存入 cache）。

#### Step 6: `chunk_gla_fwd_o_gk`

> 文件: `vllm/model_executor/layers/fla/ops/kda.py:1019-1166`

最终输出计算:
```python
# Inter-chunk contribution: Q_gated × H^T
b_o = tl.dot(b_qg, tl.trans(b_h))          # [BT, BV]

# Intra-chunk contribution: Aqk × V_new
b_o += tl.dot(b_A, b_v)                     # [BT, BV]
```

### 4.2 Decode 路径: `fused_recurrent_kda`

> 文件: `vllm/model_executor/layers/fla/ops/kda.py:32-145`
>
> Kernel: `vllm/model_executor/layers/fla/ops/fused_recurrent.py:27-176`

Decode 使用**单个 fused Triton kernel**，每个序列的每个 head 独立处理（grid: `NK × NV × N*HV`）。

#### 核心递推公式

```python
@triton.jit
def fused_recurrent_gated_delta_rule_fwd_kernel(...):
    # 加载初始 state
    b_h = tl.zeros([BV, BK], dtype=tl.float32)  # hidden state: [V_dim, K_dim]
    if USE_INITIAL_STATE:
        # 从 paged state cache 加载
        state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq)
        p_h0 = h0 + state_idx * stride_init_state_token + ...
        b_h += tl.load(p_h0, ...)

    # 逐 token 递推
    for i_t in range(T):
        b_q = tl.load(p_q, ...)    # [BK]
        b_k = tl.load(p_k, ...)    # [BK]
        b_v = tl.load(p_v, ...)    # [BV]

        # L2 normalization
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        # Gated decay: h = exp(g) * h
        if IS_KDA:
            b_gk = tl.load(p_gk, ...)       # [BK] - per-dimension gating
            b_h *= exp(b_gk[None, :])        # decay per key dimension
        else:
            b_g = tl.load(p_g)              # scalar gating
            b_h *= exp(b_g)

        # Delta rule: v_new = beta * (v - h @ k)
        b_v -= tl.sum(b_h * b_k[None, :], 1)    # [BV]  v - h·k
        b_v *= b_beta                             # * beta

        # State update: h += v_new ⊗ k
        b_h += b_v[:, None] * b_k[None, :]       # [BV, BK]

        # Output: o = h @ q
        b_o = tl.sum(b_h * b_q[None, :], 1)      # [BV]
        tl.store(p_o, b_o)

        # 写回 state cache (inplace)
        if INPLACE_FINAL_STATE:
            final_state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t)
            if final_state_idx > 0:
                p_ht = ht + final_state_idx * stride_final_state_token + ...
                tl.store(p_ht, b_h)
```

#### 数学公式总结

KDA 的 decode 核心递推:

```
h_t = exp(g_t) ⊙ h_{t-1} + β_t ⊙ (v_t - h_{t-1} @ k_t^T) ⊗ k_t
o_t = h_t @ q_t^T
```

其中:
- `h_t ∈ R^{head_dim × head_dim}` 是 recurrent hidden state
- `g_t` 是 decay gate（`-exp(A_log) * softplus(...)`, 值为负，实现遗忘）
- `β_t` 是 beta gate（sigmoid 输出，控制新信息权重）
- `v_t - h_{t-1} @ k_t^T` 是 delta rule 的修正项

#### 调用入口

```python
def fused_recurrent_kda(q, k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices, ...):
    BK = next_power_of_2(K)       # K 维度的 block size
    BV = min(next_power_of_2(V), 8)  # V 维度的 block size（最大 8）

    grid = (NK, NV, N * HV)       # N=seq_count, HV=num_heads
    fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q, k=k, v=v, g=g, beta=beta,
        h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,  # block table index
        IS_KDA=True,                         # KDA 模式: per-dimension gating
        INPLACE_FINAL_STATE=True,             # 直接更新 state cache
        USE_QK_L2NORM_IN_KERNEL=True,         # QK 做 L2 归一化
    )
```

### 4.3 辅助 Kernel

#### `fused_kda_gate`

> 文件: `vllm/model_executor/layers/fla/ops/kda.py:1272-1379`

计算衰减门 g1:
```
y = -exp(A_log) * softplus(g + dt_bias, beta=1.0)
```

```python
@triton.jit
def kda_gate_fwd_kernel(g, A, y, g_bias, beta, threshold, T, H, D, ...):
    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = -tl.exp(b_a)                        # 负指数 → 衰减因子

    b_g = tl.load(g_ptr, ...)                 # 投影结果
    if HAS_BIAS:
        b_g = b_g + b_bias[None, :]

    # softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
    # 大值时用线性近似
    g_scaled = b_g * beta
    use_linear = g_scaled > threshold
    sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + exp(g_scaled)))
    b_y = b_a * sp
    tl.store(y_ptr, b_y)
```

#### `FusedRMSNormGated`

> 文件: `vllm/model_executor/layers/fla/ops/kda.py:436-507`

融合 RMSNorm + sigmoid gating，用于输出:
```
output = RMSNorm(x) * sigmoid(g)
```

两个 Triton kernel 变体:
- `layer_norm_gated_fwd_kernel`: 用于 head_dim ≤ 512，处理多行
- `layer_norm_gated_fwd_kernel1`: 逐行处理

---

## 5. State / KV Cache 管理

KDA 层**不使用标准 PagedAttention 的 KV cache**，而是使用类似 Mamba/SSM 的 **state cache**，有 4 个独立的 state tensor。

### 5.1 四种 State 的定义

> 文件: `vllm/model_executor/layers/mamba/mamba_utils.py:237-267`

由 `MambaStateShapeCalculator.kda_state_shape()` 定义:

```python
@classmethod
def kda_state_shape(cls, tp_world_size, num_heads, head_dim,
                    conv_kernel_size=4, num_spec=0):
    proj_size = num_heads * head_dim
    local_proj = divide(proj_size, tp_world_size)

    return (
        # State 0: conv_state_q  - Q 的 1D 卷积滑动窗口
        (local_proj, conv_kernel_size - 1),

        # State 1: conv_state_k  - K 的 1D 卷积滑动窗口
        (divide(num_heads * head_dim, tp_world_size), conv_kernel_size - 1),

        # State 2: conv_state_v  - V 的 1D 卷积滑动窗口
        (divide(num_heads * head_dim, tp_world_size), conv_kernel_size - 1),

        # State 3: recurrent_state - 线性注意力的 hidden state
        (divide(num_heads, tp_world_size), head_dim, head_dim),
    )
```

| State | 形状 (每 block, 单 TP rank) | 说明 |
|-------|------------------------------|------|
| `conv_state_q` | `(local_proj_size, conv_size-1)` | Q 的 1D causal conv 滑动窗口 |
| `conv_state_k` | `(local_proj_size, conv_size-1)` | K 的 1D causal conv 滑动窗口 |
| `conv_state_v` | `(local_proj_size, conv_size-1)` | V 的 1D causal conv 滑动窗口 |
| `recurrent_state` | `(local_num_heads, head_dim, head_dim)` | 线性注意力的 hidden state **(float32)** |

其中 `local_* = */tp_size`。Conv state 的布局可以是 `SD` 或 `DS`，由环境变量 `VLLM_SSM_CONV_STATE_LAYOUT` 控制。

### 5.2 State 的数据类型

> 文件: `vllm/model_executor/layers/mamba/mamba_utils.py:119-125`

```python
@classmethod
def kda_state_dtype(cls, model_dtype, mamba_cache_dtype):
    state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
    return (state_dtype, state_dtype, state_dtype, torch.float32)
    #     conv_q     conv_k     conv_v     recurrent (强制 float32)
```

前三个 conv state 使用模型精度（bf16/fp16），**recurrent state 强制使用 float32** 以保证数值稳定性（因为 hidden state 涉及大量累加运算）。

### 5.3 State 的分页管理

KDA 层通过 `GDNAttentionBackend` 注册为 SSM 类型:

```python
class GDNAttentionBackend(AttentionBackend):
    @classmethod
    def is_ssm(cls) -> bool:
        return True
```

这意味着 KDA 层的 KV cache spec 是 `MambaSpec`（而非 `AttentionSpec`），使用 **block-table based 的分页管理**。每个请求分配一组 blocks，每个 block 存储一组 state tensors。

State 的分配和生命周期管理:
1. **分配**: 请求到达时，从 pool 中分配一组 blocks
2. **Prefill**: `chunk_kda` 计算完成后将 `final_state` 写入最后一个 block
3. **Decode**: `fused_recurrent_kda` 通过 `ssm_state_indices` 直接读写 block 中的 state
4. **释放**: 请求完成后回收 blocks

### 5.4 State Copy（Prefix Caching）

> 文件: `vllm/model_executor/layers/mamba/mamba_utils.py:299-373`

对于 prefix caching（复用已计算的 state），KDA 注册了 4 个 copy 函数:

```python
@classmethod
def kda_state_copy_func(cls):
    return (
        get_conv_copy_spec,    # conv_state_q
        get_conv_copy_spec,    # conv_state_k
        get_conv_copy_spec,    # conv_state_v
        get_temporal_copy_spec, # recurrent_state
    )
```

- `get_conv_copy_spec`: 复制卷积滑动窗口的最后 `num_accepted_tokens-1` 个位置
- `get_temporal_copy_spec`: 直接复制整个 recurrent state

---

## 6. Attention Backend 分发

### 6.1 GDNAttentionBackend（KDA 层）

> 文件: `vllm/v1/attention/backends/gdn_attn.py`

```python
class GDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GDN_ATTN"

    @classmethod
    def is_ssm(cls) -> bool:
        return True  # 标记为 SSM 类型，使用 MambaSpec
```

#### Metadata: GDNAttentionMetadata

```python
@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int          # speculative decoding 的 decode 数
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None   # [batch] bool: 是否有 prefix state

    # 非 speculative 请求的索引
    non_spec_query_start_loc: torch.Tensor   # [batch+1] cumsum of query lengths
    non_spec_state_indices_tensor: torch.Tensor  # [batch] block table index

    # Speculative 请求的索引
    spec_query_start_loc: torch.Tensor | None
    spec_state_indices_tensor: torch.Tensor | None  # [batch, num_spec+1]

    # 预计算的 FLA chunk metadata (避免 GPU→CPU sync)
    chunk_indices: torch.Tensor | None
    chunk_offsets: torch.Tensor | None

    # causal_conv1d Triton kernel 所需的元数据
    nums_dict: dict | None
    batch_ptr: torch.Tensor | None
    token_chunk_offset_ptr: torch.Tensor | None
```

#### MetadataBuilder: GDNAttentionMetadataBuilder

> 文件: `vllm/v1/attention/backends/gdn_attn.py:76-475`

这是一个复杂的 builder，核心职责:

1. **分离 prefill/decode/spec-decode 三种模式**: 通过 `split_decodes_and_prefills` 和 `spec_sequence_masks` 将请求分为三组
2. **预分配 GPU tensor**: 为 CUDA Graph 捕获预分配固定大小的 tensor:
   ```python
   self.spec_state_indices_tensor = torch.empty((max_bs, num_spec+1), dtype=torch.int32, device=device)
   self.non_spec_state_indices_tensor = torch.empty((max_bs,), dtype=torch.int32, device=device)
   self.spec_sequence_masks = torch.empty((max_bs,), dtype=torch.bool, device=device)
   # ... 更多预分配
   ```
3. **预计算 FLA chunk metadata**: 在 CPU 上计算 `chunk_indices` 和 `chunk_offsets`，然后异步拷贝到 GPU
4. **CUDA Graph 支持**: 当 batch 是 decode-only 时，使用预分配的 padded tensor 支持 CUDA Graph 捕获

### 6.2 MLA Backends（MLA 层）

MLA 层使用 vLLM 标准的 MLA backend 体系，位于 `vllm/v1/attention/backends/mla/`:

| Backend | 文件 | 适用场景 |
|---------|------|---------|
| FlashMLA | `flashmla.py` | NVIDIA GPU，使用 FlashMLA C extension |
| FlashInfer MLA | `flashinfer_mla.py` | SM100+ GPU |
| CUTLASS MLA | `cutlass_mla.py` | Blackwell (SM100) GPU |
| Triton MLA | `triton_mla.py` | 通用 fallback |
| ROCm AITER MLA | `rocm_aiter_mla.py` | AMD GPU |
| FlashAttention MLA | `flashattn_mla.py` | FlashAttention + MLA |

这些 MLA backend 管理的是压缩的 KV cache（`MLAAttentionSpec`），与 KDA 的 `MambaSpec` 完全独立。

### 6.3 Hybrid 模型的 Backend 选择机制

KimiLinearForCausalLM 实现了 `IsHybrid` 接口。vLLM 的 scheduler 为不同类型的层分配不同的 attention backend:

```
┌──────────────────────────────────────────────────┐
│                KimiLinearForCausalLM              │
│              (IsHybrid, HasInnerState)            │
├──────────────────────────────────────────────────┤
│                                                    │
│  Layer 0 (MLA):  GDN_ATTN or MLA backend?         │
│                  → 实际走 MLA backends              │
│                  → KV cache: MLAAttentionSpec      │
│                                                    │
│  Layer 1 (KDA):  GDN_ATTN backend                  │
│                  → KV cache: MambaSpec (4 states)  │
│                                                    │
│  Layer 2 (MLA):  MLA backends                      │
│                  → KV cache: MLAAttentionSpec      │
│                                                    │
│  Layer 3 (KDA):  GDN_ATTN backend                  │
│                  → KV cache: MambaSpec (4 states)  │
│                  ...                               │
└──────────────────────────────────────────────────┘
```

每种 backend 有自己的 metadata builder，metadata 通过 dict 按 layer name 索引:

```python
# 在 _forward 中:
attn_metadata_raw = forward_context.attn_metadata
assert isinstance(attn_metadata_raw, dict)
attn_metadata = attn_metadata_raw[self.prefix]  # 按层名索引
```

---

## 7. 推理全流程（End-to-End）

### 7.1 Prefill 阶段

```
Input tokens [t0, t1, t2, ..., tN]
       │
       ▼
  embed_tokens
       │
       ▼
┌─ DecoderLayer (KDA) ──────────────────────────────┐
│  input_layernorm                                    │
│       │                                             │
│  q_proj, k_proj, v_proj                            │
│  b_proj → sigmoid → beta                           │
│  f_a → f_b → fused_kda_gate → g1                  │
│  g_a → g_b → g2                                    │
│       │                                             │
│  causal_conv1d_fn (prefill mode)                    │
│    ├─ 读取 conv_state_q/k/v (如果 has_initial_state)│
│    └─ 写入 conv_state_q/k/v (更新滑动窗口)         │
│       │                                             │
│  q, k = L2_norm(q), L2_norm(k)                    │
│       │                                             │
│  chunk_kda (6步 Triton pipeline):                  │
│    1. chunk_local_cumsum(g)                         │
│    2. chunk_kda_scaled_dot_kkt → A, Aqk            │
│    3. solve_tril(A)                                │
│    4. recompute_w_u → w, u, kg                     │
│    5. chunk_gated_delta_rule_fwd_h → h, final_state│
│    6. chunk_gla_fwd_o_gk → output                  │
│       │                                             │
│  写入 recurrent_state[block_idx] = final_state     │
│       │                                             │
│  o_norm(core_attn_out, g2) = RMSNorm * sigmoid(g2) │
│  o_proj → output                                    │
└─────────────────────────────────────────────────────┘
       │
       ▼
┌─ DecoderLayer (MLA) ──────────────────────────────┐
│  input_layernorm                                    │
│       │                                             │
│  kv_a_proj_with_mqa → kv_a_norm → kv_b_proj        │
│  q_proj                                             │
│       │                                             │
│  MLA Backend (FlashMLA/FlashInfer/etc.)             │
│    ├─ Prefill: Flash Attention on compressed KV     │
│    └─ 写入 KV cache (compressed format)            │
│       │                                             │
│  o_proj → output                                    │
└─────────────────────────────────────────────────────┘
       │
       ▼ (重复所有层)
       │
  RMSNorm → lm_head → logits
```

### 7.2 Decode 阶段

```
Input token [t_new]  (single token)
       │
       ▼
┌─ DecoderLayer (KDA) ──────────────────────────────┐
│  q_proj, k_proj, v_proj → 单个 token               │
│  b_proj → beta                                      │
│  f_a → f_b → fused_kda_gate → g1                   │
│  g_a → g_b → g2                                     │
│       │                                              │
│  causal_conv1d_update (incremental mode):           │
│    ├─ 读取 conv_state_q/k/v[block_idx]              │
│    └─ 更新 conv_state_q/k/v[block_idx] += new_token │
│       │                                              │
│  fused_recurrent_kda (单 kernel):                   │
│    h = exp(g) * h_prev + beta * (v - h_prev@k) ⊗ k │
│    o = h @ q                                         │
│    ├─ 读取 recurrent_state[block_idx]                │
│    └─ 写回 recurrent_state[block_idx] = h            │
│       │                                              │
│  o_norm → o_proj → output                            │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌─ DecoderLayer (MLA) ───────────────────────────────┐
│  kv_a_proj → kv_b_proj → 写入 KV cache              │
│  q_proj                                             │
│       │                                              │
│  MLA Backend Decode (MQA on compressed KV)          │
│       │                                              │
│  o_proj → output                                     │
└──────────────────────────────────────────────────────┘
```

### 7.3 Speculative Decoding 支持

> 文件: `vllm/v1/attention/backends/gdn_attn.py:93-314`

GDNAttentionMetadataBuilder 对 speculative decoding 有完整的支持:

1. **识别 spec 请求**: 通过 `num_decode_draft_tokens_cpu` 判断哪些请求在做 speculative decoding
2. **分离 token 索引**: 将 spec tokens 和 non-spec tokens 分到不同的索引组
3. **多条 spec token**: spec 请求有 `num_spec+1` 个 tokens (draft tokens + verify token)，每条 token 有独立的 state index
4. **混合 batch 处理**: 当 batch 中同时有 prefill、decode 和 spec-decode 时:
   - non-spec decodes 被重新归类为 prefills（因为 prefill kernel 也能处理 1-token 序列）
   - 确保 `num_decodes > 0` 和 `num_spec_decodes > 0` 不同时出现

```python
# 当 spec decodes 存在时，将 non-spec decodes 重新归类为 prefills
if num_decodes > 0 and num_spec_decodes > 0:
    num_prefills += num_decodes
    num_prefill_tokens += num_decode_tokens
    num_decodes = 0
    num_decode_tokens = 0
```

---

## 8. 文件导航索引

### 核心模型文件

| 文件路径 | 说明 |
|---------|------|
| `vllm/transformers_utils/configs/kimi_linear.py` | KimiLinearConfig 配置类 |
| `vllm/model_executor/models/kimi_linear.py` | 模型主体: KimiLinearForCausalLM, KimiDecoderLayer, KimiMLAAttention, KimiMoE |
| `vllm/model_executor/layers/kda.py` | KimiDeltaAttention (KDA) 注意力层实现 |
| `vllm/model_executor/layers/mla.py` | MLAModules, MultiHeadLatentAttentionWrapper (PluggableLayer) |
| `vllm/model_executor/layers/mamba/mamba_utils.py` | MambaStateShapeCalculator, MambaStateDtypeCalculator, State Copy |

### FLA Triton Kernel

| 文件路径 | 说明 |
|---------|------|
| `vllm/model_executor/layers/fla/ops/kda.py` | `chunk_kda`, `fused_recurrent_kda`, `fused_kda_gate`, `FusedRMSNormGated` |
| `vllm/model_executor/layers/fla/ops/fused_recurrent.py` | `fused_recurrent_gated_delta_rule_fwd_kernel` (decode 核心递推 kernel) |
| `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` | `chunk_gated_delta_rule_fwd_kernel_h` (prefill chunk 间 state 递推) |
| `vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py` | chunk 内 K·K^T 计算 |
| `vllm/model_executor/layers/fla/ops/solve_tril.py` | 下三角求解 (WY 表示) |
| `vllm/model_executor/layers/fla/ops/chunk_o.py` | chunk 输出计算 |
| `vllm/model_executor/layers/fla/ops/cumsum.py` | chunk 内 cumulative sum |
| `vllm/model_executor/layers/fla/ops/l2norm.py` | L2 归一化 |
| `vllm/model_executor/layers/fla/ops/index.py` | `prepare_chunk_indices`, `prepare_chunk_offsets` |
| `vllm/model_executor/layers/fla/ops/utils.py` | `FLA_CHUNK_SIZE=64`, 平台检测, autotune 工具 |
| `vllm/model_executor/layers/fla/ops/layernorm_guard.py` | `RMSNormGated`, `layernorm_fn` |

### Attention Backend

| 文件路径 | 说明 |
|---------|------|
| `vllm/v1/attention/backends/gdn_attn.py` | GDNAttentionBackend + MetadataBuilder (KDA 层) |
| `vllm/v1/attention/backends/linear_attn.py` | LinearAttentionBackend (其他线性注意力模型) |
| `vllm/v1/attention/backends/mla/` | MLA backend 目录 (多种实现) |

### KV Cache Interface

| 文件路径 | 说明 |
|---------|------|
| `vllm/v1/kv_cache_interface.py` | `KVCacheSpec`, `MambaSpec`, `MLAAttentionSpec` 定义 |
| `vllm/v1/core/kv_cache_utils.py` | KV cache 工具函数 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | 单类型 KV cache 管理器 |

### 测试文件

| 文件路径 | 说明 |
|---------|------|
| `tests/v1/attention/test_sparse_mla_backends.py` | Sparse MLA backend 测试 |
| `tests/v1/attention/test_mla_backends.py` | MLA backend 测试 |
| `tests/kernels/attention/test_lightning_attn.py` | Lightning attention kernel 测试 |

### Benchmark

| 文件路径 | 说明 |
|---------|------|
| `benchmarks/attention_benchmarks/mla_runner.py` | MLA benchmark runner |
| `benchmarks/attention_benchmarks/configs/mla_*.yaml` | MLA benchmark 配置文件 |
