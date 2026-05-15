# Qwen3-Next (Qwen3.5) 在 vLLM 中的实现分析

> 代码基准路径: `/Users/jared/vllm-project/vllm/`
>
> 涉及核心模块: Modeling → GDN Linear Attention → FLA Triton/FlashInfer Kernels → State 管理 → Attention Backend

---

## 目录

- [1. 架构总览](#1-架构总览)
- [2. 模型族谱与配置](#2-模型族谱与配置)
  - [2.1 Qwen3-Next vs Qwen3.5 差异](#21-qwen3-next-vs-qwen35-差异)
  - [2.2 配置文件](#22-配置文件)
  - [2.3 模型注册](#23-模型注册)
- [3. Modeling 层详解](#3-modeling-层详解)
  - [3.1 DecoderLayer 分发](#31-decoderlayer-分发)
  - [3.2 Qwen3NextAttention（标准 Full Attention 层）](#32-qwen3nextattention标准-full-attention-层)
  - [3.3 GatedDeltaNetAttention（GDN 线性注意力层）](#33-gateddeltanetattentiongdn-线性注意力层)
- [4. FLA / FlashInfer Kernel 详解](#4-fla--flashinfer-kernel-详解)
  - [4.1 Prefill 路径: chunk_gated_delta_rule](#41-prefill-路径-chunk_gated_delta_rule)
  - [4.2 Decode 路径: fused_sigmoid_gating / packed_decode](#42-decode-路径-fused_sigmoid_gating--packed_decode)
  - [4.3 辅助 Kernel](#43-辅助-kernel)
- [5. State 管理](#5-state-管理)
  - [5.1 两种 State 的定义](#51-两种-state-的定义)
  - [5.2 State 的数据类型](#52-state-的数据类型)
  - [5.3 State 的分页管理与 Copy](#53-state-的分页管理与-copy)
- [6. Attention Backend 分发](#6-attention-backend-分发)
- [7. 推理全流程（End-to-End）](#7-推理全流程end-to-end)
  - [7.1 Prefill 阶段](#71-prefill-阶段)
  - [7.2 Decode 阶段](#72-decode-阶段)
  - [7.3 Speculative Decoding 与 MTP](#73-speculative-decoding-与-mtp)
- [8. 文件导航索引](#8-文件导航索引)

---

## 1. 架构总览

Qwen3-Next（又名 Qwen3.5）是阿里 Qwen 团队的**混合线性注意力模型**。在同一个 Transformer 中交替使用两种注意力机制：

| 层类型 | 注意力机制 | State / Cache | 计算复杂度 |
|--------|-----------|---------------|-----------|
| **full_attention** | 标准 GQA Attention (Q/K Norm + Output Gate) | 标准 Paged KV Cache | O(n) per token |
| **linear_attention** | Gated DeltaNet (GDN) Linear Attention | 固定大小 SSM state (conv + recurrent) | O(1) per token |

同时 MLP 部分支持 MoE（如 Qwen3-Next-80B-A3B 有 512 个 expert，top-10 路由）。

**关键设计思想:**

- GDN 层使用 **Gated Delta Rule** 线性注意力，将 KV cache 转化为固定大小的 recurrent state（类似 SSM），内存开销与序列长度无关
- Full Attention 层使用 **GQA + QK-Norm + Output Gate**，保证关键层的精确注意力
- 与 Kimi Linear 不同: Qwen3-Next/Qwen3.5 **没有 MLA**（Multi-head Latent Attention），标准 full attention 层使用传统 GQA
- GDN 层的 K/V head 维度可以不同（`linear_key_head_dim ≠ linear_value_head_dim`），支持非对称设计

**典型模型配置:**

| 模型 | Layers | Full Attn | Linear Attn | MoE |
|------|--------|-----------|-------------|-----|
| Qwen3-Next-80B-A3B | 48 | 每 4 层 1 个 (12 个) | 每 4 层 3 个 (36 个) | 512 experts, top-10 |
| Qwen3.5-0.8B | 小模型 | 按配置 | 按配置 | Dense MLP |
| Qwen3.5-397B-A17B | 大模型 | 按配置 | 按配置 | MoE |

---

## 2. 模型族谱与配置

### 2.1 Qwen3-Next vs Qwen3.5 差异

Qwen3-Next 和 Qwen3.5 共享核心的 GDN 线性注意力实现 (`GatedDeltaNetAttention`)，但有以下差异:

| 特性 | Qwen3-Next (`qwen3_next`) | Qwen3.5 (`qwen3_5` / `qwen3_5_moe`) |
|------|--------------------------|--------------------------------------|
| GDN QKV 权重布局 | Interleaved GQA (`gqa_interleaved_layout=True`) | 分离 Q/K/V/Z (`gqa_interleaved_layout=False`) |
| GDN 投影方式 | 单个 `in_proj_qkvz` + `in_proj_ba` (融合) | 支持 LoRA: `in_proj_qkv` + `in_proj_z` 分离 |
| 视觉支持 | 纯文本 | 支持 Vision (继承 Qwen3VL 架构) |
| Spec Decoding | MTP (Multi-Token Predictor) | MTP |
| LoRA | 不支持 | 支持 (`SupportsLoRA`) |
| 权重加载 | `q_proj/k_proj/v_proj` 拆分加载 | `in_proj_qkv` + `in_proj_z` 融合加载 |

### 2.2 配置文件

> 文件: `vllm/transformers_utils/configs/qwen3_next.py`

```python
class Qwen3NextConfig(PretrainedConfig):
    model_type = "qwen3_next"

    def __init__(self, ..., layer_types=None, **kwargs):
        # layer_types 决定每层的类型: "full_attention" 或 "linear_attention"
        self.layer_types = layer_types
        if self.layer_types is None:
            # 默认: 每 4 层中, 第 4 层是 full_attention, 前 3 层是 linear_attention
            self.layer_types = [
                "linear_attention" if bool((i + 1) % 4) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        # Full attention 参数
        self.head_dim = head_dim                # e.g. 256
        self.num_attention_heads = 16
        self.num_key_value_heads = 2            # GQA

        # Linear attention (GDN) 参数
        self.linear_conv_kernel_dim = 4         # 短卷积核大小
        self.linear_key_head_dim = 128          # K head 维度
        self.linear_value_head_dim = 128        # V head 维度
        self.linear_num_key_heads = 16          # K head 数量
        self.linear_num_value_heads = 32        # V head 数量 (> K heads, 非对称)

        # MoE 参数
        self.num_experts = 512
        self.num_experts_per_tok = 10
```

Qwen3.5 配置位于:
- `vllm/transformers_utils/configs/qwen3_5.py` — `Qwen3_5TextConfig`
- `vllm/transformers_utils/configs/qwen3_5_moe.py` — `Qwen3_5MoeTextConfig`

### 2.3 模型注册

> 文件: `vllm/model_executor/models/registry.py`

```python
# Qwen3-Next 系列
"Qwen3NextForCausalLM"  -> ("qwen3_next", "Qwen3NextForCausalLM")
"Qwen3NextMTP"          -> ("qwen3_next_mtp", "Qwen3NextMTP")

# Qwen3.5 系列
"Qwen3_5ForConditionalGeneration"   -> ("qwen3_5", "Qwen3_5ForConditionalGeneration")
"Qwen3_5MoeForConditionalGeneration" -> ("qwen3_5", "Qwen3_5MoeForConditionalGeneration")
"Qwen3_5MTP"            -> ("qwen3_5_mtp", "Qwen3_5MTP")
"Qwen3_5MoeMTP"         -> ("qwen3_5_mtp", "Qwen3_5MTP")
```

---

## 3. Modeling 层详解

### 3.1 DecoderLayer 分发

> Qwen3-Next: `vllm/model_executor/models/qwen3_next.py:314-450`
>
> Qwen3.5: `vllm/model_executor/models/qwen3_5.py:118-195`

两者都使用相同的 DecoderLayer 结构（Qwen3.5 继承 Qwen3-Next 的 DecoderLayer），核心分发逻辑:

```python
class Qwen3NextDecoderLayer(nn.Module):
    def __init__(self, vllm_config, layer_type, prefix=""):
        config = vllm_config.model_config.hf_config

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(
                config, vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,  # Qwen3-Next: interleaved GQA layout
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config, model_config=model_config,
                cache_config=cache_config, quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )

        # MoE / Dense MLP 分发
        if config.num_experts > 0 and (self.layer_idx + 1) % config.decoder_sparse_step == 0:
            self.mlp = Qwen3NextSparseMoeBlock(vllm_config=vllm_config)
        else:
            self.mlp = Qwen3NextMLP(hidden_size=...)
```

Qwen3.5 的 DecoderLayer 覆写了 GDN 的创建方式:

```python
class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(self, vllm_config, layer_type, prefix=""):
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(
                config=config, vllm_config=vllm_config,
                gqa_interleaved_layout=False,                  # Qwen3.5: 分离布局
                create_in_proj_qkvz=vllm_config.lora_config is None,  # LoRA 时拆分
            )
```

Decoder 层 forward（两种注意力共享）:

```python
def forward(self, hidden_states, residual, positions):
    # Self Attention
    hidden_states, residual = self.input_layernorm(hidden_states, residual)
    self_attention_output = torch.empty_like(hidden_states)

    if self.layer_type == "linear_attention":
        self.linear_attn(hidden_states=hidden_states, output=self_attention_output)
    else:
        self.self_attn(hidden_states=hidden_states, output=self_attention_output, positions=positions)

    # Optional layer_scale
    if self.layer_scale:
        hidden_states = self_attention_output * (self.attn_layer_scale + 1)
    else:
        hidden_states = self_attention_output

    # MLP
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

### 3.2 Qwen3NextAttention（标准 Full Attention 层）

> 文件: `vllm/model_executor/models/qwen3_next.py:197-311`

Full Attention 层使用标准 GQA，增加了 **Q/K-Norm** 和 **Output Gate**:

```
                   hidden_states
                        │
               qkv_proj (fused Q+gate, K, V)
                        │
           ┌────────────┼──────────┐
           │            │          │
     q + gate        k          v
     (2×q_size)   (kv_size)  (kv_size)
           │            │
     q_norm          k_norm
     (RMSNorm)      (RMSNorm)
           │            │
           └─── RoPE ───┘
                  │
         Standard GQA Attention
         (FlashAttention/FlashInfer/etc.)
                  │
            output * sigmoid(gate)
                  │
              o_proj → output
```

关键代码:

```python
class Qwen3NextAttention(nn.Module):
    def __init__(self, config, ...):
        # QKV 投影: Q 端包含 gate (2x 大小)
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),  # gate 占额外空间
            self.total_num_kv_heads,
        )
        self.q_norm = Qwen3NextRMSNorm(self.head_dim)  # Q-Norm
        self.k_norm = Qwen3NextRMSNorm(self.head_dim)  # K-Norm
        self.rotary_emb = get_rope(...)
        self.attn = Attention(...)  # 标准 vLLM Attention

    def forward(self, positions, output, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)

        # 拆分 Q+gate, K, V
        if self.attn_output_gate:
            q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)  # 拆分 Q 和 gate
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # QK-Norm (RMSNorm)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(-1, self.q_size)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(-1, self.kv_size)

        # RoPE
        q, k = self.rotary_emb(positions, q, k)

        # 标准 GQA Attention
        attn_output = self.attn(q, k, v)

        # Output gate
        if self.attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate.reshape(...))

        output[:], _ = self.o_proj(attn_output)
```

### 3.3 GatedDeltaNetAttention（GDN 线性注意力层）

> 文件: `vllm/model_executor/layers/mamba/gdn_linear_attn.py:214-1120`

GDN 是 Qwen3-Next/Qwen3.5 的核心线性注意力实现。它是一个 `PluggableLayer`，同时实现了 `MambaBase` 接口（SSM 状态管理）。

#### 3.3.1 子模块

```python
@PluggableLayer.register("gated_delta_net_attention")
class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    def __init__(self, config, vllm_config, prefix="",
                 gqa_interleaved_layout=False, create_in_proj_qkvz=True):
        # ── 维度配置 ──
        self.num_k_heads = config.linear_num_key_heads       # e.g. 16
        self.num_v_heads = config.linear_num_value_heads     # e.g. 32
        self.head_k_dim = config.linear_key_head_dim         # e.g. 128
        self.head_v_dim = config.linear_value_head_dim       # e.g. 128
        self.conv_kernel_size = config.linear_conv_kernel_dim  # e.g. 4

        # ── 短卷积 (1D causal conv) ──
        self.conv_dim = self.key_dim * 2 + self.value_dim    # Q, K 共享 key_dim
        self.conv1d = ColumnParallelLinear(self.conv_kernel_size, self.conv_dim)

        # ── 输入投影 (QKVZ + BA) ──
        # Qwen3-Next: 融合的 in_proj_qkvz + in_proj_ba (interleaved GQA)
        # Qwen3.5: 分离的 in_proj_qkv + in_proj_z + in_proj_ba (支持 LoRA)
        if create_in_proj_qkvz:
            self.in_proj_qkvz = MergedColumnParallelLinear(...)  # Q, K, V, Z
            self.in_proj_ba = MergedColumnParallelLinear(...)    # B, A (gating)
        else:
            self.in_proj_qkv = MergedColumnParallelLinear(...)   # Q, K, V (LoRA)
            self.in_proj_z = ColumnParallelLinear(...)           # Z (LoRA)
            self.in_proj_ba = MergedColumnParallelLinear(...)    # B, A

        # ── 可学习参数 ──
        self.A_log = nn.Parameter(...)    # 衰减参数, per V-head
        self.dt_bias = nn.Parameter(...)  # 偏置, per V-head

        # ── 输出 ──
        self.norm = RMSNormGated(self.head_v_dim, activation=output_gate_type)
        self.out_proj = RowParallelLinear(self.value_dim, self.hidden_size)
```

#### 3.3.2 Forward 流程

> 文件: `vllm/model_executor/layers/mamba/gdn_linear_attn.py:519-602`

`forward_cuda` 分为 3 个阶段:

```python
def forward_cuda(self, hidden_states, output):
    num_tokens = hidden_states.size(0)

    # ════════════════════════════════════════
    # Part 1: Input Projection
    # ════════════════════════════════════════
    if hasattr(self, "in_proj_qkv"):
        # Qwen3.5 LoRA 路径: 分离投影
        mixed_qkv, _ = self.in_proj_qkv(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        z, _ = self.in_proj_z(hidden_states)
        b, a = ba.chunk(2, dim=-1)
    else:
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        if self.gqa_interleaved_layout:
            # Qwen3-Next: 解包 interleaved GQA 布局
            query, key, value, z, b, a = self.fix_query_key_value_ordering(mixed_qkvz, ba)
            mixed_qkv = torch.cat((query, key, value), dim=-1)
        else:
            # Qwen3.5: 分离布局 [Q, K, V, Z] + [B, A]
            mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
            b, a = ba.chunk(2, dim=-1)

    # ════════════════════════════════════════
    # Part 2: Core Attention (Custom Op)
    # ════════════════════════════════════════
    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim), ...
    )
    torch.ops.vllm.gdn_attention_core(
        mixed_qkv, b, a, core_attn_out, _encode_layer_name(self.prefix)
    )

    # ════════════════════════════════════════
    # Part 3: Output Projection
    # ════════════════════════════════════════
    core_attn_out = self.norm(core_attn_out, z)    # RMSNormGated: RMSNorm(x) * gate(z)
    core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
    output[:num_tokens], _ = self.out_proj(core_attn_out)
```

#### 3.3.3 _forward_core 内部逻辑

> 文件: `vllm/model_executor/layers/mamba/gdn_linear_attn.py:809-1066`

`_forward_core` 是核心计算，分为 3 步:

**Step 1: 短卷积处理**

```python
# 获取 2 种 state cache
conv_state = self_kv_cache[0]   # (num_blocks, conv_dim, conv_kernel_size-1)
ssm_state = self_kv_cache[1]    # (num_blocks, num_v_heads, head_v_dim, head_k_dim)

if attn_metadata.num_prefills > 0:
    # PREFILL: 批量 causal_conv1d_fn
    mixed_qkv_non_spec = causal_conv1d_fn(
        mixed_qkv_non_spec_T, conv_weights, ...,
        conv_states=conv_state,
        cache_indices=non_spec_state_indices_tensor,
    )
elif attn_metadata.num_decodes > 0:
    # DECODE: 增量 causal_conv1d_update
    mixed_qkv_non_spec = causal_conv1d_update(
        mixed_qkv_non_spec, conv_state, conv_weights, ...,
        conv_state_indices=non_spec_state_indices_tensor,
    )
```

**Step 2: Recurrent Attention**

Prefill 和 Decode 使用不同的 kernel:

```python
if attn_metadata.num_prefills > 0:
    # PREFILL: 使用 fused_post_conv_prep 预处理 + chunk_gated_delta_rule
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=mixed_qkv, a=a, b=b,
        A_log=self.A_log, dt_bias=self.dt_bias,
        num_k_heads=..., head_k_dim=..., head_v_dim=...,
        apply_l2norm=True,
    )
    core_attn_out, final_state = self.chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=non_spec_query_start_loc,
    )
    ssm_state[non_spec_state_indices_tensor] = final_state

elif attn_metadata.num_decodes > 0:
    # DECODE: 使用 fused_sigmoid_gating_delta_rule_update (单 kernel)
    core_attn_out, final_state = fused_sigmoid_gating_delta_rule_update(
        A_log=self.A_log, a=a, b=b, dt_bias=self.dt_bias,
        q=query, k=key, v=value,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=non_spec_state_indices_tensor,
        use_qk_l2norm_in_kernel=True,
    )
```

**Step 3: Merge Output**

Spec 和 non-spec 的输出通过 `index_copy_` 合并:

```python
if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
    merged_out = torch.empty(...)
    merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
    merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
    core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
```

#### 3.3.4 Qwen3-Next Interleaved GQA Layout

> 文件: `vllm/model_executor/layers/mamba/gdn_linear_attn.py:449-498`

Qwen3-Next 的 GDN 使用 **interleaved GQA layout**: 当 `num_v_heads > num_k_heads` 时，V heads 按 K head groups 交错排列:

```python
def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
    # 按 K head group 拆分
    # 每个 K head group 包含:
    #   1 × head_k_dim (Q)
    #   1 × head_k_dim (K)
    #   (num_v_heads/num_k_heads) × head_v_dim (V)
    #   (num_v_heads/num_k_heads) × head_v_dim (Z)
    #   (num_v_heads/num_k_heads) (B)
    #   (num_v_heads/num_k_heads) (A)

    new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
        self.num_k_heads // self.tp_size,
        (self.head_k_dim + self.head_k_dim
         + (self.head_v_dim + self.head_v_dim) * self.num_v_heads // self.num_k_heads),
    )
    mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)

    (query, key, value, z) = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=2)
    (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

    # reshape V/Z/B/A 从 grouped → flat
    value = value.reshape(value.size(0), -1, self.head_v_dim)
    z = z.reshape(z.size(0), -1, self.head_v_dim)
    b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
    a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)
```

---

## 4. FLA / FlashInfer Kernel 详解

> FLA Triton kernels: `vllm/model_executor/layers/fla/ops/`
>
> 基于 [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) (MIT License)

### 4.1 Prefill 路径: `chunk_gated_delta_rule`

> 文件: `vllm/model_executor/layers/fla/ops/chunk.py`
>
> Chunk size: `FLA_CHUNK_SIZE = 64`

Prefill 有两个后端可选（通过 `ChunkGatedDeltaRule` `CustomOp` 分发）:

| 后端 | 文件 | 触发条件 |
|------|------|---------|
| **FlashInfer** | `gdn_linear_attn.py:70-116` (`fi_chunk_gated_delta_rule`) | SM90 (H100) 且 `--gdn-prefill-backend=flashinfer` |
| **Triton/FLA** | `fla/ops/chunk.py` (`fla_chunk_gated_delta_rule`) | 默认 fallback |

```python
class ChunkGatedDeltaRule(CustomOp):
    def __init__(self):
        supports_flashinfer = current_platform.is_device_capability(90)
        backend = additional_config.get("gdn_prefill_backend", "auto")
        if backend == "flashinfer":
            use_flashinfer = supports_flashinfer
        elif backend == "triton":
            use_flashinfer = False
        else:
            use_flashinfer = supports_flashinfer  # auto: 有 FlashInfer 就用
```

#### FlashInfer Prefill 路径

```python
def fi_chunk_gated_delta_rule(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, ...):
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as chunk_gated_delta_rule_fi

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)   # L2 归一化在 Python 端做
        k = l2norm_fwd(k)

    # FlashInfer 使用 exp(g) 而非原始 g
    fi_g = g.to(torch.float32)
    fi_beta = beta.to(torch.float32)
    result = chunk_gated_delta_rule_fi(
        q=q, k=k, v=v,
        g=torch.exp(fi_g),     # 传入 exp(g)
        beta=fi_beta,
        initial_state=initial_state.to(torch.float32),
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
```

#### Triton/FLA Prefill 路径

与 Kimi Linear 的 `chunk_kda` 类似的 6 步 pipeline:

1. **`chunk_local_cumsum`**: 对 g 做 chunk 内 cumsum
2. **`chunk_scaled_dot_kkt_fwd`**: 计算 `β·K·K^T` 和 `Q·K^T`
3. **`solve_tril`**: 下三角求解 (WY 表示)
4. **`recompute_w_u_fwd`**: 重计算修正后的 K 和 V
5. **`chunk_gated_delta_rule_fwd_h`**: 计算 chunk 间 state 递推
6. **`chunk_gla_fwd_o_gk`**: 计算最终输出

**与 KDA 的关键区别:**
- GDN 使用**标量 gating** (`g` per head per token)，而 KDA 使用 **per-dimension gating** (`g` per head per token per dim)
- GDN 的 `beta` 是标量 (per head per token)，KDA 的 `beta` 也是标量
- GDN 支持 **非对称 K/V head 维度** (`head_k_dim ≠ head_v_dim`)

### 4.2 Decode 路径: fused_sigmoid_gating / packed_decode

GDN 的 decode 有两条路径:

#### 路径 A: `fused_sigmoid_gating_delta_rule_update`

> 文件: `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`

用于 spec-decode 或非 packed decode:

```python
@triton.jit
def fused_sigmoid_gating_delta_rule_update_kernel(
    q, k, v, g, beta, o, h0, ht, ...
):
    # 核心递推 (与 KDA 类似但使用标量 gating):
    # h = exp(g) * h_prev
    # v_new = beta * (v - h @ k)
    # h += v_new ⊗ k
    # o = h @ q

    for i_t in range(T):
        b_q = tl.load(p_q, ...) * scale
        b_k = tl.load(p_k, ...)
        b_v = tl.load(p_v, ...)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / sqrt(sum(b_q^2) + eps)
            b_k = b_k / sqrt(sum(b_k^2) + eps)

        b_g = tl.load(p_g, ...)
        b_h *= exp(b_g)                         # 标量 decay (vs KDA 的 per-dim)

        b_v -= sum(b_h * b_k[None, :], 1)       # delta rule
        b_v *= b_beta                            # sigmoid gating
        b_h += b_v[:, None] * b_k[None, :]      # state update
        b_o = sum(b_h * b_q[None, :], 1)        # output

        tl.store(p_o, b_o)
        tl.store(p_ht, b_h)                     # inplace state update
```

#### 路径 B: `fused_recurrent_gated_delta_rule_packed_decode`

> 文件: `vllm/model_executor/layers/fla/ops/fused_recurrent.py`

当 `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1` 且无 spec-decode 时使用:

```python
def _forward_core_decode_non_spec(self, mixed_qkv, b, a, core_attn_out, attn_metadata):
    # Conv1d update
    mixed_qkv = causal_conv1d_update(mixed_qkv, conv_state, ...)

    # Packed decode: 单个 fused kernel 完成 split + rearrange + L2norm + recurrent
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv, a=a, b=b,
        A_log=self.A_log, dt_bias=self.dt_bias,
        scale=self.head_k_dim**-0.5,
        initial_state=ssm_state,
        out=out_buf,
        ssm_state_indices=non_spec_state_indices_tensor,
    )
```

这个 packed kernel 将以下操作融合为单步:
1. Split `mixed_qkv` → q, k, v
2. Rearrange to `[1, num_tokens, heads, dim]`
3. L2 normalization
4. Gating computation (`g = -exp(A) * softplus(a + bias)`, `beta = sigmoid(b)`)
5. Recurrent state update + output

### 4.3 辅助 Kernel

#### `fused_post_conv_prep`

> 文件: `vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py`

Prefill 路径的融合预处理 kernel。将 conv1d 输出后的以下操作融合为单个 Triton kernel:

```
mixed_qkv → split → rearrange → L2norm → gating → q, k, v, g, beta
```

```python
def fused_post_conv_prep(conv_output, a, b, A_log, dt_bias, num_k_heads,
                         head_k_dim, head_v_dim, apply_l2norm=True, output_g_exp=False):
    """
    Fused kernel that replaces the following chain:
      split(conv_output, [key_dim, key_dim, value_dim])
      → rearrange(q, "l (h d) -> l h d")
      → rearrange(k, "l (h d) -> l h d")
      → rearrange(v, "l (h d) -> l h d")
      → l2norm(q), l2norm(k)
      → g = -exp(A) * softplus(a + dt_bias)
      → beta = sigmoid(b)
    """
```

#### `fused_gdn_gating`

> 文件: `vllm/model_executor/layers/mamba/gdn_linear_attn.py:1165-1234`

Decode 路径的 gating 计算（在 `_forward_core` 外使用）:

```python
@triton.jit
def fused_gdn_gating_kernel(g, beta_output, A_log, a, b, dt_bias, ...):
    x = blk_a + blk_bias
    softplus_x = where(beta * x <= threshold, (1/beta) * log(1 + exp(beta * x)), x)
    g = -exp(A_log) * softplus_x       # decay gating
    beta_output = sigmoid(b)            # beta gating
```

---

## 5. State 管理

### 5.1 两种 State 的定义

> 文件: `vllm/model_executor/layers/mamba/mamba_utils.py:213-234`

GDN 层有 **2 种** state（比 Kimi KDA 的 4 种更简单）:

```python
@classmethod
def gated_delta_net_state_shape(cls, tp_world_size, num_k_heads, num_v_heads,
                                head_k_dim, head_v_dim, conv_kernel_size, num_spec=0):
    conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
    conv_state_shape = cls._orient_conv_shape(
        divide(conv_dim, tp_world_size),
        conv_kernel_size - 1 + num_spec,
    )
    temporal_state_shape = (
        divide(num_v_heads, tp_world_size),
        head_v_dim,
        head_k_dim,      # 注意: [V_dim, K_dim] 非对称
    )
    return conv_state_shape, temporal_state_shape
```

| State | 形状 (每 block, 单 TP rank) | 说明 |
|-------|------------------------------|------|
| `conv_state` | `(conv_dim/tp, conv_kernel-1+num_spec)` | Q/K/V 融合的 1D causal conv 滑动窗口 |
| `recurrent_state` | `(num_v_heads/tp, head_v_dim, head_k_dim)` | GDN hidden state，**V_dim × K_dim 非对称** |

其中 `conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads`，因为 Q 和 K 共享 `head_k_dim` 维度。

**与 Kimi KDA 的对比:**

| | Kimi KDA | Qwen GDN |
|---|---------|----------|
| State 数量 | 4 (conv_q, conv_k, conv_v, recurrent) | 2 (conv_fused, recurrent) |
| Conv state | Q/K/V 分开存储 | Q/K/V 融合存储 |
| Recurrent shape | `(num_heads, head_dim, head_dim)` 对称 | `(num_v_heads, head_v_dim, head_k_dim)` 非对称 |
| K/V heads | 相同 (`num_heads`) | 可以不同 (`num_k_heads ≠ num_v_heads`) |

### 5.2 State 的数据类型

```python
@classmethod
def gated_delta_net_state_dtype(cls, model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype):
    conv_state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
    if mamba_ssm_cache_dtype == "auto":
        temporal_state_dtype = conv_state_dtype
    else:
        temporal_state_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]
    return (conv_state_dtype, temporal_state_dtype)
```

与 Kimi KDA 不同: GDN 的 recurrent state 可以**不是 float32**，由 `mamba_ssm_cache_dtype` 配置决定（默认 `"auto"` 使用模型精度）。

Qwen3.5 的配置 (`vllm/model_executor/models/config.py:606-631`):

```python
class Qwen3_5ForConditionalGenerationConfig:
    def update(self, vllm_config):
        # 从 HF config 的 mamba_ssm_dtype 字段更新 mamba_ssm_cache_dtype
        if hasattr(self, "mamba_ssm_dtype"):
            vllm_config.cache_config.mamba_ssm_cache_dtype = self.mamba_ssm_dtype
```

### 5.3 State 的分页管理与 Copy

GDN 同样注册为 SSM 类型 (`is_ssm() -> True`)，使用 `MambaSpec` 和 block-table 分页管理。

State copy 函数:

```python
@classmethod
def gated_delta_net_state_copy_func(cls):
    return (get_conv_copy_spec, get_temporal_copy_spec)
```

- `get_conv_copy_spec`: 复制 conv sliding window 的最后 `num_accepted_tokens-1` 个位置
- `get_temporal_copy_spec`: 直接复制整个 recurrent state

**限制**: Qwen3-Next/Qwen3.5 目前不支持 `'all'` prefix caching mode:

```python
if cache_config.mamba_cache_mode == "all":
    raise NotImplementedError(
        "Qwen3.5 currently does not support 'all' prefix caching, "
        "please use '--mamba-cache-mode=align' instead"
    )
```

---

## 6. Attention Backend 分发

Qwen3-Next/Qwen3.5 作为 `IsHybrid` 模型，拥有两种 attention 类型:

```
┌──────────────────────────────────────────────────────────┐
│             Qwen3NextForCausalLM (IsHybrid)              │
│           或 Qwen3_5ForConditionalGeneration             │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Full Attention 层:                                      │
│    → 标准 vLLM Attention backends                        │
│      (FlashAttention, FlashInfer, etc.)                  │
│    → KV cache: FullAttentionSpec (Paged KV Cache)        │
│    → Metadata: 标准 AttentionMetadataBuilder             │
│                                                          │
│  Linear Attention 层 (GDN):                              │
│    → GDNAttentionBackend ("GDN_ATTN")                    │
│    → State: MambaSpec (conv + recurrent)                 │
│    → Metadata: GDNAttentionMetadata                      │
│    → Builder: GDNAttentionMetadataBuilder                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

> `GDNAttentionBackend` 和 `GDNAttentionMetadataBuilder` 的完整分析见 KimiLinear.md 第 6.1 节，两者完全共享同一套 backend。

**核心区别**: Kimi Linear 有 MLA 层需要 MLA backends，而 Qwen3-Next/Qwen3.5 的 full attention 层使用**标准 GQA**，走常规的 FlashAttention/FlashInfer backends。

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
┌─ DecoderLayer (linear_attention / GDN) ──────────────────┐
│  input_layernorm                                          │
│       │                                                    │
│  in_proj_qkvz → Q, K, V, Z                               │
│  in_proj_ba → B, A                                        │
│       │                                                    │
│  causal_conv1d_fn (prefill mode)                          │
│    ├─ 读取 conv_state[block_idx] (如果 has_initial_state) │
│    └─ 写入 conv_state[block_idx] (更新滑动窗口)           │
│       │                                                    │
│  fused_post_conv_prep (Triton 融合 kernel):               │
│    ├─ split → rearrange → L2norm                          │
│    ├─ g = -exp(A) * softplus(a + dt_bias)                 │
│    └─ beta = sigmoid(b)                                   │
│       │                                                    │
│  chunk_gated_delta_rule (FlashInfer 或 Triton):           │
│    ├─ chunk_local_cumsum(g)                                │
│    ├─ chunk_scaled_dot_kkt → A, Aqk                       │
│    ├─ solve_tril(A)                                       │
│    ├─ recompute_w_u → w, u, kg                            │
│    ├─ chunk_gated_delta_rule_fwd_h → h, final_state       │
│    └─ chunk_gla_fwd_o_gk → output                         │
│       │                                                    │
│  写入 ssm_state[block_idx] = final_state                  │
│       │                                                    │
│  RMSNormGated(core_attn_out, z) = RMSNorm(x) * gate(z)   │
│  out_proj → output                                         │
└────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ DecoderLayer (full_attention) ───────────────────────────┐
│  input_layernorm                                          │
│       │                                                    │
│  qkv_proj → Q+gate, K, V                                  │
│  q_norm(Q), k_norm(K)                                     │
│  RoPE(positions, Q, K)                                    │
│       │                                                    │
│  FlashAttention / FlashInfer (标准 GQA)                   │
│    ├─ 写入 KV cache (PagedAttention)                      │
│    └─ attn_output * sigmoid(gate)                         │
│       │                                                    │
│  o_proj → output                                           │
└────────────────────────────────────────────────────────────┘
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
┌─ DecoderLayer (linear_attention / GDN) ──────────────────┐
│  in_proj_qkvz → Q, K, V, Z                               │
│  in_proj_ba → B, A                                        │
│       │                                                    │
│  causal_conv1d_update (incremental)                       │
│    ├─ 读取 conv_state[block_idx]                           │
│    └─ 更新 conv_state[block_idx]                           │
│       │                                                    │
│  Packed decode 路径 (VLLM_ENABLE_FLA_PACKED_RECURRENT):   │
│    fused_recurrent_gated_delta_rule_packed_decode:         │
│      ├─ split qkv + L2norm + gating (fused)               │
│      └─ h = exp(g) * h_prev + beta*(v - h@k) ⊗ k         │
│         o = h @ q                                          │
│    或 Standard decode 路径:                                │
│    fused_sigmoid_gating_delta_rule_update:                 │
│      同上但分步执行                                        │
│       │                                                    │
│  RMSNormGated → out_proj → output                          │
└────────────────────────────────────────────────────────────┘
       │
       ▼
┌─ DecoderLayer (full_attention) ───────────────────────────┐
│  qkv_proj → Q+gate, K, V                                  │
│  q_norm, k_norm, RoPE                                      │
│       │                                                    │
│  FlashAttention decode (PagedAttention, read KV cache)     │
│       │                                                    │
│  o_proj → output                                           │
└────────────────────────────────────────────────────────────┘
```

### 7.3 Speculative Decoding 与 MTP

Qwen3-Next/Qwen3.5 支持 **MTP (Multi-Token Predictor)** 用于 speculative decoding:

> 文件: `vllm/model_executor/models/qwen3_next_mtp.py`, `qwen3_5_mtp.py`

MTP 模型是一个轻量级的辅助模型，预测接下来的多个 token。注册为:
- `qwen3_next_mtp` — Qwen3NextMTP
- `qwen3_5_mtp` — Qwen3_5MTP / Qwen3_5MoeMTP

GDN 层通过 `GDNAttentionMetadataBuilder` 完整支持 speculative decoding:
- Spec 和 non-spec tokens 分离索引
- Spec tokens 使用 `fused_sigmoid_gating_delta_rule_update`（多条 token 逐条处理）
- Non-spec decodes 在有 spec decodes 时被重分类为 prefills
- 支持 CUDA Graph 捕获

---

## 8. 文件导航索引

### 核心模型文件

| 文件路径 | 说明 |
|---------|------|
| `vllm/model_executor/models/qwen3_next.py` | Qwen3-Next 主模型: Attention, DecoderLayer, MoE, CausalLM |
| `vllm/model_executor/models/qwen3_next_mtp.py` | Qwen3-Next MTP (Multi-Token Predictor) |
| `vllm/model_executor/models/qwen3_5.py` | Qwen3.5 主模型: Dense + MoE + Vision-Language |
| `vllm/model_executor/models/qwen3_5_mtp.py` | Qwen3.5 MTP |
| `vllm/model_executor/models/colqwen3_5.py` | ColQwen3.5 (检索/重排模型) |

### 配置文件

| 文件路径 | 说明 |
|---------|------|
| `vllm/transformers_utils/configs/qwen3_next.py` | `Qwen3NextConfig` |
| `vllm/transformers_utils/configs/qwen3_5.py` | `Qwen3_5TextConfig` |
| `vllm/transformers_utils/configs/qwen3_5_moe.py` | `Qwen3_5MoeTextConfig` |

### GDN 线性注意力层

| 文件路径 | 说明 |
|---------|------|
| `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention`, `ChunkGatedDeltaRule`, `fused_gdn_gating` |

### FLA Triton Kernel

| 文件路径 | 说明 |
|---------|------|
| `vllm/model_executor/layers/fla/ops/chunk.py` | `chunk_gated_delta_rule` (Triton prefill) |
| `vllm/model_executor/layers/fla/ops/fused_recurrent.py` | `fused_recurrent_gated_delta_rule_*` (decode kernels) |
| `vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py` | `fused_post_conv_prep` (prefill 预处理融合) |
| `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py` | `fused_sigmoid_gating_delta_rule_update` (decode + sigmoid gating) |
| `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` | chunk 间 state 递推 |
| `vllm/model_executor/layers/fla/ops/chunk_o.py` | chunk 输出计算 |
| `vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py` | chunk 内 K·K^T |
| `vllm/model_executor/layers/fla/ops/solve_tril.py` | 下三角求解 |
| `vllm/model_executor/layers/fla/ops/cumsum.py` | chunk 内 cumsum |
| `vllm/model_executor/layers/fla/ops/wy_fast.py` | WY 表示计算 |
| `vllm/model_executor/layers/fla/ops/l2norm.py` | L2 归一化 |
| `vllm/model_executor/layers/fla/ops/index.py` | chunk indices/offsets 预计算 |
| `vllm/model_executor/layers/fla/ops/utils.py` | `FLA_CHUNK_SIZE=64`, 平台工具 |

### Attention Backend

| 文件路径 | 说明 |
|---------|------|
| `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionBackend` + MetadataBuilder (与 Kimi Linear 共享) |

### State 管理

| 文件路径 | 说明 |
|---------|------|
| `vllm/model_executor/layers/mamba/mamba_utils.py` | `MambaStateShapeCalculator.gated_delta_net_state_shape` 等 |
| `vllm/v1/kv_cache_interface.py` | `MambaSpec` 定义 |

### 测试文件

| 文件路径 | 说明 |
|---------|------|
| `tests/kernels/test_fused_gdn_post_conv.py` | `fused_post_conv_prep` kernel 测试 |
| `tests/v1/e2e/test_hybrid_chunked_prefill.py` | Hybrid chunked prefill E2E (Qwen3.5-0.8B) |
| `tests/v1/e2e/general/test_mamba_prefix_cache.py` | Mamba prefix cache 测试 |
| `tests/v1/e2e/spec_decode/test_spec_decode.py` | Spec decode E2E |
| `tests/lora/test_qwen35_densemodel_lora.py` | Qwen3.5 LoRA 测试 |
| `tests/distributed/test_eplb_spec_decode.py` | EPLB spec decode 测试 (qwen3_next_mtp) |
| `tests/v1/attention/test_kv_head_stride_canonicalization.py` | KV head stride 测试 (Qwen3.5-397B TP=8) |

### Benchmark / Eval

| 文件路径 | 说明 |
|---------|------|
| `benchmarks/attention_benchmarks/` | MLA/GDN benchmark runner 和 configs |
| `tests/evals/gsm8k/configs/Qwen3*` | GSM8K eval configs (多种模型/精度) |
| `.buildkite/scripts/scheduled_integration_test/qwen3_next_mtp_async_eplb.sh` | CI 集成测试 |
