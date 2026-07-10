# mHC (Manifold-Constrained Hyper-Connections) 在 DeepSeekV4 / vLLM 中的实现分析

> 论文: [mHC: Manifold-Constrained Hyper-Connections (arxiv 2512.24880v2)](https://arxiv.org/abs/2512.24880v2)
>
> 代码基准路径: `/Users/jared/vllm-project/vllm/`
>
> 核心文件: `vllm/model_executor/layers/mhc.py`, `vllm/model_executor/models/deepseek_v4.py`

---

## 目录

- [1. mHC 是什么: 从残差连接到流形约束](#1-mhc-是什么-从残差连接到流形约束)
  - [1.1 背景: 标准残差连接与恒等映射性质](#11-背景-标准残差连接与恒等映射性质)
  - [1.2 Hyper-Connections (HC): 扩展残差流宽度](#12-hyper-connections-hc-扩展残差流宽度)
  - [1.3 HC 的问题: 数值不稳定性](#13-hc-的问题-数值不稳定性)
  - [1.4 mHC 的核心: 流形约束恢复恒等映射](#14-mhc-的核心-流形约束恢复恒等映射)
- [2. 数学公式: 从论文到代码](#2-数学公式-从论文到代码)
  - [2.1 参数化: 论文 Eq.7-8 → vLLM `mhc_pre`](#21-参数化-论文-eq7-8--vllm-mhc_pre)
  - [2.2 Sinkhorn-Knopp 算法: 论文 Eq.9 → 双随机矩阵](#22-sinkhorn-knopp-算法-论文-eq9--双随机矩阵)
  - [2.3 三个映射的约束: 论文 Eq.17-19](#23-三个映射的约束-论文-eq17-19)
  - [2.4 mhc_post: 残差重组](#24-mhc_post-残差重组)
- [3. 整体架构位置](#3-整体架构位置)
- [4. 模型层: mHC 在 Decoder 中的使用](#4-模型层-mhc-在-decoder-中的使用)
- [5. 算法层: mHC 的三个核心操作](#5-算法层-mhc-的三个核心操作)
  - [5.1 mhc_pre: 进入 Attention/FFN 前的混合](#51-mhc_pre-进入-attentionffn-前的混合)
  - [5.2 mhc_post: Attention/FFN 后的重组](#52-mhc_post-attentionffn-后的重组)
  - [5.3 hc_head: 模型输出前的通道坍缩](#53-hc_head-模型输出前的通道坍缩)
- [6. Kernel 层: Tilelang 融合实现](#6-kernel-层-tilelang-融合实现)
  - [6.1 论文的 Kernel Fusion 策略 (Section 4.3.1)](#61-论文的-kernel-fusion-策略-section-431)
  - [6.2 mhc_pre 的 Tilelang 融合](#62-mhc_pre-的-tilelang-融合)
  - [6.3 mhc_post 的 Tilelang 融合](#63-mhc_post-的-tilelang-融合)
  - [6.4 hc_head 的 Tilelang 两 pass 融合](#64-hc_head-的-tilelang-两-pass-融合)
- [7. 基础设施优化: 从论文到实践](#7-基础设施优化-从论文到实践)
  - [7.1 Recomputing (选择性重计算)](#71-recomputing-选择性重计算)
  - [7.2 DualPipe 通信重叠](#72-dualpipe-通信重叠)
- [8. 数据流全图](#8-数据流全图)
- [9. 与标准 Transformer 和 HC 的对比](#9-与标准-transformer-和-hc-的对比)
- [10. 实验结果 (论文数据)]#10-实验结果-论文数据)
- [11. 文件导航索引](#11-文件导航索引)

---

## 1. mHC 是什么: 从残差连接到流形约束

### 1.1 背景: 标准残差连接与恒等映射性质

标准 Transformer 使用残差连接 (Residual Connection):

```
x_{l+1} = x_l + F(x_l, W_l)          ... 论文 Eq.1
```

递归展开得到:

```
x_L = x_l + Σ_{i=l}^{L-1} F(x_i, W_i)  ... 论文 Eq.2
```

**恒等映射性质** (identity mapping property) 指的是 `x_l` 项本身: 浅层信号**直接**映射到深层, 不经过任何修改。这个性质是 ResNet 训练稳定性的基石——前向信号不会爆炸或消失, 反向梯度也能畅通无阻。

### 1.2 Hyper-Connections (HC): 扩展残差流宽度

Hyper-Connections (HC) 在残差连接的基础上引入了一个新维度:

```
x_{l+1} = H_l^res · x_l + (H_l^post)^T · F(H_l^pre · x_l, W_l)  ... 论文 Eq.3
```

其中:
- `x_l ∈ R^{n×C}`: 特征维度从 C 扩展到 n×C (n 为扩展率, 如 4)
- `H_l^pre ∈ R^{1×n}`: 从 n×C 残差流中**聚合**出 C 维的层输入
- `H_l^post ∈ R^{1×n}`: 将 C 维层输出**映射回** n×C 残差流
- `H_l^res ∈ R^{n×n}`: 残差流内部的**信息混合矩阵**

HC 通过三个可学习映射来管理这个宽化的残差流:

```
x̃_l = RMSNorm(x_l)
H_l^pre  = α^pre  · tanh(θ^pre · x̃_l^T)  + b^pre
H_l^post = α^post · tanh(θ^post · x̃_l^T) + b^post
H_l^res  = α^res  · tanh(θ^res · x̃_l^T)  + b^res       ... 论文 Eq.5
```

HC 的核心价值在于: 它解耦了残差流的**信息容量** (n×C) 和层计算的**复杂度** (FLOPs 由 C 决定), 提供了一个新的缩放维度。

### 1.3 HC 的问题: 数值不稳定性

将 HC 递归展开到多层:

```
x_L = (Π_{i=1}^{L-l} H_{L-i}^res) · x_l + ...      ... 论文 Eq.4
```

**问题**: `H_l^res` 是**无约束**的, 因此复合映射 `Π H^res` 偏离恒等映射。论文实测发现:

- **Amax Gain Magnitude 峰值达 3000** (理想值应为 1)
- 前向信号指数放大/衰减, 反向梯度同样失控
- 在 27B 模型训练中, HC 在 ~12k 步出现**loss spike**, 梯度范数剧烈波动

> 论文 Fig.2-3: HC 的 composite mapping 的 Amax Gain 达到 3000×, 严重偏离恒等映射的增益 1。

此外, HC 还引入了显著的系统开销:
- 内存访问增加约 n 倍 (论文 Table 2: 从 2C+C 到 (5n+1)C + (3n+1)C)
- n 倍的流水线并行通信开销
- 中间激活占用大量显存

### 1.4 mHC 的核心: 流形约束恢复恒等映射

mHC 的核心思想: **将 `H_l^res` 投影到一个特定的流形上, 在保持多流信息交换能力的同时恢复恒等映射性质**。

具体来说, mHC 将 `H_l^res` 约束为**双随机矩阵** (doubly stochastic matrix):

```
P_M(H_l^res) = { H_l^res ∈ R^{n×n} | H_l^res · 1_n = 1_n,
                                       1_n^T · H_l^res = 1_n^T,
                                       H_l^res >= 0 }             ... 论文 Eq.6
```

即所有元素非负, 且**行和=1, 列和=1**。这实际上是 **Birkhoff 多面体** (Birkhoff polytope) 的定义。

双随机约束带来三个关键性质:

1. **范数保持**: `||H_l^res||_2 ≤ 1`, 映射是非扩张的, 有效抑制梯度爆炸
2. **复合封闭性**: 双随机矩阵的乘积仍是双随机矩阵, 因此 `Π H^res` 在任意深度都保持稳定性
3. **几何解释**: Birkhoff 多面体是所有置换矩阵的凸包, `H_l^res` 本质上是置换的凸组合, 起到信息融合作用

此外, mHC 还对 `H_l^pre` 和 `H_l^post` 施加**非负约束** (通过 sigmoid), 防止正负系数导致的信号对消。

---

## 2. 数学公式: 从论文到代码

### 2.1 参数化: 论文 Eq.7-8 → vLLM `mhc_pre`

**论文 Eq.7**: 将隐藏矩阵展平后通过线性投影计算原始系数

```
x̃_l' = RMSNorm(vec(x_l))                          # 展平为 R^{1×nC} 再归一化
H̃^pre  = α^pre  · (x̃_l' · φ^pre)  + b^pre         # R^{1×n}
H̃^post = α^post · (x̃_l' · φ^post) + b^post         # R^{1×n}
H̃^res  = α^res  · mat(x̃_l' · φ^res) + b^res       # R^{n×n}
```

其中 `φ^pre, φ^post ∈ R^{nC×n}`, `φ^res ∈ R^{nC×n²}`。

**论文 Eq.8**: 通过约束映射得到最终系数

```
H_l^pre  = σ(H̃^pre)              # sigmoid → R^{1×n}, 非负
H_l^post = 2σ(H̃^post)            # 2×sigmoid → R^{1×n}, 非负且上界为 2
H_l^res  = Sinkhorn-Knopp(H̃^res) # 双随机矩阵 → R^{n×n}
```

**在 vLLM 中的对应** (合并了 Eq.7-8 为一步):

论文将 φ 和 α 合并到 `fn` 权重矩阵中, 将所有映射的计算合并为一次 GEMM:

```python
# vllm/model_executor/layers/mhc.py:238-241 (ROCm fallback, 清晰展示数学)
x = residual_flat.view(num_tokens, hc_mult * hidden_size).float()  # vec(x_l): (T, nC)
mixes = torch.matmul(x, fn_flat.t())                                # x̃_l' · φ^T → (T, n²+2n)
sqrsum = x.square().sum(dim=-1, keepdim=True)                       # ||x||²
mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)  # RMSNorm
```

这里 `fn` 的 shape 是 `(n²+2n, nC)`, 一次性计算所有三个映射的系数:
- `mixes[:, :n]` → H̃^pre (前 n 列)
- `mixes[:, n:2n]` → H̃^post (中间 n 列)
- `mixes[:, 2n:]` → H̃^res (后 n² 列)

### 2.2 Sinkhorn-Knopp 算法: 论文 Eq.9 → 双随机矩阵

**论文 Eq.9**:

```
M^(0) = exp(H̃^res)           # 确保所有元素为正
M^(t) = T_r(T_c(M^(t-1)))    # 交替行列归一化
```

其中 `T_r` 是行归一化, `T_c` 是列归一化。当 `t_max → ∞` 时收敛到双随机矩阵。论文实验使用 `t_max = 20`。

**vLLM 实现** (`mhc.py:122-145`):

```python
# 初始 softmax (等价于 exp + 行归一化)
cm[j, k] = T.exp(cm[j, k] - row_max[j])       # 减最大值防溢出
cm[j, k] = cm[j, k] / row_sum[j] + eps         # 行归一化 + eps
cm[j, k] = cm[j, k] / col_sum[k] + eps         # 列归一化

# 迭代 (sinkhorn_repeat - 1 次)
for _ in range(sinkhorn_repeat - 1):
    row_normalize(cm)    # 行归一化
    col_normalize(cm)    # 列归一化
```

> 注: vLLM 的实现用 softmax (exp + 行归一化) 作为初始步, 而非论文的 `exp(H̃^res) + 单独归一化`, 数学等价但数值更稳定。

### 2.3 三个映射的约束: 论文 Eq.17-19

```
Eq.17:  H_l^pre  = σ(H̃^pre)           → sigmoid(x * scale[0] + base[:n]) + eps
Eq.18:  H_l^post = 2σ(H̃^post)         → sigmoid(x * scale[1] + base[n:2n]) * 2.0
Eq.19:  H_l^res  = Sinkhorn-Knopp(H̃^res) → 上面的 Sinkhorn 迭代
```

**vLLM 对应** (`mhc.py:104-149`):

```python
# Eq.18: H_l^post = 2σ(...)
post_mix[i, j] = T.sigmoid(mixes_shared[j + hc_mult] * hc_scale[1]
                            + hc_base[j + hc_mult]) * hc_post_mult_value  # hc_post_mult_value = 2.0

# Eq.19: H_l^res = Sinkhorn-Knopp(...)
cm[j, k] = mixes_shared[j*hc_mult + k + hc_mult*2] * hc_scale[2] + hc_base[...]
# ... Sinkhorn 迭代 (见 2.2) ...

# Eq.17: H_l^pre = σ(...)
pre_mix_shared[j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps

# 层输入: F_pre = H_l^pre · x_l  (加权求和各通道)
layer_input[i, ...] += pre * residual[i, i_hc, ...]
```

### 2.4 mhc_post: 残差重组

```
x_{l+1} = H_l^res · x_l + (H_l^post)^T · F(H_l^pre · x_l, W_l)  ... 论文 Eq.3
```

其中:
- `H_l^res · x_l` = `comb_mix @ residual` — 通道间信息交换 (双随机矩阵乘法)
- `(H_l^post)^T · F(...)` = `post_mix * layer_output` — 当前层输出的门控贡献

---

## 3. 整体架构位置

mHC 在 DeepSeekV4 的完整架构中的位置:

```
Input IDs
    │
    ▼
embed_tokens                                    # 词嵌入
    │
    ▼
.repeat(1, hc_mult, 1)                         # 复制为 hc_mult 个通道 ← 入口
    │                                            # n-stream residual 开始
    ▼ shape: (tokens, hc_mult, hidden_size)
┌─────────────────────────────────────────────────┐
│  DecoderLayer × N                                │
│  ┌─────────────────────────────────────────────┐ │
│  │  mhc_pre(residual, hc_attn_*)               │ │
│  │    → H^pre: σ(GEMM+RMSNorm+sigmoid)         │ │  Eq.17: 聚合出单通道输入
│  │    → H^post: 2σ(...)                         │ │  Eq.18: 门控系数
│  │    → H^res: Sinkhorn-Knopp(...)             │ │  Eq.19: 双通道混合矩阵
│  │         │                                    │ │
│  │    layer_input = H^pre · residual            │ │  加权求和 → (T, H)
│  │         │                                    │ │
│  │    attn_norm(layer_input)                    │ │
│  │         │                                    │ │
│  │    DeepseekV4Attention(positions, x)         │ │  (MLA + Sparse Indexer)
│  │         │                                    │ │
│  │  mhc_post(attn_out, residual, H^post, H^res) │ │
│  │    = H^res · residual + H^post^T · attn_out  │ │  论文 Eq.3
│  │    → new residual (hc_mult channels)         │ │
│  │         │                                    │ │
│  │  mhc_pre(residual, hc_ffn_*)                 │ │  (同结构, 不同参数)
│  │         │                                    │ │
│  │    ffn_norm → DeepseekV4MoE                   │ │
│  │         │                                    │ │
│  │  mhc_post(ffn_out, residual, H^post, H^res)  │ │
│  │    → new residual (hc_mult channels)         │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
    │
    ▼ shape: (tokens, hc_mult, hidden_size)
hc_head(residual, hc_head_*)                    # 通道坍缩 ← 出口
    │                                            # 等价于只使用 H^pre 的 mhc_pre
    ▼ shape: (tokens, hidden_size)
RMSNorm
    │
    ▼
lm_head → logits
```

---

## 4. 模型层: mHC 在 Decoder 中的使用

### 4.1 DeepseekV4Model: 初始化与入口

> 文件: `vllm/model_executor/models/deepseek_v4.py:1222-1348`

```python
class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        self.hc_mult = config.hc_mult        # 默认 4 (论文推荐值)
        self.hc_dim = self.hc_mult * config.hidden_size
        self.hc_eps = config.hc_eps

        # hc_head 参数 (模型最后的通道坍缩, 等价于只有 H^pre)
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))
```

**关键入口: embed 后立即复制通道 (构建 n-stream residual)**

```python
def forward(self, input_ids, positions, ...):
    hidden_states = self.embed_input_ids(input_ids)
    # ★ 将 hidden_states 复制 hc_mult 份, 构建 n-stream residual
    hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
    # shape: (tokens, hidden_size) → (tokens, hc_mult, hidden_size)

    for layer in self.layers:
        hidden_states = layer(hidden_states, positions, input_ids)

    # ★ 保存 pre-hc_head 的多通道状态 (给 MTP 用)
    self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

    # ★ hc_head: 多通道 → 单通道 (等价于只计算 H^pre 的 mhc_pre)
    hidden_states = hc_head(hidden_states, self.hc_head_fn, ...)
    hidden_states = self.norm(hidden_states)
    return hidden_states
```

### 4.2 DeepseekV4DecoderLayer: mHC 包裹 Attention 和 FFN

> 文件: `vllm/model_executor/models/deepseek_v4.py:1091-1218`

```python
class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix, ...):
        config = vllm_config.model_config.hf_config
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters  # 论文 t_max = 20
        self.hc_eps = config.hc_eps
        self.hc_post_alpha = 2.0  # 论文 Eq.18 的系数 2

        # 论文 Eq.10: φ_l ∈ R^{nC × (n²+2n)}, 这里 hc_mult=4 → (24, 4H)
        mix_hc = (2 + self.hc_mult) * self.hc_mult   # = n²+2n = 24
        hc_dim = self.hc_mult * self.hidden_size

        # φ 权重 (论文 Eq.10, 合并了 RMSNorm 权重)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))    # (24, 4H)
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))     # (24, 4H)

        # 论文 Eq.12: α (三个标量, 分别对应 pre/post/res)
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))

        # 论文 Eq.13: b_l ∈ R^{1 × (n²+2n)}
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
```

**Forward 流程 (论文 Eq.3 的实例化):**

```python
def forward(self, x, positions, input_ids):
    # ── Attention Block ──
    residual = x                                             # (tokens, hc_mult, H)
    x, post, comb = self.hc_pre(x, self.hc_attn_fn,         # 计算 H^pre, H^post, H^res
                                self.hc_attn_scale, self.hc_attn_base)
    # x = H^pre · residual → (tokens, H) — 聚合为单通道输入
    x = self.attn_norm(x)
    x = self.attn(positions, x, None)                         # F(H^pre · x_l, W_l)
    x = self.hc_post(x, residual, post, comb)                 # H^res · res + H^post^T · F

    # ── FFN Block ── (同结构, 不同参数)
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_ffn_fn,
                                self.hc_ffn_scale, self.hc_ffn_base)
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post(x, residual, post, comb)
    return x
```

**参数量分析:**

```
n = hc_mult = 4
H = hidden_size (e.g. 4096)
nC = hc_dim = 4H = 16384
n² + 2n = mix_hc = 24

每组 mHC 参数 (对应论文的 φ + α + b):
  φ (fn):   24 × 16384 = 393,216 (float32, tfloat32 计算) ≈ 1.5 MB
  α (scale): 3 (float32)
  b (base):  24 (float32)
  总计: ≈ 1.5 MB × 2 (attn + ffn) = ~3 MB per layer
```

---

## 5. 算法层: mHC 的三个核心操作

### 5.1 mhc_pre: 进入 Attention/FFN 前的混合

> 文件: `vllm/model_executor/layers/mhc.py:181-341`

**输入:**
- `residual`: `(tokens, hc_mult, hidden_size)` — n-stream 隐藏状态 (论文的 x_l)
- `fn`: `(n²+2n, nC)` — 论文的 φ_l 权重矩阵 (Eq.10)
- `hc_scale`: `(3,)` — 论文的 α (Eq.12)
- `hc_base`: `(n²+2n,)` — 论文的 b_l (Eq.13)

**输出:**
- `post_mix`: `(tokens, hc_mult, 1)` — H_l^post = 2σ(...) (Eq.18)
- `comb_mix`: `(tokens, hc_mult, hc_mult)` — H_l^res = Sinkhorn-Knopp(...) (Eq.19)
- `layer_input`: `(tokens, hidden_size)` — H_l^pre · x_l (Eq.17 + 应用)

**算法 (ROCm fallback, 清晰展示数学):**

```python
# ─── 论文 Eq.14-15: 计算 H̃̃ 和 RMSNorm ───
x = residual_flat.view(num_tokens, hc_mult * hidden_size).float()  # vec(x_l)
mixes = torch.matmul(x, fn_flat.t())                                # Eq.14: x̃_l · φ
sqrsum = x.square().sum(dim=-1, keepdim=True)                       # Eq.15: ||x̃||²/nC
mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)  # ÷ r

# ─── 论文 Eq.16: 加入 α 和 b ───
# mixes 已经包含了 x̃_l · φ / r 的结果, 接下来分别应用 α 和 b

# ─── 论文 Eq.17: H^pre = σ(H̃^pre) ───
pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

# ─── 论文 Eq.18: H^post = 2σ(H̃^post) ───
post_logits = mixes[:, hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult]
post_mix = torch.sigmoid(post_logits) * hc_post_mult_value  # hc_post_mult_value = 2.0

# ─── 论文 Eq.19: H^res = Sinkhorn-Knopp(H̃^res) ───
comb_logits = mixes[:, 2*hc_mult:].view(T, hc_mult, hc_mult) * hc_scale[2] + base
comb_mix = torch.softmax(comb_logits, dim=-1) + eps          # 初始行归一化
comb_mix = comb_mix / (comb_mix.sum(dim=-2) + eps)           # 初始列归一化
for _ in range(sinkhorn_repeat - 1):                          # 迭代 t_max-1 次
    comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + eps)  # 行归一化
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + eps)  # 列归一化

# ─── 应用 H^pre: F_pre = H^pre · x_l ───
layer_input = torch.sum(pre_mix.unsqueeze(-1) * residual_flat.float(), dim=1)
```

### 5.2 mhc_post: Attention/FFN 后的重组

> 文件: `vllm/model_executor/layers/mhc.py:444-468`

**输入:**
- `x`: `(tokens, hidden_size)` — F(H^pre · x_l, W_l) 的输出
- `residual`: `(tokens, hc_mult, hidden_size)` — mhc_pre 前的 x_l
- `post_mix`: `(tokens, hc_mult, 1)` — H_l^post
- `comb_mix`: `(tokens, hc_mult, hc_mult)` — H_l^res

**输出:**
- `out`: `(tokens, hc_mult, hidden_size)` — x_{l+1}

**算法 (论文 Eq.3):**

```python
# x_{l+1} = H_l^res · x_l + (H_l^post)^T · F(H_l^pre · x_l, W_l)

mixed_residual = einsum('...ij,...ih->...jh', comb_mix, residual)  # H^res · x_l
post_term = post_mix * x.unsqueeze(-2)                              # H^post^T · F(...)
out = mixed_residual + post_term
```

逐元素公式:
```
out[:, m, :] = Σ_k  comb_mix[m, k] × residual[:, k, :]  +  post_mix[m] × x[:, :]
```

由于 `comb_mix` 是双随机矩阵, 行和=1, 因此 `H^res · x_l` 是各通道特征的**凸组合**, 保证了信息守恒。

### 5.3 hc_head: 模型输出前的通道坍缩

> 文件: `vllm/model_executor/layers/mhc.py:500-688`

在所有 DecoderLayer 之后, 将 n-stream residual 坍缩回单通道。数学上等价于**只使用 H^pre 的 mhc_pre** (没有 H^post 和 H^res):

```python
x = residual.flatten(-2, -1).float()                             # vec(x_l): (T, nC)
mixes = x @ fn.T                                                  # x̃_l · φ → (T, n)
rsqrt = rsqrt(x.square().sum(-1, keepdim=True) / hc_dim + eps)   # RMSNorm
pre_mix = sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + eps   # Eq.17

out = Σ_m  pre_mix[m] × residual[:, m, :]                        # H^pre · x_l → (T, H)
```

---

## 6. Kernel 层: Tilelang 融合实现

> 文件: `vllm/model_executor/layers/mhc.py`

### 6.1 论文的 Kernel Fusion 策略 (Section 4.3.1)

论文将 mHC 的计算分为三组 kernel:

1. **Kernel 1 (Eq.14-15)**: `x̃_l · φ` (GEMM) + RMSNorm 的融合
   - 使用 DeepGEMM 的 `tf32_hc_prenorm_gemm`, 通过 split-K 并行同时输出 GEMM 结果和 x² 的部分和
   - 论文指出: RMSNorm 作用于高维 x̃_l (nC 维) 时延迟显著, 将除以范数的操作移到 GEMM 之后数学等价但更高效

2. **Kernel 2 (Eq.16-19)**: sigmoid × 2 + Sinkhorn-Knopp 的融合
   - 论文: "These lightweight operations on small coefficients are opportunistically fused into a single kernel"
   - 在 vLLM 中, 这对应 `mhc_pre_big_fuse_tilelang` 的后半部分

3. **Kernel 3 (Sinkhorn backward)**: 自定义反向传播, 在片上重计算中间结果

### 6.2 mhc_pre 的 Tilelang 融合

> 文件: `vllm/model_executor/layers/mhc.py:48-179`

**阶段 A: DeepGEMM 融合 GEMM + RMSNorm (论文 Eq.14-15)**

```python
from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

# 自定义 GEMM kernel, split-K 并行, 同时输出:
#   gemm_out_mul:    (n_splits, num_tokens, n²+2n)  — GEMM 部分和
#   gemm_out_sqrsum: (n_splits, num_tokens)          — ||x||² 部分和
tf32_hc_prenorm_gemm(residual_flat.view(T, nC), fn_flat, gemm_out_mul, gemm_out_sqrsum, n_splits)
```

**阶段 B: Tilelang 融合 RMSNorm + sigmoid + Sinkhorn + 加权求和 (论文 Eq.16-19)**

```python
@tilelang.jit(...)
def mhc_pre_big_fuse_tilelang(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base,
                               residual, post_mix, comb_mix, layer_input, ...):
    with T.Kernel(num_tokens, threads=96) as i:
        # ─── 合并 split-K + RMSNorm (Eq.15) ───
        rms[0] = sum(gemm_out_sqrsum[i_split, i])  # 合并 ||x||² 的部分和
        rms[0] = rsqrt(rms / (n * H) + eps)         # RMSNorm

        # ─── 合并 GEMM 部分和 + RMSNorm (Eq.14-15) ───
        for j in Parallel(n²+2n):
            mixes[j] = sum(gemm_out_mul[i_split, i, j]) * rms  # GEMM 合并 × 1/r

        # ─── 96 线程分叉 ───
        if thread < 32:
            # Eq.18: H^post = 2σ(...)
            post_mix[i, j] = sigmoid(mixes[j+n] * scale[1] + base[j+n]) * 2.0

            # Eq.19: H^res = Sinkhorn-Knopp(...)
            cm[j, k] = mixes[j*n+k+n*2] * scale[2] + base[...]
            for iter in sinkhorn_iters:
                row_normalize(cm)
                col_normalize(cm)
            comb_mix[i, j*n+k] = cm[j, k]
        else:
            # Eq.17: H^pre = σ(...)
            pre_mix[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps

            # 应用 H^pre → layer_input (流水线化的加权求和)
            for i0_h in Pipelined(hidden_size // hidden_block):
                layer_input[i, ...] += pre * residual[i, m, ...]
```

**关键优化:**
- GEMM 和 RMSNorm 通过 `tf32_hc_prenorm_gemm` 融合 (split-K 并行)
- 96 线程内核, 前 32 线程处理 H^post + H^res, 后 64 线程处理 H^pre + layer_input
- Sinkhorn 迭代完全在寄存器/共享内存中完成 (n=4, 矩阵只有 4×4)
- 论文指出: 应用 H^post 和 H^res 的 kernel 通过融合残差合并, 将读取从 (3n+1)C 减少到 (n+1)C

### 6.3 mhc_post 的 Tilelang 融合

> 文件: `vllm/model_executor/layers/mhc.py:392-441`

融合了论文 Eq.3 中的两个操作: `H^res · x_l` + `H^post^T · F(...)`:

```python
@tilelang.jit(...)
def mhc_post_tilelang(a, b, c, d, x, hc, hidden, n_thr=128, h_blk=1024):
    # a = H^res: (T, n, n)     b = residual: (T, n, H)
    # c = H^post: (T, n)       d = F(...): (T, H)

    with T.Kernel(num_tokens, threads=n_thr) as i_n:
        # 预加载 H^res 和 H^post 到寄存器 (很小, n=4)
        T.copy(comb_mix[i_n], a_local)    # (4, 4)
        T.copy(post_mix[i_n], c_local)    # (4,)

        # 流水线化处理 hidden 维度
        for i0_h in Pipeline(ceildiv(H, h_blk)):
            T.copy(residual[i_n, 0, ...], b_local)   # 加载 residual 块
            T.copy(x[i_n, ...], d_local)              # 加载 F(...) 块

            # ★ 融合: H^res · residual + H^post * F(...)
            for i_hco, i1_h in Parallel(n, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]  # H^post * F
                for i_hci in serial(n):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]  # H^res · residual

            T.copy(x_local, out[i_n, 0, ...])
```

### 6.4 hc_head 的 Tilelang 两 pass 融合

> 文件: `vllm/model_executor/layers/mhc.py:500-592`

```python
@tilelang.jit(...)
def hc_head_fuse_tilelang(residual, fn, hc_scale, hc_base, out, ...):
    with T.Kernel(num_tokens, threads=128) as i:
        # Pass 1: 累加 ||x||² 和 n 个点积 (需要遍历所有 n×H 元素)
        for m_c in serial(n):
            for i_h in serial(n_h):
                sqrsum_r[0] += x_local²                         # ||x||²
                mixes_r[m] += x_local * fn[m, m_c*H + ...]      # x · φ

        # RMSNorm + sigmoid
        pre_mix[m] = sigmoid(mixes_r[m] * rsqrt(sqrsum/nC) * scale + base) + eps

        # Pass 2: 加权求和
        for i0_h in Pipeline(n_h):
            out[i, ...] = sum(pre_mix[m] * residual[i, m, ...])
```

**两 pass 的原因:** Pass 1 遍历所有通道计算 mixes, 需要 mixes 才能得到 pre_mix, Pass 2 再用 pre_mix 做加权求和。两 pass 共享 residual 的读取, 避免显存中间结果。

---

## 7. 基础设施优化: 从论文到实践

### 7.1 Recomputing (选择性重计算)

> 论文 Section 4.3.2

mHC 的 n-stream residual 引入了大量中间激活。论文的策略:

1. Forward pass 后**丢弃** mHC kernel 的中间激活
2. Backward pass 时通过重新执行 mHC kernel (不含重型 F) 来重建中间结果
3. 对于连续 L_r 层, 只需保存第一层输入 x_{l_0}

**最优块大小** (论文 Eq.20):

```
L_r* = argmin [ nC × ⌈L/L_r⌉ + (n+2)C × L_r ] ≈ √(nL/(n+2))
```

当 n=4 时, L_r* ≈ √(4L/6) = √(2L/3)

> 注: 流水线并行中, 重计算块不能跨越 pipeline stage 边界。论文观察到 L_r* 通常与每个 stage 的层数对齐, 因此将重计算边界与 pipeline stage 同步。

### 7.2 DualPipe 通信重叠

> 论文 Section 4.3.3

n-stream residual 在流水线并行中引入 n 倍通信开销。论文扩展 DualPipe 调度来处理:

- FFN 层的 F_{post,res} kernel 在**专用高优先级计算流**上执行, 防止阻塞通信流
- Attention 层避免使用持久化 kernel, 允许灵活抢占
- 重计算过程与 pipeline 通信解耦 (每个 stage 的初始激活 x_{l_0} 已在本地缓存)

---

## 8. 数据流全图

```
                    n-stream 隐藏状态
                (tokens, hc_mult=4, H)
                         │
            ╔════════════╧════════════╗
            ║      mhc_pre ( attn )    ║
            ║                          ║
            ║  1. vec(x_l)            ║  展平: (T, nC)
            ║  2. GEMM: x̃_l · φ      ║  tf32_hc_prenorm_gemm (DeepGEMM)
            ║     → (sqrsum, mixes)    ║  融合 GEMM + RMSNorm (Eq.14-15)
            ║                          ║
            ║  3. RMSNorm(mixes)       ║  Tilelang 融合 kernel:
            ║                          ║  同时计算 3 组约束映射
            ║  4. 论文 Eq.17-19:       ║
            ║     H^pre  = σ(m[:4])    ║  非负门控, 聚合出单通道输入
            ║     H^post = 2σ(m[4:8])  ║  非负门控, 范围 [0, 2]
            ║     H^res = SK(m[8:])    ║  双随机矩阵, 信息守恒
            ║                          ║
            ║  5. F_pre = H^pre · x_l  ║  加权求和 → 单通道
            ╚════════════╤════════════╝
                         │
               (tokens, H) — 单通道
                         │
                    attn_norm
                         │
                 DeepseekV4Attention
                 (MLA + Indexer)  ← F(H^pre · x_l, W_l)
                         │
            ╔════════════╧════════════╗
            ║     mhc_post ( attn )     ║
            ║                           ║
            ║  论文 Eq.3:               ║  Tilelang 融合 kernel:
            ║  x_{l+1} = H^res · x_l   ║  comb_mix @ residual
            ║          + H^post^T · F   ║  + post_mix * layer_output
            ╚════════════╤════════════╝
                         │
                (tokens, 4, H) — n-stream
                         │
            ╔════════════╧════════════╗
            ║      mhc_pre ( ffn )     ║  (同 attn 的 mhc_pre, 不同参数)
            ╚════════════╤════════════╝
                         │
               (tokens, H) — 单通道
                         │
                    ffn_norm → MoE    ← F(H^pre · x_l, W_l)
                         │
            ╔════════════╧════════════╗
            ║     mhc_post ( ffn )     ║  (同 attn 的 mhc_post, 不同参数)
            ╚════════════╤════════════╝
                         │
                (tokens, 4, H) — n-stream
                         │
                        ... (重复 N 层, H^res 的复合保持双随机性)
                         │
            ╔════════════╧════════════╗
            ║        hc_head            ║
            ║                           ║
            ║  等价于只有 H^pre:        ║  Tilelang 两 pass 融合:
            ║  1. mixes = x̃_l · φ      ║  Pass 1: 累加 sqrsum + dot products
            ║  2. RMSNorm(mixes)        ║  Pass 2: H^pre · x_l (加权求和)
            ║  3. H^pre = σ(...)        ║
            ║  4. out = H^pre · x_l     ║
            ╚════════════╤════════════╝
                         │
               (tokens, H) — 单通道
                         │
                    RMSNorm → lm_head
```

---

## 9. 与标准 Transformer 和 HC 的对比

### 9.1 残差连接对比

```
标准 Transformer (Pre-Norm):
  ┌──────────────────┐
  │  x → norm → attn → + → x'           单通道, 简单加法
  │  │                    ↑              │
  │  └────────────────────┘              │  恒等映射: x 直接传到输出
  └──────────────────┘

HC (Hyper-Connections):
  ┌──────────────────────────────────────────────┐
  │  x(T,n,H) → [H^pre, H^post, H^res] → ...    n 通道, 无约束映射
  │                                               │
  │  H^res ∈ R^{n×n}: 无约束, 信号可爆炸/消失     │  ❌ Amax Gain 可达 3000
  │  H^pre, H^post: tanh, 可正可负                │  ❌ 信号对消
  └──────────────────────────────────────────────┘

mHC (Manifold-Constrained Hyper-Connections):
  ┌──────────────────────────────────────────────┐
  │  x(T,n,H) → [H^pre, H^post, H^res] → ...    n 通道, 流形约束
  │                                               │
  │  H^res ∈ Birkhoff polytope: 双随机矩阵        │  ✓ 范数保持, 复合封闭
  │  H^pre = σ(...): 非负, 范围 (0, 1)            │  ✓ 无信号对消
  │  H^post = 2σ(...): 非负, 范围 (0, 2)          │  ✓ 无信号对消
  └──────────────────────────────────────────────┘
```

### 9.2 参数开销

```
标准 Transformer:
  每层额外参数: 0
  额外显存: 0

mHC (n=4, hidden_size=H):
  每层额外参数:
    φ (fn):    2 × (24 × 4H) float32 = 768H bytes ≈ 3 MB/layer (H=4096)
    α (scale): 2 × 3 float32 = 24 bytes
    b (base):  2 × 24 float32 = 192 bytes
    总计: ≈ 3 MB per layer

  推理时额外显存:
    residual: n × 正常大小 = 4× (n-stream 张量)
    H^post + H^res: n + n² = 20 values per token (float32)

  训练开销 (论文实测):
    n=4 时仅增加 6.7% 训练时间
```

### 9.3 Kernel 融合对比

```
标准 Transformer:
  residual = x + attn_out    # 单次加法, 无需融合

mHC (3 个 fused kernel per layer):
  mhc_pre:
    GEMM + RMSNorm                # tf32_hc_prenorm_gemm (DeepGEMM)
    + sigmoid × 2 + Sinkhorn      # mhc_pre_big_fuse_tilelang (Tilelang)
    + H^pre · residual             # (同上, 融合在 kernel 中)

  mhc_post:
    H^res · residual + H^post × F  # mhc_post_tilelang (Tilelang)

  hc_head:
    两 pass: 累加 + RMSNorm + sigmoid + 加权求和  # hc_head_fuse_tilelang
```

### 9.4 MTP (Multi-Token Prediction) 支持

```python
# DeepseekV4Model.forward:
# 保存 pre-hc_head 的 n-stream 状态, 供 MTP draft model 使用
self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
# shape: (tokens, nC) → MTP 直接从 n-stream 状态开始预测
```

---

## 10. 实验结果 (论文数据)

> 论文 Section 5, 基于 DeepSeek-V3 MoE 架构

### 10.1 训练稳定性

| 方法 | Loss Gap vs Baseline | 梯度范数稳定性 | Amax Gain (峰值) |
|------|---------------------|---------------|-----------------|
| Baseline | 0.0 | 稳定 | 1.0 |
| HC | 不稳定, ~12k 步出现 spike | 剧烈波动 | **~3000** |
| mHC | **-0.021** (稳定下降) | 与 baseline 相当 | **~1.6** |

### 10.2 下游任务 (27B 模型)

| Benchmark | Baseline | HC | mHC |
|-----------|----------|-----|------|
| BBH (EM) | 43.8 | 48.9 | **51.0** |
| DROP (F1) | 47.0 | 51.6 | **53.9** |
| GSM8K (EM) | 46.7 | 53.2 | 53.8 |
| HellaSwag (Acc) | 73.7 | 74.3 | **74.7** |
| MATH (EM) | 22.0 | **26.4** | 26.0 |
| MMLU (Acc) | 59.0 | 63.0 | **63.4** |
| PIQA (Acc) | 78.5 | 79.9 | **80.5** |
| TriviaQA (EM) | 54.3 | 56.3 | **57.6** |

### 10.3 系统开销

- n=4 时训练时间开销: **仅 6.7%**
- 消融实验 (论文 Tab.1): H^res 贡献最大 (-0.022), H^pre + H^post 额外贡献 -0.005

---

## 11. 文件导航索引

### mHC 核心实现

| 文件路径 | 行号 | 说明 |
|---------|------|------|
| `vllm/model_executor/layers/mhc.py` | 1-688 | mHC 全部实现: `mhc_pre`, `mhc_post`, `hc_head` + Tilelang kernels |
| `vllm/model_executor/layers/mhc.py` | 48-179 | `mhc_pre_big_fuse_tilelang`: 融合 RMSNorm + sigmoid + Sinkhorn + 加权求和 |
| `vllm/model_executor/layers/mhc.py` | 181-341 | `mhc_pre`: ROCm fallback + CUDA Tilelang 调度 |
| `vllm/model_executor/layers/mhc.py` | 392-441 | `mhc_post_tilelang`: 融合 H^res · residual + H^post × F |
| `vllm/model_executor/layers/mhc.py` | 500-592 | `hc_head_fuse_tilelang`: 两 pass 融合 (H^pre only) |

### 模型层 (使用 mHC)

| 文件路径 | 行号 | 说明 |
|---------|------|------|
| `vllm/model_executor/models/deepseek_v4.py` | 1091-1218 | `DeepseekV4DecoderLayer`: mHC 包裹 Attention + FFN |
| `vllm/model_executor/models/deepseek_v4.py` | 1222-1348 | `DeepseekV4Model`: embed → repeat → layers → hc_head → norm |
| `vllm/model_executor/models/deepseek_v4.py` | 1317-1348 | forward: embed 后 `repeat(hc_mult)` + 最后 `hc_head` |

### 配置

| 文件路径 | 说明 |
|---------|------|
| `vllm/transformers_utils/configs/deepseek_v4.py` | `DeepseekV4Config`: `hc_mult=4`, `hc_sinkhorn_iters=20`, `hc_eps` |

### 依赖的 Kernel

| 文件路径 | 说明 |
|---------|------|
| `vllm/utils/deep_gemm.py` | `tf32_hc_prenorm_gemm`: 融合 GEMM + RMSNorm (论文 Eq.14-15) |

### 论文对应

| 论文公式 | vLLM 代码位置 | 说明 |
|---------|-------------|------|
| Eq.3 | `mhc.py` + `deepseek_v4.py` | 核心架构: H^res · x + H^post^T · F(H^pre · x, W) |
| Eq.6 | `mhc.py:122-145` | Birkhoff polytope 约束 (Sinkhorn-Knopp) |
| Eq.7 | `mhc.py:238-241` | 参数化: RMSNorm + GEMM + α·(x̃·φ) + b |
| Eq.8 | `mhc.py:104-149` | 流形投影: σ / 2σ / Sinkhorn-Knopp |
| Eq.14-15 | `deep_gemm.py` → `mhc.py:89-96` | GEMM + RMSNorm 融合 (split-K) |
| Eq.16-19 | `mhc.py:100-177` | Tilelang 融合: sigmoid + Sinkhorn + 加权求和 |
| Eq.20 | 训练框架 (非 vLLM) | 最优重计算块大小 L_r* ≈ √(nL/(n+2)) |
