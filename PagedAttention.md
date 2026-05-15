# vLLM PagedAttention 机制与各种 Attention 结构的内存管理

> 代码基准路径: `/Users/jared/vllm-project/vllm/`
>
> 核心文件: `vllm/v1/kv_cache_interface.py`, `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/single_type_kv_cache_manager.py`

---

## 目录

- [1. PagedAttention 核心思想](#1-pagedattention-核心思想)
- [2. Spec 层级体系](#2-spec-层级体系)
- [3. Standard GQA Full Attention](#3-standard-gqa-full-attention)
- [4. MLA (Multi-head Latent Attention)](#4-mla-multi-head-latent-attention)
- [5. GDN / KDA 线性注意力 (SSM State)](#5-gdn--kda-线性注意力-ssm-state)
- [6. 四种 Attention 的内存对比](#6-四种-attention-的内存对比)
- [7. 混合模型 (IsHybrid) 的分组与调度](#7-混合-model-ishybrid-的分组与调度)
- [8. Block 分配与回收策略](#8-block-分配与回收策略)
- [9. Tensor 布局与物理内存分配](#9-tensor-布局与物理内存分配)
- [10. 文件导航索引](#10-文件导航索引)

---

## 1. PagedAttention 核心思想

vLLM 的 PagedAttention 借鉴了操作系统的**虚拟内存分页**机制:

- 将 KV cache / SSM state 划分为固定大小的 **block**（页）
- 每个请求维护一个 **block table**（页表），映射逻辑位置到物理 block
- Block 按需分配，空闲时回收，不同请求共享同一个 block pool
- 通过 **slot mapping** 定位每个 token 应写入/读取的具体位置

**所有 Attention 类型都必须回答两个问题:**

1. **一个 block 存什么？** → 由 `KVCacheSpec.page_size_bytes` 决定
2. **一个请求需要多少 block？** → 由 `max_memory_usage_bytes` 和运行时分配策略决定

---

## 2. Spec 层级体系

> 文件: `vllm/v1/kv_cache_interface.py`

```
KVCacheSpec                          # 基类: block_size + 抽象 page_size_bytes
├── AttentionSpec                    # 注意力基类: num_kv_heads, head_size, dtype
│   ├── FullAttentionSpec            # 标准全注意力
│   │   └── MLAAttentionSpec         # MLA 压缩注意力 (DeepSeekV2/V3/V4)
│   ├── SlidingWindowSpec            # 滑动窗口注意力
│   │   └── SlidingWindowMLASpec     # MLA + 滑动窗口 (DeepseekV4 SWA)
│   ├── ChunkedLocalAttentionSpec    # 分块局部注意力
│   └── CrossAttentionSpec           # 交叉注意力
└── MambaSpec                        # SSM 状态 (GDN/KDA/Mamba/Mamba2)
```

每种 Spec 定义了三个关键属性:

| 属性 | 说明 |
|------|------|
| `page_size_bytes` | 一个 block 占多少字节 |
| `max_memory_usage_bytes` | 单个请求的最大内存使用 |
| `storage_block_size` | 实际存储的 token 数（可与逻辑 block_size 不同） |

---

## 3. Standard GQA Full Attention

> 适用模型: Qwen3-Next/Qwen3.5 的 full_attention 层、Llama、Qwen2 等
>
> Spec: `FullAttentionSpec` → `AttentionSpec`
>
> Block 分配器: `FullAttentionManager`

### 3.1 一个 Block 存什么

每个 block 存储 **K 和 V 两个张量**:

```
Block 布局: [2, block_size, num_kv_heads, head_size]
            ↑ K/V   ↑ tokens   ↑ heads     ↑ dim
```

**page_size_bytes 计算:**

```python
# vllm/v1/kv_cache_interface.py:265-284
@property
def real_page_size_bytes(self) -> int:
    return (
        self.block_size           # 每个 block 的 token 数 (默认 16)
        * self.num_kv_heads       # KV head 数量
        * (self.head_size + self.head_size_v)  # K dim + V dim
        * get_dtype_size(self.dtype)           # 数据类型字节数
    )
```

**示例 (Qwen3-Next-80B-A3B, bf16):**
```
block_size = 16
num_kv_heads = 2 (GQA, TP=1)
head_size = 256
head_size_v = 256

page_size = 16 * 2 * (256 + 256) * 2 = 32,768 bytes = 32 KB
```

### 3.2 一个请求需要多少 Block

```python
def max_memory_usage_bytes(self, vllm_config) -> int:
    max_model_len = vllm_config.model_config.max_model_len
    return cdiv(max_model_len, self.block_size) * self.page_size_bytes
```

**关键特征: 内存随序列长度线性增长。**

1M token 上下文 (block_size=16):
```
num_blocks = ceil(1,000,000 / 16) = 62,500 blocks
total_memory = 62,500 * 32 KB = 2,000 MB
```

### 3.3 数据写入方式

```python
# 通过 slot_mapping 定位写入位置
# slot_mapping[t] = block_id * block_size + offset_in_block

# 写入 K:
key_cache[block_id][offset] = k_value
# 写入 V:
value_cache[block_id][offset] = v_value
```

使用 `reshape_and_cache` 或 FlashAttention/FlashInfer 的专用 kernel 批量写入。

### 3.4 读取方式 (Decode)

Decode 时，通过 block table 收集所有 block 的 K/V:

```python
# FlashAttention decode:
# 读取 block_table[req_id] 指向的所有 block
# K: [num_blocks, block_size, num_kv_heads, head_size]
# V: [num_blocks, block_size, num_kv_heads, head_size]
# Paged Attention kernel 自动处理跨 block 的注意力计算
```

---

## 4. MLA (Multi-head Latent Attention)

> 适用模型: Kimi Linear 的 MLA 层、DeepSeekV2/V3/V4
>
> Spec: `MLAAttentionSpec` → `FullAttentionSpec` → `AttentionSpec`
>
> Block 分配器: `FullAttentionManager` (与标准 attention 共享)

### 4.1 一个 Block 存什么

MLA 的核心优化: **不存储完整的 K 和 V**，只存储一个**压缩后的潜在向量**。

```
标准 Attention:  K [head_size] + V [head_size_v]  = 2 个张量
MLA:            compressed_kv [kv_lora_rank + qk_rope_head_dim]  = 1 个张量
```

**page_size_bytes 计算:**

```python
# vllm/v1/kv_cache_interface.py:340-354
@property
def real_page_size_bytes(self) -> int:
    if self.cache_dtype_str == "fp8_ds_mla":
        if self.model_version == "deepseek_v4":
            # 硬编码: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B/token
            return self.storage_block_size * 584
        # DeepSeekV3.2: kv_lora_rank=512 + qk_rope_head_dim=64 = 656B/token
        return self.block_size * 656
    return (
        self.storage_block_size
        * self.num_kv_heads    # MLA: num_kv_heads = 1 (MQA)
        * self.head_size       # head_size = kv_lora_rank + qk_rope_head_dim
        * get_dtype_size(self.dtype)
    )
```

**Block 布局:**

```
Block 布局: [block_size, head_size]
            ↑ tokens   ↑ kv_lora_rank + qk_rope_head_dim

注意: 没有 K/V 分离! 没有 num_kv_heads 维度! (MLA 是 MQA: num_kv_heads=1)
```

**数据写入:**

```python
# 通过 concat_and_cache_mla 写入
# 将 kv_c_normed (压缩 KV) 和 k_pe (RoPE 位置编码) 拼接:
# [kv_lora_rank | qk_rope_head_dim]
# 一次写入一个 token 位置
ops.concat_and_cache_mla(
    kv_c_normed,       # [num_tokens, kv_lora_rank]
    k_pe,              # [num_tokens, qk_rope_head_dim]
    kv_cache,          # [num_blocks, block_size, head_size]
    slot_mapping,
)
```

**示例 (Kimi Linear MLA, bf16):**
```
kv_lora_rank = 512
qk_rope_head_dim = 64
head_size = 576
num_kv_heads = 1
block_size = 16

page_size = 16 * 1 * 576 * 2 = 18,432 bytes ≈ 18 KB
```

### 4.2 MLA vs 标准 Attention 内存对比

假设 `block_size=16`, `bf16`:

| 模型 | head_size | num_kv_heads | page_size | 压缩比 |
|------|-----------|-------------|-----------|--------|
| 标准 GQA (Llama-70B) | 128 | 8 | 16×8×256×2 = **64 KB** | 1× |
| 标准 GQA (Qwen3-Next) | 256 | 2 | 16×2×512×2 = **32 KB** | 1× |
| MLA (Kimi Linear) | 576 (=512+64) | 1 | 16×1×576×2 = **18 KB** | ~1.8× 节省 vs 标准 |
| MLA fp8 (DeepSeekV4) | 584 bytes/token | 1 | 16×584 = **9.2 KB** | ~3.5× 节省 vs 标准 |

### 4.3 Decode 读取方式

MLA 的 decode 与标准 attention 有本质区别:

```
标准 Decode:
  1. 读取 K/V cache (page cache 中)
  2. Q × K^T → attention weights → softmax → × V → output
  3. 使用 FlashAttention decode kernel (PagedAttention)

MLA Decode:
  1. 读取 compressed KV cache (page cache 中)
  2. 从 compressed KV 恢复完整的 K 和 V:
     k_nope = kv_b_proj(compressed_kv[:, :kv_lora_rank])
     k_pe   = compressed_kv[:, kv_lora_rank:]
     v      = kv_b_proj(compressed_kv[:, :kv_lora_rank]) (另一组权重)
  3. 使用 FlashMLA/FlashInfer MLA 等**专用 kernel**
     → 这些 kernel 在 kernel 内部做解压缩 + 注意力计算
```

MLA 的 block 分配器与标准 attention 完全相同 (`FullAttentionManager`)，内存随序列长度线性增长，但因为每个 token 存储的数据量更小，相同显存可以支持更长的序列。

---

## 5. GDN / KDA 线性注意力 (SSM State)

> 适用模型: Qwen3-Next/Qwen3.5 的 linear_attention 层、Kimi Linear 的 KDA 层
>
> Spec: `MambaSpec` → `KVCacheSpec`（不继承 `AttentionSpec`!）
>
> Block 分配器: `MambaManager`

### 5.1 一个 Block 存什么

SSM 层**不存储 KV cache**。每个 block 存储的是**固定大小的状态张量**，与序列位置无关。

```
标准 Attention:  每个 block 存储 block_size 个 token 的 K 和 V
                → 内存随 token 数线性增长

SSM State:      每个 block 存储一组固定大小的 state tensor (conv + recurrent)
                → 内存恒定，与 token 数无关!
```

**page_size_bytes 计算:**

```python
# vllm/v1/kv_cache_interface.py:530-548
@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]   # 各 state 的形状
    dtypes: tuple[torch.dtype]            # 各 state 的数据类型

    @property
    def page_size_bytes(self) -> int:
        return sum(
            prod(shape) * get_dtype_size(dtype)
            for (shape, dtype) in zip(self.shapes, self.dtypes)
        )
```

**GDN (Qwen3-Next/Qwen3.5) — 2 种 State:**

```python
# shapes = (
#     (conv_dim/tp, conv_kernel-1+num_spec),    # conv_state: 1D causal conv 滑动窗口
#     (num_v_heads/tp, head_v_dim, head_k_dim),  # recurrent_state: GDN hidden state
# )
# dtypes = (conv_dtype, ssm_dtype)

# 示例 (Qwen3-Next-80B-A3B, TP=1):
# conv_dim = 128*16*2 + 128*32 = 9216
# conv_state:     (9216, 3) × bf16 = 55,296 bytes
# recurrent_state: (32, 128, 128) × float32 = 524,288 bytes
# page_size = 55,296 + 524,288 = 579,584 bytes ≈ 566 KB
```

**KDA (Kimi Linear) — 4 种 State:**

```python
# shapes = (
#     (proj_size/tp, conv_kernel-1),     # conv_state_q
#     (proj_size/tp, conv_kernel-1),     # conv_state_k
#     (proj_size/tp, conv_kernel-1),     # conv_state_v
#     (num_heads/tp, head_dim, head_dim), # recurrent_state (float32)
# )

# 示例 (Kimi Linear, head_dim=128, num_heads=32, TP=1):
# conv_state_q: (4096, 3) × bf16 = 24,576 bytes
# conv_state_k: (4096, 3) × bf16 = 24,576 bytes
# conv_state_v: (4096, 3) × bf16 = 24,576 bytes
# recurrent:    (32, 128, 128) × float32 = 2,097,152 bytes
# page_size = 2,170,880 bytes ≈ 2.07 MB
```

### 5.2 一个请求需要多少 Block

```python
# vllm/v1/kv_cache_interface.py:550-557
def max_memory_usage_bytes(self, vllm_config) -> int:
    if mamba_cache_mode == "all":
        # 缓存所有 block 边界的 state
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes
    elif mamba_cache_mode == "align":
        # 只保留 2 个 block (当前 + 前一个, 用于 prefix caching 复制)
        return self.page_size_bytes * (2 + num_speculative_blocks)
    else:  # "none"
        # 只保留 1 个 block
        return self.page_size_bytes * (1 + num_speculative_blocks)
```

**关键特征: 内存恒定（在 align/none 模式下），不随序列长度增长!**

| mamba_cache_mode | Block 需求 | 内存 |
|------------------|-----------|------|
| `"none"` | 1 + spec_blocks | `page_size × (1 + spec)` |
| `"align"` | 2 + spec_blocks | `page_size × (2 + spec)` |
| `"all"` | ceil(max_len/block_size) | 与标准 attention 一样线性增长 |

**示例 (GDN, align 模式, 无 spec):**
```
max_blocks_per_req = 2
total_memory = 2 × 566 KB = 1.13 MB

对比 1M token 上下文:
- 标准 Attention: 62,500 × 32 KB = 2,000 MB
- GDN (align):    2 × 566 KB     = 1.13 MB
- 压缩比: ~1,770× !!!
```

这就是线性注意力的核心优势: 无论序列多长，SSM state 的内存占用恒定。

### 5.3 数据写入方式

SSM 的 state 更新与标准 KV cache 完全不同:

```
标准 Attention:
  - Prefill:  写入所有 token 的 K/V 到连续 block
  - Decode:   写入 1 个 token 的 K/V 到当前 block 的下一个 slot

SSM State:
  - Prefill:  chunk kernel 计算完后，将 final_state 写入当前 block
              → 覆盖整个 state (不是追加)
  - Decode:   fused recurrent kernel 直接原地更新 state (inplace)
              → 读取 state → 计算 → 写回同一个位置
```

State 的读写通过 `ssm_state_indices` (block table 的第一列) 索引:

```python
# MambaManager 中 block table 的使用:
state_index = block_table[req_id, 0]   # 只用第一个 block
# conv_state[state_index]   → conv 滑动窗口
# ssm_state[state_index]    → recurrent hidden state
```

### 5.4 Prefix Caching 的区别

```
标准 Attention Prefix Caching:
  1. Prefill "The quick brown fox..." (100 tokens)
  2. 将 100 个 token 的 K/V 存入 7 个 blocks
  3. 后续请求复用这 7 个 blocks (block table 直接指向它们)
  4. 新 token 追加到第 8 个 block

SSM Prefix Caching (align 模式):
  1. Prefill "The quick brown fox..." (100 tokens)
  2. 计算得到 final_state (一个固定大小的张量)
  3. 将 final_state 存入 block[last_idx]
  4. 后续请求:
     - 从 block[last_idx] 复制 state 到新 block
     - 在新 block 上继续 decode
  5. 只需要 2 个 blocks: [前一步 state, 当前 state]
```

**为什么 SSM 需要 "align" 模式?**

因为 SSM 的 state 是**原地更新**的。如果多个请求共享同一个 block 的 state，一个请求的更新会覆盖另一个请求的 state。Align 模式确保:
- 每个 step 最多分配 1 个新 block
- 从 `last_state_block` 复制到新 block，然后在新 block 上更新
- 旧的 `last_state_block` 可以被下一个请求复用

---

## 6. 四种 Attention 的内存对比

### 6.1 单 Block 内存占用

假设 `block_size=16`, `bf16` 模型:

| Attention 类型 | 每块存储内容 | 单块大小 (示例) |
|---------------|-------------|----------------|
| **Standard GQA** | K[16, H_kv, D] + V[16, H_kv, D] | 32 KB (H_kv=2, D=256) |
| **MLA** | compressed_kv[16, kv_lora+rope] | 18 KB (kv_lora=512, rope=64) |
| **GDN (Qwen)** | conv_state + recurrent_state | 566 KB (固定大小) |
| **KDA (Kimi)** | conv_q + conv_k + conv_v + recurrent | 2.07 MB (固定大小, float32 recurrent) |

### 6.2 单请求最大内存 (1M token 上下文)

```
block_size = 16, max_model_len = 1,000,000

Standard GQA:
  blocks = 62,500
  memory = 62,500 × 32 KB = 2,000 MB

MLA:
  blocks = 62,500
  memory = 62,500 × 18 KB = 1,125 MB  (43.8% 节省)

GDN (align mode):
  blocks = 2
  memory = 2 × 566 KB = 1.13 MB  (99.94% 节省!)

KDA (align mode):
  blocks = 2
  memory = 2 × 2.07 MB = 4.14 MB (99.79% 节省!)
```

### 6.3 内存增长特征总结

```
内存占用
  ↑
  │  Standard GQA ╲
  │                  ╲
  │  MLA              ╲
  │                    ╲
  │                      ╲
  │  GDN/KDA ───────────── ─── (水平线: 恒定)
  │
  └──────────────────────────→ 序列长度
```

- **Standard GQA**: O(n)，每个 token 都需要存储 K/V
- **MLA**: O(n)，但每个 token 存储的数据量更小（压缩）
- **GDN/KDA**: O(1)，固定大小的 state，与序列长度无关

---

## 7. 混合模型 (IsHybrid) 的分组与调度

### 7.1 分组逻辑

> 文件: `vllm/v1/core/kv_cache_utils.py:1613-1661`

混合模型（如 Kimi Linear、Qwen3-Next）有多种 Attention 层，vLLM 将相同 Spec 类型的层分为一个 **KV Cache Group**:

```
Kimi Linear (示例):
  Group 0 (FullAttentionSpec):  [layer_0_mla, layer_2_mla, layer_4_mla, ...]
    → 共享 block table, 分配器: FullAttentionManager

  Group 1 (MambaSpec):          [layer_1_kda, layer_3_kda, layer_5_kda, ...]
    → 独立 block table, 分配器: MambaManager

Qwen3-Next (示例):
  Group 0 (FullAttentionSpec):  [layer_3_full, layer_7_full, ...]
    → 共享 block table, 分配器: FullAttentionManager

  Group 1 (MambaSpec):          [layer_0_gdn, layer_1_gdn, layer_2_gdn, ...]
    → 独立 block table, 分配器: MambaManager
```

分组算法 (`_get_kv_cache_groups_uniform_page_size`):
1. 遍历所有层的 `KVCacheSpec`
2. 按 Spec 类型 (`FullAttentionSpec`, `MambaSpec`) 分桶
3. 同类型 Spec 使用 `merge()` 合并为单个 `KVCacheGroupSpec`
4. 每个 group 共享一个 block table 和一个 block pool

### 7.2 Spec 发现机制

> 文件: `vllm/v1/worker/gpu/attn_utils.py:33-41`

```python
def get_kv_cache_spec(vllm_config) -> dict[str, KVCacheSpec]:
    kv_cache_spec = {}
    # 获取所有实现 AttentionLayerBase 的模块
    attn_layers = get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
    for layer_name, attn_module in attn_layers.items():
        # 每个模块自己决定返回什么类型的 Spec
        if spec := attn_module.get_kv_cache_spec(vllm_config):
            kv_cache_spec[layer_name] = spec
    return kv_cache_spec
```

每个 Attention 模块的 `get_kv_cache_spec()` 返回值:

| 模块 | 文件 | 返回的 Spec |
|------|------|------------|
| `Attention` (标准) | `layers/attention/attention.py:538` | `FullAttentionSpec` |
| `MLAAttention` | `layers/attention/mla_attention.py:951` | `MLAAttentionSpec(num_kv_heads=1)` |
| `GatedDeltaNetAttention` (GDN) | `layers/mamba/gdn_linear_attn.py` | `MambaSpec(shapes=(conv, recurrent))` |
| `KimiDeltaAttention` (KDA) | `layers/kda.py` | `MambaSpec(shapes=(conv_q, conv_k, conv_v, recurrent))` |

### 7.3 Attention Backend 匹配

每个 KV Cache Group 有对应的 Attention Backend:

```
Group 0: FullAttentionSpec / MLAAttentionSpec
  → FlashAttention / FlashInfer / FlashMLA 等
  → block table 用于 paged KV cache 访问
  → slot_mapping 定位每个 token 的写入位置

Group 1: MambaSpec
  → GDNAttentionBackend ("GDN_ATTN")
  → block table 用于 state block 索引
  → ssm_state_indices = block_table[:, 0] (只用第一列)
```

### 7.4 Scheduler 的跨组协调

Scheduler 为每个请求在**所有 group** 中同时分配/释放 block:

```python
# 伪代码: 请求到达
for group in kv_cache_groups:
    group.manager.allocate_slots(request)

# 伪代码: 请求完成
for group in kv_cache_groups:
    group.manager.free(request)
```

这确保了不同类型的层在同一个请求上同步分配和释放。例如 Kimi Linear 的一个 prefill 请求会同时在 MLA group 分配 KV cache block，在 KDA group 分配 state block。

---

## 8. Block 分配与回收策略

> 文件: `vllm/v1/core/single_type_kv_cache_manager.py`

### 8.1 FullAttentionManager

```
分配策略:
  - 请求到达时，根据 token 数计算需要的 block 数
  - 从 free pool 中取出 block
  - 每个 token 写入 block 的下一个空 slot

回收策略:
  - 请求完成后，所有 block 归还 free pool
  - Block 的数据不立即清除 (被下一个请求覆盖)

滑动窗口优化:
  - 超出 sliding_window 的旧 block 可以提前释放
  - 使用 SlidingWindowManager 管理
```

### 8.2 MambaManager

```
分配策略 (align 模式):
  - 每个 step 最多分配 1 个新 block
  - 从 last_state_block 复制 state 到新 block
  - 在新 block 上进行 state 更新

分配策略 (none 模式):
  - 只使用 1 个 block, 原地更新 state
  - 不需要复制

回收策略:
  - 上一步的 state block 在复制后归还 free pool
  - 请求完成后归还当前 block

特殊处理:
  - cached_blocks_this_step: 防止同一 step 内跨请求的 cache 污染
  - Speculative decoding: 额外分配 spec_blocks 用于验证链
```

### 8.3 对比

```
Full Attention:
  Time ─────────────────────────────────────────→
  Blocks: [B0][B1][B2][B3][B4][B5][B6][B7]...
           ↑ 随序列增长持续分配
  每个 block 存储 block_size 个 token 的 KV

SSM State (align):
  Time ─────────────────────────────────────────→
  Blocks: [B0] → [B1] → [B0] → [B1] → ...
           ↑ 只在 2 个 block 间交替 (复制+更新)
  每个 block 存储固定大小的 state

SSM State (none):
  Time ─────────────────────────────────────────→
  Blocks: [B0][B0][B0][B0][B0]...
           ↑ 始终使用同一个 block (原地更新)
```

---

## 9. Tensor 布局与物理内存分配

> 文件: `vllm/v1/worker/gpu_model_runner.py:6612-6772`

### 9.1 内存分配流程

```
1. 计算每层的 KVCacheSpec
2. 分组 → KVCacheGroupSpec
3. 计算可用显存 → num_blocks = available_memory / page_size
4. 分配原始 int8 buffer
5. 按 Spec 类型 reshape 为具体 tensor
```

### 9.2 Attention Spec 的 Tensor Reshape

```python
# 标准 Attention / MLA
shape = attn_backend.get_kv_cache_shape(num_blocks, block_size, ...)
# 通常返回 (2, num_blocks, block_size, num_kv_heads, head_size)
# 或 MLA: (num_blocks, block_size, head_size)

tensor = raw_buffer.view(dtype).reshape(shape)
```

### 9.3 MambaSpec 的 Tensor Reshape

```python
# MambaSpec: 一个原始 buffer 被切分为多个 state tensor
num_element_per_page = page_size_bytes // dtype_size

for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
    target_shape = (num_blocks, *shape)
    # 每个 block 内部 state 是连续的
    # block 之间的 stride = num_element_per_page
    target_stride = (num_element_per_page, *stride_for_shape(shape))

    tensor = torch.as_strided(
        raw_buffer.view(dtype),
        size=target_shape,
        stride=target_stride,
        storage_offset=offset,
    )
```

**示例 (GDN, Qwen3-Next):**

```
Raw buffer: [int8] of size num_blocks × page_size_bytes

切分为:
  conv_state: as_strided → (num_blocks, 9216, 3)     bf16
  ssm_state:  as_strided → (num_blocks, 32, 128, 128)  float32
              ↑ offset = conv_state 的总字节数
```

**示例 (KDA, Kimi Linear):**

```
Raw buffer: [int8] of size num_blocks × page_size_bytes

切分为:
  conv_state_q: as_strided → (num_blocks, 4096, 3)     bf16
  conv_state_k: as_strided → (num_blocks, 4096, 3)     bf16
  conv_state_v: as_strided → (num_blocks, 4096, 3)     bf16
  recurrent:    as_strided → (num_blocks, 32, 128, 128) float32
```

这 4 个 tensor 共享同一个连续的原始 buffer，通过不同的 offset 和 stride 视图访问。

---

## 10. 文件导航索引

### Spec 定义

| 文件路径 | 说明 |
|---------|------|
| `vllm/v1/kv_cache_interface.py` | `KVCacheSpec`, `FullAttentionSpec`, `MLAAttentionSpec`, `MambaSpec` 等所有 Spec 定义 |
| `vllm/model_executor/layers/attention_layer_base.py` | `AttentionLayerBase.get_kv_cache_spec()` 基类接口 |
| `vllm/model_executor/layers/attention/attention.py:538-582` | 标准 Attention 的 Spec 生成 |
| `vllm/model_executor/layers/attention/mla_attention.py:951` | MLA 的 Spec 生成 |
| `vllm/model_executor/layers/mamba/abstract.py:43-59` | MambaBase (GDN/KDA/Mamba) 的 Spec 生成 |

### 内存分配与调度

| 文件路径 | 说明 |
|---------|------|
| `vllm/v1/core/kv_cache_utils.py` | `get_kv_cache_groups()`, `resolve_kv_cache_block_sizes()`, 内存预算计算 |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` 顶层管理器 |
| `vllm/v1/core/kv_cache_coordinator.py` | `HybridKVCacheCoordinator` 混合模型协调器 |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `FullAttentionManager`, `MambaManager` 具体分配器 |

### Model Runner

| 文件路径 | 说明 |
|---------|------|
| `vllm/v1/worker/gpu/attn_utils.py` | `get_kv_cache_spec()` — 遍历所有层收集 Spec |
| `vllm/v1/worker/gpu_model_runner.py:7034-7064` | `GPUModelRunner.get_kv_cache_spec()` |
| `vllm/v1/worker/gpu_model_runner.py:6612-6772` | Tensor 物理内存分配与 reshape |

### Backend

| 文件路径 | 说明 |
|---------|------|
| `vllm/v1/attention/backends/gdn_attn.py` | GDN SSM backend (KDA/GDN 共用) |
| `vllm/v1/attention/backend.py` | Attention 基类, `do_kv_cache_update` |
| `vllm/model_executor/layers/attention/mla_attention.py:1197` | MLA cache shape 定义 |

### State 形状与类型计算

| 文件路径 | 说明 |
|---------|------|
| `vllm/model_executor/layers/mamba/mamba_utils.py:51-267` | `MambaStateDtypeCalculator`, `MambaStateShapeCalculator` |
