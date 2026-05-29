# vLLM KV Cache 全生命周期：从 Spec 到 Tensor 的完整链路

> 追踪 KV Cache 从模型定义到 GPU tensor 分配、初始化、调度使用、清零的全过程。
>
> 核对基准：`vllm` 仓库 `glm5_next` 分支。
>
> 重点覆盖混合架构模型（如 GLM5-Next：MLA + Indexer + KDA/Mamba）在 KV cache 系统中
> 遇到的独特路径，以及 vLLM 如何处理多种不同 page size 的 KV cache spec。

## 目录

1. [全局概览](#1-全局概览)
2. [Stage 1：KV Cache Spec 生成](#2-stage-1kv-cache-spec-生成)
3. [Stage 2：KV Cache 分组](#3-stage-2kv-cache-分组)
4. [Stage 3：内存规划与配置生成](#4-stage-3内存规划与配置生成)
5. [Stage 4：Worker 侧 Tensor 分配与初始化](#5-stage-4worker-侧-tensor-分配与初始化)
6. [Stage 5：调度中的 KV Cache 分配与释放](#6-stage-5调度中的-kv-cache-分配与释放)
7. [Stage 6：KV Cache Block 清零](#7-stage-6kv-cache-block-清零)
8. [混合架构的 KV Cache 处理（GLM5-Next 案例）](#8-混合架构的-kv-cache-处理glm5-next-案例)
9. [代码阅读顺序](#9-代码阅读顺序)

---

## 1. 全局概览

KV cache 的生命周期横跨 EngineCore 和 Worker 两个进程，分为 **6 个阶段**：

```
┌──────────────────────────────────────────────────────────────────────────┐
│  EngineCore 进程                                                         │
│                                                                          │
│  Stage 1: Spec 生成                                                      │
│     │  模型的每个 AttentionLayerBase 子模块声明自己需要什么类型的 KV cache    │
│     │  MLA → MLAAttentionSpec, Indexer → MLAAttentionSpec                │
│     │  KDA/Mamba → MambaSpec                                             │
│     ▼                                                                    │
│  Stage 2: 分组                                                           │
│     │  按类型和 page size 将所有 layer 的 spec 分组                        │
│     │  同类型的 → UniformTypeKVCacheSpecs (MLA + Indexer 一组)            │
│     │  不同类型的 → 独立 group (MambaSpec 各自一组)                        │
│     ▼                                                                    │
│  Stage 3: 内存规划                                                       │
│     │  根据可用显存计算 num_blocks，生成 KVCacheConfig                     │
│     │  每个 group/layer 分配独立的 tensor                                  │
│     │  ═══════════ 跨进程传递 KVCacheConfig ═══════════                   │
│     ▼                                                                    │
│  Worker 进程 (GPU)                                                       │
│                                                                          │
│  Stage 4: Tensor 分配                                                    │
│     │  根据 KVCacheConfig 分配 GPU tensor                                 │
│     │  reshape 成 attention backend 要求的形状                            │
│     │  绑定到 static_forward_context 的各 layer                           │
│     ▼                                                                    │
│  Stage 5: 调度中的分配与释放                                              │
│     │  Scheduler 在每步调度时为请求分配/释放 KV cache blocks               │
│     ▼                                                                    │
│  Stage 6: Block 清零                                                     │
│        新分配的 block 需要清零，通过 Triton kernel 高效完成                │
└──────────────────────────────────────────────────────────────────────────┘
```

### KV Cache Spec 类型体系

```
KVCacheSpec (抽象基类)
  │  kv_cache_interface.py:94
  │  ├── page_size_bytes (property)    # 一个 page 的字节数
  │  └── max_memory_usage_bytes()      # 最大内存使用量
  │
  ├── AttentionSpec (中间基类)
  │     kv_cache_interface.py:143
  │     ├── num_kv_heads, head_size, dtype, block_size
  │     └── page_size_bytes = real_page_size + padding
  │
  │     ├── FullAttentionSpec
  │     │     kv_cache_interface.py:187
  │     │     └── 标准全注意力，支持 sliding_window / attention_chunk_size
  │     │
  │     └── MLAAttentionSpec (FullAttentionSpec 的子类)
  │           kv_cache_interface.py:337
  │           └── MLA 注意力（含 compress_ratio, cache_dtype_str 等）
  │               ├── DeepseekV2MLAAttention 的 kv_cache → MLAAttentionSpec
  │               └── DeepseekV32IndexerCache 的 kv_cache → MLAAttentionSpec
  │                   （page size 与主 MLA 不同）
  │
  ├── HiddenStateCacheSpec (MLAAttentionSpec 的子类)
  │     kv_cache_interface.py:399
  │     └── 隐藏状态缓存，用于 extract_hidden_states
  │
  ├── MambaSpec
  │     kv_cache_interface.py:563
  │     ├── shapes, dtypes, mamba_type, mamba_cache_mode
  │     ├── page_size_bytes = sum(prod(shape) * dtype_size)  ← 固定值，与 block_size 无关
  │     └── KDA (KimiDeltaAttention) 产生此 spec
  │
  └── UniformTypeKVCacheSpecs (包装器)
        kv_cache_interface.py:667
        ├── kv_cache_specs: dict[str, KVCacheSpec]  ← 多个同类型的 per-layer spec
        ├── page_size_bytes = sum(spec.page_size_bytes)  ← 所有内层 spec 之和
        ├── is_uniform_type()  ← 检查是否所有内层 spec 类型一致
        └── from_specs()  ← 工厂方法：如果所有 spec 类型一致则创建，否则返回 None
```

### 关键区别：不同 Spec 的 page_size_bytes 计算

| Spec 类型 | page_size_bytes 计算 | 与 block_size 的关系 |
|---|---|---|
| `FullAttentionSpec` | `block_size * num_kv_heads * head_size * dtype_size` | **正比于 block_size** |
| `MLAAttentionSpec` | 同上（可能含 compress 对齐 padding） | **正比于 block_size** |
| `MambaSpec` | `sum(prod(shape) * dtype_size)` — 状态张量的总大小 | **与 block_size 无关** |
| `UniformTypeKVCacheSpecs` | `sum(内层 spec.page_size_bytes)` | 取决于内层 spec |

这意味着 MLA 和 MambaSpec 的 page size 不能通过调整 block_size 来统一——这是混合架构
遇到的核心难题。

---

## 2. Stage 1：KV Cache Spec 生成

### 入口

**文件**: `vllm/v1/engine/core.py:232`

```
EngineCore._initialize_kv_caches()
  │
  ├── model_executor.get_kv_cache_specs()
  │     └── 遍历所有 Worker，收集每个 Worker 的所有 layer 的 KVCacheSpec
  │
  ├── model_executor.determine_available_memory()
  │     └── 每个 Worker 做 memory profiling，返回可用显存
  │
  └── get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)
        └── 生成最终的 KVCacheConfig（Stage 2-3）
```

### 模型侧 Spec 声明

每个 `AttentionLayerBase` 子类通过 `get_kv_cache_spec()` 声明自己的 KV cache 需求：

**文件**: `vllm/model_executor/layers/attention_layer_base.py:29`

```python
class AttentionLayerBase(ABC):
    @abstractmethod
    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec | None:
        ...
```

vLLM 在模型初始化时将所有 `AttentionLayerBase` 实例注册到 `static_forward_context`：

```
Model.__init__()
  │
  ├── 创建 N 个 TransformerLayer
  │     ├── Self-Attention (AttentionLayerBase)
  │     │     ├── MLA 类型 → get_kv_cache_spec() → MLAAttentionSpec
  │     │     ├── Indexer.k_cache (AttentionLayerBase) → get_kv_cache_spec() → MLAAttentionSpec
  │     │     └── KDA (MambaBase → AttentionLayerBase) → get_kv_cache_spec() → MambaSpec
  │     └── FFN / MoE
  │
  └── static_forward_context = {layer_name: module, ...}
        收集所有 AttentionLayerBase 子类，供后续 KV cache spec 查询
```

### GLM5-Next 的三种 Spec

```
Glm5NextModel
  │
  ├── layers[i].self_attn (Glm5NextMLAAttention, 继承 DeepseekV2MLAAttention)
  │     │
  │     ├── kv_cache (AttentionLayerBase)
  │     │     └── get_kv_cache_spec() → MLAAttentionSpec
  │     │         page_size_bytes = block_size * qk_nope_head_dim * dtype_size
  │     │         例如：block_size=64, head_dim=512, bf16 → 65536 bytes
  │     │
  │     └── indexer.k_cache (DeepseekV32IndexerCache, 继承 AttentionLayerBase)
  │           └── get_kv_cache_spec() → MLAAttentionSpec
  │               page_size_bytes = block_size * indexer_heads * indexer_head_dim * dtype_size
  │               例如：block_size=64, 不同的 head 配置 → 55296 bytes
  │
  └── layers[j].self_attn (KimiDeltaAttention, 继承 MambaBase → AttentionLayerBase)
        └── get_kv_cache_spec() → MambaSpec
            page_size_bytes = sum of fixed state tensor sizes
            例如：固定值 → 6336 bytes
```

### Spec 收集调用栈

```
ModelExecutor.get_kv_cache_specs()
  │
  └── 对每个 Worker:
        │
        └── get_kv_cache_spec_from_vllm_config(vllm_config)
              │  kv_cache_interface.py (或 gpu_model_runner.py 中的等效函数)
              │
              ├── get_layers_from_vllm_config(vllm_config, AttentionLayerBase)
              │     返回 dict[layer_name, AttentionLayerBase]
              │
              └── 对每个 layer:
                    layer.get_kv_cache_spec(vllm_config)
                    → 返回 KVCacheSpec 或 None

最终产物: dict[str, KVCacheSpec]
例如 GLM5-Next:
  {
    "layers.0.self_attn":          MLAAttentionSpec(page_size=65536),
    "layers.0.self_attn.indexer":  MLAAttentionSpec(page_size=55296),
    "layers.1.self_attn":          MambaSpec(page_size=6336),
    ...
  }
```

### Spec 生成时序图

```
EngineCore                         ModelExecutor                    Worker (GPU)
   │                                    │                               │
   │ _initialize_kv_caches()            │                               │
   │───────────────────────────────────>│                               │
   │                                    │ get_kv_cache_specs()          │
   │                                    │──────────────────────────────>│
   │                                    │                               │ 遍历 static_forward_context
   │                                    │                               │ 对每个 AttentionLayerBase:
   │                                    │                               │   layer.get_kv_cache_spec()
   │                                    │                               │   → KVCacheSpec
   │                                    │<──────────────────────────────│
   │                                    │ dict[str, KVCacheSpec]        │
   │                                    │                               │
   │                                    │ determine_available_memory()  │
   │                                    │──────────────────────────────>│
   │                                    │                               │ 执行 warmup forward
   │                                    │                               │ 计算剩余显存
   │                                    │<──────────────────────────────│
   │                                    │ available_memory: int         │
   │<───────────────────────────────────│                               │
   │ kv_cache_specs + available_memory  │                               │
```

---

## 3. Stage 2：KV Cache 分组

### 入口

**文件**: `vllm/v1/core/kv_cache_utils.py:2031`

```
get_kv_cache_configs() (行 2031)
  │
  ├── 合并所有 Worker 的 kv_cache_specs
  │
  ├── get_kv_cache_groups(vllm_config, merged_kv_cache_specs)  (行 2083)
  │     └── 将所有 layer 按 spec 类型和 page size 分组
  │
  ├── auto-fit max_model_len (如果 original_max_model_len == -1)
  │
  ├── _check_enough_kv_cache_memory()  (行 2092)
  │
  └── get_kv_cache_config_from_groups()  (行 2107)
        └── 根据 groups + available_memory 生成 KVCacheConfig
```

### get_kv_cache_groups() 分组策略

**文件**: `vllm/v1/core/kv_cache_utils.py:1660`

这是一个关键的分发函数，按优先级尝试不同的分组策略：

```
get_kv_cache_groups(vllm_config, kv_cache_spec)
  │
  ├── [1] disable_hybrid_kv_cache_manager?
  │     └── unify_hybrid_kv_cache_specs() — 强制统一所有 spec
  │
  ├── [2] is_kv_cache_type_attention_free?
  │     └── 返回 [] — 无 KV cache
  │
  ├── [3] is_kv_cache_spec_uniform? — 所有 layer 的 spec 完全相同
  │     └── _get_kv_cache_groups_uniform_spec()  (行 988)
  │         一个 group，所有 layer 共享同一个 spec
  │         例：标准 LLaMA, Qwen
  │
  ├── [4] UniformTypeKVCacheSpecs.from_specs()? — 所有 layer 类型相同但参数不同
  │     └── _get_kv_cache_groups_uniform_type()  (行 988)
  │         一个 group，用 UniformTypeKVCacheSpecs 包装不同的 per-layer spec
  │         例：DeepseekV3 (MLA + Indexer 都是 MLAAttentionSpec)
  │
  ├── [5] group_and_unify_kv_cache_specs()? — DeepseekV4 多组 UniformTypeKVCacheSpecs
  │     └── _get_kv_cache_groups_uniform_groups()  (行 1019)
  │         多个 group，每个是 UniformTypeKVCacheSpecs
  │         例：DeepseekV4
  │
  └── [6] 混合类型路径 — 有 MambaSpec + AttentionSpec
        │
        ├── 分离 hidden_specs, mamba_specs, filtered_spec
        │
        ├── filtered_spec (去掉 Mamba/Hidden):
        │     ├── UniformTypeKVCacheSpecs.from_specs(filtered_spec)?
        │     │     └── _get_kv_cache_groups_uniform_type()  ← MLA+Indexer 打包
        │     └── 否则 → unify_kv_cache_spec_page_size() → _get_kv_cache_groups_uniform_page_size()
        │
        ├── hidden_specs 按 common_page 对齐后加入
        │
        └── mamba_specs 各自独立加入，不做 page 对齐
              例：GLM5-Next (MLA + Indexer + KDA)
```

### GLM5-Next 的分组结果

```
kv_cache_spec = {
  "layers.0.self_attn":          MLAAttentionSpec(page=65536),    # MLA
  "layers.0.self_attn.indexer":  MLAAttentionSpec(page=55296),    # Indexer
  "layers.1.self_attn":          MambaSpec(page=6336),            # KDA
  "layers.2.self_attn":          MLAAttentionSpec(page=65536),    # MLA
  "layers.2.self_attn.indexer":  MLAAttentionSpec(page=55296),    # Indexer
  ...
}

分组结果:
  Group 0: UniformTypeKVCacheSpecs
    ├── "layers.0.self_attn":          MLAAttentionSpec(page=65536)
    ├── "layers.0.self_attn.indexer":  MLAAttentionSpec(page=55296)
    ├── "layers.2.self_attn":          MLAAttentionSpec(page=65536)
    ├── "layers.2.self_attn.indexer":  MLAAttentionSpec(page=55296)
    └── ...（所有 MLA + Indexer 层）
    page_size_bytes = sum(所有内层 spec) = 65536 + 55296 + ...

  Group 1: MambaSpec
    └── "layers.1.self_attn": MambaSpec(page=6336)
    page_size_bytes = 6336

  Group 2: MambaSpec
    └── "layers.3.self_attn": MambaSpec(page=6336)
    page_size_bytes = 6336

  ...
```

### 分组时序图

```
get_kv_cache_groups()
  │
  │  kv_cache_spec: dict[str, KVCacheSpec]
  │
  ├── is_kv_cache_spec_uniform?  ─── No（有不同类型）
  │
  ├── UniformTypeKVCacheSpecs.from_specs(all)?  ─── No（MLAAttentionSpec ≠ MambaSpec）
  │
  ├── group_and_unify_kv_cache_specs?  ─── No（有 MambaSpec）
  │
  ├── [分离阶段]
  │     ├── hidden_specs = {}  (GLM5-Next 无此类型)
  │     ├── mamba_specs = {"layers.1.self_attn": MambaSpec, ...}
  │     └── filtered_spec = {"layers.0.self_attn": MLA, "layers.0...indexer": MLA, ...}
  │
  ├── UniformTypeKVCacheSpecs.from_specs(filtered_spec)?  ─── Yes!
  │     │  filtered_spec 全是 MLAAttentionSpec（MLA 和 Indexer 都是子类）
  │     └── Group 0 = UniformTypeKVCacheSpecs(filtered_spec)
  │
  ├── hidden_specs 为空，跳过
  │
  └── mamba_specs:
        ├── Group 1 = KVCacheGroupSpec(["layers.1.self_attn"], MambaSpec)
        ├── Group 2 = KVCacheGroupSpec(["layers.3.self_attn"], MambaSpec)
        └── ...
```

---

## 4. Stage 3：内存规划与配置生成

### 总览

**文件**: `vllm/v1/core/kv_cache_utils.py:2031`

```
get_kv_cache_configs()
  │
  ├── 合并所有 Worker 的 kv_cache_specs
  ├── get_kv_cache_groups() → global_kv_cache_groups
  ├── _project_kv_cache_groups_to_worker() → 按每个 Worker 的 layer 过滤
  │
  ├── [auto-fit] _auto_fit_max_model_len()  (行 2084)
  │     └── 二分搜索最大 model_len 使 KV cache 放得下
  │
  ├── [override] _pool_bytes_per_block() 计算实际 bytes_per_block  (行 2074)
  │
  ├── _check_enough_kv_cache_memory()  (行 2092)
  │     ├── get_needed_memory() → _max_memory_usage_bytes_from_groups()
  │     └── 如果不够 → 估计可行 max_model_len 并 raise ValueError
  │
  └── 对每个 Worker:
        get_kv_cache_config_from_groups(vllm_config, groups, available_memory)
        → KVCacheConfig

  最终: min(num_blocks) 跨所有 Worker 同步
```

### get_kv_cache_config_from_groups() 分配策略

**文件**: `vllm/v1/core/kv_cache_utils.py:1243`

```
get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)
  │
  ├── [路径 A] 单个 UniformTypeKVCacheSpecs group
  │     │  (行 1257-1279)
  │     ├── num_blocks = available_memory // group.page_size_bytes
  │     └── 每个 layer 独立 tensor: size = per_layer_page_size * num_blocks
  │
  ├── [路径 B] 所有 groups 都是 UniformTypeKVCacheSpecs (DeepseekV4)
  │     │  (行 1280-1288)
  │     └── _get_kv_cache_config_deepseek_v4()
  │
  ├── [路径 C] 混合 UniformTypeKVCacheSpecs + 其他类型 (GLM5-Next)
  │     │  (行 1289-1316)
  │     ├── total_page_size = sum(group.page_size_bytes for all groups)
  │     ├── num_blocks = available_memory // total_page_size
  │     └── 每个 group 内每个 layer 独立 tensor
  │         ├── UniformTypeKVCacheSpecs: per_layer spec.page_size_bytes * num_blocks
  │         └── 其他 (MambaSpec): spec.page_size_bytes * num_blocks
  │
  └── [路径 D] 通用路径 — 统一 page size，共享 tensor
        │  (行 1317-1343)
        ├── group_size = max(len(group.layer_names))
        ├── page_size = get_uniform_page_size(all groups)  ← 必须一致
        └── group_size 个共享 tensor，每个被同位置 layers 共享
```

### 混合类型内存计算（GLM5-Next 路径 C）

```
假设：
  Group 0 (UniformTypeKVCacheSpecs): page_size_bytes = 120832  (65536+55296 per MLA+Indexer pair)
  Group 1 (MambaSpec):              page_size_bytes = 6336
  Group 2 (MambaSpec):              page_size_bytes = 6336
  ...

  available_memory = 55_583_554_560 bytes (约 51.71 GiB)

计算：
  total_page_size = 120832 + N * 6336
  num_blocks = 55_583_554_560 // total_page_size

  Group 0 tensor 布局：
    "layers.0.self_attn":          65536 * num_blocks bytes
    "layers.0.self_attn.indexer":  55296 * num_blocks bytes
    "layers.2.self_attn":          65536 * num_blocks bytes
    "layers.2.self_attn.indexer":  55296 * num_blocks bytes
    ...

  Group 1 tensor 布局：
    "layers.1.self_attn": 6336 * num_blocks bytes

  Group 2 tensor 布局：
    "layers.3.self_attn": 6336 * num_blocks bytes
```

### _max_memory_usage_bytes_from_groups() — 内存需求计算

**文件**: `vllm/v1/core/kv_cache_utils.py:1809`

```
_max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups)
  │
  ├── [空] 返回 0
  │
  ├── [A] 单个 UniformTypeKVCacheSpecs group
  │     └── sum(spec.max_memory_usage_bytes() for each per-layer spec)
  │
  ├── [B] 所有 groups 都是 UniformTypeKVCacheSpecs (DeepseekV4)
  │     └── 按共享 tensor layout 计算，含 padding
  │
  ├── [C] 混合类型 (GLM5-Next)
  │     └── 独立计算每个 group 的 max_memory_usage_bytes 然后求和
  │         UniformTypeKVCacheSpecs: sum(per-layer max_memory_usage_bytes)
  │         MambaSpec: spec.max_memory_usage_bytes(vllm_config)
  │
  └── [D] 通用路径
        └── group_size * uniform_page_size * sum(blocks_needed)
```

### _pool_bytes_per_block() — Override 场景

**文件**: `vllm/v1/core/kv_cache_utils.py:908`

```
_pool_bytes_per_block(kv_cache_groups)
  │
  ├── [A] 单个 UniformTypeKVCacheSpecs → group.page_size_bytes
  ├── [B] 全 UniformTypeKVCacheSpecs → DeepseekV4 layout 计算
  ├── [C] 混合类型 → sum(group.page_size_bytes for all groups)
  └── [D] 通用 → group_size * uniform_page_size
```

### 内存规划时序图

```
get_kv_cache_configs()
  │
  │  输入: kv_cache_specs (每 worker), available_memory (每 worker)
  │
  ├── 合并 specs → merged_kv_cache_specs
  │
  ├── get_kv_cache_groups()
  │     └── 返回 global_kv_cache_groups
  │
  ├── 对每个 worker: _project_kv_cache_groups_to_worker()
  │     └── 按 worker 的 layer_names 过滤 groups
  │
  ├── [auto-fit] _auto_fit_max_model_len()
  │     └── 二分搜索，每步调 _max_memory_usage_bytes_from_groups()
  │
  ├── [override] 调整 available_memory
  │     └── _pool_bytes_per_block() * num_gpu_blocks_override
  │
  ├── _check_enough_kv_cache_memory()
  │     ├── _max_memory_usage_bytes_from_groups() → needed_memory
  │     └── 如果 needed > available → 估计 max_len 并报错
  │
  ├── 对每个 worker: get_kv_cache_config_from_groups()
  │     └── 返回 KVCacheConfig {num_blocks, kv_cache_tensors, kv_cache_groups}
  │
  └── min(num_blocks) 同步所有 worker
       └── 按 min/max 比例缩小 tensor size
```

---

## 5. Stage 4：Worker 侧 Tensor 分配与初始化

### 入口

**文件**: `vllm/v1/worker/gpu_worker.py:539`

```
GPUWorker.initialize_from_config(kv_cache_config)  (行 539)
  │
  ├── cache_config.num_gpu_blocks = kv_cache_config.num_blocks
  ├── ensure_kv_transfer_initialized()
  │
  ├── [CuMem 模式] CuMemAllocator.use_memory_pool(tag="kv_cache")
  │     └── model_runner.initialize_kv_cache(kv_cache_config)
  │
  ├── model_runner.initialize_kv_cache(kv_cache_config)  (行 7104)
  │     ├── initialize_attn_backend()        → 创建 attention backends, attn_groups
  │     ├── initialize_mamba_ssu_backend()
  │     ├── prepare_kernel_block_sizes()      → 每个 group 的 kernel block size
  │     ├── initialize_metadata_builders()    → per-group metadata builders
  │     ├── may_reinitialize_input_batch()
  │     └── initialize_kv_cache_tensors()     → 分配 GPU tensor
  │
  └── model_runner._init_kv_zero_meta()  (行 571)  ← CuMem pool 外部
        └── 预计算 KV cache block 清零所需的 segment 地址
```

### initialize_attn_backend() — Attention Group 创建

**文件**: `vllm/v1/worker/gpu_model_runner.py:6517`

这一步将 `KVCacheGroupSpec` 转换为 `AttentionGroup`，每个 group 的 per-layer spec
被提取出来：

```
initialize_attn_backend(kv_cache_config)
  │
  ├── 对每个 kv_cache_group:
  │     │
  │     ├── get_attn_backends_for_group(kv_cache_group_spec)
  │     │     │
  │     │     ├── 对每个 layer_name:
  │     │     │     ├── layers[layer_name].get_attn_backend()
  │     │     │     └── 提取 per-layer spec:
  │     │     │           if UniformTypeKVCacheSpecs:
  │     │     │             layer_kv_cache_spec = uniform_spec.kv_cache_specs[layer_name]
  │     │     │           else:
  │     │     │             layer_kv_cache_spec = group_spec.kv_cache_spec
  │     │     │
  │     │     └── 按 (backend_class, layer_spec) 分组
  │     │           MLA layers → 一组 (backend=FlashAttentionMLA, spec=MLAAttentionSpec)
  │     │           Indexer layers → 一组 (backend=..., spec=MLAAttentionSpec)
  │     │           Mamba layers → 一组 (backend=..., spec=MambaSpec)
  │     │
  │     └── create_attn_groups(backends_map, kv_cache_group_id)
  │           └── 每个 (backend, spec) 组合创建一个 AttentionGroup
  │
  └── self.attn_groups[kv_cache_group_id] = [AttentionGroup, ...]
```

### initialize_kv_cache_tensors() — Tensor 分配

**文件**: `vllm/v1/worker/gpu_model_runner.py:7021`

```
initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)
  │
  ├── [uniform 路径] allocate_uniform_kv_caches()
  │     └── 所有 attention layer 共享一个大的连续 tensor
  │
  └── [通用路径]
        │
        ├── _allocate_kv_cache_tensors(kv_cache_config)  (行 6819)
        │     └── 对每个 KVCacheTensor:
        │           tensor = torch.zeros(size, dtype=int8, device=gpu)
        │           对每个 shared_by layer: raw_tensors[layer] = tensor
        │
        └── _reshape_kv_cache_tensors(raw_tensors, kernel_block_sizes)  (行 6860)
              │
              └── 对每个 AttentionGroup:
                    ├── 获取 kernel_block_size
                    └── 对每个 layer:
                          ├── raw_tensor = raw_tensors[layer_name]
                          ├── num_blocks = raw_tensor.numel() // spec.page_size_bytes
                          └── kv_caches[layer] = reshape(raw_tensor, backend 要求的形状)

最终: bind_kv_cache(kv_caches, static_forward_context)
  → 把 tensor 绑定到各 layer 的 .kv_cache 属性
```

### Worker 初始化时序图

```
EngineCore                                    GPUWorker                         GPUModelRunner
   │                                              │                                  │
   │ KVCacheConfig                                │                                  │
   │─────────────────────────────────────────────>│                                  │
   │                                              │ initialize_from_config()         │
   │                                              │─────────────────────────────────>│
   │                                              │                                  │
   │                                              │                                  │ initialize_attn_backend()
   │                                              │                                  │  ├ 创建 AttentionGroup
   │                                              │                                  │  └ 提取 per-layer spec
   │                                              │                                  │
   │                                              │                                  │ prepare_kernel_block_sizes()
   │                                              │                                  │
   │                                              │                                  │ initialize_kv_cache_tensors()
   │                                              │                                  │  ├ torch.zeros() 分配 GPU 内存
   │                                              │                                  │  ├ reshape 到目标形状
   │                                              │                                  │  └ bind_kv_cache() 绑定到 layer
   │                                              │                                  │
   │                                              │ _init_kv_zero_meta()             │
   │                                              │─────────────────────────────────>│
   │                                              │                                  │ KVBlockZeroer.init_meta()
   │                                              │                                  │  └ 预计算 segment 地址
   │                                              │                                  │
   │                                              │<─────────────────────────────────│
   │<─────────────────────────────────────────────│ 完成                             │
```

---

## 6. Stage 5：调度中的 KV Cache 分配与释放

### KVCacheManager

**文件**: `vllm/v1/core/kv_cache_manager.py:110`

```
KVCacheManager
  │
  ├── 维护 free_blocks: 每个 group 的空闲 block 列表
  ├── 维护 req_to_blocks: request_id → 各 group 的 block_ids
  │
  ├── new_step_starts()                    (行 362)
  │     └── 每步调度开始时重置状态
  │
  ├── get_computed_blocks(request)         (行 594)
  │     └── 查找 prefix cache 命中的 blocks
  │
  ├── allocate_slots(request, num_tokens)  (行 236)
  │     └── 为请求分配新的 KV cache blocks
  │         对每个 group: 从 free_blocks 取出需要的 block 数量
  │
  ├── take_new_block_ids()                 (行 882)
  │     └── 返回本轮新分配的 block IDs（需要清零）
  │
  └── free(request)                        (行 429)
        └── 释放请求占用的所有 blocks，归还 free_blocks
```

### Scheduler 中的调用

**文件**: `vllm/v1/core/sched/scheduler.py`

```
Scheduler.schedule()  (行 352)
  │
  ├── kv_cache_manager.new_step_starts()           (行 362)
  │
  ├── Phase 1: RUNNING requests
  │     ├── 计算 num_new_tokens
  │     └── kv_cache_manager.allocate_slots(...)   (行 444)
  │
  ├── Phase 2: WAITING → RUNNING
  │     ├── kv_cache_manager.get_computed_blocks() (行 594)  ← prefix cache
  │     ├── 检查内存是否足够
  │     ├── kv_cache_manager.allocate_slots(...)   (行 721)
  │     └── [preemption] kv_cache_manager.free()   (行 938)
  │
  └── new_block_ids = kv_cache_manager.take_new_block_ids()  (行 882)
        └── 这些 block IDs 传递给 Worker 进行清零

Scheduler.update_from_output()  (行 1303)
  │
  └── 如果请求完成:
        └── kv_cache_manager.free(request)          (行 1862)
```

### 调度中的 Block 管理时序图

```
Scheduler                         KVCacheManager                  Worker
   │                                    │                            │
   │ schedule()                         │                            │
   │───────────────────────────────────>│                            │
   │                                    │ new_step_starts()          │
   │                                    │                            │
   │ [Phase 1: RUNNING]                 │                            │
   │ allocate_slots(running_req)        │                            │
   │───────────────────────────────────>│                            │
   │                                    │ 从 free_blocks 取 blocks   │
   │<───────────────────────────────────│ block_ids                  │
   │                                    │                            │
   │ [Phase 2: WAITING]                 │                            │
   │ get_computed_blocks(new_req)       │                            │
   │───────────────────────────────────>│                            │
   │                                    │ 查 prefix cache            │
   │<───────────────────────────────────│ cached_block_ids           │
   │                                    │                            │
   │ allocate_slots(new_req)            │                            │
   │───────────────────────────────────>│                            │
   │                                    │ 从 free_blocks 取 blocks   │
   │<───────────────────────────────────│ new_block_ids              │
   │                                    │                            │
   │ take_new_block_ids()               │                            │
   │───────────────────────────────────>│                            │
   │<───────────────────────────────────│ 需要清零的 block IDs       │
   │                                    │                            │
   │ SchedulerOutput (含 block_ids, new_block_ids)                   │
   │────────────────────────────────────────────────────────────────>│
   │                                    │                            │ _update_states()
   │                                    │                            │ _zero_block_ids(new_block_ids)
   │                                    │                            │ model forward
   │                                    │                            │
   │ update_from_output()               │                            │
   │ [如果请求完成]                      │                            │
   │ free(request)                      │                            │
   │───────────────────────────────────>│                            │
   │                                    │ 归还 blocks 到 free_blocks │
```

---

## 7. Stage 6：KV Cache Block 清零

### KVBlockZeroer

**文件**: `vllm/v1/worker/utils.py:80`

新分配的 KV cache blocks 必须清零以避免残留数据干扰推理。

```
KVBlockZeroer
  │
  ├── init_meta()        (行 99)
  │     一次性预计算所有 attention layer 的 KV cache segment 地址
  │     按 page_size_el 分组存储（支持混合 page size）
  │
  └── zero_block_ids()   (行 196)
        对每个 page_size_el 分组，调用 Triton kernel 清零
```

### init_meta() 详细流程

```
KVBlockZeroer.init_meta(attn_groups_iter, kernel_block_sizes, cache_dtype, ...)
  │
  │  初始化:
  │    seen_ptrs: set[int]          ← 去重（共享 tensor 的 layers）
  │    page_groups: dict[int, tuple[list[int], int]]
  │       ↑ page_size_el → (seg_addrs, kernel_block_el)
  │
  ├── 对每个 AttentionGroup:
  │     ├── spec = group.kv_cache_spec
  │     ├── if not isinstance(spec, FullAttentionSpec): continue  ← 跳过 MambaSpec
  │     ├── kernel_bs = kernel_block_sizes[group.kv_cache_group_id]
  │     ├── ratio = spec.block_size // kernel_bs
  │     ├── block_dim = backend.get_kv_cache_block_dim(...)
  │     │
  │     └── 对每个 layer_name in group.layer_names:
  │           ├── kv = static_forward_context[layer_name].kv_cache
  │           ├── dp = kv.data_ptr(); if dp in seen_ptrs: continue
  │           ├── cur_page_el = (kv.stride(block_dim) * el_size // 4) * ratio
  │           │
  │           ├── page_groups[cur_page_el] 如果不存在则创建
  │           │
  │           └── 遍历 outer_dims，计算所有 segment 的绝对地址
  │                 seg_addrs.append(dp + off_bytes)
  │
  └── 对每个 page_size_el 分组:
        ├── blk_size = min(largest_power_of_2_divisor(page_size_el), 1024)
        └── _meta_list.append((seg_addrs_tensor, page_size_el, blk_size, n_segs))
```

### zero_block_ids() 清零流程

```
KVBlockZeroer.zero_block_ids(block_ids)
  │
  ├── if not block_ids or not _meta_list: return
  │
  ├── 准备 block_ids:
  │     ids_pinned[:n_blocks] = block_ids
  │     ids_gpu.copy_(ids_pinned)
  │
  └── 对每个 (seg_addrs, page_size_el, blk_size, n_segs) in _meta_list:
        grid = (n_blocks * n_segs * (page_size_el // blk_size),)
        _zero_kv_blocks_kernel[grid](
            seg_addrs, ids_gpu, n_blocks,
            N_SEGS=n_segs, PAGE_SIZE_EL=page_size_el, BLOCK_SIZE=blk_size,
        )
```

### 清零时序图

```
GPUModelRunner                  KVBlockZeroer               Triton Kernel
     │                               │                          │
     │ _init_kv_zero_meta()          │                          │
     │──────────────────────────────>│                          │
     │                               │ init_meta()              │
     │                               │  ├ 遍历 attn_groups      │
     │                               │  ├ 收集 seg_addrs        │
     │                               │  ├ 按 page_size_el 分组  │
     │                               │  └ _meta_list 准备就绪   │
     │<──────────────────────────────│                          │
     │                               │                          │
     │ (每步调度后)                    │                          │
     │ _zero_block_ids(new_ids)      │                          │
     │──────────────────────────────>│                          │
     │                               │ zero_block_ids()         │
     │                               │  ├ copy block_ids to GPU │
     │                               │  │                       │
     │                               │  ├ [page_group_0]:       │
     │                               │  │  grid launch ──────────────────>│
     │                               │  │  MLA segs, 65536 el   │  清零 MLA blocks
     │                               │  │                       │
     │                               │  ├ [page_group_1]:       │
     │                               │  │  grid launch ──────────────────>│
     │                               │  │  Indexer segs, 55296 el│  清零 Indexer blocks
     │                               │  │                       │
     │                               │  └ [可能有更多分组]       │
     │<──────────────────────────────│                          │
```

---

## 8. 混合架构的 KV Cache 处理（GLM5-Next 案例）

### 为什么 DeepseekV3/V4 没有遇到这些问题

| 模型 | MLA | Indexer | KDA/Mamba | 遇到的分组路径 |
|---|---|---|---|---|
| DeepseekV3 | ✅ | ✅ | ❌ | 全部 MLAAttentionSpec → `UniformTypeKVCacheSpecs.from_specs()` → 单组 → 路径 A |
| DeepseekV4 | ✅ | ✅ | ❌ | 多组 `UniformTypeKVCacheSpecs` → `group_and_unify_kv_cache_specs()` → 路径 B |
| Jamba/Zamba2 | ❌ | ❌ | ✅ | attention + mamba，但 attention 是标准 `FullAttentionSpec` → 通用路径 D |
| **GLM5-Next** | ✅ | ✅ | ✅ | **首次** `UniformTypeKVCacheSpecs` + `MambaSpec` 混合 → 路径 C |

**关键洞察**：GLM5-Next 是第一个同时包含 `UniformTypeKVCacheSpecs`（MLA+Indexer）和
`MambaSpec`（KDA）的模型。vLLM 的 KV cache pipeline 中多处假设"所有 group 的
page_size 一致"或"只有 UniformTypeKVCacheSpecs groups"，这些假设在混合场景下被打破。

### 需要修改的四个位置

```
kv_cache_utils.py 中混合类型处理的四处改动:

  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. get_kv_cache_config_from_groups() — 行 1289-1316             │
  │    新增 elif 分支：混合 groups 的 tensor 分配                    │
  │    每个 layer 独立 tensor，num_blocks = mem // total_page_size  │
  ├─────────────────────────────────────────────────────────────────┤
  │ 2. _pool_bytes_per_block() — 行 930-935                        │
  │    新增 elif 分支：混合 groups 的 bytes_per_block                │
  │    return sum(group.page_size_bytes)                            │
  ├─────────────────────────────────────────────────────────────────┤
  │ 3. _max_memory_usage_bytes_from_groups() — 行 1850-1864        │
  │    新增 elif 分支：混合 groups 的内存需求计算                    │
  │    独立计算每个 group 的 max_memory_usage_bytes 然后求和         │
  ├─────────────────────────────────────────────────────────────────┤
  │ 4. KVBlockZeroer.init_meta() — utils.py 行 99-192              │
  │    将 _meta 从单个元组改为 _meta_list（按 page_size_el 分组）   │
  │    zero_block_ids() 遍历所有分组分别调用 Triton kernel          │
  └─────────────────────────────────────────────────────────────────┘
```

### 完整的混合类型 KV Cache 初始化流程

```
EngineCore._initialize_kv_caches()
  │
  ├── get_kv_cache_specs()
  │     └── GLM5-Next: MLAAttentionSpec × N + MLAAttentionSpec(Indexer) × N + MambaSpec × M
  │
  ├── determine_available_memory()
  │     └── 约 51.71 GiB
  │
  └── get_kv_cache_configs()
        │
        ├── get_kv_cache_groups()
        │     ├── 分离 MambaSpec → mamba_specs
        │     ├── filtered = MLA + Indexer (全是 MLAAttentionSpec)
        │     ├── UniformTypeKVCacheSpecs.from_specs(filtered) → 成功!
        │     │     所有 MLA + Indexer 打包成一个 UniformTypeKVCacheSpecs group
        │     └── 每个 MambaSpec 独立成 group
        │
        ├── _max_memory_usage_bytes_from_groups()
        │     └── [路径 C] 混合类型：独立求和
        │         UniformTypeKVCacheSpecs: sum(per-layer max_mem)
        │         + MambaSpecs: sum(spec.max_mem)
        │
        ├── _check_enough_kv_cache_memory()
        │
        └── get_kv_cache_config_from_groups()
              └── [路径 C] 混合类型：
                  total_page_size = sum(all groups' page_size_bytes)
                  num_blocks = available_memory // total_page_size
                  每个 layer 独立 tensor

Worker.initialize_from_config()
  │
  ├── initialize_kv_cache()
  │     ├── initialize_attn_backend()
  │     │     UniformTypeKVCacheSpecs group 被拆分为多个 AttentionGroup:
  │     │       AttentionGroup(backend=FlashAttentionMLA, spec=MLAAttentionSpec(65536))
  │     │       AttentionGroup(backend=..., spec=MLAAttentionSpec(55296))  ← Indexer
  │     │       AttentionGroup(backend=..., spec=MambaSpec(6336))  ← KDA
  │     │
  │     ├── initialize_kv_cache_tensors()
  │     │     每个 layer 独立 tensor，按 spec.page_size_bytes * num_blocks 分配
  │     │
  │     └── bind_kv_cache() → 绑定到各 layer
  │
  └── _init_kv_zero_meta()
        └── KVBlockZeroer.init_meta()
              ├── 遍历 AttentionGroup，跳过非 FullAttentionSpec (MambaSpec)
              ├── 收集 MLA 和 Indexer 的 segment 地址
              ├── 按 page_size_el 分组:
              │     group 0: MLA segs, page_el=6336
              │     group 1: Indexer segs, page_el=55296
              └── _meta_list = [group_0_meta, group_1_meta]
```

---

## 9. 代码阅读顺序

建议按以下顺序阅读 KV cache 相关代码：

### 第一遍：理解数据结构

1. **`vllm/v1/kv_cache_interface.py`** — KVCacheSpec 类型体系
   - 先看 `KVCacheSpec` 基类（行 94）
   - 再看 `FullAttentionSpec`（行 187）和 `MLAAttentionSpec`（行 337）
   - 然后 `MambaSpec`（行 563）和 `UniformTypeKVCacheSpecs`（行 667）
   - 注意各类型的 `page_size_bytes` 和 `max_memory_usage_bytes` 计算差异

2. **`vllm/model_executor/layers/attention_layer_base.py`** — Spec 声明接口
   - `AttentionLayerBase.get_kv_cache_spec()`（行 29）

3. **`vllm/model_executor/layers/mamba/abstract.py`** — Mamba Spec 生成
   - `MambaBase.get_kv_cache_spec()`（行 44）返回 `MambaSpec`

### 第二遍：理解分组和配置

4. **`vllm/v1/core/kv_cache_utils.py`** — 分组与配置核心
   - `get_kv_cache_configs()`（行 2031）— 顶层入口
   - `get_kv_cache_groups()`（行 1660）— 分组策略分发
   - `get_kv_cache_config_from_groups()`（行 1243）— Tensor 分配
   - `_max_memory_usage_bytes_from_groups()`（行 1809）— 内存需求
   - `_pool_bytes_per_block()`（行 908）— Override 计算

### 第三遍：理解 Worker 侧初始化

5. **`vllm/v1/worker/gpu_worker.py`** — Worker 入口
   - `initialize_from_config()`（行 539）

6. **`vllm/v1/worker/gpu_model_runner.py`** — Model Runner
   - `initialize_kv_cache()`（行 7104）
   - `initialize_attn_backend()`（行 6517）
   - `initialize_kv_cache_tensors()`（行 7021）
   - `_reshape_kv_cache_tensors()`（行 6860）

7. **`vllm/v1/worker/utils.py`** — Block 清零
   - `KVBlockZeroer`（行 80）
   - `AttentionGroup` dataclass（行 227）

### 第四遍：理解调度中的使用

8. **`vllm/v1/core/kv_cache_manager.py`** — Block 管理
   - `allocate_slots()`（行 236）
   - `free()`（行 429）

9. **`vllm/v1/core/sched/scheduler.py`** — 调度器中的 KV cache 使用
   - `schedule()` 中的 `allocate_slots` 调用
   - `update_from_output()` 中的 `free` 调用

---

## 附录

### 压缩完整时序图

```
EngineCore               kv_cache_utils              GPUWorker              GPUModelRunner           KVBlockZeroer
   │                         │                          │                        │                        │
   │ _initialize_kv_caches() │                          │                        │                        │
   │────────────────────────>│                          │                        │                        │
   │                         │                          │                        │                        │
   │ get_kv_cache_specs()    │                          │                        │                        │
   │──────────────────────────────────────────────────────────────────────────>│                        │
   │                         │                          │  get_kv_cache_spec()   │                        │
   │                         │                          │  for each layer        │                        │
   │<──────────────────────────────────────────────────────────────────────────│                        │
   │ dict[str, KVCacheSpec]  │                          │                        │                        │
   │                         │                          │                        │                        │
   │ determine_available_memory()                       │                        │                        │
   │──────────────────────────────────────────────────────────────────────────>│                        │
   │<──────────────────────────────────────────────────────────────────────────│ available_memory (GiB) │
   │                         │                          │                        │                        │
   │ get_kv_cache_configs()  │                          │                        │                        │
   │────────────────────────>│                          │                        │                        │
   │                         │ get_kv_cache_groups()    │                        │                        │
   │                         │  ├ 分离 MambaSpec        │                        │                        │
   │                         │  ├ UniformTypeKVCacheSpecs(filtered)               │                        │
   │                         │  └ 每 MambaSpec 独立 group                         │                        │
   │                         │                          │                        │                        │
   │                         │ _check_enough_kv_cache_memory()                    │                        │
   │                         │  └ _max_memory_usage_bytes_from_groups()           │                        │
   │                         │                          │                        │                        │
   │                         │ get_kv_cache_config_from_groups()                  │                        │
   │                         │  └ num_blocks = mem // total_page_size             │                        │
   │<────────────────────────│                          │                        │                        │
   │ KVCacheConfig           │                          │                        │                        │
   │                         │                          │                        │                        │
   │─────────────────────────────────────────────────────────────────────────────────────────────────────>│
   │ KVCacheConfig           │                          │ initialize_from_config │                        │
   │                         │                          │───────────────────────>│                        │
   │                         │                          │                        │ initialize_kv_cache()  │
   │                         │                          │                        │  ├ init_attn_backend   │
   │                         │                          │                        │  ├ alloc tensors       │
   │                         │                          │                        │  └ bind_kv_cache       │
   │                         │                          │                        │                        │
   │                         │                          │ _init_kv_zero_meta()   │                        │
   │                         │                          │───────────────────────>│                        │
   │                         │                          │                        │────────────────────────>│
   │                         │                          │                        │  init_meta()           │
   │                         │                          │                        │   ├ 按 page_size_el    │
   │                         │                          │                        │   │  分组 seg_addrs    │
   │                         │                          │                        │   └ _meta_list 就绪    │
   │                         │                          │                        │<────────────────────────│
   │                         │                          │<───────────────────────│                        │
   │<─────────────────────────────────────────────────────────────────────────────────────────────────────│
   │                         │                          │ 初始化完成              │                        │
```

### 关键数据对象流转

```
AttentionLayerBase.get_kv_cache_spec()          [attention_layer_base.py:29]
    │  → 返回每层需要的 KVCacheSpec
    ▼
dict[str, KVCacheSpec]                         [每层一个 spec]
    │  例: {"layers.0.self_attn": MLAAttentionSpec, "layers.1.self_attn": MambaSpec, ...}
    ▼
get_kv_cache_groups()                          [kv_cache_utils.py:1660]
    │  → 分组为 list[KVCacheGroupSpec]
    ▼
list[KVCacheGroupSpec]                         [分组结果]
    │  例: [UniformTypeKVCacheSpecs(MLA+Indexer), MambaSpec(KDA), ...]
    ▼
get_kv_cache_config_from_groups()              [kv_cache_utils.py:1243]
    │  → 加上 available_memory 计算出 KVCacheConfig
    ▼
KVCacheConfig                                  [配置结果]
    │  ├─ num_blocks: int                      # 总 block 数
    │  ├─ kv_cache_tensors: list[KVCacheTensor] # 每个 tensor 的 size 和 shared_by
    │  └─ kv_cache_groups: list[KVCacheGroupSpec]
    │
    │  ═══════════ 传递到 Worker ═══════════
    ▼
initialize_kv_cache()                          [gpu_model_runner.py:7104]
    │  → 分配 GPU tensor, reshape, 绑定到 layer
    ▼
GPU Tensor (每层一个)                           [绑定到 layer.kv_cache]
    │
    ▼
KVBlockZeroer.init_meta()                      [utils.py:99]
    │  → 预计算清零所需 metadata
    ▼
_meta_list: list[tuple[seg_addrs, page_size_el, blk_size, n_segs]]
    │  按 page_size_el 分组，供后续清零使用
```
