# vLLM KV Cache 全生命周期：从 Spec 到 Tensor 的完整链路

> 追踪 KV Cache 从模型定义到 GPU tensor 分配、初始化、调度使用、清零的全过程。
>
> 核对基准：`vllm` 仓库 tag **`v0.25.1`**（commit `752a3a504`）。所有 `file:line` 均按该 tag 核对。
>
> **与 [execution-core.md](./execution-core.md) 的分工**：本文讲 KV cache 的**加载期**——spec 怎么收集、
> group 怎么分、`num_blocks` 怎么算、GPU tensor 怎么分配和清零（Stage 1–4、6）；运行时的
> allocate_slots 十步算法、prefix 命中/LRU、抢占、hybrid coordinator、spec→manager 全景在
> execution-core.md §3.3，本文 Stage 5 只留调用点与指针。先读本文建立"KV cache 是怎么规划和物化的"，
> 再读 execution-core.md 看"每步调度是怎么使用它的"。

## 目录

1. [全局概览](#1-全局概览)
2. [Stage 1：KV Cache Spec 生成](#2-stage-1kv-cache-spec-生成)
3. [Stage 2：KV Cache 分组](#3-stage-2kv-cache-分组)
4. [Stage 3：内存规划与配置生成](#4-stage-3内存规划与配置生成)
5. [Stage 4：Worker 侧 Tensor 分配与初始化](#5-stage-4worker-侧-tensor-分配与初始化)
6. [Stage 5：调度中的 KV Cache 分配与释放（摘要）](#6-stage-5调度中的-kv-cache-分配与释放摘要)
7. [Stage 6：KV Cache Block 清零](#7-stage-6kv-cache-block-清零)
8. [代码阅读顺序](#8-代码阅读顺序)

---

## 1. 全局概览

KV cache 的生命周期横跨 EngineCore 和 Worker 两个进程，分为 **6 个阶段**：

```
┌──────────────────────────────────────────────────────────────────────────┐
│  EngineCore 进程                                                         │
│                                                                          │
│  Stage 1: Spec 生成              core.py:240 _initialize_kv_caches       │
│     │  每个 AttentionLayerBase 子模块声明自己需要什么类型的 KV cache        │
│     │  MLA/Indexer → MLAAttentionSpec, 全注意力 → FullAttentionSpec      │
│     │  Mamba/KDA/Linear → MambaSpec                                      │
│     ▼                                                                    │
│  Stage 2: 分组                   kv_cache_utils.py:1697 get_kv_cache_groups│
│     │  早返回级联：uniform → uniform_type → DSv4 分组 → 通用混合           │
│     │  同类型等槽的层打包成 UniformTypeKVCacheSpecs 一组                   │
│     ▼                                                                    │
│  Stage 3: 内存规划               kv_cache_utils.py:2005 get_kv_cache_configs│
│     │  按可用显存算 num_blocks，生成 KVCacheConfig                        │
│     │  （内存检查 / auto-fit max_model_len / min 跨 worker 同步）          │
│     │  ═══════════ 跨进程传递 KVCacheConfig ═══════════                   │
│     ▼                                                                    │
│  Worker 进程 (GPU)                                                       │
│                                                                          │
│  Stage 4: Tensor 分配            gpu_model_runner.py:7405 initialize_kv_cache│
│     │  按 KVCacheConfig 分配 GPU tensor（支持多层共享 / packed 布局）      │
│     │  reshape 成 attention backend 要求的形状                            │
│     │  绑定到 static_forward_context 的各 layer                           │
│     ▼                                                                    │
│  Stage 5: 调度中的分配与释放      kv_cache_manager.py:114（薄门面）         │
│     │  Scheduler 每步经 KVCacheManager 分配/释放 blocks                  │
│     │  → 机制细节见 execution-core.md §3.3                                │
│     ▼                                                                    │
│  Stage 6: Block 清零             worker/utils.py:80 KVBlockZeroer        │
│        含 mamba 层的模型需要清零新分配的 block，Triton kernel 一次 launch  │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.1 KV Cache Spec 类型体系（v0.25.1）

`KVCacheSpec` 是**每层一份**的不可变描述："这层缓存什么、一个 block 多大"。运行时分配行为由
对应的 `SingleTypeKVCacheManager` 决定（spec→manager 注册表与各类 manager 的行为差异见
execution-core.md §3.3.6，本文不重复）。

```
KVCacheSpec (kv_cache_interface.py:100)
  │  ├── page_size_bytes (property, :109)      # 一个 block（一层）的字节数
  │  └── max_memory_usage_bytes() (:122)       # 单请求最大内存
  │
  ├── AttentionSpec (:164)                     # num_kv_heads, head_size, dtype
  │     └── page_size_bytes (:173) = real_page_size + scales + padding
  │
  │     ├── FullAttentionSpec (:206)           # 标准全注意力
  │     │     ├── TQFullAttentionSpec (:337)       # 三值量化
  │     │     ├── MLAAttentionSpec (:363)          # MLA latent（DSv2/V3、Indexer）
  │     │     │     └── HiddenStateCacheSpec (:434) # extract_hidden_states 用
  │     │     └── SinkFullAttentionSpec (:730)     # 带 sink block
  │     ├── RSWASpec (:441)                    # Reference SWA（prefill 全程可见 + 窗口）
  │     ├── ChunkedLocalAttentionSpec (:481)   # 分块局部注意力
  │     ├── SlidingWindowSpec (:518)           # 滑动窗口
  │     │     └── SlidingWindowMLASpec (:590)      # 滑窗 + MLA 缓存格式
  │     ├── EncoderOnlyAttentionSpec (:710)    # 不需要 KV cache
  │     └── CrossAttentionSpec (:717)          # 交叉注意力（encoder-decoder）
  │
  ├── MambaSpec (:669)                         # 状态缓存（Mamba1/2/Linear/GDN/ShortConv）
  │     ├── shapes, dtypes, mamba_type, mamba_cache_mode, num_speculative_blocks
  │     └── page_size_bytes (:678) = Σ prod(shape)×dtype_size（有 page_size_padded 时取 padded）
  │
  └── UniformTypeKVCacheSpecs (:784)           # 包装器：多个"等 token 槽数"的 per-layer spec
        ├── kv_cache_specs: dict[str, KVCacheSpec]
        ├── page_size_bytes (:795) = Σ 内层 spec.page_size_bytes
        ├── is_uniform_type (:806) / from_specs (:821)
        └── KVCacheGroupSpec (:905) / KVCacheTensor (:893) 见 Stage 3
```

### 1.2 关键区别：不同 Spec 的 page_size_bytes

| Spec 类型 | page_size_bytes 计算 | 与 block_size 的关系 |
|---|---|---|
| `FullAttentionSpec` | `block_size × num_kv_heads × (head_size+head_size_v) × dtype_size` | **正比于 block_size** |
| `MLAAttentionSpec` | `block_size × 1 × head_size(=576) × dtype_size`（无 V 那一份） | **正比于 block_size** |
| `MambaSpec` | `Σ prod(shape) × dtype_size` — 状态张量总大小 | **与 block_size 无关**（可靠 `page_size_padded` 对齐到 attention page，见 Stage 2） |
| `UniformTypeKVCacheSpecs` | `Σ 内层 spec.page_size_bytes` | 取决于内层 spec |

MLA 和 MambaSpec 的 page size 天然不一致——混合架构的分组和内存规划（Stage 2/3）很大程度上
就是在解决这个问题。

---

## 2. Stage 1：KV Cache Spec 生成

### 2.1 入口：EngineCore._initialize_kv_caches

**文件**: `vllm/v1/engine/core.py:240`

```
EngineCore._initialize_kv_caches()                                core.py:240
  │
  ├─ register_all_kvcache_specs()                                 :244
  │     把内置 spec→manager 映射注册进 KVCacheSpecRegistry
  │     （register_all_kvcache_specs，single_type_kv_cache_manager.py:1499）
  │
  ├─ kv_cache_specs = model_executor.get_kv_cache_specs()         :247
  │     └── collective_rpc 到所有 Worker，收集每层 KVCacheSpec
  │
  ├─ [non_causal 模型检查] 可能禁用 chunked prefill / prefix caching  :254-269
  │
  ├─ available_memory = model_executor.determine_available_memory()  :283
  │     └── 每个 Worker 做 memory profiling（warmup forward 后量剩余显存）
  │
  ├─ kv_cache_configs = get_kv_cache_configs(vllm_config, specs, memory)  :294
  │     └── Stage 2 + Stage 3，见下两章
  │
  ├─ [auto-fit 收缩了 max_model_len] collective_rpc("update_max_model_len")  :303
  │
  ├─ scheduler_kv_cache_config = generate_scheduler_kv_cache_config(...)   :305
  ├─ 设置 num_gpu_blocks / block_size / kv_cache_capacity                 :306-314
  │
  └─ model_executor.initialize_from_config(kv_cache_configs)         :321
        └── Stage 4：Worker 按配置分配 GPU tensor
```

Executor 侧（`v1/executor/abstract.py`）：`get_kv_cache_specs()`（`:149`）和
`determine_available_memory()`（`:146`）都是 `collective_rpc` 广播到全部 Worker。

### 2.2 模型侧 Spec 声明

每个 `AttentionLayerBase` 子类通过 `get_kv_cache_spec()` 声明自己的 KV cache 需求：

**文件**: `vllm/model_executor/layers/attention_layer_base.py:29`

```python
class AttentionLayerBase(ABC):
    @abstractmethod
    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec | None:
        ...
```

Worker 侧的收集实现：

```
GPUWorker.get_kv_cache_spec()                       gpu_worker.py:679
  └─ GPUModelRunner.get_kv_cache_spec()             gpu_model_runner.py:7561
       ├─ get_layers_from_vllm_config(vllm_config, AttentionLayerBase)   config/vllm.py:2313
       │     返回 dict[layer_name, AttentionLayerBase]（即 static_forward_context）
       ├─ 跳过 kv-sharing 层（复用别的层的 KV，自己不产 spec）
       ├─ 对每个 layer: layer.get_kv_cache_spec(vllm_config) → KVCacheSpec | None
       └─ 按 attention backend 给 spec 打 indexes_kv_by_block_stride 标记
```

> 旧版本里有一个独立的 `get_kv_cache_spec_from_vllm_config()` 函数，v0.25.1 已删除，
> 逻辑并入 `GPUModelRunner.get_kv_cache_spec()`。

各后端的典型声明：

- **`Attention`**（`layers/attention/attention.py:616`）→ `FullAttentionSpec`
- **`MLAAttention`**（`layers/attention/mla_attention.py:1000`）→ `MLAAttentionSpec`，
  `num_kv_heads=1`、cache 只含 latent（详见 execution-core.md §3.3.8）
- **`DeepseekV32IndexerCache`**（`models/deepseek_v2.py:615`，top-k 稀疏 MLA 的 indexer）
  → `MLAAttentionSpec`（`:630-631`），但 page size 与主 MLA 不同
- **`MambaBase`**（`layers/mamba/abstract.py:44`）→ `MambaSpec`：从
  `cache_config.mamba_block_size` 和 `mamba_page_size_padded` 构造——**mamba 的 page 对齐
  发生在 spec 生成期**，这是 Stage 2 混合分组能成立的前提

产物：`dict[layer_name, KVCacheSpec]`，例如 DeepSeek-V3.2：

```python
{
  "model.layers.0.self_attn":          MLAAttentionSpec(...),   # 主 MLA
  "model.layers.0.self_attn.indexer":  MLAAttentionSpec(...),   # indexer，page 不同
  ...
}
```

> 本地 GLM-5Next 模型（不在 v0.25.1 内）同理：MLA + indexer 产 `MLAAttentionSpec`，
> KDA（KimiDeltaAttention，MambaBase 子类）产 `MambaSpec`。

### 2.3 Spec 生成时序图

```
EngineCore                         ModelExecutor                    Worker (GPU)
   │                                    │                               │
   │ _initialize_kv_caches()            │                               │
   │───────────────────────────────────>│                               │
   │                                    │ get_kv_cache_specs()          │
   │                                    │──────────────────────────────>│
   │                                    │                               │ get_layers_from_vllm_config
   │                                    │                               │ 对每个 AttentionLayerBase:
   │                                    │                               │   layer.get_kv_cache_spec()
   │                                    │<──────────────────────────────│
   │                                    │ dict[str, KVCacheSpec]        │
   │                                    │                               │
   │                                    │ determine_available_memory()  │
   │                                    │──────────────────────────────>│
   │                                    │                               │ warmup forward + 量剩余显存
   │                                    │<──────────────────────────────│
   │                                    │ available_memory: int         │
   │<───────────────────────────────────│                               │
   │ kv_cache_specs + available_memory  │                               │
```

---

## 3. Stage 2：KV Cache 分组

### 3.1 get_kv_cache_groups：早返回级联

**文件**: `vllm/v1/core/kv_cache_utils.py:1697`

输入 `{layer_name: KVCacheSpec}`，输出 `list[KVCacheGroupSpec]`。分组是一条**早返回级联**，
从最特殊到最一般：

```
get_kv_cache_groups(vllm_config, kv_cache_spec)                  kv_cache_utils.py:1697
  │
  ├─ [1] disable_hybrid_kv_cache_manager?                        :1710
  │     └─ unify_hybrid_kv_cache_specs()  (def :1403)
  │          强制把 SWA/chunked-local 等同化成 FullAttentionSpec，只剩一种 spec
  │
  ├─ [2] is_kv_cache_type_attention_free?  (def :1103)           :1713
  │     └─ return []                                             :1716
  │
  ├─ [3] is_kv_cache_spec_uniform?  (def :894)                   :1718
  │     │  所有层的 spec 完全相同
  │     └─ _get_kv_cache_groups_uniform_spec()  (:1722, def :1001)
  │          → 1 个 group          例：LLaMA / Qwen 等绝大多数模型
  │
  ├─ [4] UniformTypeKVCacheSpecs.from_specs()?  (def :821)       :1723
  │     │  所有层"等 token 槽数"（类型可不同，如 MLA + indexer）
  │     └─ _get_kv_cache_groups_uniform_type()  (:1727, def :1018)
  │          → 1 个 group          例：DeepSeek-V3.2（MLA + Indexer）
  │
  ├─ [5] group_and_unify_kv_cache_specs()?  (def :1499)          :1728
  │     │  含 SlidingWindowMLASpec（即 DeepSeek-V4）：可拆成多个 UniformType 组
  │     ├─ _get_kv_cache_groups_uniform_groups()  (:1733, def :1572)
  │     └─ _annotate_eagle_groups_deepseek_v4()   (:1734, def :1673)
  │
  └─ [6] 通用混合路径                                            :1736-1763
        │  有 MambaSpec / 多种不等价的 attention spec
        ├─ 抽出 HiddenStateCacheSpec 层                           :1739-1745
        ├─ unify_kv_cache_spec_page_size()  (:1751, def :1049)
        │     把剩余所有 spec 的 page 统一（整倍放大 block_size，或对
        │     indexes_kv_by_block_stride 的 attention spec 做 padding）
        ├─ _get_kv_cache_groups_uniform_page_size()  (:1752, def :1108)
        │     按相同 spec 成桶 → 以 min(各类型层数)（1.5× 启发式可取 max）
        │     为 group_size 跨步等大切分，不足补 padding
        └─ hidden 层按 common_page 对齐后各自成组加回              :1755-1761
```

| 场景 | 命中分支 | 典型模型 |
|------|----------|----------|
| 纯 full-attn | [3] uniform_spec | Llama / Qwen 等绝大多数 |
| MLA + indexer（等槽） | [4] uniform_type | DeepSeek-V3.2 sparse MLA |
| full + 多种 SWA（等槽） | [5] grouped（DSv4） | DeepSeek-V4 |
| full/MLA + Mamba/线性（不等槽） | [6] 通用混合 | Jamba / Nemotron-H / KimiLinear |

### 3.2 通用混合路径：mamba 的 page 在哪里对齐

v0.25.1 的混合分组**不抽出 MambaSpec**——对齐责任前移到两个更早的时机：

1. **config/平台期**：`platforms/interface.py:907-910` 把 `cache_config.mamba_page_size_padded`
   预设为 attention 的 page size；
2. **spec 生成期**：`MambaBase.get_kv_cache_spec`（`mamba/abstract.py:44`）构造 `MambaSpec`
   时带上 `page_size_padded`，于是 `MambaSpec.page_size_bytes`（`kv_cache_interface.py:678-686`）
   直接返回 padded 值。

所以分组期的 `unify_kv_cache_spec_page_size` 对 mamba 是 no-op，mamba 层与 attention 层一起走
`_get_kv_cache_groups_uniform_page_size`：所有相同 `MambaSpec` 的层归一桶，再按 group_size
**等大切分**。最终每个 group 层数相等（padding 买单），这直接决定了 Stage 3 的内存核算公式。

> **版本注记**：更早的版本曾把混合模型拆成"filtered_spec（attention）走 UniformType + 每个
> mamba 层各自独立成组、不做 page 对齐"，`get_kv_cache_config_from_groups` 里也有专门的
> "混合 UniformType + 其他"分支；v0.25.1 已删除这套路径，统一走等大组方案。
> 本地 glm5_next 工作树另有一套"mamba 均衡分组 + 逐层内存核算"的改动（不属 v0.25.1），
> 见 execution-core.md §3.3.14 末尾的侧栏。

---

## 4. Stage 3：内存规划与配置生成

### 4.1 顶层：get_kv_cache_configs

**文件**: `vllm/v1/core/kv_cache_utils.py:2005`

```
get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory)   :2005
  │  （per-worker 列表输入，per-worker KVCacheConfig 列表输出）
  │
  ├─ 合并所有 Worker 的 kv_cache_specs → merged_kv_cache_specs    :2040-2050
  ├─ KVCacheSpecRegistry.check_kv_cache_spec_registry             :2056
  │
  ├─ get_kv_cache_groups(vllm_config, merged_specs)               :2060   ← Stage 2
  ├─ _project_kv_cache_groups_to_worker()                         :2066 (def :1963)
  │     按 PP stage 把全局 group 投影到每个 Worker 持有的层
  │
  ├─ [num_gpu_blocks_override]                                    :2076-2085
  │     用 _pool_bytes_per_block()（def :950）反算等效 available_memory
  │
  ├─ [original_max_model_len == -1]                               :2091
  │     _auto_fit_max_model_len()（:2093, def :1899）
  │     二分搜索最大可行 max_model_len，各 worker 取最小值，原地收缩 config
  │
  ├─ 每 worker: _check_enough_kv_cache_memory()                   :2101 (def :733)
  │     调 _max_memory_usage_bytes_from_groups()（def :1801）算需求；
  │     不够则用 _estimate_max_model_len_from_groups()（def :1864）估值后 raise
  │
  ├─ 每 worker: get_kv_cache_config_from_groups()                 :2116 (def :1318)
  │     → KVCacheConfig {num_blocks, kv_cache_tensors, kv_cache_groups}
  │
  ├─ min(num_blocks) 跨所有 worker 同步 + tensor 等比收缩          :2124-2131
  │
  └─ get_kv_cache_capacity()（:1788）打日志（KV token 容量、最大并发）
```

### 4.2 get_kv_cache_config_from_groups：num_blocks 的四种算法

**文件**: `vllm/v1/core/kv_cache_utils.py:1318`

```
get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)   :1318
  │
  ├─ [A] 空 groups（attention-free）                              :1334-1342
  │     num_blocks = 1
  │
  ├─ [B] 单个 UniformTypeKVCacheSpecs group                       :1343-1361
  │     num_blocks = available_memory // group.page_size_bytes
  │     每个 layer 独立 tensor：size = 该层 page_size_bytes × num_blocks
  │     （page_size_bytes 已经过 may_override_num_blocks，def :940）
  │
  ├─ [C] _use_packed_kv_cache_config()（def :1255）为真            :1362-1367
  │     └─ _get_kv_cache_config_packed()（def :1277）              ← DSv4 packed 布局，见 4.4
  │
  └─ [D] 通用路径                                                 :1368-1394
        group_size = max(len(group.layer_names))   ← 等大组假设（Stage 2 的 padding 保证）
        page_size  = get_uniform_page_size(all groups)
        num_blocks = get_num_blocks(...)  (def :972)
                   = available_memory // page_size // group_size
        group_size 个共享 tensor，第 i 个被每组的第 i 层 shared_by
```

**num_blocks 是所有 group 共享的块数**——同一物理 block id 被所有层复用（block_table 跨层共享，
见 execution-core.md §3.3.7）。所以可用的块数必须把"一个 block 跨所有 group/层的总字节"除干净：
分支 B 是 `avail // Σ各层page`，分支 D 是 `avail // (page × 每组层数)`，两者在等大组假设下等价。

### 4.3 内存需求与并发：三个核算函数

| 函数 | 行号 | 算什么 | 公式（通用路径） |
|------|------|--------|------------------|
| `_max_memory_usage_bytes_from_groups` | `:1801-1862` | 单请求最大 KV 占用（内存检查/auto-fit 用） | `group_size × page_size × Σ_g cdiv(g.max_mem, page_size)` |
| `_pool_bytes_per_block` | `:950-969` | 池视角每 block 字节（override 反算用） | `uniform_page_size × max(group_size)` |
| `get_max_concurrency_for_kv_cache_config` | `:919-937` | 最大并发请求数 | `num_blocks / cdiv(max_mem_per_req, bytes_per_block)`，均按 max 组大小 |

三个公式都依赖 Stage 2 的**等大组假设**（padding 保证每组层数相等），所以可以用
`max(group_size)` 一笔算出精确值，无需逐层累乘。

### 4.4 DSv4 packed 布局：多层共享一个物理 tensor

分支 D 说"每层/每槽一个 tensor"。DeepSeek-V4 打破这个默认：所有层（同 page size）的 per-block
KV 数据**打包进同一个物理 tensor**，每层只是这块内存的一个视图。

**启用判定** —— `_use_packed_kv_cache_config`（`kv_cache_utils.py:1255`）：

```
is_dsv4 = 所有 group 都是 UniformTypeKVCacheSpecs
return is_dsv4 or (enable_cross_layers_blocks=True and 多 group)   # 后者是实验 API（issue #42082）
```

**规划** —— `_get_kv_cache_config_packed`（`:1277`，别名 `_get_kv_cache_config_deepseek_v4` `:1315`）：

```
① _bucket_layers_by_page_size(groups)            :1230
     buckets = {page_size: [[layer@slot0], [layer@slot1], ...]}
     同一 (page_size, slot_idx) 的不同 group 的层共享一个 tensor
② total_num_bytes_per_block = Σ page_size × len(slots)
③ num_blocks = available_memory // total_num_bytes_per_block     :1293
④ 每个 slot 一个 KVCacheTensor(
        size=total×num_blocks, shared_by=slot 上所有层,
        offset=块内字节偏移, block_stride=total)                    :1303-1308
```

`KVCacheTensor`（`kv_cache_interface.py:893`）四字段：`size`（`:898`）、`shared_by`（`:899`）、
`offset`（`:900`，packed 块内字节偏移）、`block_stride`（`:901`，packed 每 block 总字节，
`0` = 非 packed）。Worker 侧如何把它们物化成 strided view，见 Stage 4.4。

---

## 5. Stage 4：Worker 侧 Tensor 分配与初始化

### 5.1 入口：GPUWorker.initialize_from_config

**文件**: `vllm/v1/worker/gpu_worker.py:695`

```
GPUWorker.initialize_from_config(kv_cache_config)                gpu_worker.py:695
  │
  ├─ cache_config.num_gpu_blocks = kv_cache_config.num_blocks    :700
  ├─ ensure_kv_transfer_initialized()                            :707
  │
  ├─ with self._maybe_get_memory_pool_context(tag="kv_cache"):   :709
  │     │  （def :231；CUDA 平台需 enable_cumem_allocator 才进
  │     │    CuMemAllocator.use_memory_pool，否则 nullcontext）
  │     └─ model_runner.initialize_kv_cache(kv_cache_config)     :710
  │
  ├─ [可选] init_routed_experts_capturer()                       :713
  │
  └─ [pool 外] if kv_cache_config.needs_kv_cache_zeroing:        :718-722
        model_runner._init_kv_zero_meta()                        ← Stage 6
```

### 5.2 initialize_kv_cache：七步

**文件**: `vllm/v1/worker/gpu_model_runner.py:7405`

```
GPUModelRunner.initialize_kv_cache(kv_cache_config)              gpu_model_runner.py:7405
  │
  ├─ may_add_encoder_only_layers_to_kv_cache_config()            :7419
  ├─ maybe_add_kv_sharing_layers_to_kv_cache_groups()            :7420
  │
  ├─ initialize_attn_backend(kv_cache_config)                    :7421 (def :6799)
  │     │  把 KVCacheGroupSpec 转成 AttentionGroup（见 5.3）
  │     │  嵌套函数：get_attn_backends_for_group (:6826)
  │     │            create_attn_groups (:6870)
  │     └─ self.attn_groups[group_id] = [AttentionGroup, ...]
  │
  ├─ initialize_mamba_ssu_backend()                              :7422
  │     （mamba/ops/ssu_dispatch.py:193）
  │
  ├─ prepare_kernel_block_sizes()                                :7430
  │     （worker/utils.py:331，模块级函数）→ 每个 group 的 kernel block size
  │
  ├─ initialize_metadata_builders()                              :7436 (def :6906)
  │     per-group attention metadata builders
  │
  ├─ may_reinitialize_input_batch()                              :7439 (def :7017)
  │     kernel block size 与 scheduler block size 不同时重建 InputBatch
  │
  └─ initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)  :7440 (def :7322)
        分配 + reshape + 绑定（见 5.4）
```

### 5.3 initialize_attn_backend：AttentionGroup 的创建

**文件**: `vllm/v1/worker/gpu_model_runner.py:6799`

```
initialize_attn_backend(kv_cache_config)                         :6799
  │
  ├─ 对每个 kv_cache_group:
  │     └─ get_attn_backends_for_group(kv_cache_group_spec)      :6826（嵌套 def）
  │           ├─ 对每个 layer_name:
  │           │     ├─ layers[layer_name].get_attn_backend()
  │           │     └─ 提取 per-layer spec:                       :6849-6851
  │           │           if UniformTypeKVCacheSpecs:
  │           │             layer_spec = uniform_spec.kv_cache_specs[layer_name]
  │           │           else:
  │           │             layer_spec = group_spec.kv_cache_spec
  │           └─ 按 (backend_class, layer_spec, num_heads_q) 分组
  │                 MLA layers → 一组（FlashAttentionMLA, MLAAttentionSpec）
  │                 Indexer layers → 一组
  │                 Mamba layers → 一组（MambaSpec）
  │
  ├─ create_attn_groups(backends_map, kv_cache_group_id)         :6870（嵌套 def）
  │     每个 (backend, spec) 组合创建一个 AttentionGroup
  │     （AttentionGroup dataclass：worker/utils.py:223）
  │
  └─ self.attn_groups[kv_cache_group_id] = [AttentionGroup, ...]
```

### 5.4 initialize_kv_cache_tensors：分配 → reshape → 绑定

**文件**: `vllm/v1/worker/gpu_model_runner.py:7322`

```
initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)   :7322
  │
  ├─ [uniform 路径] use_uniform_kv_cache(...) 为真                 :7339-7341
  │     （kv_connector_model_runner_mixin.py:115；仅 KV connector 且
  │      prefer_cross_layer_blocks、单 group、indexes_kv_by_block_stride 时）
  │     └─ allocate_uniform_kv_caches()  （mixin :161）
  │          所有 attention layer 共享一个大的连续 tensor
  │
  └─ [通用路径]
        ├─ _allocate_kv_cache_tensors(kv_cache_config)           :7354 (def :7081)
        │     对每个 KVCacheTensor:
        │       tensor = torch.zeros(size, dtype=int8, device=gpu)
        │       对每个 shared_by layer: raw_tensors[layer] = tensor
        │     [packed] block_stride > 0 时（:7095-7105）：
        │       packed_backing 只 new 一次，所有 packed 层的 raw_tensor 都 alias 它
        │
        ├─ _reshape_kv_cache_tensors(raw_tensors, kernel_block_sizes)  :7357 (def :7133)
        │     对每个 AttentionGroup / layer：
        │       ├─ layer_packing[layer] = (offset, block_stride)   :7158
        │       ├─ num_blocks = raw_tensor.numel() // spec.page_size_bytes
        │       └─ _reshape_attention_kv_cache(...)  （v1/worker/gpu/attn_utils.py:200）
        │            ├─ packed 分支（:214-222）：
        │            │     raw.view(-1, block_stride)[:, offset:offset+page_bytes]
        │            │         .view(dtype).view(shape)   ← 2D view 按列切片
        │            │     （as_strided 只用于 padded-page 分支 :244 和 mamba 分支）
        │            └─ mamba state tensor 走 gpu_model_runner.py:7238-7245
        │
        └─ bind_kv_cache(kv_caches, static_forward_context, ...)  :7369
              （worker/utils.py:462）把 tensor 绑定到各 layer 的 .kv_cache 属性
```

**packed 布局的内存视图**（对应 Stage 3.4 的规划）：

```
packed_backing (int8, total_size = block_stride × num_blocks):
  视作 (num_blocks, block_stride) 的 2D view：
  blk0  [layer0 | layer1 | layer2 | ...]     ← 各层按 offset 列切片
  blk1  [layer0 | layer1 | layer2 | ...]
  ...
  layer_i = view[:, offset_i : offset_i + page_size_i] 再 .view(dtype).view(shape)
```

block 寻址：layer L（offset `o`、page_size `ps`）的 block `b` 起始字节 = `b × block_stride + o`，
长度 `ps`。packed 只改物理分配，**不改 block_table 语义**——每 group/层仍各有自己的 block_table。

### 5.5 Worker 初始化时序图

```
EngineCore                                    GPUWorker                         GPUModelRunner
   │                                              │                                  │
   │ KVCacheConfig                                │                                  │
   │─────────────────────────────────────────────>│                                  │
   │                                              │ initialize_from_config()         │
   │                                              │─────────────────────────────────>│
   │                                              │                                  │ initialize_attn_backend()
   │                                              │                                  │  ├ get_attn_backends_for_group
   │                                              │                                  │  └ create_attn_groups
   │                                              │                                  │ prepare_kernel_block_sizes()
   │                                              │                                  │ initialize_metadata_builders()
   │                                              │                                  │ initialize_kv_cache_tensors()
   │                                              │                                  │  ├ torch.zeros() 分配
   │                                              │                                  │  ├ reshape（含 packed 切片）
   │                                              │                                  │  └ bind_kv_cache()
   │                                              │                                  │
   │                                              │ [needs_kv_cache_zeroing]         │
   │                                              │ _init_kv_zero_meta()             │
   │                                              │─────────────────────────────────>│ KVBlockZeroer(...)（Stage 6）
   │<─────────────────────────────────────────────│ 完成                             │
```

---

## 6. Stage 5：调度中的 KV Cache 分配与释放（摘要）

运行时的完整机制（allocate_slots 十步、prefix 命中与 LRU、free/ref_cnt、抢占、
hybrid coordinator 定点迭代、spec→manager 全景）在 **execution-core.md §3.3**，这里只留
对象关系和调用点，方便对照。

### 6.1 KVCacheManager：薄门面

**文件**: `vllm/v1/core/kv_cache_manager.py:114`

v0.25.1 的 `KVCacheManager` 几乎不存状态——`free_blocks`/`req_to_blocks` 等记账全部下沉到
`KVCacheCoordinator` / `SingleTypeKVCacheManager` / `BlockPool`（三层委托链见
execution-core.md §3.3 开头）。它的方法全是转发：

| 方法 | 行号 | 干什么 | 机制细节 |
|------|------|--------|----------|
| `get_computed_blocks` | `:206` | prefix cache 命中查找 | execution-core §3.3.2 |
| `allocate_slots` | `:248` | 为请求分配/扩容 blocks（含准入判定） | execution-core §3.3.1 |
| `free` | `:466` | 释放请求全部 blocks（ref_cnt--） | execution-core §3.3.3 |
| `take_new_block_ids` | `:637` | 收集本步新分配、需清零的 block ids | Stage 6 |
| `new_step_starts` | `:644` | 每步开始清"本步缓存"标记 | execution-core §3.3.10 |

### 6.2 Scheduler 中的调用点

**文件**: `vllm/v1/core/sched/scheduler.py`

```
Scheduler.schedule()  (:396)
  │
  ├─ kv_cache_manager.new_step_starts()                  :432
  │
  ├─ Phase A: RUNNING requests
  │     └─ kv_cache_manager.allocate_slots(...)          :535   ← 失败触发抢占（execution-core §3.3.5）
  │
  ├─ Phase B: WAITING → RUNNING
  │     ├─ kv_cache_manager.get_computed_blocks(...)     :725   ← prefix cache 命中
  │     └─ kv_cache_manager.allocate_slots(...)          :903   ← 失败停止准入
  │
  └─ new_block_ids_to_zero = kv_cache_manager.take_new_block_ids()   :1080
        （由 needs_kv_cache_zeroing 门控，scheduler.py:296；
         写进 SchedulerOutput.new_block_ids_to_zero，output.py:241）

Scheduler.update_from_output()  (:1499)
  └─ 请求结束/抢占时:
        └─ _free_request_blocks(request)  (def :2130)
              ├─ 无在飞 GPU 写 → kv_cache_manager.free(request)  (:2139)
              └─ 有在飞写 → pop_blocks_for_free 摘下暂存，延迟归还（defer 机制见 execution-core §3.3.5）
```

### 6.3 调度中的 Block 管理时序图

```
Scheduler                         KVCacheManager                  Worker
   │                                    │                            │
   │ schedule()                         │                            │
   │───────────────────────────────────>│ new_step_starts()          │
   │ [Phase A: RUNNING]                 │                            │
   │ allocate_slots(running_req)        │                            │
   │───────────────────────────────────>│ coordinator → BlockPool    │
   │<───────────────────────────────────│ KVCacheBlocks              │
   │ [Phase B: WAITING]                 │                            │
   │ get_computed_blocks(new_req)       │                            │
   │───────────────────────────────────>│ find_longest_cache_hit     │
   │<───────────────────────────────────│ 命中 blocks + tokens       │
   │ allocate_slots(new_req)            │                            │
   │───────────────────────────────────>│ touch 命中 + 分配新 block  │
   │<───────────────────────────────────│ KVCacheBlocks              │
   │ take_new_block_ids()               │                            │
   │───────────────────────────────────>│                            │
   │<───────────────────────────────────│ 待清零 block ids           │
   │ SchedulerOutput (含 block ids, new_block_ids_to_zero)           │
   │────────────────────────────────────────────────────────────────>│
   │                                    │                            │ _update_states()
   │                                    │                            │ _zero_block_ids()（Stage 6）
   │                                    │                            │ model forward
   │ update_from_output()               │                            │
   │ [请求完成] _free_request_blocks     │                            │
   │───────────────────────────────────>│ free → ref_cnt--           │
   │                                    │ （带 hash 的回 LRU 仍可命中）│
```

---

## 7. Stage 6：KV Cache Block 清零

### 7.1 什么时候需要清零：needs_kv_cache_zeroing

v0.25.1 里 block 清零是**按模型类型门控**的：`KVCacheConfig.needs_kv_cache_zeroing`
（`kv_cache_interface.py:943`）= 模型含 mamba 类层。原因：attention 的 KV slot 在被读取前
必然先被写入（读的范围不超过已 compute 的范围），残留数据无害；而 mamba 的 state block 是
递归状态快照，复用前若不清零，旧 state 会污染新序列的递归计算。

门控贯穿三处：

- 初始化：`gpu_worker.py:718-722` 决定是否构造 `KVBlockZeroer`；
- 调度：`scheduler.py:296` 决定 `take_new_block_ids()` 是否收集待清零 block
  （结果进 `SchedulerOutput.new_block_ids_to_zero`，`output.py:241`）；
- 运行时：`GPUModelRunner._update_states` 里 `_zero_block_ids(new_block_ids_to_zero)`
  （`gpu_model_runner.py:1181`，def `:1130`）。

### 7.2 KVBlockZeroer

**文件**: `vllm/v1/worker/utils.py:80`

```
KVBlockZeroer (worker/utils.py:80)
  │
  ├─ __init__(attn_groups_iter, kernel_block_sizes, cache_dtype, ...)   :88
  │     一次性预计算清零所需的 segment 地址：
  │     ├─ 遍历每个 AttentionGroup：
  │     │     ├─ if not isinstance(spec, FullAttentionSpec): continue   ← 跳过 MambaSpec
  │     │     │     （mamba state 由各自 backend 处理，不走这个 kernel）
  │     │     ├─ kernel_bs = kernel_block_sizes[group_id]
  │     │     ├─ ratio = spec.block_size // kernel_bs
  │     │     └─ 对每个 layer：
  │     │           ├─ kv = static_forward_context[layer].kv_cache
  │     │           ├─ data_ptr() 去重（共享 tensor 的层只算一次）
  │     │           └─ 按 stride 算出每个 segment 的绝对地址 seg_addrs
  │     ├─ 单一 page_size_el：所有层 assert 一致（v0.25.1 不再支持多分组）
  │     └─ 产出单个 _meta = (seg_addrs_gpu, page_size_el, blk_size, n_segs)   :170-176
  │
  └─ zero_block_ids(block_ids)   :191
        ├─ block_ids 经 pinned buffer 拷上 GPU
        └─ 单次 launch _zero_kv_blocks_kernel（Triton，utils.py:41）：
              grid = n_blocks × n_segs × (page_size_el // blk_size)
```

> 旧版本的 `init_meta()` 独立方法和"按 page_size_el 分多组、每组一次 launch"的 `_meta_list`
> 结构在 v0.25.1 已删除：预计算并入 `__init__`，全模型只保留一个 `_meta`、一次 launch。

### 7.3 清零时序图

```
GPUWorker / Scheduler              GPUModelRunner                 KVBlockZeroer
     │                                  │                            │
     │ [初始化] _init_kv_zero_meta()     │                            │
     │─────────────────────────────────>│ KVBlockZeroer(...)         │
     │                                  │───────────────────────────>│ __init__: 收集 seg_addrs
     │                                  │                            │  去重、assert 单 page_size
     │                                  │<───────────────────────────│  _meta 就绪
     │                                  │                            │
     │ [每步] SchedulerOutput           │                            │
     │  .new_block_ids_to_zero          │                            │
     │─────────────────────────────────>│ _update_states()           │
     │                                  │  └ _zero_block_ids(ids)    │
     │                                  │───────────────────────────>│ zero_block_ids()
     │                                  │                            │  └ Triton kernel 一次 launch
     │                                  │<───────────────────────────│
```

---

## 8. 代码阅读顺序

建议按以下顺序阅读 KV cache 相关代码（行号均为 v0.25.1）：

### 第一遍：理解数据结构

1. **`vllm/v1/kv_cache_interface.py`** — KVCacheSpec 类型体系
   - 先看 `KVCacheSpec` 基类（`:100`）与 `page_size_bytes`/`max_memory_usage_bytes`（`:109`/`:122`）
   - 再看 `AttentionSpec`（`:164`）、`FullAttentionSpec`（`:206`）、`MLAAttentionSpec`（`:363`）
   - 然后 `MambaSpec`（`:669`，注意 `page_size_padded`）和 `UniformTypeKVCacheSpecs`（`:784`）
   - 最后 `KVCacheGroupSpec`（`:905`）与 `KVCacheTensor`（`:893`，`offset`/`block_stride` 是 packed 的关键）
2. **`vllm/model_executor/layers/attention_layer_base.py`** — Spec 声明接口（`:29`）
3. **`vllm/model_executor/layers/mamba/abstract.py`** — `MambaBase.get_kv_cache_spec()`（`:44`）

### 第二遍：理解分组和配置

4. **`vllm/v1/core/kv_cache_utils.py`** — 分组与配置核心
   - `get_kv_cache_configs()`（`:2005`）— 顶层入口
   - `get_kv_cache_groups()`（`:1697`）— 分组决策级联
   - `_get_kv_cache_groups_uniform_page_size()`（`:1108`）— 通用混合路径的等大切分
   - `get_kv_cache_config_from_groups()`（`:1318`）— num_blocks 四分支
   - `_use_packed_kv_cache_config`（`:1255`）/ `_get_kv_cache_config_packed`（`:1277`）— DSv4 packed
   - `_max_memory_usage_bytes_from_groups()`（`:1801`）、`_pool_bytes_per_block()`（`:950`）、
     `get_max_concurrency_for_kv_cache_config()`（`:919`）— 三个核算函数

### 第三遍：理解 Worker 侧初始化

5. **`vllm/v1/worker/gpu_worker.py`** — `initialize_from_config()`（`:695`）
6. **`vllm/v1/worker/gpu_model_runner.py`** —
   - `initialize_kv_cache()`（`:7405`）七步
   - `initialize_attn_backend()`（`:6799`）
   - `initialize_kv_cache_tensors()`（`:7322`）→ `_allocate_kv_cache_tensors()`（`:7081`）→
     `_reshape_kv_cache_tensors()`（`:7133`）
   - `vllm/v1/worker/gpu/attn_utils.py:200` 的 `_reshape_attention_kv_cache`（packed 分支 `:214-222`）
7. **`vllm/v1/worker/utils.py`** — `KVBlockZeroer`（`:80`）、`AttentionGroup`（`:223`）、
   `bind_kv_cache`（`:462`）、`prepare_kernel_block_sizes`（`:331`）

### 第四遍：理解调度中的使用（与 execution-core.md 对照）

8. **`vllm/v1/core/kv_cache_manager.py`** — 薄门面
   - `allocate_slots()`（`:248`）、`get_computed_blocks()`（`:206`）、`free()`（`:466`）、
     `take_new_block_ids()`（`:637`）
   - → 内部机制读 execution-core.md §3.3.1–3.3.5 指引的 coordinator/manager/BlockPool
9. **`vllm/v1/core/sched/scheduler.py`** — `schedule()`（`:396`）中的 allocate/get_computed 调用点、
   `update_from_output()`（`:1499`）中的 `_free_request_blocks`（`:2130`）

---

## 附录

### A. 压缩完整时序图

```
EngineCore               kv_cache_utils              GPUWorker              GPUModelRunner           KVBlockZeroer
   │                         │                          │                        │                        │
   │ _initialize_kv_caches() │                          │                        │                        │
   │ (core.py:240)           │                          │                        │                        │
   │ get_kv_cache_specs()    │                          │                        │                        │
   │──────────────────────────────────────────────────────────────────────────>│                        │
   │                         │                          │  get_kv_cache_spec()   │                        │
   │                         │                          │  (gpu_model_runner:7561)│                       │
   │<──────────────────────────────────────────────────────────────────────────│                        │
   │ dict[str, KVCacheSpec]  │                          │                        │                        │
   │ determine_available_memory()                       │                        │                        │
   │──────────────────────────────────────────────────────────────────────────>│                        │
   │<──────────────────────────────────────────────────────────────────────────│ available_memory       │
   │                         │                          │                        │                        │
   │ get_kv_cache_configs()  │                          │                        │                        │
   │ (kv_cache_utils.py:2005)│                          │                        │                        │
   │────────────────────────>│                          │                        │                        │
   │                         │ get_kv_cache_groups()    │                        │                        │
   │                         │  (:1697, 早返回级联)      │                        │                        │
   │                         │ _check_enough_kv_cache_memory() (:733)           │                        │
   │                         │ get_kv_cache_config_from_groups() (:1318)        │                        │
   │                         │  └ num_blocks 四分支     │                        │                        │
   │<────────────────────────│                          │                        │                        │
   │ KVCacheConfig           │                          │                        │                        │
   │─────────────────────────────────────────────────────────────────────────────────────────────────────>│
   │                         │                          │ initialize_from_config │                        │
   │                         │                          │ (gpu_worker.py:695)    │                        │
   │                         │                          │───────────────────────>│                        │
   │                         │                          │                        │ initialize_kv_cache()  │
   │                         │                          │                        │  (gpu_model_runner:7405)│
   │                         │                          │                        │  ├ init_attn_backend   │
   │                         │                          │                        │  ├ alloc + reshape     │
   │                         │                          │                        │  └ bind_kv_cache       │
   │                         │                          │ [needs_kv_cache_zeroing]                       │
   │                         │                          │ _init_kv_zero_meta()   │                        │
   │                         │                          │───────────────────────>│────────────────────────>│
   │                         │                          │                        │  __init__ 预计算 _meta │
   │                         │                          │<───────────────────────│<────────────────────────│
   │<─────────────────────────────────────────────────────────────────────────────────────────────────────│
   │                         │                          │ 初始化完成              │                        │
```

### B. 关键数据对象流转

```
AttentionLayerBase.get_kv_cache_spec()          [attention_layer_base.py:29]
    │  → 每层声明自己需要的 KVCacheSpec
    ▼
dict[str, KVCacheSpec]
    │  例: {"layers.0.self_attn": MLAAttentionSpec, "layers.1.self_attn": MambaSpec, ...}
    ▼
get_kv_cache_groups()                           [kv_cache_utils.py:1697]
    │  → 早返回级联分组
    ▼
list[KVCacheGroupSpec]                          [kv_cache_interface.py:905]
    │  例: [UniformTypeKVCacheSpecs(MLA+Indexer), MambaSpec×N 等大组, ...]
    ▼
get_kv_cache_config_from_groups()               [kv_cache_utils.py:1318]
    │  → 结合 available_memory 算 num_blocks
    ▼
KVCacheConfig
    │  ├─ num_blocks: int                       # 所有 group 共享的块数
    │  ├─ kv_cache_tensors: list[KVCacheTensor] # size / shared_by / offset / block_stride
    │  └─ kv_cache_groups: list[KVCacheGroupSpec]
    │
    │  ═══════════ 传递到 Worker ═══════════
    ▼
initialize_kv_cache()                           [gpu_model_runner.py:7405]
    │  → AttentionGroup → torch.zeros 分配 → reshape（含 packed 切片）→ bind_kv_cache
    ▼
GPU Tensor（每层一个视图）                       [绑定到 layer.kv_cache]
    │
    ▼
[needs_kv_cache_zeroing] KVBlockZeroer.__init__  [worker/utils.py:88]
    │  → 预计算 seg_addrs，单 _meta
    ▼
运行时: SchedulerOutput.new_block_ids_to_zero → _zero_block_ids() → Triton 清零
```
