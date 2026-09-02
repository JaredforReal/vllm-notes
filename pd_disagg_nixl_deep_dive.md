# vLLM PD 分离深度篇 —— NIXL 在 TP 与 hybrid 模型下如何搬 KV

> 本文是 [`pd_disagg_readme.md`](./pd_disagg_readme.md) 的下钻续篇。
> 101 文档停在「Connector/Worker 抽象 + 逐层 `save_kv_layer`/`wait_for_layer_load`」的概念层，
> 回答了 **「KV 在 P 和 D 之间被搬」**。本文回答 **「在 TP 切分、block 几何、hybrid(mamba) 状态并存时，
> 字节到底怎么被搬、搬到哪里、为什么不能搬错」** —— 也就是 [#49612](https://github.com/vllm-project/vllm/pull/49612)
> 这类 PR 操作的基底。
>
> 阅读前提：读过 101，知道 PD 两实例、KV Connector 的 Scheduler/Worker 双角色、`start_load_kv` / `save_kv_layer` 逐层流水线。
> 本文聚焦 **NixlConnector**（RDMA 零拷贝，生产首选），其它 connector 共享上层抽象但传输细节不同。

---

## 0. 全文地图

```
                        101 README 到这里为止
                              ▲
                              │ save_kv_layer / wait_for_layer_load（概念层：把 KV 搬过去）
                              ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ 0. 一个核心矛盾：P 侧和 D 侧的 KV 不是「同一块形状的内存」                       │
   │                                                                            │
   │ 1. 两个正交的"比例"，贯穿全文                                                │
   │      tp_ratio        (P/D 的 TP 大小比)                                      │
   │      block_size_ratio(P/D 的 block token 数比)                              │
   │                                                                            │
   │ 2. KV 数据的两种"可分性"                                                    │
   │      attention KV  = token-extent，可按 token 拆                            │
   │      mamba/SSM state = 循环状态，不可拆                                     │
   │      └─ 衍生：MLA replicated vs full-attn head-sharded                      │
   │                                                                            │
   │ 3. NIXL 的内存模型：descriptor                                              │
   │ 4. Handshake：P 和 D 怎么互相认识（NixlAgentMetadata）                      │
   │ 5. Block 几何三层：block_size / kernel_block_size / ppl（hybrid 分叉的根源）  │
   │ 6. HMA：hybrid 内存池化（attention + mamba 共享 region）                     │
   │ 7. TP 映射：谁从谁读（compute_tp_mapping）                                  │
   │ 8. READ 路径详解（pull）                                                     │
   │ 9. 多源 split：P_TP > D_TP 的 per-source 切分                               │
   │ 10. Mamba 三段传输（conv sub-projection + ssm）                             │
   │ 11. 接收后处理：permute / convert / 补零                                    │
   │                                                                            │
   │ 12. 把所有轴叠起来 → heterogeneous block size for hybrid（#49612）           │
   │ 13. 安全铁律 & 坐标系速查 & 源码索引                                         │
   └────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. 一个核心矛盾：KV 不是「同一块形状的内存」

101 文档画的是「P 把 KV 传给 D」，容易让人以为两边是同形 buffer 的拷贝。**现实里几乎从不如此。** P 实例和 D 实例通常为了各自目标（TTFT vs ITL）做了不同的并行配置：

| 维度 | Prefill 侧（计算密集）常见选择 | Decode 侧（带宽密集）常见选择 |
|------|------------------------------|------------------------------|
| TP 大小 | 大（如 TP8，铺开算力降 TTFT） | 小（如 TP1/TP2，省 KV 显存、降 ITL） |
| Block size | 小（mamba 对齐 padding 因 TP 而变） | 大 |
| Attention backend | 可能与 D 不同（FlashInfer vs FlashMLA） | — |
| KV layout | HND | NHD |

于是传输面临的不是「拷贝」，而是 **「把一组按 P 几何组织、按 P 的 TP 切分过的 KV 字节，重新映射到按 D 几何组织、按 D 的 TP 切分过的 D 侧显存里，且每个字节都要落对位置」**。本文就是讲这个映射。

NIXL 把这件事拆成两个**正交**的比例来处理。

---

## 2. 两个正交的比例（全文的坐标系）

> 源码：`vllm/distributed/kv_transfer/kv_connector/utils.py` `TransferTopology.tp_ratio()` / `block_size_ratio()`

### 2.1 `tp_ratio`：TP 大小比（「谁从几个谁读」）

```python
def tp_ratio(self, remote_tp_size):
    if self.tp_size >= remote_tp_size:   # D_TP >= P_TP
        return self.tp_size // remote_tp_size      # 正数
    return -(remote_tp_size // self.tp_size)       # 负数
```

约定（**记住符号**，后面所有路径都靠它分流）：

| `tp_ratio` | 含义 | 读模式 |
|-----------|------|--------|
| **= 1** | P_TP == D_TP，一一对应 | 每个 D rank 读 1 个对应 P rank |
| **> 0**（如 +2） | D_TP > P_TP | 多个 D rank **共享**读同一个 P rank 的不同 head 切片 |
| **< 0**（如 −2） | P_TP > D_TP | 一个 D rank 要**从多个** P rank 汇聚（multi-read） |

> 例如 KimiLinear 规模：P_TP=8, D_TP=1 → `tp_ratio = -(8/1) = -8`，D 侧单个 rank 要从 8 个 P rank 把 KV 拼起来。

### 2.2 `block_size_ratio`：block token 数比（「块大小不一致」）

```python
def block_size_ratio(self, remote_block_size):
    assert self.block_size % remote_block_size == 0   # 必须整除
    return self.block_size // remote_block_size
```

| `block_size_ratio` | 含义 |
|-------------------|------|
| **= 1** | 两边 block size 相同 |
| **> 1** | P 的 block **更小**（D 一个 block 要对上 P 的 `ratio` 个 sub-block） |

> `block_size_ratio` 只允许 `local % remote == 0`，即 **nP < nD**（P 块更小）。反方向（D 块更小）不支持。

**这两个比例是正交的**：可以同时 `tp_ratio=-8` 且 `block_size_ratio=2`（既多源读、又块大小不一致）。#49612 干的事，就是让 **hybrid 模型** 在这两个比例同时 ≠ 1 时也能正确工作。

---

## 3. KV 数据的两种「可分性」（决定每个比例怎么施加）

这是理解整条链路的钥匙。要传的 KV 数据按「能不能按 token 拆」分两类：

| 数据类型 | 性质 | `block_size_ratio > 1` 时 | `tp_ratio` 时 |
|---------|------|--------------------------|---------------|
| **Attention KV** | token-extent（每 token 一组 K/V） | **可拆**：一个 local block 沿 token 轴切成 `ratio` 个 sub-block，1:1 配 P 的 block | 按 head 切（见下） |
| **Mamba/SSM state** | 循环状态（conv state + ssm state），**不按 token 组织** | **不可拆**：永远用 local page geometry，1:1 直传，不展开 | 整块按 rank shard |

### 3.1 Attention KV 又分两种 TP 行为

> 源码：`base_worker.py` `_is_region_replicated()` / `_region_is_mla`；`utils.py` `is_kv_replicated()` / `replicates_kv_cache()`

| 子类 | 特征 | TP 下的行为 | NIXL 标记 |
|------|------|-----------|----------|
| **MLA**（DeepSeek/Kimi/GLM） | KV 压成单个 latent，head 维不可切 | **replicated**：每个 rank 持有完整相同 KV，只读一次 | `_is_region_replicated = True` |
| **full-attn (MHA/GQA)** | head 跨 TP 切分 | **head-sharded**：每个 rank 只有部分 head，要从多个 rank 拼 | `_is_region_replicated = False` |
| **GQA 且 tp > kv_heads** | 多个 rank 持有相同 head | **kv replicated**（GQA 去重） | `is_kv_replicated = True` |

`replicates_kv_cache = is_mla or is_kv_replicated`。**replicated 的 region 按 REPLICATE 处理（整块、读一次），否则按 SPLIT 处理（按 head 切片、多 rank 拼）。** 这套标记后面会在 descriptor 构造和 TP 映射里反复用到。

> 一句话：**attention KV 可以沿 token 轴和 head 轴两个方向被切；mamba state 哪个轴都不能切。** heterogeneous block size 的难点，就是「沿 token 轴切（block_size_ratio）」要和「沿 head 轴切（tp_ratio 的 SPLIT）」正确叠加，而 mamba 要全程置身事外。

---

## 4. NIXL 的内存模型：descriptor

NIXL（NVIDIA Inference Transfer Library）用 **RDMA** 做 GPU 间零拷贝。它不认识「vLLM 的 block」，只认识**注册过的内存区间**。

### 4.1 三元组 descriptor

每段要传输的内存被注册成一个 descriptor：

```
(addr: u64, length: u64, device_id: int)
```

- `addr`：内存起始地址（GPU 显存指针，或 host 指针）
- `length`：字节数
- `device_id`：GPU id（CPU 记为非负）

> 源码：`base_worker.py` `register_kv_caches()` 把每个 KV cache tensor 转成 `(base_addr, size_bytes, device_id)`，`nixl_wrapper.register_memory()` 注册。

### 4.2 descriptor list（dlist）与 transfer 配对

一次传输 = 把一组 **local desc id** 和一组 **remote desc id** 一一配对：

```
local dlist:  [desc0=(0x1000, 128B), desc1=(0x1080, 128B), ...]   ← prep_xfer_dlist → handle
remote dlist: [desc0=(0x9000, 128B), desc1=(0x9080, 128B), ...]   ← 远端注册过的 region

transfer:     local_desc_ids=[0,1,...]  ←RDMA→  remote_desc_ids=[0,1,...]
              （按 index 一一对应：local desc i 的内容写到 remote desc i）
```

关键：**desc id 是在各自 dlist 里的下标**。`_compute_desc_ids()`（base_worker.py）的工作就是把 vLLM 的 block id 翻译成 dlist 下标。这个翻译必须让 local 的第 i 个 token 对上 remote 的第 i 个 token —— 否则就写错位置（见 §13 安全铁律）。

### 4.3 两种传输方向

| 模式 | 谁发起 | 数据流 | 实现文件 |
|------|--------|--------|---------|
| **Pull（READ）** | D 侧 | D 主动从 P **读** 到自己的显存 | `pull_worker.py` |
| **Push（WRITE）** | D 侧注册，P 主动**写**到 D | P→D | `push_worker.py` |

两者共用 `base_worker.py` 的几何/handshake 逻辑，只在「谁申请 buffer、谁发起 RDMA」上不同。本文以 **pull（READ）** 为主线索（#49612 的测试也是 pull 路径）。

---

## 5. Block 几何三层：hybrid 分叉的根源

> 这是最容易混淆、也是 #49612 最核心的概念。`block_size` 在 vLLM 里有 **三个不同层次**。

### 5.1 三个层次

```
┌─────────────────────────────────────────────────────────────────────┐
│ logical block (mamba 对齐的最小单位，随 TP 变！)                        │
│   = physical_blocks_per_logical (ppl)  ×  kernel_block_size          │
│                                                                       │
│   ┌───────────────────────────────────────────────────────────┐     │
│   │ kernel block (attention 后端/kernel 锁定的 page 粒度)         │     │
│   │   = block_size  (vLLM cache_config.block_size)              │     │
│   │                                                             │     │
│   │   每个含 block_size 个 token 的 KV                            │     │
│   └───────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

| 概念 | 符号 | 谁决定 | 随 TP 变？ |
|------|------|--------|----------|
| **kernel block size** | `block_size` | attention 后端 `get_supported_kernel_block_sizes()` | 否（后端固定，如 64） |
| **physical_blocks_per_logical** | `ppl` / `_physical_blocks_per_logical_kv_block` | mamba 对齐需求 | **是**（随 TP 变） |
| **logical block size** | `ppl × block_size` | 派生 | **是** |

### 5.2 为什么 hybrid 模型 logical block 随 TP 变（#41037 根因）

mamba/SSM 的状态需要对齐到一个特定的 token 间隔（mamba 的 `state_size`、conv window 等）。设 mamba 要求的对齐粒度是 `A` 个 token：

```
logical_block_size = A (mamba 对齐常量，与 TP 无关)
kernel_block_size  = K (后端固定)
ppl                = A / K
```

但 **attention 那一层** 的 block 必须和 mamba 共用同一个 logical block 划分（HMA 池化，见 §6）。当 `A` 不能被 kernel 的 `K` 整除时，vLLM 通过 padding 让 `ppl` 取一个能整除的值，而这个 padding **随 TP 改变**（因为不同 TP 下 mamba state 的 per-rank 形状不同）。

> 后果：同一个模型，P_TP=8 时 `ppl=12`，D_TP=1 时 `ppl=90` —— **logical block size 不一样**，但 kernel block size 一样。这就是 #49612 要处理的「heterogeneous **physical_blocks_per_logical**」场景（区别于单纯的 block_size 不同）。

### 5.3 内存布局：`[NB]` vs `[NB×ppl, PS/ppl]`

> 源码：`base_worker.py` `register_kv_caches()`（~L1095–1170）

当 `ppl > 1`，attention KV cache 的物理形状从 `[num_logical_blocks, page_size]` 变成 `[num_logical_blocks × ppl, page_size / ppl]` —— 即**第一维按 kernel block 拆细**，每页字节相应缩小。注释原话：

> *"When there's a mismatch between kbs<>bs, we rely on HMA to ensure caches are either `[NB, PS]` or `[NB*r, PS/r]` where r is bs/kbs."*

mamba cache 则始终是 `[num_logical_blocks, page_size]`（state 不拆）。这就是为什么 attention desc 要 ratio-expand、mamba desc 不要。

---

## 6. HMA：hybrid 内存池化

> 源码：`base_worker.py` `_is_hma_required`（L276）、`register_kv_caches` 的 `seen_base_addresses` 去重（L1135）

**HMA = Hybrid KV cache Memory Allocation**（`disable_hybrid_kv_cache_manager` 反向开关）。当模型同时有 attention 和 mamba group 时启用。

HMA 做两件事：

1. **跨 group 共享 tensor**：attention 和 mamba 的 KV 被分配在**同一个连续物理 tensor** 里（`KVCacheTensor(shared_by=[mla.0, kda_a.0, kda_b.0])`），靠 offset 区分。
2. **去重注册**：`register_kv_caches` 用 `seen_base_addresses` 去重 —— 共享同一 base_addr 的多个层只注册**一次**。结果是「**更少的 region，每个 region 更多 block**」。

```
无 HMA:  [mla.0] [mla.1] [kda.0] [kda.1]   ← 4 个独立 region，各注册一次
有 HMA:  [ ────── shared tensor 0 ────── ]  [ ─── shared tensor 1 ─── ]
         (mla.0+kda_a.0+kda_b.0 共用)        (mla.1+kda_a.1+kda_b.1 共用)
         ← 2 个 region，block 数翻倍
```

**对 NIXL 的影响**：descriptor 数量 ≠ 层数，而是 = region 数 × block 数。`num_descs = num_regions * num_blocks`（L1211，FA descs 的边界）。mamba 的 conv/ssm desc 额外拼接在 FA descs 之后。

> 这也是 #49612 里「host-buffer 模式不支持 block size ratio」的原因之一：host buffer 是 **per-layer** 的，破坏了 HMA「共享 region」的前提，去重逻辑失效。（详见 §12）

---

## 7. Handshake：P 和 D 怎么互相认识

> 源码：`base_worker.py` `add_remote_agent()` / `_validate_remote_agent_handshake()`；`metadata.py` `NixlAgentMetadata`

PD 建立连接时分两层校验：

### 7.1 兼容性 hash（架构级，必须完全一致）

`compute_nixl_compatibility_hash()`（metadata.py）对**模型架构**算 SHA-256：vLLM 版本、connector 版本、model 名、dtype、num_kv_heads、head_size、num_layers、attention backend、cache_dtype、是否 HMA。

**两边 hash 必须一致**才解码对端元数据 —— 否则报清晰错误，防止 schema 不兼容时 decoder 崩溃。

> 注意：`tp_size`、`block_size`、`kv_cache_layout` **故意不进 hash** —— 这些允许 P/D 不同（异构部署），改在运行时校验。

### 7.2 `NixlAgentMetadata`（运行时几何，允许部分不同）

握手交换的字段（metadata.py L48）：

| 字段 | 含义 | P/D 可不同？ |
|------|------|-------------|
| `kv_caches_base_addr` | 各 region 起始地址 | 是（各自的显存） |
| `num_blocks` | region 数 × kernel block 数 | 是 |
| `block_lens` | 每个 region 的 page 字节数 | 是（校验一致） |
| `block_size` | kernel block token 数 | 是（→ `block_size_ratio`） |
| `physical_blocks_per_logical_kv_block` | **ppl** | 是（→ hetero-ppl，#49612 核心） |
| `ssm_sizes` | mamba state 形状（per-rank） | 是（随 TP shard） |
| `kv_cache_layout` | HND / NHD | 是（需 permute 补救） |
| `attn_backend_name` | attention 后端名 | 是（异构后端） |

### 7.3 运行时校验（`_validate_remote_agent_handshake`）

逐条检查可异构维度的合理性，例如：

- `block_size_ratio == local_len // remote_len`（replicated region 字节匹配）
- `tp_ratio` 方向 + 是否 replicated 的组合合法
- layout 不一致时要求 `enable_permute_local_kv` 或 HND
- **#49612 之前**：`if self._is_hma_required: assert block_size_ratio == 1`（hybrid 直接禁掉 block 不一致）
- **#49612 之后**：改成 `if block_size_ratio != 1: assert not self.use_host_buffer`（放开 hybrid，只挡 host-buffer）

handshake 通过后，本地为该远端建好 **TP 映射**（§8）和 **split handles**（§9）。

---

## 8. TP 映射：谁从谁读

> 源码：`tp_mapping.py` `compute_tp_mapping()` → 产出 `TPMapping`

handshake 后，每个本地 rank 算出自己**该从哪些远端 rank 读、各读哪个 head 切片**。核心逻辑（按 `tp_ratio` 符号分两条路）：

### 8.1 `tp_ratio > 0`（D_TP ≥ P_TP）：多个 D rank 读同一个 P rank

```
P:  rank0 (holds all heads)
D:  rank0 ──┐  (读 head 切片 0)
    rank1 ──┤  (读 head 切片 1)   ← 都从 P rank0 读，但取不同 head 段
    rank2 ──┘  (读 head 切片 2)
```

- MLA（replicated）：每个 D rank 读完整 KV（latent 不可切），`attn_ranks = [tp_rank * remote_tp_size // tp_size]`，只一个源。
- full-attn：按 head 切片，`rank_to_attention_slot` 给每个源 rank 一个 head slot 索引，读时按 slot 取偏移。

### 8.2 `tp_ratio < 0`（P_TP > D_TP）：一个 D rank 读多个 P rank（multi-read）

```
P:  rank0 rank1 rank2 ... rank7   (each holds different head shard)
                  │
D:  rank0 ←───────┴──── 汇聚 8 个 rank 的 head 段，拼成完整 KV
```

- `target_remote_ranks()` 返回 `|tp_ratio|` 个源 rank。
- full-attn：每个源 rank 贡献一段 head；GQA 去重（`np.unique` 去掉持有相同 head 的冗余 rank）。
- SSM：state 跨 rank shard，每个源 rank 贡献一段 state。

### 8.3 `TPMapping` 四个字段

```python
@dataclass(frozen=True)
class TPMapping:
    source_ranks_per_group: tuple[tuple[int,...], ...]  # 每个 group 该从哪些远端 rank 读
    all_source_ranks: tuple[int, ...]                   # 所有源的并集
    rank_to_attention_slot: dict[int, int]              # 源 rank → FA head slot
    rank_offset_factor: int                             # hetero-TP 的 head 偏移因子
```

> `source_ranks_per_group` 是 per-group 的 —— attention group 和 mamba group 可以有不同的源 rank 集合（MLA 从一个 rank 读，SSM 从所有 rank 读各自 shard）。这是 hybrid 模型的关键。

---

## 9. READ 路径详解（pull）

> 源码：`pull_worker.py` `_read_blocks_for_req()` / `_read_blocks()`；`base_worker.py` `_compute_desc_ids()`

D 侧拉一个请求的 KV，分几步：

### 9.1 block id 映射（处理 block_size_ratio）

`_map_block_ids_for_block_size_ratio()`（#49612 抽到 base_worker 的共享方法）：

```
对每个 group：
  if SSM group:
      local_ids, remote_ids 直接 1:1 passthrough（state 不可拆）
  else (attention):
      local_ids 经 get_mapped_blocks() 展开成 sub-block id：
        local [1,2,3] (block_size=16) → sub-blocks [4..15] (对齐 remote block_size=4)
      若展开后多于 remote，clip 到 remote 覆盖（untransferred tail 后面补零，见 §11）
```

### 9.2 block id → desc id（`_compute_desc_ids`）

把 block id 翻译成各自 dlist 的下标。这里 FA 和 SSM 的 stride 不同：

- **FA desc**：每 region `num_blocks` 条（kernel 粒度，`block_size_ratio>1` 时按 ratio 展开）
- **SSM desc**：每 region `logical_blocks = dst_num_blocks // ppl` 条（不拆、不展开）

> #49612 修了一个这里的 bug：`logical_blocks = num_blocks // ppl` → `dst_num_blocks // ppl`（用对变量）。

### 9.3 选 local 写入 handle

```python
if tp_ratio < 0 and (not use_mla or len(read_specs) > 1):
    # multi-read：用预建好的 per-source split handle
    split_key = (tp_ratio, remote_block_size)          # ← #49612 加了 remote_block_size 进 key
    local_handle = src_xfer_handles_by_tp_ratio[split_key][i]
else:
    # 单读：用整 region handle（按 remote_block_size 选）
    local_handle = src_xfer_handles_by_block_size[remote_block_size]
```

### 9.4 发起 RDMA

```
make_prepped_xfer(READ, local_handle, local_desc_ids, remote_handle, remote_desc_ids)
transfer(handle)   # 异步发起
# 之后 get_finished() 轮询是否 DONE
```

每个 `(local_desc_id, remote_desc_id)` 配对：把 remote region 的那段字节 RDMA 写进 local region 的对应段。

---

## 10. 多源 split：P_TP > D_TP 的 per-source 切分

> 源码：`base_worker.py` `_build_local_splits_from_plan()` / `_fa_desc_replicated()` / `_build_fa_local()`

当 `tp_ratio < 0`，一个 D rank 要从 N 个 P rank 读，自己的 local region 要**预先切成 N 份**，每份对应一个源 rank 的写入目标。`_build_local_splits_from_plan` 为每个源 rank 生成一个 handle 的 desc 列表。

对每个 FA desc，按 `_fa_desc_replicated` 标记分流：

```python
for j, (addr, local_len, dev) in enumerate(src_blocks_list):
    if j < num_fa_descs:
        if fa_desc_replicated[j]:                    # REPLICATE (MLA)
            handle.append((addr, local_len, dev))    #   整块，每个源都写（实际读一次）
        else:                                        # SPLIT (head-sharded full-attn)
            chunk = local_len // fa_num_splits       #   按 head 切
            handle.append((addr + fa_slot*chunk, chunk, dev))
    else:  # SSM desc
        chunk = local_len // ssm_num_splits
        handle.append((addr + p_idx*chunk, chunk, dev))
```

每个 handle 注册成一个独立 dlist，键 `(tp_ratio, remote_block_size)`。

> **#49612 的关键改动**：split handle 的键从 `tp_ratio` 变成 `(tp_ratio, remote_block_size)`，并新增 `src_blocks_data_by_block_size`。因为不同 remote block size 要在 remote 粒度上建不同的 split desc。`num_fa_descs` 也乘以 `block_size_ratio`。

---

## 11. Mamba 三段传输 + 接收后处理

### 11.1 Mamba 的三段（conv sub-projection）

> 源码：`base_worker.py` `_build_mamba_local()` / `_conv_decomp`（`derive_mamba_conv_split`）

mamba 的 conv state 在 DS（dim, state_len）布局下，被拆成**多个连续的 sub-projection** + ssm state，每段一个 desc：

```
一个 mamba block 的传输 = [conv_sub_proj_0] [conv_sub_proj_1] ... [ssm_state]
                          └─── conv_offsets（derive_mamba_conv_split 算出）───┘
```

这是为什么 `num_descs * block_size_ratio == len(blocks_data)` 这条断言（#49612 改的）：mamba 的 desc 数 = logical_blocks（**不**乘 ratio）× (conv 段数 + 1)。

> **铁律**：mamba state block 永远不 ratio-expand（`_build_mamba_local` 删了 `assert block_size_ratio==1`，直接按 local geometry 建）。

### 11.2 接收后处理 `post_process_device_kv_on_receive`

RDMA 写完后，D 侧 cache 可能需要三种修整：

| 情形 | 操作 | 触发 |
|------|------|------|
| layout 不一致（remote HND, local NHD） | `kv_postprocess_layout_on_receive` | `enable_permute_local_kv` |
| block size 不一致（small→large） | `kv_postprocess_blksize_on_receive` | `block_size_ratio > 1` |
| 两者皆有 | `kv_postprocess_blksize_and_layout_on_receive` | 同上 |

**只作用于 attention cache**（#49612 新增 `_attention_kv_caches` cached_property，过滤掉 mamba layer —— mamba 是 1:1 传的，不 permute）。

### 11.3 补零 untransferred tail（#49612 的安全网，最重要）

传输只覆盖每个 local attention block 的前 `covered_sub_blocks` 个 remote 子块；后面被 clip 的部分是**陈旧字节**。

**为什么有陈旧字节？** 调度器在分配 block 时，会**故意跳过**覆盖 matched token 的那些 block 的 alloc-time KV 清零 —— 因为那个清零会和 RDMA 写竞争（并发写同一段）。所以传输范围之外的字节是**上一个请求残留的**，decode 长到那个尾巴时会冒 garbage/NaN。

后处理算出覆盖范围并补零：

```python
covered_blocks, sub_blocks_in_last = divmod(covered_sub_blocks, block_size_ratio)
# 部分覆盖的最后一块：token 维度补零
cache[last_block_id, sub_blocks_in_last*sub_block_tokens:].zero_()
# 完全没覆盖的后续块：整块补零
cache.index_fill_(0, stale_ids, 0)
```

`convert` flag 把「layout/blksize 转换」（MLA 不需要）和「补零」（MLA clipped 块也需要）解耦。

---

## 12. 把所有轴叠起来：heterogeneous block size for hybrid（#49612）

现在所有概念就位，可以讲清这个 PR 了。

### 12.1 它打开的组合

#49612 之前，handshake 里有一条一刀切：

```python
if self._is_hma_required:
    assert block_size_ratio == 1   # hybrid 模型 + block size 不一致 → 直接死
```

但 hybrid 模型**正是** logical block 随 TP 分叉的模型（§5.2），所以这条等于「最需要 hetero block size 的模型完全用不了」。

#49612 把它换成精确的边界：

```python
if block_size_ratio != 1:
    assert not self.use_host_buffer   # 只挡 host-buffer 路径
```

### 12.2 它怎么做到的（四条机制，对应前文）

1. **Descriptor 几何分流**（§3 + §5.3）：FA desc 按 ratio 展开（token 可拆），SSM desc 不展开（state 不可拆）。`_build_mamba_local` 删掉 `assert block_size_ratio==1`，mamba 永远按 local geometry。
2. **共享的 per-group 映射**（§9.1）：`_map_block_ids_for_block_size_ratio` 遍历所有 group，attention 展开+clip，SSM 1:1。替换掉 pull/push 里各重复一份的旧逻辑。
3. **接收后处理补零**（§11.3）：按 `covered_sub_blocks` 补零被 clip 的尾巴，只动 attention cache。
4. **多源 hetero-TP**（§10）：split handle 键升级为 `(tp_ratio, remote_block_size)`，`num_fa_descs *= ratio`，FA sub-block desc 整块 passthrough。

### 12.3 仍然不支持、显式拒绝的两个组合

#49612 不是「全支持」，而是把支持边界精确化。两条 assert 挡掉的是**当前几何数学覆盖不到 + 没测试 + 会静默损坏**的组合：

| 被挡组合 | assert | 真实阻塞 |
|---------|--------|---------|
| **host-buffer + block_size_ratio** | `assert not self.use_host_buffer` | ratio 机制全挂在「设备侧 descriptor + RDMA 直传」路径上；host-buffer 走 `copy_blocks` 按**整 block** h2d，不保留 sub-block 粒度，补零逻辑失去前提。host buffer 还是 per-layer 的，破坏 HMA 共享 region 的去重（§6）。需平行重写一套，本 PR 未做。 |
| **head-sharded attention + block_size_ratio + P_TP>D_TP** | `assert block_size_ratio==1 or fa_num_splits==1 or all(fa_desc_replicated)` | 三个轴（token 切 / head 切 / per-source 切）同时在同一个平坦 `(addr,len)` 字节区间上做切分。`addr + fa_slot*chunk` 的 head 偏移数学只在 `local_len`=完整 block 时验证过；叠加 sub-block 后既未推导也未测试。MLA（replicated）落进 `all(fa_desc_replicated)` 安然通过；目标负载（Kimi/DeepSeek/GLM）不受影响。 |

**为什么挡掉是对的**：KV 传输写错字节不会立刻崩，而是让**无关请求** decode 到一半吐 garbage —— 最难查的 bug。两条 assert 把「算不出正确几何」的组合变成握手期硬错 + 可读 message，比悄悄发错强。

---

## 13. 安全铁律 & 坐标系速查

### 13.1 安全铁律（#49612 测试守护的不变量）

> **一个请求的 RDMA 写所落入的每个 local 字节区间，必须落在该请求自己的 block 内。**
> 违反 = 静默破坏同驻留的另一个请求的 KV/mamba state（decode 中途损坏）。

这条由 `tests/v1/kv_connector/unit/test_nixl_desc_geometry.py`（619 行）端到端守护，覆盖到 KimiLinear 规模（TP8→TP1，logical block 5760/kernel 64/ppl=90 vs remote 768/ppl=12）。

### 13.2 坐标系速查

| 比例/标记 | 取值 | 含义 | 影响哪层 |
|----------|------|------|---------|
| `tp_ratio` | +1 / >0 / <0 | TP 一致 / D≥P / P>D | §8 谁读谁、§9 handle 选择、§10 split |
| `block_size_ratio` | 1 / >1 | block 一致 / P 块更小 | §9.1 展开+clip、§11.2 convert、§11.3 补零 |
| `_is_region_replicated` | T/F | MLA/复制 KV / head-sharded | §10 REPLICATE vs SPLIT |
| `ppl` (`_physical_blocks_per_logical_kv_block`) | 1 / >1 | 无 mamba 对齐 / 有 | §5 几何、§6 布局、§9.2 stride |
| `_is_hma_required` | T/F | 有非-FullAttn group / 纯 attention | §6 池化、§7 校验 |
| `hetero_ppl`（远端 ppl ≠ 本地） | T/F | 两边 ppl 不同 | §11.3 补零触发条件之一 |

### 13.3 关键源码索引

| 概念 | 文件:位置 |
|------|----------|
| `tp_ratio` / `block_size_ratio` / `is_kv_replicated` | `kv_connector/utils.py:523-570` |
| `TransferTopology` | `kv_connector/utils.py:407` |
| `TPMapping` + `compute_tp_mapping` | `nixl/tp_mapping.py:40, 65` |
| `NixlAgentMetadata` + handshake 字段 | `nixl/metadata.py:48` |
| `_is_hma_required` / `_conv_decomp` 初始化 | `nixl/base_worker.py:276, 296` |
| `register_kv_caches`（region/MLA/ppl/num_descs） | `nixl/base_worker.py:~1095-1219` |
| `_build_fa_local`（FA desc，ratio 展开） | `nixl/base_worker.py:1352` |
| `_build_mamba_local`（SSM desc，不展开） | `nixl/base_worker.py:~1277` |
| `add_remote_agent` + `_validate_remote_agent_handshake` | `nixl/base_worker.py:~1604, 1665` |
| `_compute_desc_ids`（block id → desc id） | `nixl/base_worker.py:~90` |
| `_map_block_ids_for_block_size_ratio` | `nixl/base_worker.py:~2297` |
| `_build_local_splits_from_plan` / `_fa_desc_replicated` | `nixl/base_worker.py:160, 211` |
| `post_process_device_kv_on_receive`（permute/convert/zero） | `nixl/base_worker.py:~1887` |
| `_attention_kv_caches`（过滤 mamba） | `nixl/base_worker.py:~1887`（cached_property） |
| pull READ 主路径 | `nixl/pull_worker.py:_read_blocks_for_req / _read_blocks` |
| push WRITE 主路径 | `nixl/push_worker.py:_xfer_blocks_for_req / _xfer_blocks` |
| 几何不变量测试 | `tests/v1/kv_connector/unit/test_nixl_desc_geometry.py` |

---

## 14. 一页纸总结

```
PD 分离的 KV 传输 = 把「P 几何 × P-TP 切分」的字节，重映射到「D 几何 × D-TP 切分」的显存。

两个正交比例：
  tp_ratio        → 决定「谁读谁」（一一 / 多D读一P / 一D读多P）
  block_size_ratio→ 决定「块怎么对」（一致 / P 块小，D 块拆 sub-block 配对）

两种数据可分性：
  attention KV → 可沿 token(block_size_ratio) 和 head(tp_ratio SPLIT) 切；MLA replicated 不切 head
  mamba state  → 任何轴都不切，1:1 直传，不 ratio-expand

NIXL 只认 (addr, len, dev) descriptor。block id 经 _compute_desc_ids 翻译成 dlist 下标，
local[ i ] ↔ remote[ i ] 一一 RDMA。铁律：每个 local 字节必须落在请求自己的 block 内。

hybrid(mamba) 模型 logical block 随 TP 分叉（#41037），却一度被「HMA + block_size_ratio!=1 一刀切禁」。
#49612 按「可分性分流」打开它：FA 展开+clip+补零尾巴，SSM 不展开 1:1；并把
host-buffer+ratio、head-sharded+ratio 两个未测组合显式挡掉。

读到这里，再回头看 #49612 的 diff，每一处改动都应该能对号入座到上面某一节。
```
