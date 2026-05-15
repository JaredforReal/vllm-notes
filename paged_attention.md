# vLLM PagedAttention 深度解析

> 基于源码的原理分析与代码导读，聚焦 V1 架构。

## 目录

1. [核心思想：为什么需要 PagedAttention](#1-核心思想)
2. [Block：KV Cache 的基本管理单元](#2-blockkv-cache-的基本管理单元)
3. [Block Pool：物理 Block 的分配与回收](#3-block-pool)
4. [前缀缓存：Block Hash 与共享](#4-前缀缓存)
5. [从 Scheduler 到 Worker：Block ID 的传递](#5-从-scheduler-到-worker)
6. [Worker 侧：Block Table 的 GPU 构建](#6-worker-侧block-table-的-gpu-构建)
7. [Slot Mapping：Token 到物理位置的映射](#7-slot-mapping)
8. [GPU 上的 KV Cache Tensor 布局](#8-gpu-上的-kv-cache-tensor-布局)
9. [Reshape and Cache：写入 KV Cache 的 Kernel](#9-reshape-and-cache)
10. [Attention Kernel 如何使用 Block Table](#10-attention-kernel-如何使用-block-table)
11. [Sliding Window 与 Null Block](#11-sliding-window-与-null-block)
12. [Cascade Attention：共享前缀优化](#12-cascade-attention)
13. [跨层 KV Sharing](#13-跨层-kv-sharing)
14. [代码阅读顺序](#14-代码阅读顺序)

---

## 1. 核心思想

### 传统方法的问题

LLM 推理时，每个请求需要为已生成的所有 token 存储 Key 和 Value 向量（KV Cache）。传统方法为每个请求**预分配一块连续的显存**，大小为 `max_seq_len × hidden_dim`：

```
传统方式：
请求A: [████████████████████░░░░░░░░░░]  预分配 max_len
请求B: [████░░░░░░░░░░░░░░░░░░░░░░░░░]  预分配 max_len
请求C: [█████████████░░░░░░░░░░░░░░░░░]  预分配 max_len

问题：
- 大量碎片化空闲空间（░部分）
- 请求完成后整块释放，产生外部碎片
- 无法在请求间共享相同前缀的 KV cache
```

### PagedAttention 的解决方案

借用操作系统**虚拟内存分页**的思想：

1. 将 KV Cache 切分为固定大小的 **Block（页）**，每个 Block 存储 `block_size` 个 token 的 K/V 向量
2. 维护一个 **Block Table（页表）**，将请求的逻辑 Block 索引映射到物理 Block ID
3. 物理 Block **不需要连续**，可以分布在显存的任意位置
4. 不同请求可以**共享**相同的物理 Block（前缀缓存）

```
PagedAttention：

物理 Block Pool:  [B3][B7][B1][B5][B9][B2][B8][B4][B6]...  (全局共享)

请求A 的 Block Table:  [B3][B7][B1][B5][B9]   →  逻辑连续，物理不连续
请求B 的 Block Table:  [B3][B7][B2]            →  B3,B7 与请求A 共享！
请求C 的 Block Table:  [B3][B7][B1][B8][B4]    →  B3,B7,B1 与请求A 共享！

请求A/B/C 有相同 prompt 前缀 → 共享物理 Block → 省内存 + 省计算
```

---

## 2. Block：KV Cache 的基本管理单元

### Block 元数据

**文件**: `vllm/v1/core/kv_cache_utils.py:110-155`

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int                    # 物理 Block ID (0 到 num_gpu_blocks-1)
    ref_cnt: int = 0                 # 引用计数（多个请求共享时 > 1）
    _block_hash: BlockHashWithGroupId | None = None  # 前缀缓存的 hash key
    prev_free_block: KVCacheBlock | None = None       # 双向链表：前驱
    next_free_block: KVCacheBlock | None = None       # 双向链表：后继
    is_null: bool = False            # 是否为 null block（sliding window 用）
```

### Block Size

Block size 是可配置的（通常为 16）。每个 Block 存储 `block_size` 个 token 的 K 和 V 向量。一个 Block 占用的字节数：

```
block_bytes = 2 × block_size × num_kv_heads × head_size × dtype_size
            = 2 × 16 × 64 × 128 × 2 (FP16) = 524,288 bytes = 512 KB
```

### 空闲 Block 队列

**文件**: `kv_cache_utils.py:158-`

```python
class FreeKVCacheBlockQueue:
    """空闲 Block 的双向链表。按 LRU 顺序排列：头部最久未用，尾部最近使用。"""
```

使用双向链表而非 Python deque，是为了支持 O(1) 从**中间**移除（当 Block 被缓存命中时）。

---

## 3. Block Pool：物理 Block 的分配与回收

**文件**: `vllm/v1/core/block_pool.py:130-`

### 初始化

```python
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        # 创建所有 Block 对象
        self.blocks = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
        # 初始化空闲队列（双向链表）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # 前缀缓存：hash → block 映射
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        # Block 0 作为 null block（sliding window 占位符）
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
```

### 分配 Block

**`get_new_blocks()` (行 322)**：

```python
def get_new_blocks(self, num_blocks):
    blocks = []
    for _ in range(num_blocks):
        # 从空闲队列头部取出（LRU 优先淘汰）
        block = self.free_block_queue.popleft()
        # 如果 block 还在前缀缓存中，淘汰它
        self._maybe_evict_cached_block(block)
        block.ref_cnt = 1
        blocks.append(block)
    return blocks
```

### 缓存命中（Touch）

**`touch()` (行 391)**：

当发现前缀缓存命中时，不分配新 Block，而是增加已有 Block 的引用计数：

```python
def touch(self, blocks):
    for block in blocks:
        block.ref_cnt += 1
        if block.ref_cnt == 1:
            # 从空闲队列中移除（不再可被分配）
            self.free_block_queue.remove(block)
```

### 释放 Block

**`free_blocks()` (行 408)**：

```python
def free_blocks(self, blocks):
    for block in blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0:
            # 追加到空闲队列尾部（最近释放，不优先淘汰）
            self.free_block_queue.append(block)
            # 注意：如果 block 有 hash，保留在前缀缓存中
            # 未来可能有请求命中这个缓存
```

**关键设计**：释放的 Block 如果仍有 hash，会留在前缀缓存中。只有当新请求需要 Block 且空闲队列不足时，才会通过 `_maybe_evict_cached_block()` 淘汰最老的缓存 Block。

### 分配流程图

```
Scheduler 需要分配 Block
     │
     ▼
BlockPool.get_new_blocks()
     │
     ├── 从 free_block_queue 头部取出
     │
     ├── Block 在前缀缓存中？
     │     ├── 是 → _maybe_evict_cached_block() → 移除 hash
     │     └── 否 → 直接使用
     │
     └── ref_cnt = 1 → 返回

═══════════════════════════════════

前缀缓存命中
     │
     ▼
BlockPool.touch(cached_blocks)
     │
     ├── ref_cnt += 1
     │
     └── 从 free_block_queue 移除（如果之前是空闲的）
```

---

## 4. 前缀缓存：Block Hash 与共享

### Hash 链

**文件**: `kv_cache_utils.py:535-562`

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys):
    """链式 hash：每个 Block 的 hash 依赖于父 Block 的 hash。"""
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
```

链式 hash 保证了**前缀唯一性**：

```
Block 0 hash = hash(None, [t0..t15])
Block 1 hash = hash(Block0_hash, [t16..t31])
Block 2 hash = hash(Block1_hash, [t32..t47])

两个请求共享前缀 [t0..t31] → Block 0 和 Block 1 的 hash 完全相同
                          → 可以安全复用相同的物理 Block
```

### Extra Keys

Hash 不仅仅基于 token IDs，还包括额外的 key：
- **LoRA adapter ID**：不同 LoRA 的 KV cache 不同
- **多模态内容 hash**：不同图片/视频的 KV cache 不同
- **Cache salt**：用户指定的缓存隔离

### 查找前缀缓存

**文件**: `vllm/v1/core/kv_cache_manager.py:176-216`

```python
def get_computed_blocks(self, request):
    """为请求查找前缀缓存命中。"""
    computed_blocks, num_new_computed_tokens = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )
    return computed_blocks, num_new_computed_tokens
```

**文件**: `vllm/v1/core/single_type_kv_cache_manager.py:421-`

`FullAttentionManager.find_longest_cache_hit()`：从左到右逐个匹配 Block hash，返回最长前缀匹配。

### 缓存插入

**文件**: `block_pool.py:211-`

`cache_full_blocks()`：当一个 Block 被填满（`block_size` 个 token 全部计算完毕），计算其 hash 并插入 `cached_block_hash_to_block`：

```python
def cache_full_blocks(self, ...):
    for block in new_blocks:
        block_hash = hash_block_tokens(hash_fn, parent_hash, token_ids, extra_keys)
        hash_with_group = make_block_hash_with_group_id(block_hash, group_id)
        self.cached_block_hash_to_block[hash_with_group] = block
        block.block_hash = hash_with_group
```

---

## 5. 从 Scheduler 到 Worker：Block ID 的传递

### Scheduler 侧

```
KVCacheManager.allocate_slots(request, num_new_tokens)
  │  (kv_cache_manager.py:257)
  │
  ├── 阶段 1：释放不需要的 Block（sliding window 超出部分）
  ├── 阶段 2：处理前缀 token（touch 缓存命中 / 分配新 Block）
  ├── 阶段 3：为新 token 分配 Block
  │
  └── 返回 KVCacheBlocks { blocks: tuple[tuple[KVCacheBlock, ...], ...] }
        每个 tuple 对应一个 KV cache group
        每个 KVCacheBlock 有 block_id（物理 ID）
```

### SchedulerOutput 传递

Block ID 通过 `SchedulerOutput` 传递给 Worker：

```python
# NewRequestData 包含：
block_ids: tuple[list[int], ...]       # 每个 KV cache group 的 block ID 列表
new_block_ids: tuple[list[int], ...]   # 新分配的 block ID（增量）
```

### Worker 侧接收

**文件**: `gpu_model_runner.py`

```
_update_states(scheduler_output)
  │
  ├── 新请求 → add_requests()
  │     └── block_tables.append_block_ids(req_index, new_block_ids, overwrite=True)
  │           将完整的 block ID 列表写入 StagedWriteTensor
  │
  └── 已缓存请求 → update_requests()
        └── block_tables.append_block_ids(req_index, new_block_ids, overwrite=False)
              追加新分配的 block ID 到已有列表

block_tables.apply_staged_writes()
  → 异步将 staged 的 block ID 从 CPU 拷贝到 GPU
```

---

## 6. Worker 侧：Block Table 的 GPU 构建

**文件**: `vllm/v1/worker/gpu/block_table.py:13-`

### 数据结构

```python
class BlockTables:
    # 每个 KV cache group 一个 StagedWriteTensor
    # 形状: [max_num_reqs, max_num_blocks]
    self.block_tables: list[StagedWriteTensor]

    # 持久化的 GPU tensor（用于模型 forward pass 和 CUDA Graph）
    self.input_block_tables: list[torch.Tensor]  # 形状同上

    # Slot mapping tensor
    # 形状: [num_kv_cache_groups, max_num_batched_tokens]
    self.slot_mappings: torch.Tensor
```

### Block Table 的两阶段构建

```
阶段 1：Stage Write（CPU 侧，异步）
  block_tables.append_block_ids(req_index, new_block_ids)
    → StagedWriteTensor.stage_write()
    → 数据写入 CPU staging buffer

阶段 2：Apply Write（异步拷贝到 GPU）
  block_tables.apply_staged_writes()
    → StagedWriteTensor.apply_write()
    → Triton kernel 将 CPU 数据 scatter-write 到 GPU tensor
```

### Gather：按 Batch 顺序重排

```
gather_block_tables(idx_mapping, num_reqs_padded)
  → _gather_block_tables_kernel
```

请求在 GPU 上的排列顺序可能与 `block_tables` 中的存储顺序不同。`gather_block_tables()` 使用 Triton kernel 按 `idx_mapping`（batch index → request state index）重排行顺序：

```python
# Triton kernel: _gather_block_tables_kernel (block_table.py:173)
# 对于每个 (kv_cache_group, batch_idx)：
batch_idx = program_id(1)

if batch_idx >= num_reqs:
    # Padding 行填零（CUDA Graph 需要）
    zero out dst_row
else:
    # 从源 block table 按 request state index 读取
    req_idx = idx_mapping[batch_idx]
    copy src[req_idx, :num_blocks] → dst[batch_idx, :num_blocks]
```

---

## 7. Slot Mapping：Token 到物理位置的映射

### 计算

**文件**: `block_table.py:133-158`

`compute_slot_mappings()` 启动 Triton kernel `_compute_slot_mappings_kernel`：

```python
# 简化后的核心逻辑 (block_table.py:252-274)：
for each (group_id, batch_idx):
    for token_position in [start_idx, end_idx):
        # 1. 从 position 计算逻辑 block index 和 block 内偏移
        block_index = position // block_size
        block_offset = position % block_size

        # 2. 从 block table 查找物理 block ID
        block_number = block_table[req_state_idx][block_index]

        # 3. 计算物理 slot ID
        slot_id = block_number * block_size + block_offset

        # 4. 存储 slot mapping
        slot_mapping[group_id][token_position] = slot_id
```

### Slot Mapping 的用途

Slot mapping 被 `reshape_and_cache` kernel 使用，将模型计算出的 K/V 向量写入 KV cache 的正确物理位置。

### Padding

最后一个 program（`batch_idx == num_programs(1) - 1`）负责将剩余 token 位置的 slot mapping 填充为 `PAD_SLOT_ID = -1`（行 234-242），这对 CUDA Graph 正确性至关重要。

---

## 8. GPU 上的 KV Cache Tensor 布局

### FlashAttention 布局

**文件**: `vllm/v1/attention/backends/flash_attn.py:134-146`

```python
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    return (2, num_blocks, block_size, num_kv_heads, head_size)
    #       ↑  K/V   ↑ 物理    ↑ block    ↑ KV头数   ↑ 头维度
    #       分开     block ID  内偏移
```

即：`kv_cache[0]` = Key, `kv_cache[1]` = Value。

```
kv_cache[0]  (Key):  shape = [num_blocks, block_size, num_kv_heads, head_size]
kv_cache[1]  (Value): shape = [num_blocks, block_size, num_kv_heads, head_size]

对于 block_size=16, num_kv_heads=64, head_size=128, FP16:
  一个 Block 的 K = 16 × 64 × 128 × 2 bytes = 256 KB
  K + V = 512 KB per Block
```

### FlashInfer 布局

**文件**: `vllm/v1/attention/backends/flashinfer.py:351-361`

```python
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    return (num_blocks, 2, block_size, num_kv_heads, head_size)
    #       ↑ 物理       ↑ K/V
    #       block ID     在第二维
```

### NHD vs HND 布局

通过 `VLLM_KV_CACHE_LAYOUT` 环境变量控制 stride 顺序：

- **NHD**（默认）：`[num_blocks, block_size, num_kv_heads, head_size]` — token 维度连续
- **HND**：`[num_blocks, num_kv_heads, block_size, head_size]` — head 维度连续

某些 attention kernel（如 FlashAttention 3）对特定布局有更好的性能。

### 内存分配

**文件**: `vllm/v1/worker/gpu/attn_utils.py:127-`

```python
def _allocate_kv_cache(kv_cache_spec, memory):
    # 分配原始 int8 tensor
    raw_tensor = torch.empty(memory, dtype=torch.int8, device=device)

def _reshape_kv_cache(raw_tensor, kv_cache_spec, attn_backend):
    # 根据 backend 的 shape 和 stride order reshape
    shaped = raw_tensor.view(shape).permute(stride_order)
```

---

## 9. Reshape and Cache：写入 KV Cache 的 Kernel

### 写入时机

模型前向推理后，每个 attention 层需要将新计算的 K/V 写入 KV cache。这是通过 `do_kv_cache_update()` 完成的。

### Triton Kernel

**文件**: `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py:18-`

```python
@triton.jit
def reshape_and_cache_kernel_flash(
    key, value, slot_mapping, kv_cache, ...):
    """
    对每个 token：
    1. slot_idx = slot_mapping[token_position]
    2. block_idx = slot_idx // block_size
    3. block_offset = slot_idx % block_size
    4. kv_cache[block_idx, block_offset, head, dim] = key/value[token, head, dim]
    """
    token_idx = tl.program_id(0)
    slot_idx = tl.load(slot_mapping + token_idx)
    if slot_idx == PAD_SLOT_ID:
        return  # padding token，跳过

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # 写入 Key
    for each (head, dim):
        kv_cache[block_idx, block_offset, head, dim] = key[token_idx, head, dim]
    # 写入 Value
    # ... 类似
```

### CUDA Kernel

**文件**: `csrc/cache_kernels.cu:280-`

`reshape_and_cache_flash_kernel` 是 Triton kernel 的 CUDA 实现，支持向量化加载（vectorized loads）以提高显存带宽利用率。

### 写入流程图

```
模型 forward pass → 每层 Attention 输出 key, value 张量 [num_tokens, num_heads, head_size]
     │
     ▼
do_kv_cache_update(key, value, slot_mapping, kv_cache)
  │  (flash_attn.py:843 / flashinfer.py:1654)
  │
  └── reshape_and_cache_flash(key, value, slot_mapping, kv_cache)
        │
        └── Kernel: 对每个 token
              slot_idx = slot_mapping[token_idx]
              block_idx = slot_idx // block_size
              offset = slot_idx % block_size
              kv_cache[block_idx, offset, head, dim] = key[token_idx, head, dim]
```

---

## 10. Attention Kernel 如何使用 Block Table

### FlashAttention

**文件**: `flash_attn.py:674-`

```python
class FlashAttentionImpl:
    def forward(self, query, key, value, attn_metadata, ...):
        # block_table 直接传给 flash_attn_varlen_func
        # 该库内部使用 block_table 进行 paged KV lookup
        output = flash_attn_varlen_func(
            q=query,
            k=key_cache,         # [num_blocks, block_size, ...]
            v=value_cache,       # [num_blocks, block_size, ...]
            block_table=block_table,  # [num_reqs, max_num_blocks_per_req]
            ...
        )
```

FlashAttention 库内部：
1. 对于请求 `i`，逻辑 token `j` 的 K/V 位于 `kv_cache[block_table[i][j // block_size]][j % block_size]`
2. 使用 block table 进行间接寻址（gather）读取 K/V

### FlashInfer

**文件**: `flashinfer.py:794-`

FlashInfer 使用不同的 paged KV 格式：

```python
def _compute_flashinfer_kv_metadata(block_table, seq_lens):
    """
    将 block_table 转换为 FlashInfer 的格式：
    - paged_kv_indptr: [0, num_blocks_req0, num_blocks_req0+num_blocks_req1, ...]
    - paged_kv_indices: 展平的所有 block ID
    - paged_kv_last_page_len: 每个请求最后一个 block 的有效长度
    """
```

`_copy_page_indices_kernel` (行 1776) 是一个 Triton kernel，将 2D block table 展平为 1D `paged_kv_indices`。

---

## 11. Sliding Window 与 Null Block

### 问题描述

Sliding Window Attention（如 Mistral）只关注最近 `W` 个 token 的 KV cache。早期的 Block 不再需要。

### Null Block

**文件**: `kv_cache_utils.py:176`

Block Pool 初始化时，Block 0 被设为 `null_block`（`is_null=True`）。它的 KV cache 内容全为零。

### 滑动窗口 Block 管理

**文件**: `single_type_kv_cache_manager.py:481-`

`SlidingWindowManager`：

```python
class SlidingWindowManager(SingleTypeKVCacheManager):
    def get_num_skipped_tokens(self, num_computed_tokens):
        """计算有多少 token 在 sliding window 之外。"""
        return max(0, num_computed_tokens - self.sliding_window)

    def remove_skipped_blocks(self, request, ...):
        """将超出 window 的 Block 替换为 null_block。"""
        for block in skipped_blocks:
            # 替换为 null_block (block_id=0)
            # 释放原始 block 回空闲池
```

在 Block Table 中，被跳过的位置存储 `null_block`（block_id=0）。Attention kernel 读取 block 0 时得到全零的 K/V，等效于忽略这些位置。

---

## 12. Cascade Attention：共享前缀优化

**文件**: `flash_attn.py:1045-`

当多个请求共享相同的前缀（如相同的 system prompt），可以将 attention 分为两步：

```
普通方式（N 个请求，共享前缀 P）：
  请求 0: Q₀ × [P | S₀]  →  attention([共享前缀 + 请求0后缀])
  请求 1: Q₁ × [P | S₁]  →  attention([共享前缀 + 请求1后缀])
  ...
  请求 N: Qₙ × [P | Sₙ]  →  attention([共享前缀 + 请求N后缀])
  → 共享前缀 P 被计算了 N 次！

Cascade Attention：
  步骤 1: [Q₀,Q₁,...,Qₙ] × P  →  prefix_output（只计算一次！）
  步骤 2: Q₀ × S₀, Q₁ × S₁, ..., Qₙ × Sₙ  →  suffix_output（各计算各的）
  步骤 3: merge(prefix_output, suffix_output)  →  最终输出
```

### 共享 Block 识别

**文件**: `kv_cache_manager.py:476`

```python
def get_num_common_prefix_blocks(self, ...):
    """判断 Block 是否是所有活跃请求共享的（ref_cnt == num_active_requests）。"""
```

### Cascade Attention 实现

**文件**: `flash_attn.py:1123-`

```python
def cascade_attention(self, query, kv_cache, block_table, ...):
    # 步骤 1：共享前缀 attention（block_table 的第一行代表共享前缀）
    prefix_output = flash_attn_varlen_func(
        q=query,
        k=kv_cache[0], v=kv_cache[1],
        block_table=block_table[:1],  # 只取第一行（共享前缀 blocks）
        ...
    )

    # 步骤 2：各请求后缀 attention
    suffix_output = flash_attn_varlen_func(
        q=query,
        k=kv_cache[0], v=kv_cache[1],
        block_table=block_table[:, num_common_kv_blocks:],  # 跳过共享部分
        ...
    )

    # 步骤 3：合并
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
```

---

## 13. 跨层 KV Sharing

某些模型（如某些 GQA 架构）中，多个 attention 层共享相同的 KV cache。

**文件**: `vllm/model_executor/layers/attention/attention.py`

```python
class Attention:
    def __init__(self, ..., kv_sharing_target_layer_name=None):
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
```

当 `kv_sharing_target_layer_name` 被设置时：
1. **Scheduler 侧**：不为该层分配独立的 KV cache blocks，复用目标层的 blocks
2. **Worker 侧**：该层的 attention 直接读取目标层的 KV cache tensor
3. **Cache 更新**：该层跳过 `reshape_and_cache`，不写入新 K/V

这意味着 Block Table 中的 Block 同时服务多个 attention 层。

---

## 14. 代码阅读顺序

### 第一阶段：理解 Block 元数据

| 顺序 | 文件 | 关键内容 |
|------|------|----------|
| 1 | `v1/core/kv_cache_utils.py:110-155` | `KVCacheBlock` — Block 的元数据 |
| 2 | `v1/core/kv_cache_utils.py:158-` | `FreeKVCacheBlockQueue` — 空闲 Block 双向链表 |
| 3 | `v1/core/kv_cache_utils.py:535-562` | `hash_block_tokens()` — 链式 hash |

### 第二阶段：理解 Block 分配与管理

| 顺序 | 文件 | 关键内容 |
|------|------|----------|
| 4 | `v1/core/block_pool.py:130-` | `BlockPool` — 分配、释放、前缀缓存 |
| 5 | `v1/core/block_pool.py:322` | `get_new_blocks()` — 分配 |
| 6 | `v1/core/block_pool.py:408` | `free_blocks()` — 释放 |
| 7 | `v1/core/block_pool.py:211` | `cache_full_blocks()` — 缓存插入 |

### 第三阶段：理解调度器侧的 KV Cache 管理

| 顺序 | 文件 | 关键内容 |
|------|------|----------|
| 8 | `v1/core/kv_cache_manager.py:106` | `KVCacheManager` — 顶层管理器 |
| 9 | `v1/core/kv_cache_manager.py:176` | `get_computed_blocks()` — 前缀缓存查找 |
| 10 | `v1/core/kv_cache_manager.py:257` | `allocate_slots()` — Block 分配（含布局图） |
| 11 | `v1/core/kv_cache_manager.py:429` | `free()` — Block 释放 |
| 12 | `v1/core/single_type_kv_cache_manager.py:421` | `find_longest_cache_hit()` — 最长前缀匹配 |

### 第四阶段：理解 Worker 侧 Block Table

| 顺序 | 文件 | 关键内容 |
|------|------|----------|
| 13 | `v1/worker/gpu/block_table.py:13` | `BlockTables` — GPU block table 管理 |
| 14 | `v1/worker/gpu/block_table.py:87` | `append_block_ids()` — 写入 block ID |
| 15 | `v1/worker/gpu/block_table.py:106` | `gather_block_tables()` — 重排到 batch 顺序 |
| 16 | `v1/worker/gpu/block_table.py:133` | `compute_slot_mappings()` — 计算 slot mapping |
| 17 | `v1/worker/gpu/block_table.py:212` | `_compute_slot_mappings_kernel` — Triton kernel |

### 第五阶段：理解 KV Cache Tensor 和写入

| 顺序 | 文件 | 关键内容 |
|------|------|----------|
| 18 | `v1/attention/backends/flash_attn.py:134` | `get_kv_cache_shape()` — Tensor 形状 |
| 19 | `v1/attention/ops/triton_reshape_and_cache_flash.py:18` | reshape_and_cache kernel |
| 20 | `csrc/cache_kernels.cu:280` | CUDA 版 reshape_and_cache |

### 第六阶段：理解 Attention 如何使用 Block Table

| 顺序 | 文件 | 关键内容 |
|------|------|----------|
| 21 | `v1/attention/backends/flash_attn.py:674` | `FlashAttentionImpl.forward()` |
| 22 | `v1/attention/backends/flashinfer.py:794` | FlashInfer paged KV 转换 |
| 23 | `v1/attention/backends/flash_attn.py:1123` | Cascade Attention |

### 第七阶段：进阶主题

| 顺序 | 文件 | 关键内容 |
|------|------|----------|
| 24 | `v1/core/single_type_kv_cache_manager.py:481` | Sliding Window Block 管理 |
| 25 | `v1/kv_cache_interface.py` | `KVCacheSpec`, `FullAttentionSpec` 等类型定义 |
| 26 | `v1/worker/gpu/buffer_utils.py:101` | `StagedWriteTensor` — 异步写入机制 |

---

## 附录：完整数据流图

```
┌─────────────────────── Scheduler 侧（CPU）─────────────────────────┐
│                                                                    │
│  Request: prompt = [t0, t1, t2, ..., t127]                       │
│                                                                    │
│  Step 1: 计算前缀缓存命中                                          │
│    get_computed_blocks()                                           │
│      └── find_longest_cache_hit()                                  │
│            遍历 block_hashes → 查找 cached_block_hash_to_block     │
│            结果: Block 0-3 命中（t0-t63 已缓存）                   │
│                                                                    │
│  Step 2: 分配新 Block                                              │
│    allocate_slots(request, num_new_tokens=64)                      │
│      ├── touch(cached_blocks) → ref_cnt++ on Block 0-3            │
│      └── get_new_blocks(4) → 从空闲队列取 Block 7,8,9,10         │
│                                                                    │
│  Block Table（逻辑视图）:                                          │
│    [B0, B1, B2, B3, B7, B8, B9, B10]                              │
│     ↑── 前缀缓存命中 ──↑  ↑── 新分配 ──────↑                      │
│                                                                    │
│  返回 KVCacheBlocks: block_ids = [0,1,2,3,7,8,9,10]              │
│                                                                    │
└────────────────────────────┬───────────────────────────────────────┘
                             │ SchedulerOutput
                             ▼
┌─────────────────────── Worker 侧（GPU）───────────────────────────┐
│                                                                    │
│  Step 3: 写入 Block Table                                          │
│    block_tables.append_block_ids() → StagedWriteTensor            │
│    block_tables.apply_staged_writes() → async GPU copy            │
│                                                                    │
│  GPU Block Table Tensor:                                           │
│    row[req_idx] = [0, 1, 2, 3, 7, 8, 9, 10, 0, 0, ...]          │
│                                                                    │
│  Step 4: Gather 到 batch 顺序                                      │
│    gather_block_tables(idx_mapping)                                │
│    → input_block_tables[batch_idx] = [0,1,2,3,7,8,9,10,0,...]    │
│                                                                    │
│  Step 5: 计算 Slot Mapping                                         │
│    compute_slot_mappings()                                         │
│    → 对每个 token:                                                 │
│      token 64: pos=64, block_idx=4, offset=0                      │
│                 block_number = block_table[4] = 7                  │
│                 slot_id = 7 * 16 + 0 = 112                        │
│      token 65: pos=65, block_idx=4, offset=1                      │
│                 slot_id = 7 * 16 + 1 = 113                        │
│      ...                                                           │
│                                                                    │
│  Step 6: 模型前向推理                                               │
│    model(**inputs) → key, value 张量                                │
│                                                                    │
│  Step 7: 写入 KV Cache                                             │
│    reshape_and_cache_flash(key, value, slot_mapping, kv_cache)    │
│    → kernel: kv_cache[block_idx, offset, head, dim] = key[...]    │
│      token 64 → kv_cache[7, 0, :, :] = key[64, :, :]             │
│      token 65 → kv_cache[7, 1, :, :] = key[65, :, :]             │
│      ...                                                           │
│                                                                    │
│  Step 8: Attention 使用 Block Table                                │
│    flash_attn_varlen_func(q, k_cache, v_cache, block_table)       │
│    → 库内部：通过 block_table 间接寻址读取 KV                      │
│      query[64] 对应的 key/value 在:                                │
│        k_cache[block_table[req][64//16]][64%16] = k_cache[4][0]   │
│        = k_cache[7][0]  （因为 block_table[4]=7）                  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

KV Cache Tensor 物理布局 (FlashAttention NHD):
  k_cache shape: [num_blocks, block_size, num_kv_heads, head_size]

  Block 0: [t0_K,  t1_K,  ..., t15_K ]    ← 请求A 和 请求B 共享（前缀缓存）
  Block 1: [t16_K, t17_K, ..., t31_K]     ← 请求A 和 请求B 共享
  Block 2: [t32_K, t33_K, ..., t47_K]     ← 请求A 和 请求B 共享
  Block 3: [t48_K, t49_K, ..., t63_K]     ← 请求A 和 请求B 共享
  Block 7: [t64_K, t65_K, ..., t79_K]     ← 请求A 独占
  Block 8: [t80_K, t81_K, ..., t95_K]     ← 请求A 独占
  Block 9: [t96_K, t97_K, ..., t111_K]    ← 请求A 独占
  Block 10:[t112_K,t113_K,..., t127_K]    ← 请求A 独占

  Block Table (逻辑→物理映射):
    请求A: [0, 1, 2, 3, 7, 8, 9, 10]   ← 8 个 Block
    请求B: [0, 1, 2, 3]                  ← 4 个 Block，复用 Block 0-3！
```

---

## 关键概念速查表

| 概念 | 源码位置 | 说明 |
|------|----------|------|
| Block 元数据 | `kv_cache_utils.py:110` KVCacheBlock | block_id, ref_cnt, hash, 双向链表指针 |
| 空闲 Block 队列 | `kv_cache_utils.py:158` FreeKVCacheBlockQueue | 双向链表，LRU 排序 |
| Block Pool | `block_pool.py:130` BlockPool | 分配、释放、前缀缓存管理 |
| 链式 Hash | `kv_cache_utils.py:535` hash_block_tokens | parent_hash + token_ids → block_hash |
| 前缀缓存查找 | `kv_cache_manager.py:176` get_computed_blocks | 最长前缀匹配 |
| Block 分配 | `kv_cache_manager.py:257` allocate_slots | 含完整 Block 布局图 |
| GPU Block Table | `block_table.py:13` BlockTables | StagedWriteTensor + 持久化 GPU tensor |
| Gather | `block_table.py:106` gather_block_tables | Triton kernel 按 batch 顺序重排 |
| Slot Mapping | `block_table.py:133` compute_slot_mappings | pos → block_idx → block_number → slot_id |
| KV Cache 形状 | `flash_attn.py:134` get_kv_cache_shape | (2, num_blocks, block_size, heads, dim) |
| 写入 Kernel | `triton_reshape_and_cache_flash.py:18` | slot_id → block + offset → 写入 K/V |
| Attention 使用 | `flash_attn.py:674` forward | block_table 传给 flash_attn 库 |
| Cascade Attn | `flash_attn.py:1123` cascade_attention | 共享前缀只计算一次 |
| Null Block | `block_pool.py:176` null_block | block_id=0，全零，sliding window 用 |
| KV Sharing | `attention.py:372` kv_sharing_target_layer_name | 跨层共享 KV cache |
| NHD/HND 布局 | `flash_attn.py:146` get_kv_cache_stride_order | 环境变量控制 stride 顺序 |
