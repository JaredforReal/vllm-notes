# vLLM Profiler 输出分析指南

## 基于 GLM5Next (TP8, MTP=3, CUDA Graphs) 的实际数据

---

## 一、profiler_out_0.txt 解读

### 列定义

| 列名 | 含义 |
|------|------|
| **Name** | 操作名称（kernel、Python 函数、通信操作等）|
| **Self CPU %** | 该操作自身 CPU 时间占总 CPU 时间的百分比 |
| **Self CPU** | 该操作自身的 CPU 执行时间（不含子操作）|
| **CPU total %** | 该操作及其子操作的 CPU 时间占比 |
| **CPU total** | 该操作及其子操作的总 CPU 时间 |
| **CPU time avg** | 每次调用的平均 CPU 时间 |
| **Self CUDA** | 该操作自身的 GPU kernel 执行时间 |
| **Self CUDA %** | 该操作 GPU 时间占总 GPU 时间的百分比（**最关键的列**）|
| **CUDA total** | 该操作及其子操作的 GPU 总时间 |
| **CUDA time avg** | 每次调用的平均 GPU 时间 |
| **# of Calls** | 被调用的次数 |

**核心关注点**：**Self CUDA %** 和 **Self CUDA**。它们告诉你 GPU 时间花在了哪里。

### 你的数据逐行分析

```
Name                                                Self CUDA    Self CUDA %    # of Calls
──────────────────────────────────────────────────  ──────────   ───────────    ──────────
execute_context_0(0)_generation_1(1)                 34.449s       105.28%           254
```

这是 **decode iteration 的标注**（0 个 prefill、1 个 decode 请求）。CUDA % 超过 100% 是因为
PyTorch profiler 的计时方式——多个 CUDA stream 上有重叠执行。这个是顶层容器，包含下面所有 kernel。

```
cross_device_reduce_1stage<__nv_bfloat16,...>         29.503s        90.17%        11648
```

**这是最大的瓶颈——TP 通信（Tensor Parallel All-Reduce）占了 90% 的 GPU 时间！**
TP=8 意味着每层 attention/MLP 之后都要做 all-reduce，通信量很大。

```
record_param_comms                                      2.219s         6.78%          128
ncclDevKernel_AllGather_RING_LL(...)                    2.219s         6.78%          128
nccl:_all_gather_base                                   2.219s         6.78%          128
```

**AllGather 通信**——可能来自 MoE 的 expert 并行或 EP（Expert Parallelism）的参数同步。
128 次调用对应模型层数。

```
execute_context_1(5)_generation_0(0)                   1.811s          5.54%           43
```

**Prefill iteration**（1 个 prefill 请求 5 tokens，0 个 decode）。开销远小于 decode。

```
fused_moe_kernel                                      189.133ms         0.58%        10752
```

MoE 的 fused kernel。10752 次调用 = 43 层 MoE × 多个 batch iteration。本身开销不大（0.58%）。

```
nvjet_tst_64x8_64x16_*                                 ~218ms         ~0.66%        ~46000
```

这些是 **JIT 编译的 Triton kernel**（nvjet = nvFuser/triton JIT），用于 element-wise 操作。

```
mhc_pre_big_fuse_tilelang_kernel                       66.625ms         0.20%        11520
mhc_post_tilelang_kernel                               37.803ms         0.12%        11520
```

**MHC (Multi-Head Checkpointing) 的 pre/post 操作**——你模型的特有组件。11520 次 ≈
层数 × batch iterations × 2（每层有 attn + ffn 两个 HC 操作）。

```
_causal_conv1d_update_kernel                           28.791ms         0.09%        12954
```

**KDA 层的因果卷积更新**——decode 阶段的 1D conv 状态更新。

```
deep_gemm::fp8_gemm_kernel_swapAB                      ~58ms          ~0.18%        ~10000
```

FP8 GEMM kernel，用于 MoE expert 的矩阵乘法。

```
vllm::moe::grouped_topk_fused_small_expert_count       22.334ms         0.07%         5376
```

MoE 的 TopK 路由选择 kernel。

### 性能结论

```
GPU 时间分布（近似）：

TP All-Reduce (cross_device_reduce)   ████████████████████████████████████████  90.2%
AllGather (EP/参数通信)               ████                                      6.8%
Prefill iterations                    ███                                       5.5%
MoE + KDA + MHC + 其他                ██                                        ~2%
```

**关键发现**：**TP=8 的 All-Reduce 通信是绝对瓶颈**，占了 90% 的 GPU 时间。
实际计算（attention、MoE、MLP）只占约 2-3%。这意味着 GPU 大部分时间在等通信完成。

---

## 二、Perfetto UI 使用指南

### 1. 加载 Trace

1. 打开 https://ui.perfetto.dev/
2. 将 `.pt.trace.json.gz` 文件拖入页面（不需要解压）
3. 建议先加载 **rank 0** 的 trace（`dp0_pp0_tp0_dcp0_ep0_rank0.*.json.gz`）

### 2. 界面导航

```
┌─────────────────────────────────────────────────────────────────────┐
│  Perfetto UI                                                        │
│                                                                     │
│  ┌─ 搜索栏 ─────────────────────────────────────────────────────┐  │
│  │ 搜索 kernel 名、op 名等                                       │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ 时间线区域 ─────────────────────────────────────────────────┐  │
│  │                                                               │  │
│  │  ▸ cpu:0  ████████  ██  ████████████  ██  ██████████        │  │
│  │  ▸ cpu:1  ████████  ██  ████████████  ██  ██████████        │  │
│  │  ▸ ...                                                        │  │
│  │  ▸ CUDA (gpu)  ████████████████████████████████████████████  │  │
│  │      ├── aten::linear                                         │  │
│  │      ├── nccl:_all_reduce                                     │  │
│  │      ├── fused_moe_kernel                                     │  │
│  │      └── ...                                                  │  │
│  │                                                               │  │
│  │  ←───── 时间轴 ──────→（鼠标滚轮缩放，拖拽平移）              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─ 详情面板（点击某个 slice 后显示）────────────────────────────┐  │
│  │  Name: fused_moe_kernel                                       │  │
│  │  Duration: 17.591us                                           │  │
│  │  Thread: worker-0                                             │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  快捷键: W/A 放大/缩小, S/D 左右移动, F 聚焦选中区域               │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. 关键操作

| 操作 | 方法 |
|------|------|
| 缩放时间轴 | 鼠标滚轮，或 W（放大）/ A（缩小）|
| 平移 | 鼠标拖拽，或 S（左移）/ D（右移）|
| 聚焦某个区域 | 选中一个 slice 后按 F |
| 搜索 | Ctrl+F，输入 kernel 名 |
| 查看 slice 详情 | 点击任意 slice |
| 看调用栈 | 点击 slice 后看下方的 Stack 标签 |

### 4. 找 Bubble（性能空隙）

**Bubble = GPU 时间线上的空白区域**，即 GPU 空闲、没在执行任何 kernel。

在 Perfetto 中：
1. 找到 **CUDA** 那一行（或 `stream 7` 等 GPU stream）
2. 看连续 kernel 之间的间隙
3. 常见的 bubble 来源：
   - **通信等待**：TP all-reduce 完成前 GPU 空闲
   - **CPU dispatch 延迟**：CPU 端发起 kernel 的延迟
   - **CUDA Graph capture**：首次 capture 时有额外开销（运行时无）
   - **调度延迟**：scheduler 准备下一个 batch 的时间

### 5. 常用搜索关键词

```
搜索内容              →  看什么
──────────────────────────────────────
all_reduce            →  TP 通信开销
all_gather            →  EP/参数通信
fused_moe             →  MoE 计算
kda_attention         →  KDA 线性注意力
causal_conv1d         →  KDA 卷积
mhc_pre / mhc_post    →  MHC 操作
execute_context       →  Prefill iteration
execute_generation    →  Decode iteration
```

### 6. 火焰图（Flame Graph）

Perfetto 中的 "火焰图" 就是时间线上的 **slice 嵌套**：

```
层级结构（从外到内）：

execute_generation_1(1)                    ← iteration 标注
  └─ Glm5NextDecoderLayer (L0)            ← 模型层
       ├─ self_attn (MLA 或 KDA)           ← attention
       │    ├─ q_proj / k_proj / v_proj
       │    ├─ attention kernel
       │    └─ o_proj
       ├─ mlp (MoE 或 Dense)              ← FFN
       │    ├─ gate_proj / up_proj
       │    ├─ fused_moe_kernel
       │    └─ down_proj
       └─ cross_device_reduce             ← TP 通信
```

每一层是一个 slice，嵌套关系反映调用栈。点击外层 slice 可以折叠/展开。

> **注意**：默认情况下 vLLM 不会给每个模型层加 NVTX 标注。要看到上面这种层级结构，
> 需要开启 `--observability-config '{"enable_layerwise_nvtx_tracing": true}'`
> （但这与 CUDA Graph 不兼容）。

### 7. 对比多个 Rank

可以同时加载多个 rank 的 trace 文件来对比：
1. 在 Perfetto 中点 "Open trace" 再加载 rank 1 的文件
2. 两个 trace 会上下排列显示
3. 可以对比不同 rank 的 kernel 执行时间是否均衡（load balance）

---

## 三、针对你的模型的性能建议

### 核心发现：TP=8 通信是瓶颈

90% 的 GPU 时间花在 `cross_device_reduce`（TP All-Reduce）上，说明：
- 模型的 per-layer 计算量相对于 TP=8 的通信开销太小
- GPU 计算和通信没有充分 overlap

### 可能的优化方向

| 方向 | 说明 |
|------|------|
| **减少 TP 大小** | 如果单卡能放下，TP=4 或 TP=2 能大幅减少通信 |
| **通信/计算 overlap** | 检查 vLLM 的 TP 是否支持 pipeline overlap（如 `--tp-enable-parallel-softmax`）|
| **AllGather 优化** | 128 次 AllGather 可能来自 EP，检查是否有不必要的同步 |
| **FP8 量化** | 已经在用 FP8 GEMM，通信也可以考虑压缩 |

### 下一步排查

1. 在 Perfetto 中搜索 `cross_device_reduce`，看每次通信的耗时
2. 看通信和计算是否有 overlap（时间线上是否并行）
3. 如果通信是完全串行的（没有 overlap），这是主要优化点
4. 用 `nvidia-smi` 确认 GPU 利用率（如果利用率低，说明 GPU 在等通信）
