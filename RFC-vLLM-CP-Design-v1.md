# RFC: vLLM Context Parallel (CP) 目标架构设计

> 状态：Draft v2 / 供审视（已按 vLLM **main** 分支现状重写）
> 范围：PCP（Prefill CP）+ DCP（Decode CP），覆盖 GQA / MLA / DSA / SFA / Linear(Mamba/GDN/KDA) 及 Hybrid 模型。
> 参考实现：vLLM main（`vllm/v1/attention/backends/flash_attn.py`、`vllm/v1/attention/ops/dcp_alltoall.py`、`vllm/v1/worker/cp_utils.py`、`vllm/v1/worker/gpu/cp_utils.py`、`vllm/distributed/parallel_state.py`）、`vllm-ascend`、`sglang`。源码拆解见 `learning_doc/CP-learning-doc/`。

---

## 1. 背景与动机（按 main 重述）

**先纠正一个常见误解：vLLM main 已经有 CP 了，而且设计相当克制。** main 的现状：

- 配置：`parallel_config.prefill_context_parallel_size` / `decode_context_parallel_size`，`cp_size = pcp_size * dcp_size`。
- 通信域：`vllm/distributed/parallel_state.py` 提供 `get_pcp_group()` / `get_dcp_group()`（标准 `GroupCoordinator`）。
- 基础集合通信：**复用 `vllm/distributed` 已有的** `all_gather / all_gatherv / all_reduce / reduce_scatter`——CP 不另造。
- CP 专属的"算法复合原语"（online-softmax out/lse 合并这类 CP 配方）：`vllm/v1/attention/ops/dcp_alltoall.py`（`dcp_a2a_lse_reduce`、`_lse_weighted_combine`）。
- **CP 折叠进 attention backend**，没有独立 CP backend：`flash_attn.py` 里 `if self.dcp_world_size > 1:` 内联处理。
- worker 侧只有轻量 `cp_utils.py`（兼容性检查 + `prepare_dcp_local_seq_lens`），**无 fat Manager 对象**。

**真正的 gap 不是"没 CP"，而是：**
1. **广度不足**：main 的 CP 基本围绕 dense GQA 的 DCP（flash_attn）。MLA / DSA / SFA / Linear(Mamba/GDN/KDA) / Hybrid 的 CP 支持缺失或不完整。
2. **编排分散、缺统一抽象**：CP 逻辑散在 `flash_attn.py` 内联 + 两个 `cp_utils.py`，没有一处统一承载"切分策略 + 各类型编排"，导致扩展新类型时无处下手、易重复实现。
3. 相比之下 vllm-ascend / sglang 在 hybrid、纯线性、稀疏注意力上已落地（参考 `learning_doc/CP-learning-doc/`）。

本 RFC 的主张：**保留 main 的克制美德（折叠 backend、复用 distributed 原语、CP 复合算子独立成模块），补一个轻量的统一编排抽象 `PCPManager`（只管 Layout + Strategy），并把广度扩到 MLA/DSA/Linear/Hybrid。** 不照搬 ascend 的重 PCPManager + 独立 CP backend。

---

## 2. 现状三方对比

| 维度 | vLLM **main** | vllm-ascend | sglang |
|---|---|---|---|
| 基础集合通信 | ✅ 复用 `vllm/distributed` | 自有（NPU） | 复用 `dp_attention` group |
| CP 复合算子 | `v1/attention/ops/dcp_alltoall.py` 等 | `common_cp.py` | `layers/utils/cp_utils.py` |
| CP backend | **折叠进 flash_attn**（无独立 CP backend）✅ | 独立 CP impl per 类型（4 个）❌ | 折叠进 flashattention backend ✅ |
| 编排抽象 | 散在 flash_attn 内联 + 两个 cp_utils ❌ | fat `PCPManager`（含内核索引）❌ | 纯函数 + 闭包，但**泄漏进 modeling** ❌ |
| model CP 透明度 | 高（model 不感知 CP）✅ | 高 ✅ | **低**（每模型 forward 写 `cp_split_*`）❌ |
| metadata 形状 | 偏 layout 级 | 混了 kernel 索引 ❌ | layout/policy 级 ✅ |
| 广度 | dense GQA 为主 | GQA/MLA/DSA/SFA/Linear/Hybrid 全 | dense + DSA，无 Linear/Hybrid CP |

**结论：main 的"折叠 backend + 复用 distributed + 复合算子成模块 + model 透明"四点是正确的方向，要保留。** 要改的只有两件：①补统一编排抽象（轻量 PCPManager）；②扩广度。ascend 的 fat Manager + 独立 CP backend 是要避免的；sglang 的 modeling 泄漏是要避免的。

---

## 3. 目标架构

### 3.1 不可消除的两段切分

CP 职责天然分两段：

1. **Token 划分**（哪个 rank 处理哪些 token、positions）——影响整个 forward，必须在 **batch 准备阶段**定好（main 现在的 `prepare_dcp_local_seq_lens` 就是干这事的雏形）。
2. **每层 gather/merge / 状态传播**——发生在 attention 层内部，折叠进 backend（main 现状）。

两段必须分别归位，但统一收口在一个轻量对象 `PCPManager` 名下。

### 3.2 PCPManager 契约（**只拥有 Layout + Strategy，不拥有通信原语**）

```
PCPManager                         ← CP 的唯一编排化身（轻量）
├─ CPLayout（token 划分）：batch + policy → 每 rank token 切片 / positions / cp_rank / cp_size
│     · policy 可插拔：Zigzag / Linear / HeadSplit / RoundRobin
│     · 整合 main 现有的 prepare_dcp_local_seq_lens 等
└─ CPStrategy（编排选择）：为给定 backend + policy 选出"用哪些复合算子、按什么顺序"
      · 全注意力 decode：DCP head-ag Q → 本地 attn → dcp_a2a_lse_reduce
      · 全注意力 prefill：AllGather-KV / AllGather-Q（按类型）
      · Linear：conv1d 边界 allgather + ssm state 传播
      · hybrid FullAttention：enter/exit FA 布局转换
      ── 这些"配方"由 PCPManager 选择与排序，但其执行调用的是下面两类既有设施 ──

【PCPManager 调用、但不拥有的两类设施】
• 基础集合通信：vllm/distributed 的 GroupCoordinator.all_gather/all_to_all/reduce_scatter（已存在）
• CP 复合算子：vllm/v1/attention/ops/ 的 dcp_a2a_lse_reduce / merge_attn_states / conv1d 边界…（已存在或待补）
```

**关键：PCPManager 是编排者，不是通信库。** 它消费 `vllm/distributed` 和 `v1/attention/ops`，自身不实现 all_gather/a2a。这样既不重复造轮子（你担心的点），又给 CP 一个单一编排入口。

### 3.3 ⚑ 关键设计纪律（写死）

> **PCPManager 只产出 layout 级 metadata，绝不产出 kernel 级索引。**

这是「能消灭独立 CP backend」的命门。ascend 的 `generate_pcp_metadata` 还在生成 `q_head_idx`/`kv_tail_proj_idx`/`pcp_enter_fa_restore_idx`… —— 只要 kernel 专属索引在 Manager 里，base backend 就无法 CP-uniform，就得开独立 CP backend。

| | 保留在 PCPManager（layout 级） | 移出，归 base backend（kernel 级） |
|---|---|---|
| token 切片 / positions / cp_rank / 每 rank 长度 | ✅ | |
| prev/next 序列边界 | ✅ | |
| GQA 的 nomask/mask 拆分索引 | | ✅ forward 内算 |
| MLA tail-only `kv_b_proj` 索引 | | ✅ MLA backend 内算 |
| hybrid 的 enter/exit FA restore/scatter 索引 | | ✅ 该层 strategy 内算 |

main 现状的 metadata 偏 layout 级（`dcp_context_kv_lens` 等），这点方向是对的，要保持。

### 3.4 三方最终边界

```
[Modeling]          纯净：self.attn(q,k,v,...)，不知 CP 存在（main 现状已如此）
                        │  tokens 由 batch-prep 按 CPLayout 切好
                        ▼
[batch-prep]        pcp_manager.layout(batch) → 每 rank token 视图（整合 main 的 cp_utils）
                        │
                        ▼
[Attention Backend] 一个类（无 CP 变体，main 现状已如此）：
                    o = pcp_manager.strategy_for(self).apply(q,k,v,...)
                      · 内部按 strategy 调用 distributed 原语 + v1/attention/ops 复合算子
                      · kernel 专属索引在本 forward 内算
                      · cp_size==1 时 strategy.apply 是 no-op
                        ▼
[vllm/distributed]   all_gather / all_to_all / reduce_scatter（既有，CP 复用）
[v1/attention/ops]   dcp_a2a_lse_reduce / merge_attn_states / conv1d 边界…（既有/待补）
```

### 3.5 对照硬要求

| 要求 | 落实 |
|---|---|
| PCPManager **不**拥有通信原语 | ✅ 只做 Layout + Strategy 编排；通信走 `vllm/distributed`，复合算子走 `v1/attention/ops` |
| 消灭 CP 特殊 backend | ✅ main 现状已是折叠 backend；靠 §3.3 纪律保持，新类型不破例开 CP backend |
| modeling 无 CP if/else | ✅ main 现状已透明；扩展时保持 |
| 统一编排入口 | ✅ PCPManager 收口 main 散落的 cp_utils + flash_attn 内联编排 |

---

## 4. 各方借鉴清单（修正版）

| 模块 | 采纳来源 | 说明 |
|---|---|---|
| 基础集合通信 | **main 既有** `vllm/distributed` | CP 复用，不另造（**PCPManager 不碰**） |
| CP 复合算子模块 | **main 既有** `v1/attention/ops/` | `dcp_a2a_lse_reduce` 等已存在；为 Linear/hybrid 补 conv1d 边界、enter/exit 等 |
| CP backend 归属 | **main 现状**（折叠进 backend） | 保持，不学 ascend 的独立 CP impl |
| model 透明度 | **main 现状** | 保持，不学 sglang 的 modeling 泄漏 |
| metadata 形状 | **main/sglang**（layout 级） | 不学 ascend 的 kernel 索引混入 |
| 统一编排抽象 | **新增** PCPManager（Layout+Strategy） | 收口 main 散落逻辑 |
| per-type strategy 配方 | 借 **ascend** 的实现经验 | 但配方里的算子调 distributed + ops，不调 fat Manager |
| 单点 adaptor + 薄闭包 | 借 **sglang** 的 `cp_attn_forward_extend` 思路 | 泛化成 policy 驱动 |
| policy 可插拔（L1） | **新增** | Zigzag/Linear/HeadSplit/RoundRobin |

> 注：本节不再引用任何"upstream plain mode / kimi_linear 模型级 allgather"——那是试验分支内容，非 main。main 上 kimi_linear 等混合模型的 CP 属于本 RFC 要**新增**的广度，不是既有的。

---

## 5. L1 策略层（policy）—— main/sglang 都缺、RFC 要补

把切分方式抽成可插拔 `ShardingPolicy`，policy 决定「token 怎么切、strategy 配方调几次、metadata 什么形状」。

| Policy | 适用 | 切法 |
|---|---|---|
| **Zigzag**（DualChunkSwap） | dense 全注意力（负载均衡） | head/tail 配对，pad 到 2·cp |
| **Linear**（contiguous） | hybrid（迁就线性层递归） | rank r 拿 `[r·c,(r+1)·c)` 连续段 |
| **HeadSplit** | pure-linear（head 独立） | rank r 持 num_heads/cp head，跑完整 seq |
| **RoundRobin** | DSA / 特定稀疏 / DCP KV 分片 | 按 interleave 粒度轮流（main 的 `cp_kv_cache_interleave_size` 已有雏形） |

policy 是**唯一**按模型拓扑注册的东西。加 KimiLinear = 注册 Linear policy，不碰 backend、不碰 Manager 内核逻辑。

---

## 6. 线性/递归注意力的 CP：按拓扑选策略

线性注意力（Mamba/GDN/KDA）head 互相独立，两条合法路线：

| | seq-split + 状态传播 | head-split |
|---|---|---|
| 跨 rank 依赖 | 有（串行链 r0→r1） | 无（head 独立） |
| 状态 cache 显存 | 不省（全 head 状态 cp 份冗余） | 省 cp× |
| conv1d 路径 | 需注入跨 rank 初态 | 可用 fused 算子 |
| decode | 零通信 | 每层 reduce output |
| 与 hybrid 注意力层布局 | 一致 | 不一致（要布局转换） |

**决策规则：**
- **Hybrid（qwen3-next / GLM / Kimi-Linear）→ seq-split**：和相邻全注意力层共享 Linear-split 布局，decode 零通信。
- **纯线性（gpt-oss / jamba / 纯 Mamba2）→ head-split**：无状态链、省状态显存、简单。

落地优先级：**先 seq-split**（覆盖主流 hybrid），抽象层允许 head-split 后续可选。

---

## 7. 各注意力类型的 strategy 配方映射

| 类型 | prefill | decode | 配方用到的设施 |
|---|---|---|---|
| GQA（dense） | Mode1: AllGather-KV + nomask/mask 拆分 | Mode2: DCP head-ag Q + `dcp_a2a_lse_reduce` | `distributed.all_gather` + `ops/dcp_alltoall` |
| MLA | Mode1: AllGather kv_c+k_pe，tail-only `kv_b_proj` | 同 GQA 骨架 + v_up_proj | `distributed` + `ops/merge_attn_states` |
| DSA | TP 域线性切片 + local indexer/compressor metadata | output TP all_to_all 还原 | `distributed.all_to_all` |
| Linear（Mamba/GDN/KDA） | seq-split: conv1d 边界 allgather + ssm state | state 已分片，零额外通信 | 新增 `ops`：conv1d 边界 + ssm state |
| Hybrid FullAttention | Linear policy + enter/exit FA 布局转换 | 复用 GQA decode | `distributed` + 新增 enter/exit idx（backend 内算） |

通信密度：**decode > 混合 > prefill** → 优化重心放 decode 的 out/lse 合并。

---

## 8. 三种 batch 场景的回归基线

| 场景 | Mode | 关键通信 |
|---|---|---|
| 纯 Prefill | Mode1 | PCP allgather KV（每层） |
| 纯 Decode | Mode2 | DCP ag Q(head) + `dcp_a2a_lse_reduce`（含 PCP ag）|
| Prefill+Decode 混合 | Mode1+2 串行 | decode 段 + prefill 段各自配方，共享 metadata，slot_mapping 同时表达两段 |

详见 `learning_doc/CP-learning-doc/vllm-ascend-PCP-DCP-batch-scenarios.md`。

---

## 9. 分阶段路线

1. **Phase 0 — 摸清 main 现有 CP 边界**：确认 main 上 PCP（prefill）在 flash_attn 的覆盖度（目前看到 DCP 较完整，PCP 待核实）、`v1/attention/ops` 已有哪些复合算子。这是改造基线。
2. **Phase 1 — 引入轻量 PCPManager（Layout+Strategy）**：把 main 散落的 `worker/cp_utils.py` + flash_attn 内联编排收口到 PCPManager；**不引入通信原语**（继续走 distributed）。保证 dense GQA 行为不变。
3. **Phase 2 — policy 抽象（L1）**：把 main 现有切分抽成 Zigzag/RoundRobin policy；验证 §8 三场景。
4. **Phase 3 — 广度：MLA + Linear(seq-split)**：复用 PCPManager + distributed + ops，新增 MLA tail-proj 配方、Linear conv1d 边界/ssm state 复合算子。
5. **Phase 4 — Hybrid（qwen3-next/3.5/Kimi-Linear）**：注册 Linear policy + hybrid FullAttention 的 enter/exit 配方。
6. **Phase 5 — 纯线性 head-split policy**（可选）+ DSA + chunked prefill / spec decode 收尾。

---

## 10. 待审视的开放问题

1. **PCPManager 与 main 散落 cp_utils 的关系**：是"新建 PCPManager 并把 `worker/cp_utils.py`/`worker/gpu/cp_utils.py` 逻辑迁入"，还是"把现有 cp_utils 重命名/重组"？倾向前者（新增统一抽象，渐进迁移），但需评估对已有调用方的影响。
2. **PCPManager stateful 边界**：token 划分是 per-batch 函数性的，但 cp_rank/group、图捕获参数需持久 state。确认"薄 stateful 持有 + 函数性 per-batch 方法"的度，且不持有通信句柄（那归 distributed）。
3. **DCP KV 分片归属**：`cp_kv_cache_interleave_size` 相关的 block-table slot_mapping 重写，归 PCPManager（提供 layout 信息）还是 block-table 自己（执行分片）？倾向后者，PCPManager 只提供 layout。
4. **DSA 的特殊性**：DSA 把 CP 折进 TP 域、contiguous 切片 + TP all_to_all，不符合"PCP 独立通信域"假设。作为 strategy 的特例还是单独说明？
5. **`cp_size==1` 零开销保证**：strategy.apply 在 cp_size==1 时必须是 early-return 级别 no-op，decode 热路径不留运行时分支。
6. **复合算子归属**：新增的 Linear（conv1d 边界/ssm state）、hybrid（enter/exit）复合算子放 `v1/attention/ops/`（与 `dcp_alltoall` 同级）还是 `v1/cp/`？倾向 `v1/attention/ops/` 保持一致。

---

## 附：核心主张一句话

> **保留 main 的克制（折叠 backend、复用 `vllm/distributed` 通信、CP 复合算子独立成 `v1/attention/ops` 模块、model 透明），新增一个轻量 `PCPManager` 只承载 Layout + Strategy 编排——它消费既有通信与算子设施、自身不碰通信原语、只产 layout 级 metadata。以此把 CP 广度扩到 MLA/DSA/Linear/Hybrid，既不学 ascend 的 fat Manager + 独立 CP backend，也不学 sglang 的 modeling 泄漏。**
