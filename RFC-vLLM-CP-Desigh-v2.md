# RFC: vLLM Context Parallel (CP) 目标架构设计

> 状态：Draft（完全重写版，基于 vLLM **main** 分支）
> 范围：PCP（Prefill CP）+ DCP（Decode CP），覆盖 GQA / MLA / DSA / SFA / Linear(Mamba/GDN/KDA) 及 Hybrid 模型。
> 历史版本：`with-PCPManager/`（基于"PCPManager 作为唯一编排对象"的旧主张，已废弃——理由见 §3）。
> 参考实现：vLLM main（`vllm/v1/attention/backends/flash_attn.py`、`vllm/v1/attention/ops/dcp_alltoall.py`、`vllm/v1/worker/cp_utils.py`、`vllm/v1/worker/gpu/cp_utils.py`、`vllm/distributed/parallel_state.py`）、`vllm-ascend`、`sglang`。源码拆解见 `learning_doc/CP-learning-doc/`。

---

## 1. 背景与动机（按 main 重述）

**先纠正一个常见误解：vLLM main 已经有 CP 了，而且设计克制、方向正确。**

main 现状：
- **配置**：`parallel_config.prefill_context_parallel_size` / `decode_context_parallel_size`，`cp_size = pcp_size * dcp_size`。
- **通信域**：`vllm/distributed/parallel_state.py` 提供 `get_pcp_group()` / `get_dcp_group()`（标准 `GroupCoordinator`）。
- **基础集合通信**：**复用 `vllm/distributed` 已有的** `all_gather / all_gatherv / all_reduce / reduce_scatter`——CP 不另造。
- **CP 复合算子**（online-softmax out/lse 合并这类 CP 专属配方）：`vllm/v1/attention/ops/dcp_alltoall.py`（`dcp_a2a_lse_reduce`、`_lse_weighted_combine`）。
- **CP 折叠进 attention backend**，没有独立 CP backend：`flash_attn.py` 里 `if self.dcp_world_size > 1:` 内联处理。
- **worker 侧只有轻量 `cp_utils.py`**（兼容性检查 + `prepare_dcp_local_seq_lens`），**无 manager 对象**。
- **modeling 对 CP 透明**：模型只调 `self.attn(...)`。

**真正的 gap 不是"没 CP"，而是两件事：**
1. **广度不足**：main 的 CP 基本围绕 dense GQA 的 DCP（flash_attn）。MLA / DSA / SFA / Linear(Mamba/GDN/KDA) / Hybrid 的 CP 缺失或不完整。
2. **编排内联、缺共享配方层**：CP 逻辑散落在 `flash_attn.py` 的内联 `if` 块里，没有一处共享的"配方函数"，导致扩展新类型时无处复用、易重复实现。

**本 RFC 主张**：保留 main 的克制美德（折叠 backend、复用 distributed 原语、CP 复合算子独立成模块、model 透明），**不引入任何 manager 对象**，而是补一个**共享设施层**（layout 函数 + 配方函数 + 一个值对象 `CPContext`），把广度扩到 MLA/DSA/Linear/Hybrid。既不学 vllm-ascend 的 fat `PCPManager` + 独立 CP backend，也不学 sglang 的 modeling 泄漏。

---

## 2. 现状三方对比

| 维度 | vLLM **main** | vllm-ascend | sglang |
|---|---|---|---|
| 基础集合通信 | ✅ 复用 `vllm/distributed` | 自有（NPU） | 复用 `dp_attention` group |
| CP 复合算子 | `v1/attention/ops/dcp_alltoall.py` | `common_cp.py` | `layers/utils/cp_utils.py` |
| CP backend 归属 | **折叠进 flash_attn**（无独立 CP backend）✅ | 独立 CP impl per 类型（4 个）❌ | 折叠进 flashattention backend ✅ |
| 编排抽象 | 散在 flash_attn **内联** + 两个 cp_utils（待去内联）⚠️ | fat `PCPManager`（含内核索引、权威过载）❌ | 纯函数 + 闭包，但**泄漏进 modeling** ❌ |
| model CP 透明度 | 高 ✅ | 高 ✅ | 低（每模型 forward 写 `cp_split_*`）❌ |
| metadata 形状 | 偏 layout 级 ✅ | 混了 kernel 索引 ❌ | layout/policy 级 ✅ |
| 广度 | dense GQA 为主 | GQA/MLA/DSA/SFA/Linear/Hybrid 全 | dense + DSA，无 Linear/Hybrid CP |

**结论**：main 的"折叠 backend + 复用 distributed + 复合算子成模块 + model 透明"四点是对的，要保留。要补的只有：①共享配方层（去内联）；②广度。ascend 的 fat Manager + 独立 CP backend、sglang 的 modeling 泄漏，都是要避免的。

---

## 3. 核心原则：CP 的统一性来自共享设施，不来自一个 manager 对象

这是本 RFC 与旧版（`with-PCPManager/`）的根本分歧，也是审查时要重点确认的一条。

**为什么废弃"PCPManager/CPManager 编排对象"：**

CP 的职责天然分两段，归属不同的层：
1. **Token 布局**（哪个 rank 处理哪些 token、positions）——必须在 **batch 准备（worker）**定好，影响整个 forward。
2. **逐层 gather/merge 配方**——发生在 **attention 层内部**，每层每类型不同。

一个对象同时拥有这两件 = god-object：在 worker 里有权威（重写 token 计划），又在 attention 里有权威（选配方），生命周期还不一样（layout 每 batch 一次、配方每层每 backend 一次）。main **没有** manager、靠散落函数就跑通了，证明这个对象不是必需的。引入它只会带来跨层耦合，换不来收益。

**因此：消灭 manager 编排对象。"CP 的单一化身"是一套共享设施，不是一个对象。** CP 的"统一"来自大家都用同一套设施（distributed 原语 + ops 复合算子 + layout 函数 + 配方函数），而非一个中央 manager。

---

## 4. 目标架构：四块设施 + CPContext 值对象

消灭 manager 后，CP 的职责回到各自天然归属——**就是 main 现有结构的形式化，不另起炉灶**：

```
┌─────────────────────────────────────────────────────────────────┐
│ ① Token 布局  —— vllm/v1/worker/cp_utils.py（无状态函数）        │
│    batch + policy → 每 rank token 切片 / positions / per-rank 长度│
│    （扩展 main 现有的 prepare_dcp_local_seq_lens）               │
├─────────────────────────────────────────────────────────────────┤
│ ② CP 配方函数 —— 新建 vllm/v1/attention/cp.py（无状态函数）      │
│    cp_gather_q_for_decode / cp_merge_attn_out_lse /              │
│    cp_linear_state_propagate / cp_hybrid_enter_fa / ...          │
│    （把 flash_attn 的内联 CP 块抽出来，所有 backend 共用）       │
├─────────────────────────────────────────────────────────────────┤
│ ③ 基础集合通信 —— vllm/distributed/parallel_state.py（既有）     │
│    GroupCoordinator.all_gather / all_to_all / reduce_scatter     │
│    （CP 复用，不动）                                             │
├─────────────────────────────────────────────────────────────────┤
│ ④ CP 复合算子  —— vllm/v1/attention/ops/（既有 + 待补）          │
│    dcp_alltoall.dcp_a2a_lse_reduce / merge_attn_states /         │
│    conv1d 边界 / ssm state 传播（Linear 待补）                   │
└─────────────────────────────────────────────────────────────────┘
                    ▲ 共同消费 ▲
        ⑤ CPContext（dataclass 值对象，跨层传递，不是 orchestrator）
           pcp_group / dcp_group / pcp_rank / dcp_rank /
           pcp_size / dcp_size / policy
```

- **没有任何叫 Manager 的对象。** ①是 worker 里的函数，②是 attention 层的函数，⑤是值对象。三者通过 `CPContext` 解耦传递。
- **CPContext 只是配置载体**（持有 group 句柄、rank、policy），无编排逻辑、无 token 改写权威。它替代 main 现在各处散取 `get_dcp_group()` 的写法，集中配置。

### 4.1 ⚑ 关键纪律（写死）

> **布局层（①）只产出 layout 级 metadata，绝不产出 kernel 级索引。**

这是「能保持折叠 backend、不开独立 CP backend」的命门。ascend 的 `generate_pcp_metadata` 生成 `q_head_idx`/`kv_tail_proj_idx`/`pcp_enter_fa_restore_idx`… —— 只要 kernel 专属索引进到布局层，base backend 就无法 CP-uniform，就得开独立 CP backend。

| | 保留在布局层（layout 级） | 移出，归 backend（kernel 级） |
|---|---|---|
| token 切片 / positions / cp_rank / 每 rank 长度 | ✅ | |
| prev/next 序列边界 | ✅ | |
| GQA 的 nomask/mask 拆分索引 | | ✅ forward 内算 |
| MLA tail-only `kv_b_proj` 索引 | | ✅ MLA backend 内算 |
| hybrid 的 enter/exit FA restore/scatter 索引 | | ✅ 该 backend 内算 |

main 现状的 metadata 偏 layout 级（`dcp_context_kv_lens` 等），方向正确，要保持。

---

## 5. Attention Backend 怎么改：去内联 → 委托点

main 现状是 flash_attn 里散落的 `if dcp_world_size > 1:` 内联块（Q gather、`dcp_a2a_lse_reduce` 夹在 forward 中间）。改造 = **把内联块抽成对配方函数（②）的委托**，backend forward 只保留"在 CP 敏感点委托"：

```python
# vllm/v1/attention/cp.py —— 配方函数（消费 ③ distributed + ④ ops）
def cp_gather_q_for_decode(ctx, q):
    return ctx.dcp_group.all_gather(q, dim=1) if ctx.dcp_size > 1 else q

def cp_merge_attn_out_lse(ctx, out, lse):
    # 顺序：DCP a2a → PCP ag → online-softmax merge
    return dcp_a2a_lse_reduce(out, lse, ctx.dcp_group, ctx.pcp_group, ...)

def cp_linear_state_propagate(ctx, conv_states, ...): ...
def cp_hybrid_enter_fa(ctx, qkv, ...): ...
```

```python
# flash_attn backend forward（去内联）
q = cp_gather_q_for_decode(ctx, q)              # CP 敏感点 1
out, lse = core_attn(q, k, v, ...)              # 干净的内核调用
out = cp_merge_attn_out_lse(ctx, out, lse)      # CP 敏感点 2
```

要点：
- backend **不持有 CP 逻辑**，只在 CP 敏感点委托；`cp_size==1` 时配方函数 early-return no-op（热路径干净，无运行时分支）。
- 配方函数集中在 `v1/attention/cp.py`，**所有 backend 共用，消除跨 backend 重复**（这正是旧版想用 manager 解决的问题，但共享函数就够）。
- **不开独立 CP backend**——保持 main 的"折叠进 backend"。

---

## 6. PCP 与 DCP：同一 layout + 配方的两个正交维度

消灭 manager 后，"PCP 谁管、DCP 谁管"是伪问题。二者是**同一套设施的两个 group 维度**，正如 main 早已编码的 `cp_size = pcp_size * dcp_size`：

- **布局函数（①）**同时吃 `(pcp_size, dcp_size)`，产每 rank token 视图 + per-rank 长度（main 的 `prepare_dcp_local_seq_lens` 已是 DCP 维度的雏形，补 PCP 维度即可）。
- **配方函数（②）**按正确顺序编排两个域——DCP 域 `a2a`（`dcp_a2a_lse_reduce`）在前，PCP 域 `all_gather` 在后（main 的现有顺序）。
- **`CPContext`** 同时持有 `pcp_group` / `dcp_group`，配方按需各取所需。

**不存在"PCP 的 manager / DCP 的 manager"之分。** 这也是旧版名"PCPManager"被否定的原因——名字本身漏了 DCP。正确模型：**一份布局 + 一套配方 + 一个 context，PCP/DCP 是其中两个 group 维度**。

---

## 7. L1 策略层（policy）：把切分方式做成可插拔

布局函数（①）的切分方式抽成 `ShardingPolicy`，policy 决定「token 怎么切、配方调几次、metadata 什么形状」：

| Policy | 适用 | 切法 | main 现状 |
|---|---|---|---|
| **Zigzag**（DualChunkSwap） | dense 全注意力（负载均衡） | head/tail 配对，pad 到 2·cp | 待补 |
| **Linear**（contiguous） | hybrid（迁就线性层递归） | rank r 拿 `[r·c,(r+1)·c)` | 待补 |
| **HeadSplit** | pure-linear（head 独立） | rank r 持 num_heads/cp head，跑完整 seq | 待补 |
| **RoundRobin** | DCP KV 分片 / 特定稀疏 | 按 interleave 粒度轮流 | `cp_kv_cache_interleave_size` 已有雏形 |

policy 是**唯一**按模型拓扑注册的东西。加 KimiLinear = 注册 Linear policy，不碰 backend、不碰配方函数、不碰任何 manager。

---

## 8. 线性/递归注意力的 CP：按拓扑选策略

线性注意力（Mamba/GDN/KDA）head 互相独立，两条合法路线：

| | seq-split + 状态传播（ascend 路线） | head-split（sglang 哲学） |
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

## 9. 各注意力类型的配方映射

| 类型 | prefill | decode | 配方用到的设施 |
|---|---|---|---|
| GQA（dense） | Mode1: AllGather-KV + nomask/mask 拆分 | Mode2: DCP head-ag Q + `dcp_a2a_lse_reduce` | ③ `all_gather` + ④ `dcp_alltoall` |
| MLA | Mode1: AllGather kv_c+k_pe，tail-only `kv_b_proj` | 同 GQA 骨架 + v_up_proj | ③ + ④ `merge_attn_states` |
| DSA | TP 域线性切片 + local indexer/compressor metadata | output TP all_to_all 还原 | ③ `all_to_all` |
| Linear（Mamba/GDN/KDA） | seq-split: conv1d 边界 allgather + ssm state | state 已分片，零额外通信 | ④ 新增 conv1d 边界 + ssm state |
| Hybrid FullAttention | Linear policy + enter/exit FA 布局转换 | 复用 GQA decode | ③ + ④ 新增 enter/exit（backend 内算） |

通信密度：**decode > 混合 > prefill** → 优化重心放 decode 的 out/lse 合并。

---

## 10. 三种 batch 场景的回归基线

作为 CP 正确性回归基线（最易暴露 bug）：

| 场景 | Mode | 关键通信 |
|---|---|---|
| 纯 Prefill | Mode1 | PCP allgather KV（每层） |
| 纯 Decode | Mode2 | DCP ag Q(head) + `dcp_a2a_lse_reduce`（含 PCP ag）|
| Prefill+Decode 混合 | Mode1+2 串行 | decode 段 + prefill 段各自配方，共享 metadata，slot_mapping 同时表达两段 |

详见 `learning_doc/CP-learning-doc/vllm-ascend-PCP-DCP-batch-scenarios.md`。

---

## 11. 分阶段路线

1. **Phase 0 — 摸清 main 现有 CP 边界**（改造基线）：确认 main 上 PCP（prefill）在 flash_attn 的覆盖度（目前看到 DCP 较完整，PCP 待核实）、`v1/attention/ops` 已有哪些复合算子、哪些 `if dcp_world_size>1` 内联块需要去内联。
2. **Phase 1 — 引入 `CPContext` + `v1/attention/cp.py` 配方层（去内联）**：把 flash_attn 内联 CP 块抽成配方函数；行为不变，只是结构化。**不引入任何 manager，不碰 distributed。**
3. **Phase 2 — policy 抽象（L1）**：把现有切分抽成 Zigzag/RoundRobin policy；验证 §10 三场景。
4. **Phase 3 — 广度：MLA + Linear(seq-split)**：新增 MLA tail-proj 配方、Linear conv1d 边界/ssm state 复合算子（进 `v1/attention/ops`）。
5. **Phase 4 — Hybrid（qwen3-next/3.5/Kimi-Linear）**：注册 Linear policy + hybrid FullAttention 的 enter/exit 配方。
6. **Phase 5 — 纯线性 head-split policy**（可选）+ DSA + chunked prefill / spec decode 收尾。

---

## 12. 待审视的开放问题

1. **main 现有 PCP 覆盖度**（影响 Phase 0/1）：main 上 flash_attn 的 PCP（prefill 切分）是完整支持还是主要只 DCP？这决定 Phase 1 是"去内联重组"还是"顺带补 prefill"。
2. **配方层落点**：`v1/attention/cp.py`（与 backends 同级）vs `v1/attention/ops/cp.py`（与 `dcp_alltoall` 同级）？倾向前者——`ops` 是底层算子，`cp.py` 是组合配方的编排，层级不同。
3. **DCP KV 分片归属**：`cp_kv_cache_interleave_size` 相关的 block-table slot_mapping 重写，归布局函数（提供 layout 信息）还是 block-table 自己（执行分片）？倾向后者，布局只提供 layout。
4. **DSA 的特殊性**：DSA 把 CP 折进 TP 域、contiguous 切片 + TP all_to_all，不符合"PCP 独立通信域"假设。作为配方的特例还是单独说明？
5. **`cp_size==1` 零开销保证**：配方函数在 `cp_size==1` 时必须 early-return no-op，decode 热路径不留运行时分支（可用 config 驱动的空对象/直接短路两种实现，待定）。
6. **Linear 复合算子归属**：conv1d 边界 / ssm state 传播放 `v1/attention/ops/`（与 `dcp_alltoall` 同级）还是单独？倾向 `v1/attention/ops/` 保持一致。

---

## 附：核心主张一句话

> **保留 main 的克制（折叠 backend、复用 `vllm/distributed` 通信、CP 复合算子独立成 `v1/attention/ops` 模块、model 透明）。不引入任何 manager 对象——CP 的统一性来自共享设施层：`worker/cp_utils` 布局函数 + 新建 `v1/attention/cp.py` 配方函数 + `CPContext` 值对象，三者复用既有 distributed 原语与 ops 算子。布局层只产 layout 级 metadata（绝无 kernel 索引）。以此把 CP 广度扩到 MLA/DSA/Linear/Hybrid，既不学 ascend 的 fat Manager + 独立 CP backend，也不学 sglang 的 modeling 泄漏。**
