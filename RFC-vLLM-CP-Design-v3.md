# RFC: vLLM Context Parallel (CP) Architecture

> Status: Draft v3  
> Scope: Prefill Context Parallelism (PCP) + Decode Context Parallelism (DCP) for FullAttention/GQA, MLA, sparse MLA/DSA/SFA, Linear attention (Mamba/GDN/KDA), and hybrid models.  
> Baseline: vLLM main branch CP design, vLLM-Ascend PCP/DCP implementation, and SGLang GLM5Next CP implementation.  
> Goal: propose an upstream-friendly vLLM core architecture, not a direct port of any downstream implementation.

---

## 1. Executive Summary

vLLM already has important CP infrastructure:

- `prefill_context_parallel_size` / `decode_context_parallel_size`
- `get_pcp_group()` / `get_dcp_group()`
- DCP-aware FullAttention/GQA paths in FlashAttention
- LSE-based CP merge ops such as `dcp_a2a_lse_reduce`
- `cp_kv_cache_interleave_size` and DCP local KV length metadata

The gap is not "vLLM has no CP". The gap is that the existing support is narrow and not yet structured for broad PCP across attention families and hybrid models.

This RFC proposes:

1. Keep vLLM's current direction:
   - no separate CP backend class hierarchy
   - model code remains CP-transparent
   - communication uses existing `vllm.distributed` groups
   - numerical merge uses shared ops under `vllm/v1/attention/ops`
2. Add explicit CP metadata:
   - `CPContext`: static group/rank/config information
   - `CPBatchLayout`: per-forward layout metadata, inspired by but stricter than vLLM-Ascend's `AscendPrefillContextParallelMetadata`
3. Add shared CP recipe helpers:
   - `vllm/v1/attention/cp.py`
   - helpers for Q gather, KV gather/restore, out+LSE merge, chunked context merge, linear-state boundaries
4. Add pluggable layout policies:
   - Zigzag
   - Continuous
   - Round-robin / interleaved
   - Head-split
5. Expand support in phases:
   - FullAttention/GQA PCP first
   - dense MLA
   - sparse MLA/indexer
   - linear/KDA/Mamba and hybrid models
   - partial-KV PCP+DCP extensions

This RFC intentionally does **not** introduce a stateful `PCPManager` / `CPManager` orchestrator. CP should be unified by shared metadata and shared recipe helpers, not by a central manager object.

---

## 2. Current vLLM Main Baseline

### 2.1 What Already Exists

vLLM main already has several CP concepts:

- Configuration:
  - `parallel_config.prefill_context_parallel_size`
  - `parallel_config.decode_context_parallel_size`
  - `parallel_config.cp_kv_cache_interleave_size`
  - `parallel_config.dcp_comm_backend`
- Distributed groups:
  - `get_pcp_group()`
  - `get_dcp_group()`
- Compatibility checks:
  - `vllm/v1/worker/cp_utils.py`
  - DCP requires attention impls to return decode LSE
  - PCP requires impls to opt in via `supports_pcp`
- FullAttention/GQA DCP:
  - `vllm/v1/attention/backends/flash_attn.py`
  - `FlashAttentionImpl._forward_with_dcp`
  - `FlashAttentionMetadataBuilder.build`
- LSE merge ops:
  - `vllm/v1/attention/ops/common.py`
  - `cp_lse_ag_out_rs`
  - `vllm/v1/attention/ops/dcp_alltoall.py`
  - `dcp_a2a_lse_reduce`
  - `vllm/v1/attention/ops/merge_attn_states.py`

The existing DCP design is important: it shows how CP can be folded into a normal backend without creating a separate CP backend.

### 2.2 What Is Missing

The missing pieces are:

- FullAttention PCP current prefill layout and KV restore
- Dense MLA PCP
- sparse MLA / DSA / SFA PCP with global indexer correctness
- linear attention / Mamba / GDN / KDA CP
- hybrid attention + linear-attention layer contracts
- chunked prefill PCP+DCP context merge
- unified per-forward CP layout metadata
- reusable recipe helpers instead of backend-local CP code

### 2.3 Main Design Values To Preserve

We should preserve these main-branch values:

- model implementations should not contain CP-specific layout code
- CP should not create a parallel backend hierarchy
- communication should go through `GroupCoordinator`
- low-level numerical kernels should stay in `v1/attention/ops`
- backend forward methods should call small CP recipe helpers at CP-sensitive points

---

## 3. Lessons From vLLM-Ascend

vLLM-Ascend has broader CP coverage than vLLM main. It supports multiple attention families and several PCP+DCP scenarios.

### 3.1 What To Borrow

Borrow these ideas:

- PCP decisions happen in runner/batch preparation, not ad hoc in model code.
- A per-forward metadata object is necessary.
- Zigzag / DualChunkSwap is useful for prefill load balance.
- Decode/context partial-KV attention requires out+LSE merge.
- Chunked prefill has different strategies for current chunk and historical context.
- Hybrid models need explicit enter/exit layout handling between linear-attention and full-attention layers.

### 3.2 What Not To Borrow Directly

Do not directly copy these parts:

- fat `PCPManager` as a central orchestrator
- separate CP backend classes for every attention type
- kernel-specific indices inside core layout metadata

`AscendPrefillContextParallelMetadata` is the most useful reference, but it is too broad for vLLM core metadata. It contains both layout metadata and backend/kernel-specific tensors.

Examples of fields that are useful at the layout level:

- full and local query lengths
- padded token count
- unpad mask
- restore index
- PCP/DCP local context lengths
- active policy

Examples of fields that should remain backend-local:

- `q_head_idx_tensor`
- `q_tail_idx_tensor`
- `kv_tail_proj_idx_tensor`
- `kv_with_q_head_mask_idx_tensor`
- `kv_with_q_tail_nomask_idx_tensor`
- `pcp_enter_fa_restore_idx`
- `pcp_exit_fa_scatter_idx`

The vLLM core metadata should be a clean subset/generalization of Ascend's metadata, not an exact port.

---

## 4. Lessons From SGLang

SGLang GLM5Next provides a useful hybrid-model reference.

Important ideas:

- layer-to-layer residual stream should have a stable layout
- linear/KDA layers should not consume zigzag token order directly
- MLA can convert from a stable residual layout into an attention-friendly layout at layer boundaries
- sparse MLA/indexer correctness is easier with Partial Q + Full K
- FlashMLA sparse kernel can remain CP-unaware when upper layers provide correct Q/KV/indexer metadata

SGLang's weakness is that layout conversion leaks into model code. vLLM should avoid that by pushing layout decisions into runner metadata and attention backend recipes.

---

## 5. Core Abstractions

This RFC proposes three core abstractions:

1. `CPContext`
2. `CPBatchLayout`
3. CP recipe helpers

No stateful CP manager is introduced.

### 5.1 CPContext

`CPContext` is a lightweight value object. It carries static or near-static CP runtime facts.

Sketch:

```python
@dataclass(frozen=True)
class CPContext:
    pcp_size: int
    pcp_rank: int
    dcp_size: int
    dcp_rank: int
    pcp_group: GroupCoordinator | None
    dcp_group: GroupCoordinator | None
    cp_kv_cache_interleave_size: int
    dcp_comm_backend: Literal["ag_rs", "a2a"]

    @property
    def total_cp_size(self) -> int:
        return self.pcp_size * self.dcp_size

    @property
    def total_cp_rank(self) -> int:
        return self.pcp_rank * self.dcp_size + self.dcp_rank
```

`CPContext` is not a manager:

- it does not mutate token scheduling
- it does not build backend-specific indices
- it does not own communication primitives
- it does not dispatch backend strategy dynamically

It simply prevents every backend from independently calling `get_pcp_group()` and `get_dcp_group()`.

### 5.2 CPBatchLayout

`CPBatchLayout` is the vLLM core equivalent of a cleaned-up `AscendPrefillContextParallelMetadata`.

It is per-forward/per-batch metadata. It describes what the current rank sees and how to restore or merge results.

Sketch:

```python
@dataclass
class CPBatchLayout:
    active: bool
    policy: CPShardingPolicy

    pcp_size: int
    pcp_rank: int
    dcp_size: int
    dcp_rank: int

    # Full batch/request lengths.
    query_lens_full_cpu: torch.Tensor
    seq_lens_full_cpu: torch.Tensor | None

    # Local lengths for this rank.
    query_lens_local_cpu: torch.Tensor
    seq_lens_local_cpu: torch.Tensor | None

    # Token counts.
    num_actual_tokens: int
    num_actual_tokens_padded: int
    local_num_tokens: int

    # Restore/padding metadata.
    restore_idx: torch.Tensor | None
    unpad_mask: torch.Tensor | None

    # Context KV local lengths for partial-KV CP.
    cp_local_seq_lens: torch.Tensor | None
    cp_local_seq_lens_cpu: torch.Tensor | None

    # Optional high-level segment metadata.
    # This may describe boundaries, but should not contain kernel-specific
    # q/k/v index tensors.
    segment_boundaries: CPLayoutSegments | None
```

Important rule:

> `CPBatchLayout` is layout-level metadata only. It must not contain backend/kernel-specific index tensors.

Backend-local indices should be derived inside the backend from `CPBatchLayout`.

### 5.3 CP Recipe Helpers

Add a shared helper module:

```text
vllm/v1/attention/cp.py
```

This module contains stateless CP recipes. It can consume `CPContext`, `CPBatchLayout`, existing distributed groups, and low-level ops.

Examples:

```python
def cp_gather_query_heads_for_dcp(ctx, query): ...

def cp_merge_out_lse(
    ctx,
    attn_out,
    softmax_lse,
    *,
    return_lse: bool = False,
): ...

def cp_gather_kv_to_full_current_chunk(ctx, layout, key, value): ...

def cp_restore_padded_tokens(layout, tensor): ...

def cp_merge_context_and_current(
    output,
    context_out,
    context_lse,
    current_out,
    current_lse,
): ...
```

These helpers are recipes, not kernels. They should call:

- `GroupCoordinator` for communication
- `v1/attention/ops` for numerical primitives such as LSE merge

---

## 6. Why No PCPManager / CPManager

A central manager object is attractive because it creates an obvious place to "put CP". But it creates the wrong ownership boundary.

CP has two different lifetimes:

- layout is per batch
- backend recipes are per layer / per backend forward

A stateful manager would either:

- live in the runner and know too much about backend internals, or
- live in attention and know too much about batch scheduling

vLLM main already has natural boundaries:

- worker/batch code builds metadata
- attention backend consumes metadata
- distributed groups do communication
- ops modules do numerical merge

The RFC should formalize these boundaries rather than introduce a central orchestrator.

Therefore:

- no `PCPManager`
- no `CPManager`
- no manager-owned communication
- no manager-produced kernel-specific indices

The unifying layer is:

```text
CPContext + CPBatchLayout + shared CP recipe helpers
```

---

## 7. Layout Policies

The CP layout policy determines how query tokens and/or KV ownership are partitioned.

### 7.1 Zigzag / DualChunkSwap

Best for dense prefill attention load balance.

Sequence is padded to `2 * cp_size` chunks. Rank `r` gets:

```text
chunk r
chunk 2 * cp_size - 1 - r
```

Pros:

- balances causal attention work better than continuous slicing
- works well for dense FullAttention/MLA prefill

Cons:

- not natural token order
- linear/KDA/Mamba layers cannot consume this layout directly
- requires restore indices and unpad masks

### 7.2 Continuous

Rank `r` gets a contiguous slice:

```text
[r * chunk, (r + 1) * chunk)
```

Pros:

- natural token order
- good for linear attention, KDA, Mamba, causal conv, recurrent scan
- simple layer-to-layer residual contract

Cons:

- imbalanced for causal full attention
- later ranks do more work

### 7.3 Round-Robin / Interleaved

Rank `r` gets token positions:

```text
r, r + cp_size, r + 2 * cp_size, ...
```

or interleave-size variants of the same idea.

Pros:

- good causal attention load balance
- useful for sparse MLA/indexer token ownership
- naturally matches `position % cp_size`

Cons:

- token order is unnatural
- needs layout conversion around linear layers
- non-divisible lengths need padding/ragged handling

### 7.4 Head-Split

Rank `r` holds a subset of heads and runs full sequence.

Pros:

- works for pure linear/head-independent models
- no sequence dependency across ranks
- can save state memory

Cons:

- output reduction is required
- less natural for hybrid models that alternate linear attention and full attention

---

## 8. Enabled vs Active

CP must distinguish enabled and active.

```text
enabled: configured world size > 1
active: current batch actually uses CP layout/recipe
```

Examples:

- PCP enabled, short prefill: active may be false.
- PCP enabled, decode-only batch: prefill split is inactive.
- DCP enabled, decode/context path: active is true for partial-KV context merge.
- PCP enabled, long prefill with non-divisible length: active should still be possible via padding/ragged metadata.

This distinction should be represented in `CPBatchLayout.active` and in backend metadata.

---

## 9. Attention Family Strategies

### 9.1 FullAttention / GQA

#### Prefill PCP

Initial target should be Mode 1:

```text
Partial Q + Full current KV
```

Each rank computes attention for local Q tokens. Current chunk K/V is all-gathered and restored into full natural order.

For zigzag, backend derives head/tail query and nomask/mask KV ranges from `CPBatchLayout`.

Backend must handle GQA through:

```text
num_heads
num_kv_heads
num_queries_per_kv = num_heads // num_kv_heads
```

#### Decode / Context DCP

Existing vLLM DCP path should be reused:

1. all-gather Q heads across DCP
2. attend to local context KV shard
3. return output and LSE
4. merge across ranks using LSE
5. reduce-scatter output heads back to local rank

For GQA/MQA, the existing constraint remains:

```text
num_q_per_kv % dcp_size == 0
```

#### Chunked Prefill

Chunked prefill should combine:

- current chunk: Partial Q + Full current KV
- historical context: Partial KV + LSE merge

This should reuse `merge_attn_states` or equivalent LSE merge helpers.

### 9.2 Dense MLA

Initial target:

```text
Partial Q + Full current latent KV
```

For MLA, current chunk all-gather should restore:

```text
kv_c + k_pe
```

Then backend can derive tail-only `kv_b_proj` indices locally.

Do not put `kv_tail_proj_idx` into `CPBatchLayout`.

### 9.3 Sparse MLA / DSA / SFA

Correctness-first target:

```text
Partial Q + Full K for indexer/topk
Partial Q + Full KV for sparse attention
```

The indexer must see global K. Local topk over local K is not correct unless a distributed topk merge is implemented.

Sparse kernel can remain CP-unaware if:

- Q is already local
- K/V cache or page table represents global visible KV
- sparse indices are computed against global K

DSA is special:

- vLLM-Ascend DSA uses a DSA-specific continuous/local-token metadata path
- SGLang GLM5Next sparse/indexer is closer to round-robin Partial Q + Full K

The RFC should not force all DSA paths into one policy. DSA should be allowed to register a backend-specific CP recipe while still using common `CPContext` and `CPBatchLayout`.

### 9.4 Linear Attention / Mamba / GDN / KDA

Linear attention often needs natural token order:

- causal conv
- recurrent scan
- state cache update

Zigzag or round-robin token order cannot be fed directly into these layers.

Two valid strategies:

#### Sequence Split

Use continuous token layout and propagate boundary state.

Best for hybrid models.

Needs:

- conv/state boundary exchange
- correct state cache update
- chunk/recurrent scan across rank boundaries

#### Head Split

Each rank owns a subset of heads and runs full sequence.

Best for pure linear/head-independent models.

Needs:

- output reduction
- careful state sharding

First upstream target should be sequence split for hybrid models only after FullAttention/MLA PCP metadata is stable.

### 9.5 Hybrid Models

Hybrid models need an explicit layer-to-layer layout contract.

Recommended contract:

- residual stream uses continuous/plain layout
- full attention layers may convert to zigzag/round-robin internally
- linear/KDA/Mamba layers consume natural-order continuous layout
- conversions are implemented in backend recipes, not model forward

This follows SGLang's GLM5Next insight while avoiding SGLang's modeling-level CP leakage.

---

## 10. Metadata Boundary Rules

### 10.1 Allowed in CPBatchLayout

`CPBatchLayout` may include:

- CP active flag
- policy name
- pcp/dcp ranks and sizes
- full query lengths
- local query lengths
- full sequence lengths
- local context sequence lengths
- padded token count
- restore index
- unpad mask
- high-level segment boundaries

### 10.2 Not Allowed in CPBatchLayout

`CPBatchLayout` must not include:

- backend-specific Q index tensors
- backend-specific K/V index tensors
- MLA tail projection indices
- sparse kernel block selection indices
- NPU-specific enter/exit FA tensors
- any metadata that only one backend can interpret

These are derived by the backend from layout metadata.

This is the key difference from `AscendPrefillContextParallelMetadata`.

---

## 11. Batch Scenarios

### 11.1 Pure Prefill

Likely Mode 1 for first implementation:

```text
Partial Q + Full current KV
```

Communication:

- gather current K/V or compressed K/V
- local attention
- gather/restore outputs if needed

### 11.2 Pure Decode

Likely Mode 2:

```text
Partial KV + output/LSE merge
```

Communication:

- Q head all-gather
- local KV attention
- out+LSE merge
- reduce-scatter/all-to-all result

### 11.3 Chunked Prefill

Mixed:

- current chunk may use full current KV
- historical context should use local KV shard + LSE merge
- final output merges context and current chunk with LSE

### 11.4 Mixed Prefill + Decode

Should be treated as two segments sharing a batch:

- decode segment uses decode/context recipe
- prefill segment uses prefill recipe
- metadata must preserve segment boundaries

---

## 12. Proposed File/Module Layout

### 12.1 New or Expanded Modules

```text
vllm/v1/worker/cp_utils.py
    CP layout construction functions.
    Builds CPBatchLayout or CP-related fields in CommonAttentionMetadata.

vllm/v1/attention/cp.py
    Shared CP recipe helpers.
    Stateless functions used by attention backends.

vllm/v1/attention/ops/
    Low-level numerical CP ops.
    Existing: dcp_alltoall, merge_attn_states, common LSE helpers.
    Future: linear state boundary ops if needed.
```

### 12.2 Existing Modules To Reuse

```text
vllm/distributed/parallel_state.py
    PCP/DCP groups.

vllm/v1/attention/backend.py
    Attention metadata base classes and backend capability flags.

vllm/v1/attention/backends/flash_attn.py
    Existing FullAttention/GQA DCP reference path.

vllm/v1/attention/backends/mla/
    Future dense/sparse MLA PCP integration.
```

---

## 13. Compatibility and Invariants

### 13.1 `cp_size == 1`

All CP helpers must early-return with no observable overhead.

### 13.2 Backend Opt-In

Keep explicit capability flags:

- `can_return_lse_for_decode`
- `supports_pcp`
- `supports_mtp_with_cp_non_trivial_interleave_size`

Add more specific flags only if needed, for example:

- `supports_pcp_full_kv_prefill`
- `supports_pcp_partial_kv_context`
- `supports_cp_zigzag`

### 13.3 GQA/DCP Constraint

For GQA/MQA DCP:

```text
num_q_per_kv % dcp_size == 0
```

This must remain a config validation rule.

### 13.4 Sliding Window / Local Attention

Existing DCP/PCP restrictions around sliding window and local attention should remain until tested.

### 13.5 Hybrid KV/Mamba Cache

Hybrid cache managers must be CP-aware before enabling prefix cache hits with PCP/DCP.

---

## 14. Phased Roadmap

### Phase 0: Normalize Current DCP Code

No behavior change.

- introduce `CPContext`
- extract small DCP helper functions from `FlashAttentionImpl._forward_with_dcp`
- preserve existing tests and behavior
- no PCP behavior changes

### Phase 1: Introduce CPBatchLayout

Add layout-level metadata.

- build `CPBatchLayout` or equivalent fields from worker
- represent enabled vs active
- support padding/restore/unpad metadata
- support DCP local context lengths
- do not add kernel-specific indices

### Phase 2: FullAttention/GQA PCP

First real PCP target.

- Zigzag or round-robin prefill policy
- Partial Q + Full current KV
- backend-local head/tail and nomask/mask index derivation
- tests against TP-only baseline

### Phase 3: Chunked Prefill Context

Extend FullAttention PCP+DCP.

- current chunk full-KV path
- context partial-KV path
- out+LSE merge
- final context/current merge

### Phase 4: Dense MLA

Reuse CPBatchLayout and recipe helpers.

- Partial Q + Full latent KV
- backend-local tail projection selection
- decode/context LSE merge when partial-KV is used

### Phase 5: Sparse MLA / Indexer

Correctness-first sparse path.

- indexer sees global K
- sparse kernel receives local Q and full/global KV view
- no distributed topk in first version

### Phase 6: Linear / KDA / Mamba and Hybrid

Only after layout metadata is stable.

- continuous residual layout
- state boundary propagation
- hybrid enter/exit full-attention layout recipes

### Phase 7: Advanced Policies

Optional:

- head-split for pure linear models
- partial-KV PCP prefill for more memory savings
- distributed sparse topk
- DCP+PCP with non-trivial interleave and speculative decode

---

## 15. Testing Strategy

### 15.1 Correctness Baselines

For each supported model/backend:

- TP-only
- TP + PCP
- TP + DCP
- TP + PCP + DCP where supported

Compare:

- logits top-k
- generated tokens at temperature 0
- per-layer output stats
- attention output finiteness
- LSE validity

### 15.2 Batch Shapes

Test:

- pure prefill
- pure decode
- mixed prefill/decode
- chunked prefill
- short prefill fallback
- non-divisible query lengths
- multiple requests

### 15.3 Layout Tests

Unit-test:

- zigzag split/restore
- continuous split/restore
- round-robin split/restore
- padding/unpad mask
- local seq lens with interleave

### 15.4 Backend Tests

Backend-specific tests:

- FullAttention/GQA DCP existing behavior unchanged
- FullAttention/GQA PCP matches TP-only
- MLA tail projection indices derived correctly
- sparse indexer sees global K
- KDA/Mamba state order preserved

---

## 16. Open Questions

1. Should `CPBatchLayout` live inside `CommonAttentionMetadata`, or should it be a separate field referenced by backend metadata?
2. Should `vllm/v1/attention/cp.py` be one module or a package with per-family helpers?
3. Should first FullAttention PCP use zigzag or round-robin?
4. How should non-divisible lengths be represented: padded equal chunks or ragged chunks?
5. Should `CPBatchLayout.segment_boundaries` be generic enough for zigzag head/tail and chunked context without becoming kernel-specific?
6. How should DSA register exceptions without breaking common CP abstractions?

---

## 17. Source Map

vLLM main:

- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/utils.py`
- `vllm/v1/attention/ops/common.py`
- `vllm/v1/attention/ops/dcp_alltoall.py`
- `vllm/v1/attention/ops/merge_attn_states.py`
- `vllm/v1/worker/cp_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/distributed/parallel_state.py`
- `vllm/config/model.py`

vLLM-Ascend:

- `vllm_ascend/worker/pcp_utils.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/attention/context_parallel/common_cp.py`
- `vllm_ascend/attention/context_parallel/attention_cp.py`
- `vllm_ascend/attention/context_parallel/mla_cp.py`
- `vllm_ascend/attention/context_parallel/dsa_cp.py`
- `vllm_ascend/attention/context_parallel/sfa_cp.py`

SGLang:

- `python/sglang/srt/models/glm5_next.py`
- `python/sglang/srt/layers/attention/dsa/utils.py`
- `python/sglang/srt/layers/communicator_dsa_cp.py`
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`
- `python/sglang/srt/layers/attention/dsa_backend.py`

---

## 18. One-Sentence Position

vLLM should extend CP by adding explicit `CPContext`, layout-level `CPBatchLayout`, and shared stateless CP recipe helpers, while preserving main's existing ownership boundaries: model-transparent execution, CP folded into normal attention backends, communication through `vllm.distributed`, and numerical merge through `v1/attention/ops`.
