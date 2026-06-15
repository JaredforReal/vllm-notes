# RFC: vLLM Context Parallel (CP) — Final Design

> Status: Final draft (rewrite, gap-first). Supersedes v1/v2/v3.
> Scope: PCP (Prefill CP) + DCP (Decode CP) for GQA/FullAttention, dense MLA, sparse MLA/DSA/SFA, Linear (Mamba/GDN/KDA), hybrid models.
> Grounding: vLLM `main` (commit `7852e50e4`), vLLM-Ascend `attention/context_parallel/`, SGLang GLM5Next. All `file:line` refs are real.
> Method: this document opens by enumerating, concretely and from source, **what `main` is missing**. Every subsequent abstraction exists only to close one of those gaps — there is no design for its own sake.

---

## 1. Executive summary

vLLM `main` already has restrained, well-placed CP foundations: model-transparent execution, CP folded into a normal attention backend (not a parallel hierarchy), communication through `vllm.distributed` groups, and numerical merge primitives under `v1/attention/ops`. The DCP decode path for GQA works end-to-end.

The work to do is not "build CP" — it is to **close six concrete gaps** while preserving those foundations. The closures are: a `CPContext` value object, a layout-level `CPBatchLayout`, a stateless recipe module `v1/attention/cp.py`, a pluggable layout policy, an `enabled`/`active` distinction, and the breadth recipes (PCP prefill, MLA tail-proj, linear state-propagation, hybrid enter/exit). **No stateful manager is introduced** — the gaps are closed by shared metadata + shared recipes + existing distributed/ops facilities, not by a central orchestrator.

---

## 2. What vLLM is currently missing

Each gap is stated concretely against `main`, with the source reference that proves it. These six gaps are the entire motivation for the design in §4 onward.

### Gap 1 — PCP is absent in the V2 runner
`config/vllm.py:1991` lists `prefill context parallelism` under V2-model-runner unsupported features. `supports_pcp = False` on every `AttentionImplBase` (`v1/attention/backend.py:716`). There is no token splitting, no prefill KV all-gather, no zigzag/round-robin layout, and `check_attention_cp_compatibility` (`v1/worker/cp_utils.py:39`) will reject `pcp_size > 1` for any current impl. **PCP prefill is net-new work**, not a refactor.

### Gap 2 — CP code is inlined, not reusable
The entire DCP decode logic lives inside `FlashAttentionImpl._forward_with_dcp` (`v1/attention/backends/flash_attn.py:930`) as one ~100-line method. The two merge backends (`dcp_a2a_lse_reduce`, `cp_lse_ag_out_rs`) and `merge_attn_states` are properly factored into `v1/attention/ops`, but the **orchestration** around them (Q-head all-gather → context attention → combine → query attention → merge) is copy-pasted per backend. Adding MLA/sparse/linear CP means duplicating this orchestration N times.

### Gap 3 — CP rank/layout info is scattered, with no layout-level metadata
CP facts are spread across three places, none of them a coherent "what does this rank own this batch" object:
- `AttentionImplBase.__new__` (`backend.py:746`) sets `dcp_world_size/dcp_rank/pcp_world_size/pcp_rank/total_cp_*` per impl, by calling `get_dcp_group()`/`get_pcp_group()` independently.
- `CommonAttentionMetadata` (`backend.py:362`) carries `dcp_local_seq_lens` / `dcp_local_seq_lens_cpu` (a single field).
- `FlashAttentionMetadata` (`flash_attn.py:236`) carries `max_dcp_context_kv_len` / `dcp_context_kv_lens` (FA-specific).
There is no place that holds padding counts, restore indices, unpad masks, segment boundaries, or an `active` flag. The worker injects only `dcp_local_seq_lens` (`gpu_model_runner.py:2332`).

### Gap 4 — Breadth: MLA, sparse, linear, hybrid are unsupported
- **MLA**: no DCP/PCP. The expensive `kv_b_proj` would run over the full all-gathered sequence without the tail-only optimization.
- **Sparse MLA / DSA / SFA**: no CP. The indexer/topk must see global K; a naive split computes local topk over local K — incorrect.
- **Linear (Mamba/GDN/KDA)**: no CP. These layers need natural token order for causal-conv/recurrent-scan; zigzag/round-robin token order cannot be fed to them, and there is no cross-rank state-propagation primitive.
- **Hybrid (qwen3-next/GLM/Kimi-Linear)**: no CP, and no enter/exit layout contract between linear and full-attention layers.

### Gap 5 — Chunked-prefill PCP+DCP context merge is missing
DCP today handles decode/context against the paged cache. The chunked-prefill case — current chunk as Partial Q + Full current KV, historical context as Partial KV + LSE merge — is not implemented. `merge_attn_states` exists but is only wired for the decode suffix merge.

### Gap 6 — `enabled` vs `active` is not represented
Config has `pcp_size`/`dcp_size` (enabled). But a PCP-enabled run with a short prefill, or a decode-only batch, should not split. There is no `active` flag to let a backend cheaply skip the CP path per batch; the only gate is `world_size > 1`, which is too coarse.

### Gap → Closure map
| Gap | Closure (section) |
|---|---|
| 1 PCP absent | §4.4 PCP prefill recipe + policy wiring |
| 2 Inlined code | §4.3 `v1/attention/cp.py` recipes |
| 3 Scattered info | §4.1 `CPContext` + §4.2 `CPBatchLayout` |
| 4 Breadth | §4.5 MLA tail-proj, §4.6 sparse, §4.7 linear/hybrid |
| 5 Chunked-prefill merge | §4.8 context+current merge |
| 6 enabled/active | §4.2 `active` field + §5 |

The remainder of this document is the closure of these gaps, in order.

---

## 3. Preserved invariants (what we do NOT change)

Before the design, the boundaries that close the gaps without breaking main:

- **Model-transparent**: model code contains zero CP branches (model calls `self.attn(...)`).
- **No parallel backend hierarchy**: CP stays folded into normal backends. No `*CPBackend` classes.
- **Comms in `vllm.distributed`**: CP calls `GroupCoordinator` (`all_gather`/`reduce_scatter`/`send_tensor_dict`) and `dist.all_to_all_single` — never owns primitives.
- **Numerics in `v1/attention/ops`**: LSE merge, merge_attn_states stay there.
- **No stateful manager**: layout (per-batch, worker) and recipes (per-layer, backend) have different lifetimes and stay separated.

---

## 4. The closures

### 4.1 `CPContext` — closes Gap 3 (scattered rank info)

One frozen value object, computed once, replacing the six loose attributes set in `AttentionImplBase.__new__` (`backend.py:746`).

```python
# vllm/v1/attention/backend.py
@dataclass(frozen=True)
class CPContext:
    pcp_size: int; pcp_rank: int
    dcp_size: int; dcp_rank: int
    pcp_group: GroupCoordinator | None
    dcp_group: GroupCoordinator | None
    interleave: int                       # == cp_kv_cache_interleave_size
    dcp_comm_backend: Literal["ag_rs","a2a"]
    policy: CPShardingPolicy              # §5

    @property
    def total_cp_size(self): return self.pcp_size * self.dcp_size
    @property
    def total_cp_rank(self): return self.pcp_rank * self.dcp_size + self.dcp_rank
    @property
    def cp_active(self):    return self.total_cp_size > 1

    @classmethod
    def build(cls, vllm_config) -> "CPContext":
        # same try/except group access as __new__ today
```
It is a carrier: no token mutation, no backend dispatch, no owned comms. `AttentionImplBase` exposes `self.cp: CPContext` instead of the loose attrs. Cap flags (`can_return_lse_for_decode`, `supports_pcp`, `supports_mtp_...`) stay as class attributes — unchanged.

### 4.2 `CPBatchLayout` — closes Gap 3 (no layout metadata) + Gap 6 (active)

Built in the worker (`v1/worker/cp_utils.py`, extending `gpu_model_runner._build_attention_metadata:2332`), carried on `CommonAttentionMetadata`. **Layout-level only** — the rule that keeps backends CP-uniform.

```python
@dataclass
class CPBatchLayout:
    active: bool                           # enabled AND this batch actually splits (Gap 6)
    policy: CPShardingPolicy

    # full (pre-split) lengths
    query_lens_full_cpu: torch.Tensor     # [num_reqs]
    seq_lens_full_cpu:   torch.Tensor | None
    # local (this rank) lengths
    query_lens_local_cpu: torch.Tensor
    seq_lens_local_cpu:   torch.Tensor | None
    dcp_local_seq_lens:   torch.Tensor | None   # [num_reqs] int32 GPU (absorbs existing field)

    num_actual_tokens: int
    num_actual_tokens_padded: int              # padded to policy granularity
    local_num_tokens: int

    restore_idx: torch.Tensor | None           # restore natural order after a gather
    unpad_mask:  torch.Tensor | None           # real tokens in padded buffer
    max_local_kv_len: int                      # = ceil(max_seq_len/(N*I))*I, no GPU sync

    segments: CPSegments | None                # high-level chunk sizes/offsets (zigzag head/tail,
                                               #   round-robin stripes); NEVER per-kernel index tensors
```

**The layout/kernel boundary** — adapted from vLLM-Ascend's `AscendPrefillContextParallelMetadata` (`pcp_utils.py:1066`) by splitting every field. This classification is exhaustive:

*Allowed (layout-level, L)*: `num_actual_tokens_pcp_padded`, `num_computed_tokens_of_pcp_dcp[req][pcp][dcp]`, `query_lens_pcp_full_cpu`, `max_query_len_pcp_full`, `pcp_unpad_mask`→`unpad_mask`, `pcp_allgather_restore_idx`→`restore_idx`, `pcp_padded_tokens_fla`, `pcp_enter_fa_restore_idx`/`pcp_exit_fa_scatter_idx` (generic residual↔FA enter/exit conversions), `max_num_tokens_across_pcp`, `total_num_scheduled_tokens`, `attn_chunk_seqlens`, `dcp_mtp_attn_mask`.

*Forbidden; derived in-backend (kernel-level, K)*: `q_head_idx`, `q_tail_idx`, `q_full_idx`, `kv_with_q_head_nomask_idx`, `kv_with_q_head_mask_idx`, `kv_with_q_tail_*`, `kv_tail_proj_idx`, `kv_with_q_*_attn_idx_in_tail`, `attn_mask_seqlens`, `head/tail_attn_nomask_seqlens`, `head/tail_actual_seq_lengths_kv`.

> Rule: **if only one backend/kernel can interpret a tensor, it is K and stays out of `CPBatchLayout`.** This single rule is why Ascend needs a fat `PCPManager` + per-type CP backends and this design needs neither.

`active` resolves Gap 6: `active = cp_enabled and policy_applies(batch)` — e.g. PCP enabled but `max_query_len` below the split threshold ⇒ `active=False`, backend skips the whole CP path.

### 4.3 `v1/attention/cp.py` — closes Gap 2 (inlined code)

Stateless recipes; backends call them at CP-sensitive points. Each body is a 1:1 lift of code already in production.

```python
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

# ---- DCP (extracted from flash_attn._forward_with_dcp) ----
def cp_gather_query_heads(ctx, query):
    """[B,H,D] -> [B,H*dcp,D]. No-op if dcp_size==1."""
    return ctx.dcp_group.all_gather(query, dim=1) if ctx.dcp_size > 1 else query

def cp_combine_out_lse(ctx, out, lse, *, return_lse=False):
    """[B,H_total,D]+[B,H_total] -> [B,H_local,D](+[B,H_local]).
       Contract obeyed by both ops backends."""
    fn = dcp_a2a_lse_reduce if ctx.dcp_comm_backend == "a2a" else cp_lse_ag_out_rs
    return fn(out, lse, ctx.dcp_group, return_lse=return_lse)

def cp_merge_prefix_suffix(output, p_out, p_lse, s_out, s_lse):
    merge_attn_states(output, p_out, p_lse, s_out, s_lse)   # intra-rank online-softmax

# ---- PCP prefill (NEW; KV-allgather Mode-1, no manager) ----
def cp_allgather_kv_restore(ctx, layout, key, value):
    """Each rank computed its segment K/V; all-gather to full + restore order.
       [T_local,kv_h,D] -> [T_full,kv_h,D] per K,V."""
    if ctx.pcp_size <= 1: return key, value
    kv = torch.cat([key, value], dim=-1)
    full = ctx.pcp_group.all_gather(kv, dim=0).index_select(0, layout.restore_idx)
    return full[..., :key.shape[-1]], full[..., key.shape[-1]:]

def cp_zigzag_q_split(layout, query):
    """Backend-local head/tail Q derivation (K-level, NOT in layout).
       rank r: head=[r*c:(r+1)*c), tail=[(2P-1-r)*c:(2P-r)*c)."""
    head_idx, tail_idx = _derive_head_tail_idx(layout)
    return query.index_select(0, head_idx), query.index_select(0, tail_idx)
```

**The merge primitive contract** (why `cp_combine_out_lse` is one line): both `cp_lse_ag_out_rs` (`ops/common.py:212`) and `dcp_a2a_lse_reduce` (`ops/dcp_alltoall.py:393`) obey
```
combine(out [B,H_total,D], lse [B,H_total], group, return_lse)
    -> out_local [B,H_total/N,D], lse_local [B,H_total/N]
```
- ag_rs: `all_gather(lse)→[N,B,H]` → `correct_attn_out` rescales `out*=exp(lse_local−lse_global)` → `reduce_scatter(out,dim=1)`. 2 collectives.
- a2a: pack `[D]+bitcast(fp32 lse)` into `[N,B,H/N,D+pack_dim]` → one `dist.all_to_all_single` → Triton unpack+combine. 1 collective.

One contract ⇒ every backend (GQA, MLA, future) reuses `cp_combine_out_lse` unchanged.

### 4.4 PCP prefill recipe — closes Gap 1

GQA prefill, Mode-1 (Partial Q + Full current KV), ported from ascend `attention_cp.py:485` but with K-level indices derived in-backend:
```python
def _forward_prefill_pcp(self, query, key, value, md):
    ctx = self.cp; layout = md.cp_layout
    key_f, value_f = cp_allgather_kv_restore(ctx, layout, key, value)     # full current KV
    q_head, q_tail = cp_zigzag_q_split(layout, query)                     # backend-local
    k_h_nomask, k_h_mask = _select_kv_for_chunk(key_f, layout, ctx.pcp_rank, "head")
    k_t_nomask, k_t_mask = _select_kv_for_chunk(key_f, layout, ctx.pcp_rank, "tail")
    # nomask = causal-free prefix (fast sparse_mode=0); mask = local causal chunk
    o_head, lse_head = _attn_nomask_then_mask(q_head, k_h_nomask, v_h_nomask, k_h_mask, v_h_mask)
    o_tail, lse_tail = _attn_nomask_then_mask(q_tail, k_t_nomask, v_t_nomask, k_t_mask, v_t_mask)
    return _restore_order([o_head,o_tail], [lse_head,lse_tail], layout)    # q_full_idx=argsort
```
The nomask/mask split (ascend `_attention_with_nomask_and_mask`, `attention_cp.py:421`) is a real compute saver: the prefix a rank fully attends to runs `atten_mask=None`; only the local chunk runs causal. The two are merged by the same online-softmax primitive (`merge_attn_states`-equivalent / `_npu_attn_out_lse_update`), two operands. No cross-rank LSE merge is needed in prefill (KV is complete).

### 4.5 Dense MLA — closes Gap 4 (MLA), with the tail-only `kv_b_proj`

Port ascend's `mla_preprocess_prefill` (`mla_cp.py:416`) exactly. The single biggest MLA cost saver:
```python
# all-gather compressed kv_c + k_pe (small), restore order
kv_c_k_pe = ctx.pcp_group.all_gather(torch.cat([kv_c_normed, k_pe], -1), dim=0)
kv_c_k_pe = kv_c_k_pe.index_select(0, layout.restore_idx)
# project ONLY the tail span through expensive kv_b_proj:
#   kv_tail_proj_idx = [kv_offset : kv_offset + chunk*(q_tail_chunk_id+1)]
tail = kv_c_k_pe.index_select(0, _derive_tail_proj_idx(layout))   # K-level, in-backend
k_nope, value = kv_b_proj(tail).split([qk_nope, v_dim], -1)
```
Because the head chunk's causal attention only needs KV up to its own head chunk (a prefix of the tail span), projecting only the tail span suffices for both halves — `kv_b_proj` runs over ~half the all-gathered sequence. Decode reuses §4.3's recipe with `head_size=kv_lora_rank` in the merge, then `v_up_proj`.

### 4.6 Sparse MLA / DSA / SFA — closes Gap 4 (sparse)

Correctness-first: **indexer must see global K**. Local topk over local K is wrong unless a distributed topk merge is added. First version: Partial Q + Full K for indexer/topk, Partial Q + Full KV for sparse attention. The sparse kernel stays CP-unaware if Q is local, the page table represents global KV, and sparse indices are computed against global K. DSA's TP-folded topology (`transpose(3,4)`-style) registers a backend-specific recipe but still uses common `CPContext`/`CPBatchLayout`.

### 4.7 Linear / hybrid — closes Gap 4 (linear + hybrid)

Linear layers need natural order (causal conv, recurrent scan, state cache). Two recipes:
- **Sequence-split + boundary state** (CONTINUOUS policy, hybrid): conv1d last-`(width−1)` boundary all-gathered across PCP; rank r uses rank r−1's boundary as `initial_state` (ascend `ops/triton/mamba/causal_conv1d.py:135`); ssm state persists in cache.
- **Head-split** (HEADSPLIT policy, pure-linear): rank owns `num_heads/N` heads over full seq; output via `cp_combine_out_lse`; saves state memory N×.

**Hybrid contract** (SGLang GLM5Next insight, without leaking into modeling): residual stream is CONTINUOUS; full-attention layers convert to zigzag/round-robin internally via enter/exit (`pcp_enter_fa_restore_idx`/`pcp_exit_fa_scatter_idx`, kept layout-level in `segments`), then convert back. Conversions live in backend recipes, never in model `forward`.

### 4.8 Chunked-prefill context merge — closes Gap 5

Current chunk: Partial Q + Full current KV (§4.4). Historical context: Partial KV + LSE merge. The two merge via `cp_merge_prefix_suffix` (`merge_attn_states`). The partial-KV context combine goes through `cp_combine_out_lse` (DCP a2a/ag_rs across the context shards), exactly the existing contract.

---

## 5. Layout policies — the one pluggable axis (closes Gap 1/4 cleanly)

`CPShardingPolicy` decides how tokens/KV are partitioned. Arithmetic is concrete.

```python
class CPShardingPolicy(Enum):
    ZIGZAG = "zigzag"          # dense prefill load balance
    CONTINUOUS = "continuous"  # hybrid / linear
    ROUNDROBIN = "roundrobin"  # DCP KV interleave; sparse indexer
    HEADSPLIT = "headsplit"    # pure-linear
```
- **Zigzag** (DualChunkSwap, ascend `pcp_utils.py:604`): pad req to `2*P`; `chunk=tokens//2`; rank r head=`[r*c:(r+1)*c)`, tail=`[(2P−1−r)*c:(2P−r)*c)`; `restore_idx=argsort(concat ranks)`.
- **Continuous**: rank r gets `[r*c:(r+1)*c)`. Required for Linear/hybrid.
- **Round-robin**: token i → rank `(i//I)%N`, `I=interleave`; matches `get_dcp_local_seq_lens` (`attention/backends/utils.py:862`) and `position%N` sparse ownership.
- **Head-split**: rank r owns `num_heads/N` heads, full seq; reduce via `cp_combine_out_lse`.

Policy is the **only** thing registered per model topology. Adding KimiLinear = register CONTINUOUS; no backend or manager change.

---

## 6. The fixed collective sequences

**Decode/context (Mode-2, Partial KV)** — canonical recipe (ascend `_forward_decode_pcp_dcp` `attention_cp.py:566`; main `_forward_with_dcp` DCP half):
```
DCP all_gather Q (heads,dim=1)     [B,H,D]->[B,H*N,D]
  local attn vs paged KV (seqused_k = dcp_local_seq_lens)   ->[B,H*N,D]+[H*N,B]
DCP combine (a2a|ag_rs)            ->[B,H,D]+[B,H]           cross-rank LSE merge
PCP all_gather(out,seq,dim=0)      if pcp>1                  ->[B,H,D]
merge with query-part              (merge_attn_states)
```
**Prefill (Mode-1, Full current KV)**:
```
PCP all_gather KV + restore_idx    each rank sees full current KV
  head/tail Q vs nomask/mask KV (backend-local indices)
  merge nomask+mask locally; restore Q order (q_full_idx=argsort)
```

Invariants: `cp_size==1` ⇒ every recipe early-returns, decode hot path zero extra ops; GQA/MQA `num_q_per_kv % dcp_size == 0` (`config/model.py:1222`); `block_size % interleave == 0` (`config/vllm.py:2101`).

---

## 7. Backend integration — before/after (closes Gap 2, concretely)

Phase-0 refactor of `_forward_with_dcp` (`flash_attn.py:930`), **behavior-identical**:
```python
def _forward_with_dcp(self, query, key, value, key_cache, value_cache, output, md):
    ctx = self.cp; layout = md.cp_layout
    q = cp_gather_query_heads(ctx, query)                         # [B,H,D]->[B,H*N,D]
    ws = alloc((B, H*ctx.dcp_size, D))
    c_out, c_lse = flash_attn_varlen_func(q, key_cache, value_cache, out=ws,
        seqused_k=layout.dcp_local_seq_lens, max_seqlen_k=layout.max_local_kv_len,
        causal=False, return_softmax_lse=True)
    c_out, c_lse = cp_combine_out_lse(ctx, c_out, c_lse.transpose(0,1), return_lse=True)
    c_lse = c_lse.transpose(0,1).contiguous()
    ws2 = alloc((B, H, D))
    q_out, q_lse = flash_attn_varlen_func(query, key, value, out=ws2,
        causal=md.causal, return_softmax_lse=True)
    cp_merge_prefix_suffix(output, c_out, c_lse, q_out, q_lse)
```
Same ops, same shapes, same order — `self.dcp_combine`/`self.dcp_world_size`/`md.dcp_context_kv_lens` collapse into `ctx`/`layout`. Zero behavior change; this de-risks everything and is the Phase-0 deliverable.

---

## 8. Phased roadmap (each phase closes specific gaps)

- **Phase 0 — Normalize DCP** (Gap 2). Introduce `CPContext`; extract `cp_gather_query_heads`/`cp_combine_out_lse`/`cp_merge_prefix_suffix`; rewrite `_forward_with_dcp` as §7. Tests unchanged.
- **Phase 1 — `CPBatchLayout` + `active`** (Gap 3, 6). Build it in the worker from `dcp_local_seq_lens` + (PCP) restore/unpad/segments. No kernel indices.
- **Phase 2 — GQA PCP prefill** (Gap 1). Port `cp_allgather_kv_restore`+zigzag+nomask/mask (§4.4); set `supports_pcp=True`; remove PCP from V2-unsupported list (`vllm.py:1991`).
- **Phase 3 — Chunked-prefill PCP+DCP** (Gap 5). Context partial-KV via `cp_combine_out_lse`; final `merge_attn_states`.
- **Phase 4 — Dense MLA** (Gap 4/MLA). Tail-only `kv_b_proj` (§4.5).
- **Phase 5 — Sparse MLA/DSA/SFA** (Gap 4/sparse). Indexer sees global K; kernel stays CP-unaware.
- **Phase 6 — Linear (seq-split) + hybrid (enter/exit)** (Gap 4/linear+hybrid). CONTINUOUS policy; conv1d boundary + ssm state ops in `v1/attention/ops`.
- **Phase 7 — Optional**: head-split (pure-linear), distributed sparse topk, partial-KV PCP prefill.

---

## 9. Testing
- **Baselines** per model/backend: TP-only, TP+DCP, TP+PCP, TP+PCP+DCP. Compare logits top-k, greedy tokens, per-layer stats, LSE finiteness.
- **Batch shapes**: pure prefill, pure decode, mixed, chunked prefill, short-prefill fallback (`active=False`), non-divisible lengths (padding), multi-request.
- **Layout units**: zigzag `restore_idx==argsort`, continuous, round-robin interleave, unpad mask, `get_dcp_local_seq_lens` kernel-vs-eager parity.
- **Backend**: FA DCP unchanged after Phase 0; FA PCP matches TP-only; MLA tail-proj index correctness; sparse indexer global-K; KDA/Mamba state order.

---

## 10. Open questions
1. PCP seq all-gather: fuse into `cp_combine_out_lse` (ascend fuses it in `_process_attn_out_lse`) or keep as a separate `ctx.pcp_group.all_gather` in the recipe (main keeps the combine DCP-only)? Lean separate, for clarity.
2. `CPBatchLayout` as a field on `CommonAttentionMetadata` vs standalone? Lean field on Common.
3. First FA PCP policy: zigzag (load-balanced, matches ascend) vs round-robin (matches DCP KV layout, simpler indexing)? Lean zigzag.
4. Non-divisible lengths: padded equal chunks vs ragged? Lean padded + `unpad_mask`.
5. Are `pcp_enter_fa_restore_idx`/`pcp_exit_fa_scatter_idx` truly layout-level? They are the most K-like L fields (generic enter/exit, but shaped by FA). Kept in `segments`; revisitable.
6. DSA TP-folded topology — ROUNDROBIN recipe variant, or its own policy?

---

## 11. Source map
**vLLM main:** `v1/attention/backend.py` (`AttentionImplBase.__new__:746`, cap flags `:716`), `v1/attention/backends/flash_attn.py` (`_forward_with_dcp:930`, builder DCP `:489`, metadata `:236`), `v1/attention/backends/utils.py` (`get_dcp_local_seq_lens:862`), `v1/attention/ops/dcp_alltoall.py:393`, `v1/attention/ops/common.py:212`, `v1/attention/ops/merge_attn_states.py:9`, `v1/worker/cp_utils.py:14`, `v1/worker/gpu/cp_utils.py:8`, `v1/worker/gpu_model_runner.py:2332`, `distributed/parallel_state.py` (`GroupCoordinator:351`, DCP group `:1774`, PCP group `:1796`), `config/parallel.py` (`:124/:331/:351/:343`), `config/vllm.py:1991`.
**vLLM-Ascend:** `worker/pcp_utils.py` (`update_tokens_for_pcp:502`, `generate_pcp_metadata:1066`, `_get_cp_local_seq_lens:1042`), `attention/context_parallel/common_cp.py` (`_process_attn_out_lse:108`, `_npu_attention_update:130`), `attention/context_parallel/attention_cp.py` (`_forward_prefill_cp:485`, `_forward_decode_pcp_dcp:566`, all-gather-KV `:832`), `attention/context_parallel/mla_cp.py:416`, `ops/triton/mamba/causal_conv1d.py:135`.
**SGLang:** `python/sglang/srt/models/glm5_next.py`, `python/sglang/srt/layers/attention/dsa/*`.

---

## 12. One-sentence position
vLLM closes its CP gaps — PCP prefill, reusable recipes, layout-level metadata, MLA/sparse/linear/hybrid breadth, chunked-prefill merge, enabled/active — by adding `CPContext` + `CPBatchLayout` + `v1/attention/cp.py` recipes and porting vLLM-Ascend's *recipes* (zigzag prefill, DCP/PCP LSE-merge, MLA tail-proj, linear state propagation), while preserving main's boundaries (model-transparent, CP folded into normal backends, comms via `vllm.distributed`, numerics in `v1/attention/ops`) and adopting **none** of Ascend's fat `PCPManager`, per-type CP backends, or kernel-level metadata.
