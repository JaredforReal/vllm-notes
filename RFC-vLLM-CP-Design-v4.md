# RFC: vLLM Context Parallel Design, Starting From What Is Missing

> Goal: start from what vLLM is currently missing, then define the implementation
> design around those gaps.

## 1. What vLLM Is Currently Missing

vLLM is not starting from zero. It already has:

- `prefill_context_parallel_size`
- `decode_context_parallel_size`
- PCP and DCP process groups
- CP-aware KV cache slot mapping
- FullAttention/GQA DCP
- MLA DCP
- LSE-based CP merge ops
- coarse `supports_pcp` compatibility checks

The missing pieces are:

1. **A per-forward CP layout contract**
   - There is no shared object that says whether PCP is active, how tokens were
     split, how to restore them, or how local lengths relate to global lengths.
2. **Enabled-vs-active semantics**
   - `pcp_size > 1` means PCP is configured, but a batch may still fall back to
     non-PCP because it is decode-only, too short, or unsupported by a backend.
3. **Runner-level token splitting and restore metadata**
   - PCP changes `input_ids`, positions, query lengths, logits indices, slot
     mapping, multimodal embeddings, and MRoPE metadata. This cannot live only
     inside attention kernels.
4. **A PCP slot-mapping and KV-cache-update contract**
   - vLLM has CP-aware physical KV ownership, but not the contract for restored
     current K/V under PCP prefill.
5. **Attention-family PCP support**
   - FullAttention/GQA lacks PCP prefill.
   - Dense MLA lacks PCP prefill.
   - Sparse MLA/indexer needs global-K correctness.
   - Linear/Mamba/KDA hybrid models need layout transitions.
6. **Chunked prefill PCP+DCP**
   - MLA chunked context and DCP exist, but PCP+DCP chunked context does not
     have shared metadata or workspace rules.
7. **Structured backend capability declarations**
   - `supports_pcp: bool` is too coarse for full-KV prefill, partial-KV LSE,
     sparse indexers, and hybrid layout transitions.
8. **A phased upstream plan and test matrix**
   - PCP is too broad for one PR.

This RFC is organized around those missing pieces.

## 2. Current vLLM Baseline

### 2.1 Config and Groups

Current files:

- `vllm/config/parallel.py`
- `vllm/engine/arg_utils.py`
- `vllm/distributed/parallel_state.py`

Important existing config:

```python
prefill_context_parallel_size: int = 1
decode_context_parallel_size: int = 1
dcp_comm_backend: Literal["ag_rs", "a2a"] = "ag_rs"
cp_kv_cache_interleave_size: int = 1
```

Existing CP rank convention:

```python
total_cp_rank = pcp_rank * dcp_world_size + dcp_rank
total_cp_world_size = pcp_world_size * dcp_world_size
```

This convention should be preserved everywhere.

### 2.2 KV Cache Ownership Already Knows CP

Current file:

- `vllm/v1/worker/block_table.py`

`BlockTable.compute_slot_mapping` already uses:

```python
TOTAL_CP_WORLD_SIZE = pcp_world_size * dcp_world_size
TOTAL_CP_RANK = pcp_rank * dcp_world_size + dcp_rank
CP_KV_CACHE_INTERLEAVE_SIZE = cp_kv_cache_interleave_size
```

So physical KV cache ownership is already CP-sharded. PCP design should not
introduce a second ownership model.

### 2.3 FullAttention/GQA DCP Exists

Current file:

- `vllm/v1/attention/backends/flash_attn.py`

`FlashAttentionImpl._forward_with_dcp` already implements partial-KV decode
context attention:

1. gather query heads across DCP
2. run attention over local historical KV
3. return output + LSE
4. combine DCP partials with `cp_lse_ag_out_rs` or `dcp_a2a_lse_reduce`
5. run current query-token attention
6. merge historical-context and current-token states with `merge_attn_states`

This is the reference path for future Partial-KV PCP.

### 2.4 MLA DCP Exists

Current files:

- `vllm/model_executor/layers/attention/mla_attention.py`
- `vllm/v1/attention/backends/mla/flashattn_mla.py`
- `vllm/v1/attention/backends/mla/flashmla.py`

MLA already has:

- `MLACommonMetadataBuilder`
- prefill/decode split
- chunked context metadata
- MLA prefill backend abstraction
- DCP local context lengths
- DCP all-gather/reorg for chunked context
- DCP query-head gather for decode
- LSE correction after decode

PCP should extend this common path, not bypass it.

### 2.5 vLLM-Ascend Shows the Missing PCP Mechanics

Reference files:

- `vllm_ascend/worker/pcp_utils.py`
- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/attention/context_parallel/common_cp.py`
- `vllm_ascend/attention/context_parallel/attention_cp.py`
- `vllm_ascend/attention/context_parallel/mla_cp.py`
- `vllm_ascend/attention/context_parallel/sfa_cp.py`

Useful lessons:

- runner-level PCP token splitting is required
- DualChunkSwap/zigzag is practical for dense prefill
- restore indices and unpad masks are required
- decode and prefill need different PCP behavior
- hybrid models need enter/exit full-attention layout indices
- chunked context needs local PCP/DCP lengths

Do not directly copy `PCPManager` into vLLM core. It mixes scheduling mutation,
buffers, layout metadata, backend-specific indices, slot mapping, logits,
hybrid transitions, and MTP masks. Core vLLM should express the reusable parts
as metadata plus stateless helpers.

## 3. Missing Piece 1: Per-Forward CP Layout Contract

### 3.1 Current State

`CommonAttentionMetadata` contains query starts, seq lens, block table, slot
mapping, DCP local seq lens, positions, and prefill/decode flags. It does not
describe PCP token layout.

### 3.2 Target Design

Add `CPBatchLayout` and attach it to `CommonAttentionMetadata`:

```python
@dataclass
class CommonAttentionMetadata:
    ...
    cp_layout: CPBatchLayout | None = None
```

`CPBatchLayout` is layout-level metadata only. It should group these fields:

- state: `pcp_enabled`, `dcp_enabled`, `pcp_active`, `dcp_active`, `policy`
- rank: `pcp_size/rank`, `dcp_size/rank`, `total_cp_size/rank`,
  `cp_kv_cache_interleave_size`
- batch shape: `num_reqs`, `num_decodes`, `num_prefills`,
  `num_decode_tokens`
- full query shape: `query_lens_full_cpu`, `query_start_loc_full_cpu`,
  `max_query_len_full`
- local query shape: `query_lens_local_cpu`, `query_start_loc_local_cpu`,
  `max_query_len_local`, `num_tokens_local`
- PCP padding/restore: `num_tokens_pcp_padded`, `max_tokens_across_pcp`,
  `pcp_unpad_mask(_cpu)`, `pcp_allgather_restore_idx`
- CP local context: `cp_local_seq_lens(_cpu)`
- hybrid transitions: `enter_attention_restore_idx`,
  `exit_attention_scatter_idx`, `attention_query_idx`

Allowed in `CPBatchLayout`:

- active/enabled state
- rank/world size
- policy
- global/local query lengths
- padding count
- restore index
- unpad mask
- local CP sequence lengths
- generic hybrid enter/exit indices

Not allowed:

- FullAttention head/tail KV index tensors
- MLA projection indices such as `kv_tail_proj_idx`
- sparse indexer topk results
- backend-specific masks
- graph workspaces

Those are backend-local derived metadata.

### 3.3 Implementation Location

Add one core module:

```text
vllm/v1/attention/cp.py
```

It should contain:

- `CPContext`
- `CPPolicy`
- `CPBatchLayout`
- layout builders
- gather/restore helpers
- local-seq-lens helpers

## 4. Missing Piece 2: Enabled vs Active Semantics

### 4.1 Current State

vLLM mostly checks:

```python
if pcp_size > 1:
    assert layer_impl.supports_pcp
```

This conflates configured PCP with actually using PCP for the current forward.

### 4.2 Target Design

Separate:

```python
pcp_enabled = pcp_size > 1
pcp_active = layout is not None and layout.pcp_active
```

Backends must branch on `pcp_active`, not only `pcp_enabled`.

### 4.3 Fallback Rules

Initial zigzag fallback:

```python
can_pcp_split = all(
    query_len >= 2 * pcp_size
    for each prefill request
)
```

If false:

- `pcp_enabled=True`
- `pcp_active=False`
- local input remains unsplit
- metadata remains normal
- backend runs normal prefill

This avoids crashes for short prompts, prefix-cache hits, decode-only batches,
and unsupported shapes.

## 5. Missing Piece 3: Runner-Level Token Splitting

### 5.1 Current State

The runner builds inputs and attention metadata assuming one local token layout.
DCP local lengths exist, but PCP does not yet alter `input_ids`, positions,
query lengths, slot mapping, or logits indices.

### 5.2 Target Design

Add a pure layout builder:

```python
def build_pcp_layout_and_local_positions(
    query_lens_full_cpu: torch.Tensor,
    num_decodes: int,
    cp_context: CPContext,
    policy: CPPolicy,
) -> tuple[CPBatchLayout, torch.Tensor]:
    ...
```

The returned `local_positions_cpu` indexes into the full scheduled-token order:

```python
input_ids_local = input_ids_full[local_positions_cpu]
positions_local = positions_full[local_positions_cpu]
query_lens_local = layout.query_lens_local_cpu
query_start_loc_local = layout.query_start_loc_local_cpu
```

### 5.3 Zigzag / DualChunkSwap

For each prefill request:

```python
segments = 2 * pcp_size
aligned_len = ceil_div(query_len, segments) * segments
chunk = aligned_len // segments

head_chunk = pcp_rank
tail_chunk = segments - 1 - pcp_rank

local = head_chunk_tokens + tail_chunk_tokens
```

Decode tokens are not zigzag-split in the first implementation.

### 5.4 Restore Index

After PCP all-gather:

```text
rank0 local || rank1 local || ... || rankN local
```

`pcp_allgather_restore_idx` restores original request-major token order.

It is used for:

- hidden-state restore before logits
- K/V restore before cache update
- MLA `kv_c_normed/k_pe` restore
- future sparse global-K indexer
- hybrid enter-attention restore

### 5.5 Logits

Initial implementation should use the simple correct path:

- restore hidden states across PCP before logits
- reuse existing logits code

Local logits optimization can come later.

## 6. Missing Piece 4: PCP Slot Mapping and KV Cache Update

### 6.1 Current State

`BlockTable.compute_slot_mapping` maps logical token positions to CP-local
physical slots. But PCP active prefill creates local K/V before attention. Cache
update must see K/V in the same logical order as slot mapping.

### 6.2 Initial Target: Partial Q + Full Current KV

Initial PCP prefill mode:

```text
Partial Q + Full current KV
```

Properties:

- each PCP rank computes only local Q rows
- current K/V are all-gathered and restored before attention/cache update
- each rank writes only KV entries owned by its CP slot mapping
- no LSE merge is needed for current prefill

### 6.3 Core Helpers

Add:

```python
pcp_all_gather_restore(x, layout, group, dim=0, unpad=True)
```

Behavior:

1. pad local tensor to `layout.max_tokens_across_pcp`
2. all-gather across PCP
3. restore with `layout.pcp_allgather_restore_idx`
4. optionally unpad with `layout.pcp_unpad_mask`

Add:

```python
build_pcp_padded_slot_mapping(slot_mapping, layout)
```

Behavior:

- fill padded entries with `PAD_SLOT_ID` / `-1`
- place real slot mappings at restored positions
- return the shape expected by cache update

### 6.4 Cache Ownership Invariant

Keep existing invariant:

```text
logical sequence is full
physical KV cache is CP-sharded by total_cp_rank
slot_mapping decides which tokens this rank stores
```

Do not create a second KV ownership model for PCP.

## 7. Missing Piece 5: Attention-Family PCP Support

This section combines the original gaps for FullAttention/GQA, dense MLA, sparse
MLA/indexer, and Linear/Mamba/KDA hybrid models.

### 7.1 FullAttention/GQA PCP Prefill

Current state:

- normal prefill exists
- DCP decode/context exists
- PCP prefill does not exist

Initial target:

```text
Partial Q + Full current KV
```

Metadata: derive backend-local `FlashAttentionPCPMetadata` from
`CPBatchLayout`, containing restore/unpad tensors, padded token count, max tokens
across PCP, and local/full query start locations.

Forward recipe: pack local K/V, PCP all-gather, restore global token order,
unpack full K/V, then call `flash_attn_varlen_func` with local Q start locs and
full K/V start locs.

Cache update: build padded restored slot mapping, then call the existing
`reshape_and_cache_flash` with restored full current K/V.

Optional later optimization: Ascend-style head/tail split with no-mask and mask
subcalls. Those derived indices stay backend-local.

### 7.2 Dense MLA PCP Prefill

Current state:

- `MLACommonMetadataBuilder` exists
- `MLACommonImpl.forward_mha` handles prefill
- `MLACommonImpl.forward_mqa` handles decode-style path
- MLA DCP exists
- PCP prefill does not exist

Initial target:

```text
Partial Q + Full current KV
```

Extend `MLACommonMetadata` with `cp_layout`. Extend prefill metadata with
`MLAPCPPrefillMetadata`, derived from `CPBatchLayout`, containing restore/unpad
tensors, padded token count, and local/full query start locations.

Forward recipe: keep Q local, PCP all-gather/restore `kv_c_normed` and `k_pe`,
project restored `kv_c_normed` through `kv_b_proj`, build full `k/v`, then call
`prefill_backend.run_prefill_new_tokens` for local Q against full current KV.

The first MLA PCP PR should not include sparse MLA, chunked context PCP+DCP,
FP8 PCP, or partial-KV PCP.

### 7.3 Sparse MLA / Indexer PCP

Current state:

- sparse MLA/indexer code exists
- no shared CP contract guarantees global-K topk correctness

Initial target:

```text
Partial Q + Full K / Full KV visibility
```

Indexer contract:

- local Q is split by PCP
- projected/index K is all-gathered/restored across PCP
- topk is computed against global K
- sparse attention runs for local Q
- kernel can remain PCP-unaware if metadata is correct

Future policy:

- round-robin/interleaved layouts may be better for sparse/indexer models
- do not block dense FullAttention/MLA PCP on round-robin

### 7.4 Linear/Mamba/KDA Hybrid Models

Current state:

- `LinearAttentionMetadata` and `BaseMambaAttentionMetadata` assume token order
  is valid for state updates
- they have no CP layout field

Problem:

- zigzag order is unsafe for causal conv, recurrent state updates, chunk scan,
  Mamba state indexing, and KDA-style linear attention

Target hybrid contract:

```text
between layers: continuous local token layout
inside FullAttention/MLA: backend may enter zigzag/restored attention layout
after attention: scatter back to continuous local token layout
```

Generic fields already reserved in `CPBatchLayout`:

```python
enter_attention_restore_idx
exit_attention_scatter_idx
attention_query_idx
```

These correspond to vLLM-Ascend's:

- `pcp_enter_fa_restore_idx`
- `pcp_exit_fa_scatter_idx`
- `pcp_fa_query_idx`

Do not implement full hybrid PCP in the first FullAttention/MLA PR. Add the
metadata hooks now, then support one hybrid model in a dedicated PR.

## 8. Missing Piece 6: Chunked Prefill PCP+DCP

### 8.1 Current State

MLA chunked prefill already tracks:

- chunk cu-seqlens
- starts
- max seq lens
- token-to-seq
- workspace
- DCP local context lengths
- DCP padded local chunk lengths

### 8.2 Missing

PCP adds:

- PCP local context length per request
- restore/recover indices for chunks
- workspace sizing across PCP and DCP
- current-chunk vs historical-context ownership rules

### 8.3 Target Design

Generalize local length metadata:

```python
cp_local_seq_lens_cpu: [num_reqs, pcp_size, dcp_size]
```

Metadata builders derive:

- local context starts
- local chunk lengths
- padded local chunk lengths
- chunk restore indices
- context output LSE merge plan

### 8.4 Rollout

Phase 1:

- disable PCP for chunked context
- fallback cleanly

Phase 2:

- current prefill chunk uses Partial Q + Full current KV
- historical context remains DCP-only

Phase 3:

- full PCP+DCP chunked context with LSE merge

## 9. Missing Piece 7: Structured Backend Capabilities

### 9.1 Current State

`supports_pcp: bool` is too coarse.

Backends differ by:

- full-KV prefill support
- partial-KV LSE support
- DCP decode support
- zigzag support
- continuous-layout support
- sparse global-indexer support
- hybrid enter/exit support

### 9.2 Target

Add:

```python
@dataclass(frozen=True)
class PCPBackendSupport:
    full_kv_prefill: bool = False
    partial_kv_prefill_lse: bool = False
    decode_dcp: bool = False
    zigzag: bool = False
    contiguous: bool = False
    round_robin: bool = False
    sparse_global_indexer: bool = False
    hybrid_enter_exit: bool = False
```

Initial PR can keep `supports_pcp` and add structured support later, but the
target should be policy-aware validation:

```python
if layout.policy == CPPolicy.ZIGZAG:
    assert impl.pcp_support.zigzag
if layout.requires_full_kv_prefill:
    assert impl.pcp_support.full_kv_prefill
```

## Main Invariants

1. If `pcp_active=False`, behavior matches current vLLM.
2. PCP fallback must not crash short prefills.
3. Initial PCP prefill uses Partial Q + Full current KV.
4. Physical KV cache remains CP-sharded by total CP rank.
5. FullAttention DCP decode remains unchanged.
6. MLA DCP decode remains unchanged.
7. Backend-specific derived indices stay out of `CPBatchLayout`.
8. Linear/Mamba/KDA layers do not consume zigzag order unless explicitly
   designed for it.
