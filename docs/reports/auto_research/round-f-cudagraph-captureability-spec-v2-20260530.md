# Round-F CUDA Graph Captureability Spec — v2 (Revised) — 2026-05-30

## Status of this document

This revises and corrects `round-f-cudagraph-captureability-spec-20260530.md`
(the "v1 draft"). It keeps v1's core engineering conclusion — *do not force the
whole GDN/tree verifier into one monolithic full graph this iteration; keep the
fixed-shape scaffolding capture-clean and let vLLM run the rest piecewise* — but
fixes several load-bearing factual claims and rewrites the mitigation plan around
what vLLM's CUDA-graph machinery actually does.

Every non-obvious claim below is backed by a primary source (PyTorch / NVIDIA
docs, or vLLM source/PRs at a pinned ref). See **Sources**. Where a fact is
version-sensitive it is flagged inline.

> Version note up front. The serving image is tagged `v0.19.0`
> (`DEFAULT_VLLM_IMAGE = lumo-flywheel-vllm:26.01-py3-v0.19.0`), but there is no
> public vLLM release literally numbered 0.19; current upstream `main` self-
> versions as `0.16.x`. Treat "our 0.19.0" as *a recent vLLM that already has the
> V1 `cudagraph_mode` + `AttentionCGSupport` system (PR #20059) and mamba/GDN
> full-decode CUDA graphs (PR #21401, #22594)*. **Action:** pin the exact upstream
> commit the image was built from; several corrections below depend on which side
> of specific PRs that commit lands.

---

## 1. Corrections to the v1 draft

### C1 — vLLM already full-captures the GDN *decode* step; it is NOT eager

v1 frames `GDNAttentionBackend` as "not full-capture safe → eager GDN." That is
imprecise and it changes the whole goal.

`GDNAttentionMetadataBuilder` declares `_cudagraph_support =
AttentionCGSupport.UNIFORM_BATCH`, and it ships a `build_for_cudagraph_capture`
whose docstring says *"Currently, only decode is supported for full cudagraphs
with Mamba"* with an assert that capture is decode-only. With
`cudagraph_mode = FULL_AND_PIECEWISE` (the V1 default), the **uniform-decode
batch is captured as a true FULL graph that includes the GDN/Mamba `causal_conv1d`
+ `selective_state_update` kernels**; only prefill/mixed batches run piecewise.

The definitive statement is in merged PR #34571:

> "Only FULL decode captures are affected because they run GDN/Mamba layers on
> the decode path (`max_query_len=1`), which calls `causal_conv1d_update`.
> PIECEWISE captures run GDN/Mamba ops eagerly on the prefill path
> (`max_query_len=num_tokens`), so `causal_conv1d_update` is never called."

So: decode-side GDN is captured; prefill-side GDN is eager. "Eager GDN" is only
the *prefill/mixed* half, and only because that half is piecewise by design.

**Impact:** "make CUDA graph work full" should not mean "capture prefill+decode
in one graph." It should mean "make sure the **uniform-decode** GDN/tree step is
*actually* in the full graph and is not being kicked back out to eager by our own
patches." That reframes Tracks C/E below.

### C2 — `FULL_AND_PIECEWISE` is the correct target, not a compromise

A single monolithic `FULL` graph spanning prefill+decode requires the minimum
`AttentionCGSupport` across *all* layers to be `ALWAYS`. Even FlashAttention only
reaches `ALWAYS` on FA3; FA2 is deliberately pinned to `UNIFORM_BATCH`. A hybrid
model whose linear-attention/GDN mixer must handle variable-length prefill cannot
honestly advertise `ALWAYS` for mixed batches. vLLM's own docs call
`FULL_AND_PIECEWISE` *"the most performant mode for most models and is the
default."*

**Impact:** the success condition is **`FULL` on the uniform-decode bucket +
`PIECEWISE` on prefill/mixed**, with the decode bucket genuinely capturing the
GDN core. Treat a measured `FULL_AND_PIECEWISE` with a captured decode step as
*success*, and treat decode that silently runs eager/piecewise as the failure.

### C3 — the `cudagraph_unsafe` tag is real, but only bites on an experimental path

v1 leans on tagging `gdn_attention_core` with
`tags=(torch._C.Tag.cudagraph_unsafe,)`. The tag is real:

> `aten/src/ATen/native/tags.yaml`: *"cudagraph_unsafe … This operator does not
> support cudagraphs. The presence of this tag on an operator will cause Inductor
> to split the graph around this operator."*

But vLLM has **two** split mechanisms and the tag is consulted by only one:

- **Default (`use_inductor_graph_partition=False`)** — vLLM splits the FX graph at
  a *hardcoded* `_attention_ops` list (Dynamo-level). That list already contains
  `vllm::qwen_gdn_attention_core`, `vllm::mamba_mixer2`, `vllm::linear_attention`,
  etc. The `cudagraph_unsafe` tag is **not consulted** on this path.
- **Inductor partition (`use_inductor_graph_partition=True`, requires
  `torch>=2.9`, experimental)** — vLLM partitions at Inductor level and *does*
  key off the `cudagraph_unsafe` tag.

Also note: upstream `main` registers the op **without** any `tags=` argument, and
the op is named **`qwen_gdn_attention_core`** (renamed from `gdn_attention_core`).

**Impact (two actions):**
1. Confirm `compilation_config.use_inductor_graph_partition`. If it is `False`
   (the default), `LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE=1` adding the tag is likely a
   **no-op**, and the FULL→FULL_AND_PIECEWISE downgrade you observed is driven
   purely by GDN's `UNIFORM_BATCH` support — not by the tag. Do not attribute the
   runtime mode to the tag without checking this.
2. Where the tag *is* honored, it **excludes the GDN core from the decode full
   graph** (C1). So the tag is the lever that *forfeits* decode-side full capture
   — it is a deliberate correctness/stability guard, not a neutral setting, and
   removing it is exactly the goal of Mitigation B.

### C4 — vLLM's mamba/GDN spec-decode is a *state-copy*, and the upstream fix is unmerged

v1 calls this "state-rollback." vLLM does **not** roll back or recompute. In
`vllm/v1/worker/mamba_utils.py` it does a **block-aligned state copy + conv-offset
read** keyed by the accepted position:

```text
new_num_computed_tokens = num_computed_tokens + num_scheduled_tokens
                          - num_draft_tokens + num_accepted_tokens - 1
```

The `-1` is because `num_accepted_tokens` counts the accepted drafts **plus** the
bonus/sampled token, so the last valid state row is `num_accepted_tokens - 1`
from the running-state anchor. This is the same convention the closeout's
`final_row = base + accepted_count - 1` already uses — good, it matches upstream.

Critically: the upstream fix for the GDN spec-decode state corruption
(**issue #39273**, **PR #40738 "Fix GDN conv + SSM state corruption with ngram
spec decode"**) is, as of this writing, **OPEN / not merged** (`needs-rebase`),
and it only covers the all-non-spec branch (a mixed-batch hole remains).

**Impact:** pulling a recent `main` does **not** hand you a correct GDN
spec-decode state path — the `LUMO_FA_*` patches must carry that correctness
themselves (which is exactly what the tree-delta state-copy work does). Cite
#39273/#40738 as *corroborating the bug class*, not as a shipped fix.

### C5 — the `causal_conv1d` assert fix is real but may be redundant on a recent base

The `num_cache_lines >= batch` assert (your `_CAUSAL_CONV_CUDAGRAPH_ASSERT_FIX`
prelaunch patch) is a known upstream bug — **issue #35945**, reported on exactly
this stack (vLLM 0.16.1rc1, Qwen3.5/GDN). Root cause matches your patch comment:
`num_cache_lines` is the Mamba state-pool width while `batch` is the padded
cudagraph capture size; the real selector is `conv_state_indices`, so the correct
invariant is `batch == conv_state_indices.shape[0]`.

On current `main` the offending asserts are already gated behind
`validate_data=False`, and the durable fix (**PR #34571**) instead *caps capture
sizes to the Mamba cache-block count* via
`CompilationConfig.adjust_cudagraph_sizes_for_mamba_cache(num_mamba_cache_blocks)`,
called from `GPUModelRunner._check_and_update_cudagraph_mode` when any
`MambaSpec` group is present.

**Impact:** keep your assert patch only if the pinned base predates these
changes; otherwise it may be dead code or conflict. Prefer adopting #34571's
size-capping over re-patching the assert, and confirm your
`cudagraph_capture_sizes` never exceed `num_mamba_cache_blocks`.

### C6 — tree-attention is `AttentionCGSupport.NEVER`, and the obstacle is host syncs, not the mask

In the era where `TreeAttentionBackend` exists (verified at tag v0.11.1),
`TreeAttentionMetadataBuilder` declares **no** `_cudagraph_support`, so it inherits
the base default `AttentionCGSupport.NEVER`. Two consequences:

- A model whose minimum backend capability is `NEVER` is forced to `PIECEWISE`
  (or eager) for that attention op — tree-verify cannot run under FULL. This is
  *why* branched/tree decode is piecewise.
- The capture obstacle is **not** a dynamic mask: the tree bias is precomputed
  once in `__init__` from a fixed/regular `speculative_token_tree`. The real
  blockers are host-side `.item()` / `.max().item()` reductions in `build()` and
  the variable decode/prefill split — i.e. data-dependent host work, not topology.

Also flagged: on current `main`, `tree_attn.py` / the `TREE_ATTN` enum appear to
have been **removed**, and the EAGLE drafter runs a *chain* of forward passes with
tree-spec marked `FIXME`. So tree-verify capturability is partly a moving target;
pin your base before investing.

**Impact:** the branched-tree path remains research-only for capture reasons that
are structural in this vLLM line. Focus capture effort on the **F-spine** path
(C2), where the topology is a fixed chain.

---

## 2. General capture-safety rules (every new change must satisfy these)

These are the hard constraints. A captured graph **records GPU work and replays
the identical kernels on the identical memory addresses**; it does not re-run
Python and does not re-decide control flow.

1. **Static addresses.** Inputs/outputs/state must live in long-lived buffers;
   feed new data with in-place `copy_()`, never by rebinding the Python variable.
   (PyTorch CUDA Graphs guide.)
2. **No host↔device sync inside capture.** No `.item()`, `.cpu()`, `.tolist()`,
   `print(tensor)`, `.nonzero()`, `torch.unique`, `cuda.synchronize()`, or any
   stream/event query/sync. Each forces a CPU↔GPU sync and breaks/invalidates the
   capture. (NVIDIA "Prohibited and Unhandled Operations"; PyTorch sync-free
   guidance.)
3. **No data-dependent Python control flow on tensor values**, and **no
   dynamic/variable shapes** — topology and shapes are frozen at capture; varying
   them between replays is undefined. Handle batch variation by capturing a
   discrete set of sizes and padding up (bucketing). (NVIDIA; PyTorch.)
4. **No allocation inside capture.** Allocate during init/warmup; graphs use a
   private memory pool and capture-time allocations get frozen into it.
5. **Warm up on a side stream before capture** so allocator/JIT/autotune settle
   outside the captured region. (PyTorch: ≥3 warmup iters; the side stream waits
   on, then is waited on by, the current stream.)
6. **Detect capture with `torch.cuda.is_current_stream_capturing()`** to gate any
   path that might do host work. (PyTorch 2.12 adds the backend-agnostic alias
   `torch.Stream.is_capturing()`; the `cuda.` form remains valid.)

Forward-looking, **not actionable today:** CUDA 12.4+/12.8 *conditional graph
nodes* (IF/WHILE/SWITCH) can do data-dependent control flow on-device without a
host sync, and GB10/Blackwell+CUDA 12.8 supports them. But PyTorch only exposes
them via `torch.cond` on the eager/cudagraphs backends (Inductor support not yet
shipped), and vLLM does not use them — so they are not a lever for vLLM's
Inductor-compiled model graph right now. Do not design around them yet.

---

## 3. vLLM capture machinery (what actually decides the runtime mode)

- **`CUDAGraphMode`** (`vllm/config/compilation.py`): `NONE`, `PIECEWISE`,
  `FULL`, plus compounds `FULL_DECODE_ONLY = (FULL, NONE)` and
  `FULL_AND_PIECEWISE = (FULL, PIECEWISE)` (tuples = `(decode_mode, mixed_mode)`).
  V1 default is `FULL_AND_PIECEWISE`.
- **`AttentionCGSupport`** (`vllm/v1/attention/backend.py`), ordered:
  `ALWAYS(3) > UNIFORM_BATCH(2) > UNIFORM_SINGLE_TOKEN_DECODE(1) > NEVER(0)`.
  `UNIFORM_BATCH` = "all query lengths equal," which is exactly the spec-decode
  decode shape `1 + num_speculative_tokens`. A builder advertises its level via
  the private `_cudagraph_support` ClassVar (read through
  `get_cudagraph_support()`).
- **Downgrade logic** lives in
  `GPUModelRunner._check_and_update_cudagraph_mode()`: it takes the **minimum**
  `AttentionCGSupport` over all backends/KV-cache groups, then resolves the mode.
  Requesting `FULL` with a `UNIFORM_BATCH` minimum and attention in
  `splitting_ops` ⇒ **`FULL_AND_PIECEWISE`** (warning:
  `"CUDAGraphMode.FULL is not supported with <backend> backend (support:
  AttentionCGSupport.UNIFORM_BATCH) ; setting cudagraph_mode=FULL_AND_PIECEWISE"`).
  A `NEVER` minimum ⇒ `PIECEWISE`, or `NONE` if attention isn't compiled
  piecewise. This is the resolution you must *measure*, never assume.
- **Two split mechanisms** (see C3): default Dynamo split on the hardcoded
  `_attention_ops` list vs. experimental Inductor partition keyed on the
  `cudagraph_unsafe` tag.
- **Capture sizes / padding**: `cudagraph_capture_sizes` +
  `max_cudagraph_capture_size`; at runtime `CudagraphDispatcher` pads the batch up
  to a captured size and publishes the mode via
  `set_forward_context(..., cudagraph_runtime_mode, batch_descriptor)`. For mamba,
  cap sizes to `num_mamba_cache_blocks` (PR #34571).
- **Spec-decode alignment**: when spec decode is on, capture sizes for the uniform
  decode must be divisible by `uniform_query_len = 1 + num_spec_tokens`
  (PR #23679). This is what makes the spec "decode" shape static and thus
  capturable.
- **Drafter capture**: the proposer/drafter forward is captured **separately** and
  *only* supports `PIECEWISE` (per `llm_base_proposer.initialize_cudagraph_keys`;
  full-CG-for-drafter is open feature request #33341). So the *target verify* is
  where decode-full capture lives, not the drafter.

---

## 4. The corrected target and boundary

**Boundary (unchanged in spirit, corrected in detail):**

- Capture-compatible: persistent metadata buffers sized by
  `decode_cudagraph_max_bs`; `copy_`/`fill_`/`index_select`/`index_copy_` and
  kernel launches over stable addresses; fixed F-spine chain shape; packed capture
  sizes; capture sizes divisible by `1 + num_spec_tokens` and ≤
  `num_mamba_cache_blocks`.
- Deliberately *outside* one monolithic graph: prefill/mixed GDN/tree (piecewise),
  CPU/debug telemetry (must be capture-guarded), branched-tree topology (research
  only).
- **The real target:** the **uniform-decode F-spine bucket** runs as a captured
  FULL graph that *includes* the GDN core — i.e. we earn the right to **drop the
  `cudagraph_unsafe` tag** for that bucket, instead of keeping the GDN core eager
  on decode.

Success ≠ "monolithic FULL." Success = measured `FULL` on the decode bucket +
`PIECEWISE` on mixed, with zero capture-time cache misses and an unchanged
acceptance distribution.

---

## 5. Mitigation A — eliminate cache-miss-during-capture (the main risk)

The hazard, confirmed in the patch surface: the GDN-core conv path
(`relaunch_qwen36_round.py`, the `LUMO_FA_UNIQUE_NODES` conv update block, ~lines
3838–3897) still **builds tensors on cache miss inside the forward** via
`_lumo_fa_conv_depth_clip_cache` (`torch.tensor(...)` / `torch.arange(...)` at
~3880–3890), has a `.detach().cpu().tolist()` parent-traversal fallback when
`fa_tree_depth_rows is None` (~3846), and iterates a **variable-length**
`zip(_depth_rows, _depth_row_tensors, _depth_query_start_tensors)` loop (~3893).
`fb6c99fa` made the depth-row tensors cache-backed — necessary but not sufficient:
a miss *during active capture* still allocates and still fails closed only if we
make it.

### A1 — hit-only + fail-closed cache guard (apply at every captured-path cache)

```python
_capturing = torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
if _capturing and key not in cache:
    raise RuntimeError(
        f"LUMO_FA capture cache miss: key={key}; "
        "run prime_fa_tree_metadata_cache() during warmup before capture"
    )
if key not in cache:                       # eager path only
    cache[key] = build_tensors_for_key(...)
```

Apply to `_lumo_fa_tree_depth_cache` **and** `_lumo_fa_conv_depth_clip_cache`.
The point is to convert "silently works after an eager miss, then corrupts/dies
under capture" into a loud, early, eager-time failure.

### A2 — explicit warmup priming contract

Add `prime_fa_tree_metadata_cache()` that runs during model warmup (before vLLM
starts stream capture) and pre-populates **every key that can appear during
capture**, enumerated over the cudagraph buckets:

- `num_spec_decodes`, `num_spec_decode_tokens`, `actual_conv_rows`
- tree topology id (`spine_d3`, `spine_d4`, …)
- padded token count / batch size from vLLM's `BatchDescriptor`
- ⇒ build `fa_tree_parent_indices_tensor`, `fa_tree_depth_row_tensors`,
  `fa_tree_depth_query_start_tensors`, `_lumo_fa_tree_depth_cache`,
  `_lumo_fa_conv_depth_clip_cache`.

Because vLLM pads to discrete capture sizes and (with spec) those sizes are
multiples of `1 + num_spec_tokens`, the bucket set is finite and enumerable.
Prime exactly those, not the open-ended runtime space.

### A3 — count misses and gate on them

Maintain `self._lumo_fa_tree_depth_cache_misses += int(key not in cache)` (even
eager), surface it in the artifact, and **fail any "captureable" claim whose final
artifact reports nonzero post-warmup cache misses.**

### A4 — kill the host-sync fallbacks on the captured path

Remove/guard the `.detach().cpu().tolist()` parent traversal (~3846) so it can
*never* execute while capturing — the depth-row tensors must arrive pre-built from
metadata, not be re-derived from a host round-trip inside the forward. (The
telemetry guard already in the code at ~3795 — `None if … is_current_stream_capturing() else …cpu().tolist()`
— is the correct pattern; apply it consistently, and for the *compute* path prefer
"precompute in build(), assert-hit in forward" over "guard to None.")

### A5 — make the loop topology static (prerequisite for FULL)

The `zip(...)` over depth rows has tree-dependent length. Full capture needs a
**fixed** loop: iterate `range(max_depth)` and pass count `0` for unused rows
(masked, not `continue`-skipped). If empty-row dispatches are too costly, capture
**one graph per depth bucket** instead of varying the loop.

---

## 6. Mitigation B — make the decode F-spine step truly FULL-capturable (drop the unsafe tag safely)

This is the corrected Track C. The goal is **not** to raise
`AttentionCGSupport` above `UNIFORM_BATCH` — that is already sufficient for
decode-full capture (C1). The goal is to make our **custom GDN-core op clean
enough to stay inside the existing FULL_AND_PIECEWISE decode graph**, so we can
stop excluding it.

Preconditions (Mitigation A done): hit-only caches, primed warmup, static loop,
no host syncs on the captured path.

1. **Tensorize the accepted-row state-copy.** No `int(tensor.item())`, no
   `.cpu().tolist()`, no Python parent traversal over GPU tensors inside the
   captured forward. Accepted count is a device tensor (or a fixed scalar bucket
   value); state copy is a fixed-shape kernel over `[batch, max_nodes]`; rejected
   suffix rows are **masked**, not skipped by Python control flow. Reuse the
   `num_accepted_tokens - 1` convention (C4) — pass `num_accepted_tokens` as a
   persistent device buffer the way upstream `gdn_attn.py` already does for its
   `non_spec_state_indices_tensor` / `num_accepted_tokens` buffers.
2. **Move topology derivation out of the captured forward** into `build()` /
   `build_for_cudagraph_capture` (metadata construction runs before the captured
   region). The forward should *receive* `fa_tree_*` tensors, never derive them.
3. **Adopt the upstream static-index-buffer pattern.** Preallocate
   `*_state_indices_tensor` at `decode_cudagraph_max_bs`, `copy_` real indices in,
   `fill_` the padded tail with `NULL_BLOCK_ID`, slice to padded batch. Kernels
   skip null rows via the `HAS_NULL_BLOCK` sentinel. This is exactly how vLLM keeps
   the conv/ssm update graph-safe: a fixed-address, fixed-shape index table whose
   *contents* (not shape) encode the valid batch.
4. **One full-capture bucket per fixed topology**: `spine_d3` first, `spine_d4` a
   separate graph; branched trees stay piecewise (C6).
5. **Then, and only then, drop the tag for that bucket** behind a new flag (e.g.
   `LUMO_FA_GDN_FULL_CAPTURE_EXPERIMENT=1`) — *and first confirm
   `use_inductor_graph_partition`* (C3), since on the default path the tag is inert
   and step 5 reduces to "verify the decode graph already captures the core."

If after this the decode bucket still reports `PIECEWISE`, inspect the **minimum
backend support** across layers (a *non-GDN* attention layer below `UNIFORM_BATCH`
will hold the whole model down) rather than treating it as a GDN-core failure.

---

## 7. Track B alternative — fixed buffer tables instead of dynamic caches

The strongest version of Mitigation A removes runtime cache-building entirely for
the F-spine. Replace tuple/dict caches with module-owned buffers sized by the max
capture bucket:

```text
parent_indices_buf:     [max_nodes]
depth_row_indices_buf:  [max_depth, max_nodes]
depth_row_counts_buf:   [max_depth]
depth_query_start_buf:  [max_depth, max_nodes + 1]
clip_row_indices_buf:   [max_actual_rows, max_depth, max_nodes]
clip_row_counts_buf:    [max_actual_rows, max_depth]
```

The captured path then consumes fixed-address buffers + count tensors; empty depth
rows become count `0`, not missing Python entries. This removes `torch.tensor`,
`torch.arange`, tuple reconstruction, cache-key hashing, and variable loop length
from the captured path in one move. Prefer this for `spine_d3` once A1–A3 prove
the bucket set is small and static.

---

## 8. What not to do

- **Don't rely on `capture_error_mode="relaxed"`** to get past capture errors — it
  masks unsafe side effects and makes correctness regressions hard to attribute.
  Fix the host work instead.
- **Don't reach for `cudaMallocAsync` graph memory nodes / manual graph
  construction** as the first lever. This stack is PyTorch/vLLM and already relies
  on stable replay addresses, fixed `BatchDescriptor`s, and cacheable metadata; the
  low-risk fix is preallocate-and-prime.
- **Don't design around CUDA conditional nodes** (§2) — not exposed through
  Inductor/vLLM yet.
- **Don't attribute the runtime mode to `LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE=1`
  without checking `use_inductor_graph_partition`** (C3).
- **Don't assume pulling `main` fixes GDN spec-decode state** — PR #40738 is
  unmerged (C4).

---

## 9. Runtime enforcement and artifact gates

Emit a `capture_contract` block in the debug JSONL and final summary. For a
genuine full-decode bucket claim:

```json
{
  "capture_contract": {
    "requested_mode": "full",
    "use_inductor_graph_partition": false,
    "expected_runtime_modes": ["CUDAGraphMode.FULL", "CUDAGraphMode.PIECEWISE"],
    "decode_bucket_runtime_mode": "CUDAGraphMode.FULL",
    "metadata_cache_misses_after_warmup": 0,
    "capture_cache_misses": 0,
    "gdn_core_cudagraph_unsafe": false,
    "fixed_buffer_metadata": true,
    "topology_bucket": "spine_d3",
    "capture_sizes_divisible_by_uniform_query_len": true,
    "capture_sizes_le_num_mamba_cache_blocks": true,
    "full_capture_claim": true
  }
}
```

For the current deliberate boundary (GDN core still tagged unsafe / not yet
clean), expect instead:

```json
{
  "expected_runtime_modes": ["CUDAGraphMode.FULL", "CUDAGraphMode.PIECEWISE"],
  "decode_bucket_runtime_mode": "CUDAGraphMode.PIECEWISE",
  "gdn_core_cudagraph_unsafe": true,
  "full_capture_claim": false
}
```

Fail the run if the **decode bucket** mode is worse than expected, if runtime-mode
telemetry is missing, or on any unplanned eager-fallback spike — *not* merely
because the overall mode is `FULL_AND_PIECEWISE` (which is expected).

---

## 10. Implementation sequence

1. Add cache-miss counters + fail-closed `is_current_stream_capturing()` guards
   around `_lumo_fa_tree_depth_cache` and `_lumo_fa_conv_depth_clip_cache`
   (Mitigation A1/A3).
2. Add `prime_fa_tree_metadata_cache()` warmup priming for all configured packed /
   `LUMO_CUDAGRAPH_CAPTURE_SIZES`, enumerated over buckets (A2).
3. Record `use_inductor_graph_partition` and per-bucket
   `cudagraph_runtime_mode` telemetry; rerun current F-spine and require: zero
   capture cache misses, expected `FULL_AND_PIECEWISE`, acceptance unchanged within
   noise.
4. Make the depth loop static (A5) and/or replace tuple caches with fixed buffer
   tables for `spine_d3` (§7).
5. Tensorize the accepted-row state-copy and move topology derivation to `build()`
   (Mitigation B1–B3).
6. Behind `LUMO_FA_GDN_FULL_CAPTURE_EXPERIMENT=1`, stop excluding the GDN core for
   `spine_d3`; rerun and require `decode_bucket_runtime_mode == FULL` with
   acceptance unchanged. If still downgraded, inspect minimum backend support
   (B last paragraph) before claiming success.
7. Expand to additional fixed buckets (`spine_d4`, …) only after the first bucket
   is invariant-clean.

---

## 11. Measurement requirements

Every captureability claim records: requested `LUMO_CUDAGRAPH_MODE`;
`use_inductor_graph_partition`; actual `cudagraph_runtime_summary.runtime_modes`
with `full_count` / `piecewise_count` / `eager_fallback_count`, **broken out by
decode vs mixed bucket**; acceptance distribution; invariant failures; whether
`LUMO_FA_CUDAGRAPH_UNSAFE_GDN_CORE=1` and `LUMO_FA_PACKED_CUDAGRAPH_SIZES=1` were
set; depth-row + clip-cache miss counters; and — if Nsight is used — whether the
sqlite contains CUDA kernel tables, not just GPU-metric tables. Also assert the
cross-workload caveat from the closeout: re-establish E3-FULL / spine / branched
on **one** frozen SWE-Verified subset before any speed ranking.

---

## 12. Recommendation (updated)

The v1 recommendation stands with one correction. *Yes*, keep the deliberate
boundary this iteration — but stop calling the GDN decode core "not full-capture
safe." It **is** captured on the uniform-decode path by upstream vLLM; what keeps
it out today is (a) our own `cudagraph_unsafe` exclusion and (b) host work
(`.cpu().tolist()`, cache-miss allocation, variable loop) in our custom tree-delta
forward. The next useful work, in order:

1. Make the scaffolding strictly **hit-only during capture** + fail-closed
   (Mitigation A) — this retires the *main risk* the closeout flagged.
2. Prove zero post-warmup cache misses and an unchanged acceptance distribution at
   the expected `FULL_AND_PIECEWISE`.
3. Tensorize the state-copy and move topology to `build()` so the GDN core becomes
   capture-clean (Mitigation B), then drop the unsafe tag for `spine_d3` behind a
   flag and confirm `decode_bucket = FULL`.
4. Treat the branched-tree path as research-only — its capture obstacles
   (`TreeAttentionBackend = NEVER`, host `.item()` in build, and the backend's
   removal on recent `main`) are structural in this vLLM line.

The bug to avoid is unchanged: claiming full capture while cache misses, CPU
telemetry, or dynamic state-repair leak into the captured path. The addition this
revision makes: don't under-sell what you already have — the decode step is one
clean refactor away from legitimately dropping the unsafe tag.

---

## Sources

PyTorch / capture rules
- PyTorch, *Accelerating PyTorch with CUDA Graphs* — https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
- PyTorch CUDA semantics (CUDA Graphs) — https://docs.pytorch.org/docs/2.12/notes/cuda.html
- `torch.cuda.is_current_stream_capturing` — https://docs.pytorch.org/docs/2.12/generated/torch.cuda.is_current_stream_capturing.html
- PyTorch op tags (`cudagraph_unsafe`) — https://raw.githubusercontent.com/pytorch/pytorch/main/aten/src/ATen/native/tags.yaml
- NVIDIA "CUDA Graph Best Practice for PyTorch" (sync-free) — https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/sync-free-code.html

NVIDIA CUDA
- CUDA C++ Programming Guide, CUDA Graphs (stream capture, prohibited ops, graph update, memory/conditional nodes) — https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- *Constructing CUDA Graphs with Dynamic Parameters* — https://developer.nvidia.com/blog/constructing-cuda-graphs-with-dynamic-parameters/
- *Dynamic Control Flow in CUDA Graphs with Conditional Nodes* — https://developer.nvidia.com/blog/dynamic-control-flow-in-cuda-graphs-with-conditional-nodes/

vLLM design / source (pin your base commit; `main` refs below)
- Design: CUDA Graphs — https://github.com/vllm-project/vllm/blob/main/docs/design/cuda_graphs.md
- Design: torch.compile integration (splitting_ops) — https://github.com/vllm-project/vllm/blob/main/docs/design/torch_compile.md
- `CompilationConfig` / `CUDAGraphMode` / `splitting_ops` / `_attention_ops` / `use_inductor_graph_partition` / `adjust_cudagraph_sizes_for_mamba_cache` — https://github.com/vllm-project/vllm/blob/main/vllm/config/compilation.py
- `AttentionCGSupport`, `AttentionMetadataBuilder` — https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backend.py
- `GDNAttentionMetadataBuilder` (UNIFORM_BATCH, decode-only capture, state-index buffers) — https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/gdn_attn.py
- `_check_and_update_cudagraph_mode` — https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py
- mamba spec-decode state copy (`num_accepted_tokens - 1`) — https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/mamba_utils.py
- `causal_conv1d` ops (NULL_BLOCK_ID, validate_data) — https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
- `direct_register_custom_op` (tags plumbing) — https://github.com/vllm-project/vllm/blob/main/vllm/utils/torch_utils.py

vLLM PRs / issues
- PR #20059 — full cudagraph orthogonal to compilation; `CUDAGraphMode` / `AttentionCGSupport` — https://github.com/vllm-project/vllm/pull/20059
- PR #21401 — full (decode-only) CUDA graph for Mamba — https://github.com/vllm-project/vllm/pull/21401
- PR #22594 — full CUDA graph default for hybrid models — https://github.com/vllm-project/vllm/pull/22594
- PR #23679 — spec-decode cudagraph uniform-size alignment — https://github.com/vllm-project/vllm/pull/23679
- PR #34571 — cap cudagraph sizes to Mamba cache blocks; "FULL decode runs GDN/Mamba kernels" — https://github.com/vllm-project/vllm/pull/34571
- Issue #35945 — `causal_conv1d` `num_cache_lines >= batch` assert under capture — https://github.com/vllm-project/vllm/issues/35945
- Issue #39273 / PR #40738 — GDN conv+SSM state corruption with spec decode (PR UNMERGED) — https://github.com/vllm-project/vllm/issues/39273 , https://github.com/vllm-project/vllm/pull/40738
- Issue #33341 — full CUDA graph for the drafter (open feature request) — https://github.com/vllm-project/vllm/issues/33341
- `TreeAttentionMetadataBuilder` (inherits `NEVER`) — https://github.com/vllm-project/vllm/blob/v0.11.1/vllm/v1/attention/backends/tree_attn.py
