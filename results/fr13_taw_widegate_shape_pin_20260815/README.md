# FR13 TAW sampler lever — widened byte gate + shape-pinned batched candidate

Phase 1 (build only). **Nothing is armed by this change.** The diagnostic and
production selectors remain default-off, the production arm still requires a
PASS bundle, and no PASS bundle exists at this HEAD. Arming is phase 2 and runs
only on the operator's explicit green light.

Branch: `codex/fr13-taw-batched-widegate-20260815`
Parent: `codex/fr13-single-launch-b4-scope-20260814` @ `d44f456ce`

## Evidence this change is built on

Device probe (`/home/mark/shared/tmp-scratch/taw_probe/`, NVIDIA GB10, CUDA
13.1, torch 2.10.0a0, V=248320):

| operator | behaviour across row counts |
| --- | --- |
| `torch.softmax` | bit-identical at every row count (1..68) |
| `torch.cumsum` | bit-identical for rows >= 2; **non-reproducible run-to-run at rows == 1** (20/20 reruns differ, up to 2-3 ULP, including under CUDA-graph replay) |
| `sum(dim=-1)` | run-to-run deterministic, but drifts up to **2 ULP across row counts** |

Shape-pin recipe (`E_shape_pin` in `taw_bit_probe2_results.json`):
`torch.cat([X[i:i+W].sum(dim=-1) for i in range(0, N, W)])` reproduces the
rows=W reference **bitwise** for W in {1,2,3,4}, where a single fused `[N, V]`
reduction does not (`naive_batched_bit_identical: false`, 2 ULP).

Consequence (`D_flip_decomposition`): a 2-ULP sum drift flips ~0.09%-0.29% of
inverse-CDF indices. A drifted normalization sum is a token flip waiting to
happen, and the pre-widening gate could not see it.

## What was built

### 1. Widened byte gate

The old gate compared int-views of **pre-normalization** softmax rows plus the
five committed integer products. It was blind to everything between them. The
gate now walks the whole chain, per level, on the rows the reference walk
actually reads (`leaf` / `has_kids` masked), in
`_fr13_fixed32_taw_gate_level`:

* pre-normalization self/target rows (as before)
* **post-normalization** self/target rows (after the `/sum`)
* the **residual** rows and the `overlaps` rows
* the **post-cumsum threshold inputs** — `total` and `thresh` from
  `_fr13_taw_inv_cdf_parts` — for the self, source, and residual draws
* the **accept decisions and emitted tokens**: `self_token`, `source`,
  `selected_token`, `rejected_token`, `accepted`, and the emitted token

The first four groups feed `native_ab_probability_mismatches`; the decision
group feeds `native_ab_accept_decision_mismatches`, which is now a real
comparison rather than a duplicate of the product count. Every verdict path
reads it and **fails**: the uncaptured root check, the graph-replay gate, and
the final candidate-acceptance census.

`_fr13_fixed32_taw_gate_is_complete` refuses a half-wired gate, so the gate is
all-or-nothing and the reference is always the served result.

### 2. Shape-pinned batched candidate

`_fr13_fixed32_taw_pinned_row_sum(rows, width=B)` implements the proven recipe.
All three full-vocab normalization sums now run at the served batch width:

| reference site | candidate site |
| --- | --- |
| self `self_prob / self_prob.sum(-1)` | `_fr13_fixed32_taw_pinned_normalized_caches` |
| target `target_prob / target_prob.sum(-1)` | `_fr13_fixed32_taw_pinned_normalized_caches` |
| residual `residual.sum(-1, keepdim=True)` | pinned `[B, V]` chunks in `_fr13_fixed32_taw_all_parent_decisions` |

The softmaxes stay batched (row-count invariant) and the cumsums stay batched
(bit-identical for rows >= 2). No pinning is needed for either.

### 3. B=1 refusal

At B=1 the reference operator itself is non-reproducible, so no byte-exact
batched candidate can exist. The walk stays unbatched at B=1, enforced in four
places:

* `_fr13_fixed32_taw_native_selector(batch_size=1)` returns `"reference"` for
  **both** the diagnostic and production arms
* `_fr13_fixed32_taw_all_parent_decisions` and
  `_fr13_fixed32_taw_execute_all_parent` raise
* `_fr13_fixed32_taw_native_live_entry` **declines to gate** B=1 — it returns
  the reference-passthrough sentinel (`None`), it does not raise. See the
  correction below; the original build raised here and that was a defect.
* `_fr13_fixed32_taw_native_live_pass_emit` refuses to record B=1

### 4. Required production batches: `(1, 4)` -> `(4,)`

`qualified_batches` should cover exactly the batches the candidate will serve.
B=1 is refused. B=2 and B=3 only occur as transient partial batches and fall
back to the reference route through the existing per-batch qualification check
(`batch_size not in bundle["qualified_batches"]` -> `"reference"`). The fixed32
B4 campaign is the only route that serves the candidate, so the narrower `(4,)`
is correct; a bundle only has to prove what it will actually be used for.

## Identity

Changing the candidate's math changes its identity. The pinned TAW source
contract digest moved, which invalidates every existing v7 PASS bundle by
construction — including
`results/fr13_fixed32_taw_source_v7_b1_b4_bound_20260805/`. That is the
intended fail-closed state: the new candidate has no live evidence and cannot
arm production until phase 2 produces it. Per repo precedent (commit
`87344abdc`) the schema string stays `fr13-fixed32-taw-all-parent-v7` and only
the digest is re-pinned; the digest is the identity.

Re-pinned digests chain kernel -> runtime overlay -> packed wrapper -> gate
runners, and every consumer was chased to a fixpoint.

## Deviations from the build spec

1. **Record- and bundle-level rejection of a B=1 qualification claim was not
   added.** It is redundant — the selector refuses to route B=1 to the
   candidate before any bundle content is consulted, and the emitter cannot
   write a B=1 record — and adding it would have broken the B1 credential
   evidence chain (`scripts/fr13_taw_b1_credential.py`) which reads historical
   B1 records. The merge path in the live-PASS emitter *does* reject a stale
   bundle that claims B1.
2. **`scripts/fr13_run_b4_tail23_all_parent_live_gate.sh` was updated** from
   `qualified_batches == [1,2,3,4]` / `required == [1,4]` to `[2,3,4]` / `[4]`.
   This is a direct consequence of the required-batches change; leaving it
   would have left phase 2 with a provably unsatisfiable gate. The runner is
   still default-off and still sets
   `FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0`.
3. `scripts/fr13_taw_b1_credential.py` and
   `tests/test_fr13_k64_physical32_fullstack_route.py` now read
   `_FR13_FIXED32_TAW_REQUIRED_PRODUCTION_BATCHES` from the module instead of
   hard-coding `[1, 4]`.

## Tests

`tests/test_fr13_fixed32_taw_widegate_shape_pin.py`. The centrepiece pair,
same fixture, same injected 2-ULP normalization-sum drift, both gate widths:

* `test_two_ulp_normalization_drift_passes_the_old_narrow_gate` — 0
  probability mismatches, 0 product mismatches. The old gate called this a
  PASS.
* `test_two_ulp_normalization_drift_fails_the_widened_gate` — 602 byte
  mismatches on the same run, while the products and accept decisions are
  still identical. The widened gate refuses the drift *before* it becomes a
  token flip.

GPU-dependent bitwise reduction claims are marked and skip without CUDA; the
shape semantics are asserted on CPU.

## Correction — the B=1 refusal was applied at the wrong layer (2026-08-15)

The first phase-2 live gate run
(`output/fr13_taw_widegate_b4_gate_20260815T162842Z`, superseded) killed the
engine 19 minutes in. Not a byte mismatch — the instrument.

`_fr13_fixed32_taw_native_live_entry` is called by the runtime overlay
(`_fr13_fixed32_observed_begin` -> `_fr13_fixed32_taw_full_graph_begin`) on
**every measured forward**, before any per-batch selector runs. The build had
it RAISE below the pinned minimum. So when `astropy__astropy-12907` finished at
16:45:38 and the campaign drained from four running requests to one, the next
engine step hit the entry hook at B=1 and took EngineCore down with
`FR13 fixed32 device row-map failed`. Draining to a partial batch is ordinary
scheduling, not a defect, and a diagnostic must never be able to kill the
engine it is only supposed to observe.

The fix is refuse-at-the-right-layer, matching how the FA2 dual gate treats its
untreated widths:

* `_fr13_fixed32_taw_native_live_entry` returns `None` — no gate, no capture,
  no record, the engine keeps serving the stock reference.
* `fr13_fixed32_taw_native_live_gate_begin` / `..._on_replay` and the overlay's
  `_fr13_fixed32_taw_full_graph_begin` / `..._on_replay` propagate a `no_gate`
  status instead of arming or reading counters.
* Every refusal that guards a real violation still RAISES: the all-parent
  decision and execute paths (the candidate must never SERVE below B=2), the
  pass-record emitter (must never RECORD below B=2), and the final acceptance
  census (a B=1 census means the caller lost track of what was gated).

Tests: `test_live_entry_declines_to_gate_batch_one_instead_of_raising` (entry
returns the sentinel, both hooks report `no_gate`, no live-pass file is
written), plus `test_pass_emitter_still_refuses_to_record_batch_one` and
`test_final_census_still_refuses_an_ungated_batch_one` holding the raises. The
superseded `test_live_gate_and_pass_emitter_refuse_batch_one` asserted the
defect and was replaced.

Changing the entry re-pinned the identity chain to a fixpoint: TAW source
contract `654503f8...` -> `c8e32edf...`, kernel file `f50dd60d...` ->
`6e1f09f5...`, packed CFWD overlay `8cfd0455...` -> `65be6890...` (regenerated,
byte-identical across two emissions).

What the wrecked run did prove: its live bundle reached `status=production_ready`
at 16:38 with `qualified_batches [2,3,4]` and **zero probability and zero product
mismatches on every widened surface for all three batches**. The shape-pinned
math held; only the untreated-width path was defective.
