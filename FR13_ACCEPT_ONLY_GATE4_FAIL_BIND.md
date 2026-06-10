# FR13 accept-only publish — gate-4 LIVE FAIL bind (2026-06-10)

**Verdict: PARKED on branch `fr13-accept-only-wip` (1a566d41). NOT merged — the live multi-step path is NOT bit-identical.** Main is unchanged; the serving default remains the proven all-rows publish.

## What passed (offline/static, artifacts in `output/fr13_accept_only_20260610T002243Z/`)
- **Gate 1** raw scan: `out_vs_native_max_abs = 0.0`, bit-exact at N_PAD=1 AND N_PAD=16 (n_actual=10).
- **Gate 2** publish equivalence (single forward, offline): accepted rows vs old publish `torch_equal, max_abs 0.0` per batch incl. the empty-accepted-path edge; rejected rows untouched; durable rows 30→5.
- **Gate 3** regular-decode: fork==pristine `torch_equal 0.0` (fp16 decode + prefill rows).

## What FAILED (gate 4 — live B=4 CUDA-captured, the regime that matters)
| metric | baseline (194841Z) | accept-only live | direction |
|---|---|---|---|
| accept/event (tree-arm-internal) | 2.024 | **1.521** | WORSE |
| real-loss outside self-noise | 0.4751 | **0.7315** (109/149) | WORSE |
| emitted bag-TV | 0.2335 | **0.5347** | WORSE |
| gate exit | — | 2 (FAIL) | — |

The accept/event collapse is **internal to the tree arm** (accepted/draft-events in its own probe; seed 1313, spp=4, 128-tok, same shape as baseline) — so this is NOT a comparison/pairing artifact. A bit-identical change cannot move these numbers at all; they moved drastically ⇒ the live path corrupts state.

## Prime suspect (for the fix workflow)
Deferred publish ordering across steps: gate-2 validated ONE forward offline, but live serving interleaves
`scan → (defer publish) → sample/commit → NEXT step h0 seed read + launch_tree_state_linear_remap → publish?`.
If the next step's h0 read or the remap consumes bank rows BEFORE the deferred accepted-row publish lands (or
the remap's src rows now hold stale data from a prior step because rejected-row slots are no longer refreshed),
states are corrupted mid-sequence — consistent with depth-collapse at pos 17 and the accept collapse. The
remap reads `pid_k < accepted_len` from `spec_state_indices` rows: under all-rows publish those are always
fresh; under accept-only they may be stale. B=2313-class co-residency makes the interleaving worse.

## Lesson (binds to the gate methodology)
**Offline single-forward bit-identical (gate 2) ≠ live multi-step bit-identical.** Gate-4-at-the-deployed-regime
is load-bearing and caught what gates 1-3 structurally could not. Any retry must add a LIVE single-step
ordering probe (publish-before-next-h0-read assert) before burning a full B=4 campaign.

## Status
- Patch + gate script preserved on `fr13-accept-only-wip` (pushed).
- Fix design will be informed by the in-flight `w78aq6xum` flow (full traffic accounting + every reader of
  the publish + the ordering contract).
- Codex_fr19 fully stood down (contract: workflows are the worker).

## ROOT CAUSE IDENTIFIED (w78aq6xum adversarial verify, 2026-06-10 — supersedes "prime suspect")
Two concrete mechanisms, both consistent with the live collapse:
1. **Stale row-0 on zero-accept events**: the next-event h0 read clamps `accepted_len-1` to column 0 (kernel :261-266), so bank row 0 is LIVE even when the accepted path is empty — the path-only publish never refreshes it ⇒ the next event seeds from a stale state. Gate-2's offline empty-path case checked "rejected rows untouched" but NOT the next event's h0 correctness.
2. **Non-graph-stable pending dict under FULL capture** (the gate-4 regime): `_FR10_PENDING_TREE_STATE_PUBLISH` pins the per-step `tree_state_all` allocations (48×201.3 MB ≈ 9.4 GiB, never popped) and per-batch-size graphs alias different allocations ⇒ the committer publishes whichever buffer was captured last = silent wrong-buffer publish.
Fix shape (for the retry, on the branch): ONE persistent preallocated staging bank per layer (the patch :184-197 pattern), pop-on-publish, and an explicit zero-accept row-0 publish path; then the full replay route deletes the scratch entirely.
