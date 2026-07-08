# FR13 — Tree near-neighbor GARBLE: reproduction gate + fix ladder

## The pathology (wf_2629d92b, 2026-07-08, supervisor-confirmed)
Tree spec-decode commits **plausible-but-wrong NEIGHBOR tokens** inside otherwise-correct code:
`wcs_dict`↔`wcs_header`, `astrop`↔`astropy`, `readFile`↔`read_file`, `result_slice`↔`result_sliced`, `20`↔`ny` → NameError / "tool not found" → typo-fix **loop** → budget burn → empty patch. Reproducible in BOTH tree arms (cat6, cat8), ABSENT in native. Comprehension intact; **emission** degrades.
**Root:** the tree's VERIFY forward is numerically drifted (tree-attention + GDN co-residency) → it commits a near-neighbor the TRUE (no-spec) target wouldn't. Confirmed size-robust (cat8 ⊃ cat6, both do it).

## Why the OLD gates missed it
Per-token argmax/TV "within-floor" gates AVERAGE over the whole stream — a ~within-floor flip on a *prose* token is harmless, but the SAME flip on a *code identifier* is fatal (NameError). Scalar metrics are blind to this ([[reference_scalar_metric_per_token_blindspot]]). **The gate must score CODE-CORRECTNESS, not per-token TV**, and ideally weight/restrict to identifier tokens.

## Reproduction GATE — ladder (cheap → expensive)
Every rung compares **3 arms on identical prompts+seeds**: `no-spec` (ground truth) / `native MTP-5` / `tree`. **Tree FAILS iff its code-error rate > native's** (both should ≈ no-spec). This isolates the tree from temp-0.6 noise (no-spec is the truth; native is the non-tree control).

| gate | what | metric | cost | role |
|---|---|---|---|---|
| **G3 replay** | teacher-force the EXACT context where the tree emitted a garble token (extracted from committed traces) | does tree commit the wrong neighbor where no-spec commits the right one? at the exact positions | ~min, GPU-solo | **confirm reproduction, deterministic** |
| **G1 identifier-probe** | synthetic code prompts that DEFINE distinctive identifiers + require heavy reuse (mimic SWE density); N=100–200/arm, temp 0.6 | **AST undefined-name rate** + exec NameError rate | ~min–1h | **fix-iteration gate (high-n, controlled, no agent loop / no SWE eval)** |
| **G2 code-bench** | HumanEval / MBPP subset, single-turn, N samples/arm | pass@1 + identifier-error rate in failures | ~1h | real-code confirmation (deterministic eval, high-n) |
| **G4 live SWE** | the full agentic gate (what we have, n=4) | resolves / semantic give-ups | hours–days | binding confirmation only |

**Recommend:** G3 (confirm) → **G1 is the fix-iteration gate** → G2/G4 confirm. All GPU-solo. Note the give-up "count" confound: exclude engine stream-hangs (`No stream activity 600000ms after 0 chunks`) — those are a separate tree failure mode, not garble.

## FIX ladder
Root = verify-forward numerical drift tilts the committed token to a near-neighbor. Fixes in order of cost:
1. **Commit-confidence gate (cheap dial, NO kernel change):** commit a spec token only where the target verify is confident (top ≫ 2nd logit); at low-confidence / near-neighbor positions, don't commit the spec token → fall to a clean sample. Trades accept (speed) for correctness — a direct speed↔quality knob, first thing to test on G1. Caveat: fixes marginal flips, not a fully drifted top-token.
2. **Targeted fp32 at drift-critical accumulations** (GDN scan, conv1d, tree-attn softmax/rescale) — cut the ~1-ULP/layer compounding ([[reference_diffuse_gdn_accumulation_explained]]) → verify logits closer to truth. **Primary kernel lever.**
3. **Batch-invariant tree-attention** (`FR13_BI_TREE_ATTN`, partially built) — kill the branches-perturb-spine co-residency batch-variance → spine logits independent of co-resident branches ([[project_fr13_22flip_carrier_l0gdn]]).
4. **M-invariant spine** — spine logits independent of tree width M.
5. **Correctness backstop (expensive):** re-verify the committed SPINE tokens against a clean non-co-resident forward; roll back mismatches. Catches all drift, costs an extra forward.

Iterate each against **G1** (garble rate) — keep what cuts garble at acceptable accept-cost. These are the queued amplification-reduction levers ([[project_fr13_amplification_levers_queued]]), now with a **cheap high-n gate** that actually sees the pathology.

## Sequencing
GPU busy (native+nocache → serialization shot). Build the G1/G3 harness now (CPU); run when GPU frees. A targeted kernel-read (where exactly the drift enters the verify forward) picks fp32 site #2 precisely.
