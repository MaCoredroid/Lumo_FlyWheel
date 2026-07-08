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
Root = verify-forward drift tilts the committed token to a near-neighbor (`flip = drift > margin`). **KEY (spine-drift study wsvy4vn5k):** the ~1.166×/layer residual amplification is **SHARED with native** — reducing it lowers the shared floor but does NOT close the tree>native gap. The tree's *differential* (garble-causing) drift = **co-residency M-dependence at the L0 GDN birth-amplitude** (spine logits depend on tree width / co-resident branch count M; native has M=1). So attack the DIFFERENTIAL source first:

**PRIMARY (removes the tree's extra drift — the actual gap):**
1. **Spine M-invariance** — make the committed-spine logits independent of M (co-resident branch count). The named carrier ([[project_fr13_22flip_carrier_l0gdn]]); primary lever from the spine-drift study.
2. **Batch-invariant tree-attention** (`FR13_BI_TREE_ATTN`, partially built) — kill the branches-perturb-spine batch-variance so the spine sees identical numerics regardless of co-resident branches.

**SECONDARY (margin buffer — lowers the SHARED floor, helps both arms stay under margin; the "amplification-reduction levers" [[project_fr13_amplification_levers_queued]]):**
3. **Targeted fp32 at amplification hotspots** — deep full-attn L35/47/51/62 + the **GDN gate `1/rms` blow-up** (targeted, NOT whole-model, default-OFF byte-identical).
4. **Clamp the gate rms denominator** + **periodic residual re-anchor** (cap the compounding past K layers).

**CHEAP DIAL / BACKSTOP:**
5. **Commit-confidence gate** — commit a spec token only where the verify is confident (top ≫ 2nd); at near-tie positions fall to a clean sample. Speed↔quality knob (lower accept). NOTE: doesn't fix a *drifted top-token*, only marginal ties.
6. **Re-verify backstop (expensive)** — re-run committed spine tokens through a clean non-co-resident forward; roll back mismatches.

Iterate each against **G1** (garble rate) — keep what cuts garble at acceptable accept-cost. The win vs. the earlier queued work: we now have a **cheap high-n gate that actually SEES the pathology** (code-correctness), where the old scalar per-token gates were blind — and the pathology (near-neighbor identifier flip) is the *concrete face* of `drift > margin`.

## Sequencing
GPU busy (native+nocache → serialization shot). Build the G1/G3 harness now (CPU); run when GPU frees. A targeted kernel-read (where exactly the drift enters the verify forward) picks fp32 site #2 precisely.
