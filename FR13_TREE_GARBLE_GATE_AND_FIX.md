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

## FIX ladder — leading ranking from the (native-referenced) spine-drift study, G0 re-confirms
Root = verify-forward drift tilts the committed token to a near-neighbor (`flip = drift > margin`).

> **Naming note (see `FR13_E5_CONFOUND.md`):** "E5" is overloaded. The spine-drift study `wsvy4vn5k` that ranked these levers used `E5` = a **native-MTP-5 capture** (fr10, num_spec=5), NOT chain5 — so its conclusion IS native-referenced (ladder: native≈3 flips < chain5-tree-kernels≈5 < cat9-branches≈17). So the ranking below is the **leading hypothesis, not confounded**; the only caveat is that capture's exact config isn't confirmed and it predates our clean arms → **G0 re-confirms cheaply** against a baseline we control.

**G0 — ground-up drift re-confirm (cheap insurance, run first):** on one fixed code-heavy input, capture per-layer hidden states for `tree` (cat6/cat8, TREE_ATTN) vs `native-MTP-decode` (`flash_ns5_nocache`, FLASH_ATTN, known config) — same tokens. If the tree's excess is a **branch/M-dependent jump** over a native-matched amplification slope → confirms M-invariance is primary. If instead the tree's **slope** is higher → amplification-reduction is more differential than the study found. Either way it re-grounds the ranking on a config we fully control.

**PRIMARY (spine-drift ladder: branches/co-residency = the ~+12–14 excess):**
1. **Spine M-invariance** ([[project_fr13_22flip_carrier_l0gdn]]) — spine logits independent of co-resident branch count M.
2. **Batch-invariant tree-attention** (`FR13_BI_TREE_ATTN`, partially built) — kill branches-perturb-spine batch-variance.

**SECONDARY (shared-floor margin buffer; the tree kernels add only ~2 over native → amplification is largely shared):** [[project_fr13_amplification_levers_queued]]
3. **Targeted fp32** at hotspots (deep full-attn + GDN gate `1/rms`), **rms-clamp**, **residual re-anchor**.

**CHEAP DIAL / BACKSTOP:**
4. **Commit-confidence gate** — commit only where verify is confident; near-tie → clean sample (speed↔quality knob; fixes marginal ties, not drifted top-tokens).
5. **Re-verify backstop** (expensive) — re-run committed spine through a clean non-co-resident forward, roll back mismatches.

Iterate each against **G1** (garble rate). The win vs. the earlier queued work: we now have (a) a **cheap high-n gate that SEES the pathology** (code-correctness, where scalar per-token gates were blind) AND (b) a **known-config native-MTP baseline** for G0 to re-ground the ranking.

## Sequencing
GPU busy (native+nocache → serialization shot). Build the G1/G3 harness now (CPU); run when GPU frees. A targeted kernel-read (where exactly the drift enters the verify forward) picks fp32 site #2 precisely.
