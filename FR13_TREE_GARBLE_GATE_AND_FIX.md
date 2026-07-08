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

## FIX ladder — levers UN-RANKED pending a ground-up drift measurement (do NOT trust the old spine-drift ranking)
Root = verify-forward drift tilts the committed token to a near-neighbor (`flip = drift > margin`).

> ⚠️ **CONFOUND (2026-07-08, user caught).** The spine-drift study `wsvy4vn5k` that concluded *"amplification is SHARED with native, so the differential is only co-residency M-dependence"* used **`E5` = `chain5`, which is a FORKED TREE_ATTN linear spine — NOT native-MTP-decode** (`fr13_apc_3way_gate.sh:6`: "chain5 = the E5/spine baseline"; `chain5` KIND is `LAUNCHER=forked`). So "shared amplification" only shows it's shared between **two TREE_ATTN things** — it says **nothing** about whether the tree's amplification differs from **native-MTP-decode** (FLASH_ATTN, `naive_mtp`), which is the actual garble baseline. **The tree-vs-native-MTP per-layer drift comparison has NEVER been run against the real native-MTP baseline.** We now HAVE it: `nativemtp5_exseed` / `flash_ns5_nocache` (forked, FLASH_ATTN, naive_mtp) — so the garble differential is TREE_ATTN-tree vs FLASH_ATTN-native, both forked (NOT launcher-confounded). **Do NOT pre-rank the levers.**

**G0 — GROUND-UP DRIFT MEASUREMENT (run first, decides the ranking):** on one fixed code-heavy input, capture per-layer hidden states for `tree` (cat6/cat8, TREE_ATTN) vs `native-MTP-decode` (flash_ns5_nocache, FLASH_ATTN) — same tokens. Plot per-layer drift(tree→native). Then it tells us which lever is differential:
- if the tree's drift **slope/amplification** is higher than native-MTP's (not just offset) → **amplification-reduction IS differential** (fp32 hotspots / rms-clamp / re-anchor) — possibly PRIMARY, contra the confounded study.
- if the tree's drift is a **specific-layer / M-dependent jump** (e.g. L0 GDN birth-amplitude) over a native-matched slope → **M-invariance / batch-invariant tree-attn** is the differential lever.
- likely BOTH contribute; G0 apportions.

**Candidate levers (rank AFTER G0):**
- **Spine M-invariance** ([[project_fr13_22flip_carrier_l0gdn]]) — spine logits independent of co-resident branch count M.
- **Batch-invariant tree-attention** (`FR13_BI_TREE_ATTN`, partially built) — kill branches-perturb-spine batch-variance.
- **Amplification-reduction** ([[project_fr13_amplification_levers_queued]]) — targeted fp32 at hotspots (deep full-attn + GDN gate `1/rms`), rms-clamp, residual re-anchor. **NOTE its "secondary" status came from the confounded E5=chain5 study — treat as an open candidate until G0.**
- **Commit-confidence gate** (cheap dial): commit only where verify is confident; near-tie → clean sample. Speed↔quality knob; fixes marginal ties not drifted top-tokens.
- **Re-verify backstop** (expensive): re-run committed spine through a clean non-co-resident forward, roll back mismatches.

Iterate each against **G1** (garble rate) — keep what cuts garble at acceptable accept-cost. The win vs. the earlier queued work: we now have (a) a **cheap high-n gate that SEES the pathology** (code-correctness, where scalar per-token gates were blind) AND (b) the **true native-MTP baseline** the old drift studies never used.

## Sequencing
GPU busy (native+nocache → serialization shot). Build the G1/G3 harness now (CPU); run when GPU frees. A targeted kernel-read (where exactly the drift enters the verify forward) picks fp32 site #2 precisely.
