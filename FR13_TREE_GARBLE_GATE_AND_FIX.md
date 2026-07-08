# FR13 — Tree near-neighbor GARBLE: reproduction gate + fix ladder

## ✅ G1 BASELINE MEASURED (2026-07-08) — GARBLE CONFIRMED, tree ≫ native

Ran `scripts/fr13_garble_gate.py` (identifier-consistency probes + AST undefined-name scorer),
**N=15 × 3 prompts = 45 gens/arm**, identical prompts+seeds, `enable_thinking=false`, temp 0.6, NO cache.
Only the backend differs.

| arm | undefined-name rate | samples w/ undef | syntax-err | verdict |
|---|---|---|---|---|
| **native** (FLASH_ATTN, MTP-5) | **0.00%** | 0/45 | 0/45 | clean |
| **tree** (cat6root, TREE_ATTN) | **9.56%** | **32/45 (71%)** | 0/45 | **GARBLE** |

The tree systematically corrupts EVERY multi-word identifier into a near-neighbor — dropped underscore
(`input_wcsheader`←input_wcs_header, `expected_rowcount`←expected_row_count), truncation
(`verification`←verification_passed_flag, `applied_index`←applied_entry_index), abbreviation
(`crpix_ref_pixel`←crpix_reference_pixel, `crval_ref_value`←…_value), **wrong word**
(`expected_row_length`←…_count, `input_wcs_helper`←…_header), camelCase (`finalReconciled_rows`),
doubling (`applied_index_index`). Native emits all of them correctly (0 undefined). This is the
tree-verify-drift → near-neighbor commit → NameError → agentic give-up mechanism, isolated at emission.

**Gate build notes (this run):** (1) `enable_thinking=false` REQUIRED — with thinking on, the qwen3
parser + codex template think endlessly on synthetic single-turn prompts (3000 tok, content empty →
vacuous 0.00% for BOTH arms). (2) Prompts shortened to complete in-budget (the long "do real work"
versions truncated 55/90 at 700 tok → confounded). (3) Gate writes incrementally (`.partial`) — the
harness kills long background tasks; foreground N=15 (~8 min/arm) is the reliable cadence.
Artifacts: `output/fr13_garble_gate/{tree_n15,native_n15}.jsonl`.

**This is now the STAGE-E fix gate:** apply the fix (M-invariance / batch-invariant tree-attn), re-run
the tree arm, KEEP iff the undefined-name rate drops toward native's 0.00% at acceptable accept.

## ✅ STAGE-E LEVER 1 RESULT (2026-07-08) — pad-block M-invariance cuts garble 68%

The cron's named "primary" `FR13_BI_TREE_ATTN` (num_splits=1 under BI) is **inert for tree shapes**
(max_seqlen_q=tree_len>1 ⇒ num_splits stays 0) and needs full `VLLM_BATCH_INVARIANT`, which is
**counterproductive on GB10** (reduced override branch perturbs fp8/scan; prior cat9+BI=34). The
launcher's actual **targeted** M-invariance is the **pad-block in_proj GEMM** (`LUMO_FB_KERNEL_ROWS=1`,
#42960-authorized, lossless): pads the VERIFY-path (`num_spec_decodes>1`) projection to a fixed
`LUMO_FB_PROJ_PAD_ROWS`(=16) row group ⇒ GEMM tiling independent of co-resident branch count M.

| arm (N=15×3, thinking-off, no-cache) | undefined-name | samples w/ undef | accept |
|---|---|---|---|
| tree baseline | 9.56% | 32/45 | ~4.1 |
| **tree + `LUMO_FB_KERNEL_ROWS=1`** | **3.10%** | **15/45** | **4.55** |
| native | 0.00% | 0/45 | — |

**−68% garble, accept NOT tanked** (lossless-by-design). This **localizes the garble mechanism**: the
tree-verify wrong-accepts are driven by M-dependent (co-residency) batch-variance in the in_proj GEMM
tiling — making it M-invariant makes the verify reject near-neighbors it used to accept. Confirms the
"drafter-proposes / drifted-tree-verify-wrongly-accepts" mechanism (recurrent-oracle clear-margin flips
= these wrong-accepts).

**Caveats:** cross-boot A/B (baseline vs fix are different boots) — but −68% with the *same* residual
identifiers (input_wcsheader 10→7, applied_index 6→4) is far beyond autotune noise. **Not yet
same-boot-provable** (LUMO_FB is a container-start env, not per-request toggleable).

**RESIDUAL 3.10% ⇒ next levers** (the batch-variance also enters elsewhere in the verify forward):
o_proj / MLP GEMM pad-block, tree-attention reduction, GDN scan. Each: add M-invariant padding at the
site, re-gate. Target: 3.10% → native 0.00%. **DIRECT confirm option** (per user 2026-07-08): on a
garbling seed, trace `(drafter proposal, tree-verify accept/reject, recurrent-oracle argmax)` at the
garble position to show wrong-accept-of-a-draft vs bonus-resample.

---


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
