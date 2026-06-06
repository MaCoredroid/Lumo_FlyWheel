# FR12 RED-TEAM — argmax-lag hypothesis [REFUTED 2026-06-06]

> **RESOLUTION (2026-06-06): the one-depth LAG was a measurement artifact, NOT a real structural bug.** Codex's `fr12_compare_argmax_lag.py` re-ran on the same splice-ON captures with the **parent-traced spine rows `[0,1,2,4,6]`** (passing both methodology gates below: correct `tree_rows`, `draft_match=True`). Result: **call2 tree==native at all 5 depths; call3 matches at d1/d2/d4 with genuine mismatches at d0 and d3; `lag_match_depths=[]`, `lag_persists=False`.** The earlier "tree_argmax[d]==native_argmax[d−1]" came from the OLD table reading the **contiguous rows `[0,1,2,3,4]`** — i.e. **branch row 3** (a child of the depth-2 spine node, which predicts ≈native's depth-2 distribution) read as "depth 3." With correct spine rows there is **no lag**. The deficit is **diffuse per-depth/per-event drift from accumulated 64-layer numeric mismatch** (FR11 confirmed), NOT a fixable structural seam. → Proceed with the numerics-alignment program (conv-L0 done; target the dominant cross-layer propagator next), gating on splice-OFF per-depth final argmax. The methodology gates below did their job — they killed a phantom.

---

## [ORIGINAL HYPOTHESIS — kept for the record, now refuted] the argmax "flip" is a one-depth LAG (structural), not residual rounding

**Claude red-team, 2026-06-06.** Re: `FR12_PARITY_RESULTS.md` §"Token-Level Argmax Gate With Conv Splice" (L387-445).

## The observation
That argmax gate ran with **the conv splice ON** (`FR12_TREE_CONV_NATIVE_SPINE=1`). Its per-depth argmax tables show a systematic pattern, not drift:

Call 2 (spine rows `[0,1,2,4,6]`, native rows `[0,1,2,3,4]`):
- d0 tree 248068 == native 248068 ✓
- d1 tree 12305 == native 12305 ✓
- d2 tree **12305** vs native 198   → tree[2] == native[1]
- d3 tree **198**   vs native 1005  → tree[3] == native[2]
- d4 tree **1005**  vs native 9637  → tree[4] == native[3]

Call 3:
- d1 tree 248069 == native 248069 ✓
- d2 tree **248069** vs native 271   → tree[2] == native[1]
- d3 tree **271**    vs native 71093 → tree[3] == native[2]

**tree_argmax[d] == native_argmax[d−1] for d ≥ 2, in BOTH calls.** Rounding drift cannot reproduce "exactly native's previous-depth distribution." This is a **one-depth lag**, and it begins **exactly at the first branch-row gap** (spine rows are contiguous 0,1,2 then jump to 4 — depth 2→3 is the first place the tree row index stops equaling the depth).

## Why this matters
1. The verdict "remove the remaining post-core/logit residual that changes argmax" assumes the flips are **numeric residual**. They are not — at least depths 2–4 are a structural lag. Chasing more rounding alignment will not fix a row/position misalignment.
2. The run had the **splice ON**. The splice `index_select`s path0 rows, calls native conv, and `index_copy_`s back. If its path0 row identification assumes contiguous rows `[0,1,2,3,4]` but the real spine rows are `[0,1,2,4,6]`, the splice itself can shift downstream state by one at the first gap. **So the lag may be a splice artifact, not a property of our real kernel.**

## Required next measurement (before any more rounding work)
Re-run the per-depth argmax gate with the **REAL kernel**: splice **OFF** (`FR12_NATIVE_SPINE_ORACLE=0`), `FR12_TREE_CONV_NATIVE_BF16_TAPS=1` ON, same matched event, per-depth argmax tree vs native. Two outcomes:
- **Lag vanishes** → it was a splice `index_copy` artifact; trust the splice-OFF argmax as the real gate; conv bf16-taps may already lift L0; continue numerics alignment on the next real seam.
- **Lag persists** → it is a real tree-row→depth alignment bug (spine logit capture row mapping, or position-ids / causal mask at branch boundaries). Fix THAT before any rounding — it is a far bigger lossless lever than 1-ULP residuals, and it would explain the whole accept deficit better than diffuse drift.

Either way: **the honest gate is splice-OFF per-depth argmax on the same event.** Do not conclude "argmax still flips, keep aligning rounding" from a splice-ON table.

## Theory backs this read (online research, 2026-06-06, primary-source cited)
The branch-losslessness research independently predicts this exact signature. SpecInfer (arXiv:2305.09781 Def 4.1) + STree (arXiv:2505.14969 §3 Eq.4-6): a node's verify output equals the target run on its path-to-root **only if** the ancestor mask / state-accumulation folds in **exactly that node's ancestors and no others**. The named failure mode (research caveat 2): *"any cross-branch sharing below the fork (state from a sibling/non-ancestor branch bleeding in) violates Def 4.1 and silently corrupts the oracle — detectable as a per-node argmax/logit mismatch against the path-rerun, NOT a fundamental limit."* A **one-depth lag that begins at the first branch-row gap** is precisely a path/ancestor-set construction error (the spine node at depth d accumulating the wrong ancestor count), i.e. a mask/row-mapping/position bug — **categorically different from 1-ULP rounding residual.** Rounding cannot yield "exactly native's previous-depth distribution."

Corollary: the correct gate is **per-depth argmax / distributional equivalence vs the recurrent path-rerun** (research confirms: gate on argmax, not bit-exact max_abs). And STree's diagonal-A shortcut does NOT apply to our non-diagonal `(I−βkkᵀ)` term — so there is no shared-accumulator excuse; each spine/branch node needs its correct ancestor-ordered operator. If the lag is real (splice-OFF), it is the single biggest lossless lever on the board.

## Splice-OFF argmax result (2026-06-06, run `fr12_spliceoff_argmax_20260606T072901Z`) — progress, NOT lossless yet
Our bf16-conv kernel, splice OFF, `tree_has_parent_idx=True`, spine rows `[0,1,2,4,6]`, `draft_match=True`: **per-depth argmax == native at ALL depths (call2 + call3), `argmax_mismatch_depths=[]`.** Codex also root-caused the old lag to a real bug (`fr10_layer_hidden_spine_compare.py` indexed `target_logits` by row-id instead of via `target_logits_indices`; fixed). Genuine — our kernel computes, no splice.
**Two reasons this is NOT a lossless verdict (delivered to codex):**
1. **Coverage:** ONE one-prompt event (10 spine positions). Old splice-ON data flipped argmax on *other* events → one green event ≠ lossless. Need the splice-OFF per-depth argmax gate over MANY events with a reported mismatch RATE.
2. **Distributional drift persists under matching argmax:** call3 d3 tree `prob_draft=0.7307` vs native `1.0`; call2 d3 `0.1438` vs `0.1200`. Argmax is the user's gate (right vs max_abs), but **temp=0.6 losslessness is distributional** — the rejection-sampler accept prob uses the full dist (SpecInfer Thm 4.2 MSS). Our kernel logits still differ from native's; the drift just didn't flip the top token here. Add a TV/KL or accept-rate metric; keep aligning the next propagator (gdn_scan FLA chunk-64/bf16, fp8 GEMM batch-invariance) to shrink the prob gap.

## Scan lever REJECTED; the dominant propagator is the fp8 GEMM (2026-06-06, boot-free probe `fr12_scan_bf16_boundary_probe_20260606T083638Z`)
Boot-free probe of our scan vs native: **baseline (no bf16 boundaries) `out_max_abs = 5.96e-08`** — our scan is ALREADY ~bit-exact to native (first mismatch tree −8.01e-05 vs native −8.06e-05). Forcing FLA bf16 boundaries → **`0.0078125` = ~4 orders WORSE.** So native does NOT bf16-round the scan there; our fp32 baseline already matches, and the "scan bf16 boundary" lever (commit `f1270afb`) is **counterproductive → default OFF, do not ship.** I interrupted the in-server run that was about to burn ~20 min GPU on it.
**Where the drift actually lives:** conv=0.0 (fixed), scan~5e-8 (already aligned) → the only notable L0 residual is **`o_proj_out = 1.2e-4`, an fp8 GEMM**, compounding over 64 layers into the TV p90=1.0 tail. The first candidate was the **fp8 per-token-group quant batch-invariance** issue (#42960): the tree's branched row-layout might make per-token-group fp8 scale round differently than native's linear chain. **Cost-gate result:** boot-free `scripts/fr12_fp8_gemm_batch_invariance_probe.py` on `output/fr12_fp8_gemm_batch_invariance_20260606T084909Z/fp8_gemm_batch_invariance_l0_o_proj.json` refutes that activation-quant sub-hypothesis for the captured L0 `o_proj`: tree full-batch vs row-only has **0 fp8 byte mismatches and 0.0 scale drift**, native full-batch vs row-only also **0 / 0.0**, and reversed row context is also **0 / 0.0**. Tree row-only vs native row-only has only **2 fp8 byte mismatches** from the already tiny `gate_out` input delta (`1.9e-6`) with **0.0 scale drift**. So the live activation quantizer is row-independent here. The `o_proj_out = 1.2e-4` residual remains real, but full fp8 GEMM replay is still unmeasured because the captures lack RowParallelLinear fp8 weights/block scales; `in_proj` is also unmeasured because the captures start after the input projections.

## ⚠️ The OUTCOME metric is still far from goal (2026-06-06, 40-event SWE4 run `fr12_spliceoff_many_argmax_20260606T075933Z`)
`accepted_per_draft_event = spec_acc/spec_drafts` (per speculation EVENT, comparable across tree-9-draft / native-5-draft — verified in `fr10_quick_decode_tps_probe.py:246`):
- **tree_mtp = 0.659**, **naive_mtp (native) = 1.739** → tree is **0.38× native**, the OPPOSITE of superset.
This is the honest aggregate over 40 SWE-Verified events (the earlier all-depths argmax match was ONE easy event). It **unifies with TV>0**: the verify-distribution drift makes the tree reject, at temp=0.6, spine tokens native accepts → accept/event collapses. **Lossless and the accept prize are the same problem** — drive TV→0 and accept/event should climb to native (spine parity), then branches add superset.
Caveats: max_tokens=16 (short gen → both numbers deflated vs canonical native ~3.076); metrics ON (accept counters valid, speed not); native logit `.pt` were missing (codex re-running) but `spec_acc/spec_drafts` come from /metrics so the accept ratio is valid.
**ATTRIBUTION codex must do (decompose before more numerics work):** is the 0.659 from (a) verify numerics drift, or (b) drafter topology (`project_fr10_drafter_topology_mismatch`: stock propose_tree gives parallel chains not a caterpillar)? **Measure SPINE-ONLY (path0) accept vs native's chain accept.** If spine-accept < native → verify drift (TV→0 fixes it). If spine-accept ≈ native but tree-total < native → it's the branches/drafter, and numerics alignment alone will NOT reach superset. This decomposition decides whether the current path can win. Gate the OUTCOME on accept/event ≥ native, splice-OFF; do NOT declare any progress-toward-goal on argmax/TV alone.

## Distributional gate added (2026-06-06, commit `bf786881`) — sound metric, confirms residual drift; needs a NOISE FLOOR
Codex built the per-depth distributional gate I asked for: `argmax_mismatch_rate`, **`tv`** (total variation), **`js_nats`** (Jensen-Shannon), `draft_prob_abs_delta`, each with max/mean/p50/p90/p99. Smoke (`argmax_distribution_smoke.json`, 6 events): depth 0 `argmax_mismatch_rate=0.0` but **tv max=0.099 / mean=0.033, js max=0.0052** — distributions differ ~3-10% where the argmax still matches; depths 1-2 tv=0.0. Quantitatively confirms: argmax-match ≠ temp-0.6-lossless; the diffuse drift is real and now measurable (the lever for numerics alignment).
**Open refinement (deliver next):** a tv of 0.033 has NO pass/fail meaning without a **noise floor**. Native decode is nondeterministic → native-vs-native verify-dist tv is nonzero. Lossless = tree-vs-native tv **within the native-vs-native (or our-kernel-vs-itself, same prompts) self-noise floor** (FR9-style; acceptance-length floor was ~0.0188, but per-position verify-dist tv needs its OWN measured floor). Establish that floor as the bar before declaring any tv "lossless." Then align the next propagator (gdn_scan FLA chunk-64/bf16, fp8 GEMM batch-invariance) and show tv shrinking toward the floor.

## Methodology gate for `fr12_compare_argmax_lag.py` (validate BEFORE trusting its verdict)
The lag verdict is only meaningful if the compared rows are the real spine path and the same event:
1. **`tree_rows` MUST equal the spine path `[0,1,2,4,6]`, NOT `[0,1,2,3,4]`.** The script selects tree rows by dedup-order of `target_logits_indices` (first-N-unique). The verify forward emits target logits for ALL tree nodes, so that dedup likely yields contiguous `[0,1,2,3,4]` — which picks **branch row 3** for depth 3 and can MANUFACTURE a false lag (or mask a real one). The spine must be traced from `tree_parent_indices` (path0), the way the original `[0,1,2,4,6]` was derived. **Check the `tree_rows` field in the output JSON.** If it is `[0,1,2,3,4]`, the row selection is wrong — fix it to the parent-traced spine before drawing any conclusion.
2. **`draft_match` MUST be true at the compared depths** — else tree and native are different decode events and the comparison is vacuous. Use the same matched event where spine draft tokens align (the original lag was seen on the `[71093,12305,198,727,9637]` / `[271,248069,271,71093,12305]` matched calls).

Only if both gates pass does `lag_match_depths == [2,3,4]` mean a real structural lag (→ fix mask/row-mapping/position). A wrong `tree_rows` makes the lag flag uninterpretable either way.

## Also pending
- conv bf16-taps = 0.0 was shown **offline (boot-free replay)** only. Confirm in-server: splice OFF, bf16-taps ON ⇒ `conv1d_out` max_abs == 0.0 live.
- Branch-path oracle (per `FR12_LOSSLESS_PLAN.md`): off-spine branch nodes have no native MTP-5 counterpart; validate each branch's logits against **no-MTP native run on that branch's linear ancestor-path** (depth-based RoPE). Add to the parity harness once the spine argmax gate is clean splice-OFF.
