# FR13 — Robust B=4 drift-tracking (byte-exact impossible → statistical-within-floor)
Salvaged from workflow wet1mdi93 (3 readers completed + cross-consistent; the synth/verify agent died on a transient socket error, not a content gap). Byte-exact is impossible at B=4 — native itself is non-deterministic — so "lossless" = **the tree's drift is within native's own self-noise, measured robustly**.

## Why the CURRENT bar is fragile (reader A — confirmed file:line)
- `BAG_TV_FLOOR = 0.0593` (`scripts/fr13_corruption_gate.py:123`) is **ONE** native-a vs native-b draw (`native_e5_self_compare.json`, seed 1313). One realization of a random variable used as a hard threshold.
- The per-position self-noise mask `_self_noise_mask()` (`fr13_corruption_gate.py:170-198`) is built from exactly **TWO** native arms — the user's "137/256 single seed-pair mask". One pair UNDER-estimates the true mask → the tree gets charged for positions that are actually native run-noise.
- Asymmetry: the **superset/accept-event side is ALREADY distribution-robust** — `evaluate_strict_win_gate` uses a 10 000-sample paired-bootstrap 95% CI lower-bound>0 (`fr10_superset_gate.py:312-381`). Only the **lossless/bag-TV side** is a single-draw scalar needing the N-run upgrade.

## The three-piece robust design (readers A, C, D — all reuse existing infra)

### (1) Robust self-noise FLOOR — multi-sample, not single-draw (reader A)
- Run native **N=6-8 times at B=4** varying only the seed → C(N,2) native-vs-native draws.
- Floor = **upper 95th-percentile (or bootstrap upper-CI)** of those draws, NOT the mean/single-draw.
- Tree PASSES iff: `tree-vs-native bag-TV ≤ floor_p95` AND `tree real-loss-rate ≤ native-self rate + tol` AND `tree-vs-native chi-square ≤ native-self chi-square upper-quantile`.
- Tools already present: `chi_square_2sample` (`fr10_equivalence_gate.py:268-290`), `total_variation` (:232), `sampling_distribution_distance`+`summarize` (:308-362), CLI `scripts/fr10_sampling_noise_floor.py`. Design intent is literally documented at `fr10_equivalence_gate.py:7-12` ("floor-relative statistical tests: cross-regime distances vs same-regime distances at matched sample count and batch shape").

### (2) Verify-drift ISOLATION — forced-decode logit-distribution (reader D)
- Feed the **SAME fixed token prefix** to BOTH arms; compare the next-token **LOGIT VECTOR** per position — NOT the sampled/served token. Removes the RNG/path divergence that makes native ~53% non-det, leaving (mostly) the kernel/verify numerical drift.
- Metrics: `KL(native‖tree)` + symmetric TV over softmax(logits) (the within-floor continuous quantity that predicts temp>0 acceptance) + per-depth argmax-match (the lossless gate) + max_abs logit drift.
- **~90% already built:** `scripts/measure_spec_teacher_forced.py` forces a fixed reference stream into both arms (temp0/top_p1) with a paired `compare`; logit capture exists at `FR13_FINAL_LOGIT_CAPTURE` (`fr10_phase4_patch_vllm_tree_gdn.py:7071-7152`, torch.save full fp32 logit vector); reducer `fr13_argmax_lcp_localize.py` already loads both arms' `*_final_logits.call*.pt` + owns the row→depth map `TREE_SPINE_PATH=[0,1,3,5,7]`. **Only extension:** add KL/TV on softmax(logits) per row. The input is hard-pinned (`_prompt_identity()` RAISES on prompt mismatch — the bug that burned 3-4 boots).
- **CRITICAL HONEST CAVEAT (reader D, this IS the adversarial-verify answer):** forced-decode does **NOT fully isolate** verify-drift at B=4 — the co-residency non-det is in the **reduction ORDER** (batch-composition-dependent, non-associative fp), so it persists *even with identical operands*. `VLLM_BATCH_INVARIANT=1` is BLOCKED for the tree arm (requires FLASH_ATTN/TRITON_ATTN, not TREE_ATTN; on native it cut bag-TV 0.152→0.086 but did NOT drop the raw positional mask 137→139). **Therefore: forced-decode reduces the path-divergence component, and the residual irreducible co-residency non-det is absorbed into the multi-sample p95 floor (1).** That combination is what makes the metric trackable.

### (3) The reusable committed TRACKER (reader C) — `scripts/fr13_drift_tracker.py`
- A **thin orchestrator over `fr13_corruption_gate.run_gate()`** (the per-seed kernel — keep it importable, do NOT re-implement its metrics, so the two never disagree). The corruption gate is single-run boolean → inherits B=4 non-det → not trackable; the tracker adds the averaging + scalar layer.
- Loops over **K seeds**, emits ONE comparable scalar **`D = excess drift over native floor`** (weighted: real_loss 0.40, bag_tv 0.25, accept 0.25, verify_def 0.10 → served reality carries 0.90; `D=0` ⟺ indistinguishable from the native-vs-native floor) + a stable sub-scalar vector.
- Two channels: **A (verify-drift)** = forced-decode per-event acceptance deficit (`_superset_from_traces`, gate:284); **B (served-token drift)** = self-noise-corrected `real_loss_rate` (`_argmax_match`, :201) + bag-TV excess.
- Appends a **per-commit row bindable to `FR13_LADDER_LOG.md`** via `--ladder-append`.
- Composes: `total_variation`, `evaluate_superset_hard_gate`+`load_tree_accept_trace`, `fr13_argmax_lcp_localize._compute_prompt_identity`, `_self_noise_mask`. Input layout = exactly what `fr13_e2e_measure.capture_arms` (:135) writes.

## Implementation cost
Small — both halves are ~90% present. (2) = add KL/TV to `fr13_argmax_lcp_localize`; (1)+(3) = a K-seed wrapper over `run_gate` + p95 floor. **GPU cost = the N=6-8 native + K tree runs** (the real expense; CPU reduction is free). This is the guardrail the user asked for: a stable, trackable `D` that a regression can't hide behind native noise.
