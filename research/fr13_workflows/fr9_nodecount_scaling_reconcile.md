# Reconcile: "3->5 nodes ~free" vs FR9-era node/depth scaling

## The claim, restated precisely

"The verify forward is ~weight-load-bound; going 3->5 nodes is almost free (+1.7%), ~0.0015 s/node." The supporting numbers are `s_per_fwd_gpu`: E3 (num_spec=3, 4 verify rows) = 0.13473 vs E5 (num_spec=5, 6 verify rows) = 0.13696, a +1.66% delta; plus cat555 (15 spec, 16 rows) = 0.15630.

The metric is verified to be `d(fr13_decode_forward_gpu_seconds_total)/d(fr13_decode_forward_gpu_drafts_total)` — async-CUDA-event GPU time of **only** `self._model_forward` (the single TREE_ATTN/MTP-VERIFY pass over the spec-verify rows), per draft event. It **excludes** the drafter's spine forwards.

## What FR9/FR10 actually found (numbers + citations)

FR9/FR10's "depth costs more" was measured on the **full per-step / per-request basis**, not the verify alone:

- FR9 B=4 decode-seconds decomposition: the 2.336x total gap "factors EXACTLY into ... (a) MORE FORWARDS x (b) MORE TIME PER FORWARD = (spec_drafts 620/433 = 1.432x) x (decode_seconds/spec_drafts 0.4814/0.2950 = 1.632x)" — `research/fr13_workflows/why_slower_wacoxe6i2.raw.json` (gap_root_measured). Factor (a) MORE FORWARDS is the drafter/accept side; the verify metric only ever sees a slice of factor (b).
- The single largest per-forward tax FR10 ever localized was a **DRAFTER** cost: "+5.45 cuBLAS gemvx calls/draft × 15.05 ms = +81.9 ms/draft = ~87-93% of the measured per-forward gap" — the caterpillar drafter computing the full-vocab lm-head twice **per drafter step** (depth-scaling) — `research/fr13_workflows/speed_gpu_kernel_attrib_wf_c3b79cf7.raw.json`. This is entirely outside `s_per_fwd_gpu`.
- The verify side, independently, FR9/FR10 found weight-load-bound and row-nearly-flat: dense (non-MoE) model streams 26.9 GB weights **once per forward regardless of M**; CAT10 measured +2.9 ms/added verify row but that row's true byte cost is ~0.013 ms — "a ~220x gap, so even the per-row cost is overhead, not bytes" (`FR13_CAT10_BIND.md:84`; `speed_tax_tree_scaling_wrt469m5h.raw.json`). Deployed clean B=1 verify-only: native E5 (6 rows) 0.137, cat6root (7 rows) 0.138 ≈ ZERO tax, cat9 (10 rows) 0.144 ≈ +5% (`b1_depth5_speed_verdict.raw.json`).
- FR13 Stage-D states the rule explicitly: "DEPTH-bound, not node-bound: cat6 (6 nodes) and cat9 (9 nodes) have ~IDENTICAL wall/step ... because both are depth-5 -> both do 4 sequential draft forwards. **Node count only changes rows/forward (HBM-bound, ~free).**" — `research/fr13_workflows/stage_d_overhead_is_stock_depthbound.md:29-31`.

## The forward-cost model: what `s_per_fwd_gpu` measures vs what FR9 measured

Per decode step (native MTP-D), verified against live container source:

1. **DRAFTER (D sequential forwards):** `EagleProposer.propose` runs 1 initial forward (`eagle.py:576`) + a loop `for token_index in range(self.num_speculative_tokens - 1)` (`eagle.py:2136`), one `self.model(...)` each (`eagle.py:2222`); `parallel_drafting=False` for MTP (`eagle.py:92-94`) ⇒ **D drafter forwards**. These run separately/later under `record_function 'gpu_model_runner: draft'` (`gpu_model_runner.py:5170-5183`), **outside** the timed span. Correction to the original framing: `self.model` here is the **draft model** (MTP head / 1 nextn layer, `eagle.py:83-84`), so each is a small head read, **not** a full-27B weight-load.
2. **VERIFY (1 forward over 1+D rows):** `self._model_forward` called exactly once (`gpu_model_runner.py:5015`); the FR13 timer brackets exactly this (`_fr13_sfwd_begin :4990` / `_fr13_sfwd_end :5022`), pure-decode steps only.

So the depth term lives in the **drafter** (D forwards), not the verify. `s_per_fwd_gpu` = factor (b) for the verify pass only. The full-step basis FR9/FR10 used sees both: native E5 verify-only 0.137 vs full-step `request_decode_time/drafts` 0.2182 (`FR13_SPEED_HISTORY_RECONCILE.md:136,139`) — the ~0.081 s gap is the D drafter forwards + committer + idle.

## Does it contradict?

**NO — they measure different things, AND the specific +1.7% number is within noise.** Two distinct points:

1. **No contradiction in principle.** "Verify is weight-load-bound and row-flat" is exactly what FR9/FR10 found (26.9 GB read once; per-row tax is launch overhead, not bytes). "Depth costs more" is also what FR9/FR10 found — but on the **total-forwards** basis, because depth-5 adds ~2 extra **drafter** forwards (and historically a double-lm-head drafter cost) that `s_per_fwd_gpu` structurally cannot see. Both are true on different bases. The claim becomes *misleading* only if restated as "depth-5 total speed ≈ depth-3," which FR9 shows is false.

2. **The +1.7% itself is not a clean scaling signal.** The E3-vs-E5 delta (0.13473 vs 0.13696 = +1.66%) sits inside the documented GB10 measurement floor:
   - **Cross-boot:** E3 live vs E5 banked on a different boot; no cross-boot byte gate (`feedback_no_cross_boot_byte_gate`). Canonical re-boot vs banked floor is ±0.4–0.7% (native E5 −0.38%, cat9 +0.69%, B4 −0.42%; `build_speed_measure_infra_wycxas2x6.raw.json`). Same-boot, identical-6-row anchor cat6root 0.13798 vs native E5 0.13704 = +0.69% with **identical** row count — ~0.7% of inter-arm variance is not row-driven.
   - **Cross-trajectory:** E3 = 3 tasks PARTIAL vs E5 = 4. The missing astropy-13398 is the slowest-per-forward task in every banked arm (E5 0.13991 vs ~0.135x siblings); dropping it from E5's draft-weighted aggregate shifts it down ~0.99% (0.13696→0.13561), ~60% of the claimed delta (`b1_depth5_raw/nativeE5_b1_deploy_speed.json` per_task).
   - Combined floor ≈ ±1.0–1.5%, so +1.66% is ~1–1.7 noise floors. Also note E3 native MTP-3 s/fwd was flagged **UNMEASURED** and the protocol forbids judging a depth-3 arm before a controlled E3 capture (`speed_history_reconcile_wqunplqqm.raw.json`).

## The CORRECT statement to use going forward

The verify forward is weight-load-bound (26.9 GB read once per forward, M-independent), so adding verify **rows** is genuinely cheap — but the +1.7% E3→E5 number is within the GB10 cross-boot (±0.4–0.7%) + cross-trajectory (slow astropy-13398 missing from the 3-task E3) noise floor, NOT a measured per-row slope; and it says nothing about **depth-5-vs-depth-3 total step cost**, which is materially higher because depth-5 runs ~2 extra sequential drafter forwards that this verify-only metric excludes (FR9/FR10's "depth costs more" lives there, not in the verify).

## Was my number a hand-rolled artifact?

**Retracted as a quantitative model; the qualitative direction survives.** "+0.0015 s/node weight-load-bound" is a 2-point extrapolation off noisy cross-boot/cross-trajectory endpoints, and its slope is internally inconsistent: E3→E5 gives 0.00112 s/row but E3→cat555 gives 0.00180 s/row — a 60% swing; 0.0015 matches neither. Presenting it as a measured s/node figure is exactly the hand-rolled per-forward decomposition `feedback_dont_handroll_speed_defer_tuning` warns against (cf. the 1.412x / 1.29x retractions).

What **is** defensible: (a) the verify forward is weight-load-bound and adding rows is qualitatively near-flat — independently confirmed by FR9/FR10 (dense 26.9 GB read once; per-row tax is overhead not bytes); and (b) a small **real** per-row verify cost exists, proven not by E3/E5 but by cat555 = +14.1% over E5 across 12 extra rows (well above the ±1.5% floor) — but its magnitude cannot be pinned from the E3/E5 endpoints. Resolve the actual per-row slope only with a **same-boot, same-trajectory** E3-vs-E5 pair under the controlled protocol.