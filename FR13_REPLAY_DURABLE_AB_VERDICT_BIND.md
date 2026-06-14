# FR13 — Replay durable-state A/B VERDICT: SSM recurrent is NOT the back-loaded carrier

Date 2026-06-14. Workflow w2vaqcsmx (replay-durable-state-vs-native-MTP A/B), captured + verified. The PRIME
lead (replay durable-state = the back-loaded carrier of the 21 cat9 flips) is **NOT supported**. Pivot to conv
(ww22n39bi launched).

## The capture (infra SUCCESS — first GDN cross-event A/B to actually capture)
3264 records = 68 commit events × 48 GDN layers, ONE eager greedy decode (300 tokens). Infra worked where the
conv/scan SUBOP A/B failed 5×: sidecar env-to-worker ✓, eager ✓, `ref_kernel=fused_sigmoid_gating_delta_rule_update`,
all `ref_final_finite=true`, **zero arm-fail, NO device assert** (the linear B=1 varlen chain + no
ssm_state_indices dodged the reduced-row geometry that poisoned the CUDA context 5×). Observe-only confirmed
(served bank byte-untouched, `inplace_final_state=False`, cloned h0/rings, no reroute).

## VERDICT: not the carrier (converges with the monitor red-team)
- **`grows_across_events = FALSE`**: per-event L0 max_abs is FLAT/slightly-DECREASING (slope −0.011/event,
  corr −0.17; first-half mean 2.70 > second-half 2.27). The 21 flips ARE back-loaded (norm-mean 0.696); the
  replay divergence is not → it **cannot** be their carrier. Also: the A/B resets native to OUR h0 each event,
  so it measures per-event kernel divergence, not accumulation — and even that doesn't grow.
- **The dominant L0 `4.17` is a HARNESS ARTIFACT** (both the workflow Verdict and the monitor red-team, agreeing):
  - The Verdict explicitly says "align the AB harness ring-gather to the replay kernel's exact clamped-column
    convention to separate a harness artifact" before attributing the 4.17.
  - Monitor red-team (3264-record analysis): the large values are **bank-row + deepest-row LOCKED** —
    `h0_src_row ∈ {1–5}` (bank 0) → max 1.97–4.17, but `{11–15, 21–25}` (banks 1–2) → 0.11–0.65; and only
    `dst_row == M_chain−1` (deepest) spikes. A real kernel divergence cannot depend on which physical bank row
    the state occupies. Decisive consistency check: a real 4.17 in the committed durable state would garbage
    the next event, but serving is coherent at 21 small flips ⇒ artifact.
  - Per-layer medians are ~0.0; the honest SSM divergence is small (~0.05–0.2) = minor BV/warps codegen.
    `feedback_check_artifact_before_concluding`, bug-class #11 (measurement trap).
- This MATCHES the prior finding (project_fr13_conv_priorwindow_root / FR13_DRIFT_LOCALIZE_BIND): h0/SSM
  recurrent state was BYTE-EXACT, **recurrent-drift REFUTED**. SSM-faithful was always the prior.

## Two corrections to the workflow Verdict
1. Its `nextAction = "pivot to TREE_ATTN"` is **STALE** — TREE_ATTN is closed (FR13_FA2_FORK_IS_DECODE_KERNEL_CORRECTION:
   the FA2-fork is the decode kernel at its 0.0039 lossless floor; full-attn is NOT the carrier).
2. The L0 4.17 must NOT be banked as a real contributor (it's the ring-gather artifact). Lesson for the next
   A/B: record relative error + state-norm, and match the gather column convention exactly.

## Pivot → CONV cross-event path (ww22n39bi)
The evidence already named it: FR13_DRIFT_LOCALIZE_BIND pinned **"the conv prior-window READ as the carrier"**
(conv1d_out diverges 18.375 at num_accepted>1) with h0/SSM byte-exact; sglang #25587 corroborates **conv-state**
corruption after partial accept (~100-token divergence = the 21-flip back-loading). Caveat: conv fixes
(c0b53f5d, 06-10) landed AFTER the 18.375 capture, so the investigation re-verifies whether conv is STILL a
live cross-event carrier in the CURRENT locked build, and designs the conv-state A/B (reusing this proven
harness, + relative-error + gather-convention-match + the linear-geometry assert-prevention).

## Net
Replay (SSM recurrent durable-state) = ~faithful, NOT the carrier. The proven observe-only cross-event A/B
harness is reusable for conv. The 21-flip carrier search moves to the conv-state cross-event handoff.
