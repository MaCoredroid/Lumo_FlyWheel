# FR13_RESHAPE_WIDE sweep verdict — does wider-not-deeper help? (B=1 temp-0.6, SWE-Verified)

Sweep of the general width-N caterpillar drafter (top-5) vs depth-matched native MTP,
on subset_b4_four.json (4 astropy tasks), B=1 temp-0.6, deploy = codex+SWE-Verified.

## Headline: NO — wider-not-deeper does not help. Moderate width (cat6) wins; more width erodes toward a tie.

## CRITICAL methodology (user-caught): the aggregate accept/tps is TRAJECTORY-CONFOUNDED
Two of the four tasks (astropy-13236, 13398) are retry-heavy: the codex agent `gave_up`
(empty patch) and retried, and the retry is nondeterministic — so each arm's 13236/13398
are DIFFERENT codex turns (different tokens), NOT apples-to-apples. Token-weighting the
aggregate over those diverged trajectories deflated cat55221's accept (its per-task accept
is 3.70/3.97 on the clean tasks but 3.14/2.88 on the diverged ones) and produced a spurious
verdict. The apples-to-apples basis is the CLEAN tasks (12907, 13033) where all arms ran the
same lossless-determined trajectory. (`feedback_check_artifact_before_concluding`;
`feedback_depth_matched_accept_compare` "cross-trajectory accept is NOT apple-to-apple".)

Per-task accept (smoking gun — cat55221 accepts the MOST on clean tasks, confirming the
superset drafter is correct, NO bug):

| task | nativeE5 | cat6root | cat9 | cat55221 |
|---|---|---|---|---|
| 12907 (clean) | 2.806 | 3.672 | 3.227 | **3.700** |
| 13033 (clean) | 3.513 | 3.822 | 3.722 | **3.971** |
| 13236 (diverged) | 3.249 | 3.896 | 3.863 | 3.143 |
| 13398 (diverged) | 2.997 | 3.681 | 3.621 | 2.880 |

## Depth-5 (clean tasks 12907+13033) — derived_tps_gpu = (accept+1)/s_fwd_gpu (prefill-indep)
| arm | nodes | clean accept | s_fwd_gpu | tps_gpu | vs E5 |
|---|---|---|---|---|---|
| nativeE5 | 5 | 3.087 | 0.1370 | 29.84 | baseline |
| **cat6root** | 6 | 3.756 | 0.1377 | **34.53** | **+15.7%** |
| cat9 | 9 | 3.437 | 0.1440 | 30.82 | +3.3% |
| cat55221 (wide) | 15 | 3.850 | 0.1631 | 29.73 | −0.4% (tied) |

The curve PEAKS at cat6 (6 nodes) and erodes with more nodes: the wide tree's superset accept
is real (cat55221 has the highest accept, 3.85) but the verify tax grows faster (s_fwd_gpu
0.138→0.144→0.163 as nodes 6→9→15), so past ~6 nodes the accept gain no longer pays.
cat55221 (15-node wide) is NOT worse than native E5 — it TIES — but it loses to the moderate cat6.

## Depth-3 (clean tasks) — cat555 (15-node wide) vs E3 (3-node chain)
| arm | nodes | clean accept | s_fwd_gpu | tps_gpu | vs E3 |
|---|---|---|---|---|---|
| nativeE3 | 3 | 2.214 | 0.1356 | 23.70 | baseline |
| cat555 (wide) | 15 | 2.583 | 0.1563 | 22.92 | −3.3% |

cat555's superset accept (2.58 > E3 2.21) doesn't pay for its +15% heavier verify → slight loss.
(Aggregate read −6.9% was the same trajectory confound; clean basis −3.3%.)

## Verdict
- **Wider-not-deeper does NOT help.** At both depths the 15-node wide tree ties (d5) or slightly
  loses (d3) to depth-matched native, and loses outright to the moderate-width deployed cat6.
- The optimal on this workload is a MODERATE tree (cat6, 6 nodes, +15.7% vs E5) — enough width
  to lift accept, light enough verify to keep the per-forward cost low.
- The wide drafter is CORRECT (superset accept confirmed on clean tasks) — the sweep was a
  speed question, answered: width past ~6 nodes is a net negative.

## Caveats / next
- derived_tps_gpu is VERIFY-only (ignores the drafter forwards). The per-length TPS-attribution
  instrument (drafter+committer GPU timers, FR13_DFWD_GPU_TIMER built) will add that; the
  drafter cost grows with width too, so the full-step picture can only widen cat6's lead.
- LOSSLESS is the GATE, not yet passed: this whole comparison assumes the clean-task trajectories
  are lossless (same output across arms). The temp-0.6 lossless gate (US vs no-spec recurrent
  oracle, per-token argmax-vs-clean within native floor) must confirm cat555+cat55221 before any
  ship. Speed losing/tying does not change the lossless requirement.
