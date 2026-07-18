# FR13 BEAT-NATIVE LADDER — the bar is a WIN, not parity (2026-07-18)

Framing (user directive): we control the WHOLE pipeline — drafter, verify kernel, committer, tree shape,
scheduler. Stock native MTP-5 is the same model served naively; the deliverable must BEAT it, not tie it.

Reference (B=4 speed-gate basis: per-forward GPU + decode-bracketed accept, qwen-code nudge-free):
- native MTP-5: fullstep 27.9 tok/s, tps_gpu 75.96, per_req 5.49, accept 3.415, step ~158ms
- tail6 today:  fullstep 18.5,      tps_gpu 56.9,  per_req ~5.06, accept 4.317, step ~289ms

## Rungs (compounding; every rung gated: same-session A/B, matched pf/eff-concurrency, lossless, OFF=byte-identical)

| # | rung | mechanism | expected fullstep | status |
|---|------|-----------|-------------------|--------|
| 0 | native-committer bake | committer replay 99→75ms (linear fused path) | ~20 (+8% vs today) | cng16 GATE IN FLIGHT (CFWD 75.1ms @1650 spans, 4/16 tasks) |
| 1 | **ASYNC-SCHEDULING bake** | `--async-scheduling` overlaps host schedule/prepare/sample N+1 with GPU forward N (hides ~250ms host stall) | +14% cross-run (as1 n=16: fullstep 21.1 vs b7 18.5; accept 4.953; per_req 5.03 measured) | **PROMISING, NOT YET VALIDATED** — task #40's A/B lost its baseline arm (reaped at arm-2 start, 0 fatal; the "5.9>5.49" in its commit was the HYPOTHESIS, not measured). Async arm itself clean 16/16 on GDN-hybrid. => the async pair in the combined campaign is REQUIRED (same-session confirm + lossless gate), then bake |
| 2 | PIGGYBACK | committer →~16ms (fold accepted-path advance into next forward's fused scan) | ~27.3 alone; **~31 with async (+11% vs native)** | bundle scouts running (woi1w5mxi); seams 0/3/1a landed |
| 3 | verify tree-tax trim | 88→~70ms/draft (tree-scan cost toward native's 66) | ~34 (+23%) | after rung 2 proves |
| 4 | drafter CUDA-graph capture (LEVER 5) | drafter ~100→~50ms (collapse sequential M=1 launches); hardest invariant (in_proj_ba M-dep) ALREADY SOLVED by FR13_SLOT_REORDER | ~40+ (+45%) | design exists (plan L5); de-risked, not started |
| 5 | accept levers (numerator ×) | tail6realloc (zero-node d6 realloc, prepped); tail6-pb hybrid (K=8 chain + rare-overflow replay, P(accept>7)≈0.27); MTP-d6 seam (cost-gated) | multiplies all above | queued behind 2 |
| 6 | (different axis) CONC oversubscription | effective concurrency 1.3→~4 at max_num_seqs=4 (HBM amortization) | 2–3× aggregate tasks/hour, NOT per-stream | deployment economics; one sweep, queued |

## Sequencing (bundled, per user directive)
- NOW (GPU busy with cng16): apply the piggyback BUNDLE (all seams, flag-gated) + wire cat9_pb + bake
  --async-scheduling into the serve-variant arms for the validation campaign.
- WHEN cng16 lands: ONE combined campaign = {cat9_pb-ON, cat9-OFF} (piggyback gates) + {tail6+async, tail6}
  (clean same-session async confirm) — 4 arms; short-subset first for engagement/lossless, 16-task for winners.
- THEN rungs 3→4→5 in order, each on the proven predecessor.

Honest guards: async's accept 4.953 is trajectory-bound (cross-run) — only the same-session delta counts;
native+async should also be measured eventually (the fair endgame bar); piggyback lossless is within-floor
(1.19e-7 state-carry) → trajectory gate, not byte gate.

## RESOLVE GATE (user-mandated, 2026-07-18): verdict pass-count ~8/16-ish; drifting below = issue signal
Measured (subset_b4_sixteen, WALL=1800): native 6/16 passed (2 wall-truncated, 5 tests_failed = completed
attempts); tail6 b7 3/16 (12 TRUNCATED mid-work); async as1 1/16 (9 truncated, 6 ended-text); cng16 interim
1/8 (5 truncated). TRACE CLASSIFICATION: zero give-up texts, zero garble — truncated traces end with clean
mid-investigation tool calls. => the tree's resolve deficit is WALL-CENSORING (34% slower => agent gets
fewer turns in the fixed 30min wall => truncated => empty_patch), NOT token-quality degradation.
IMPLICATIONS: (1) the speed deficit ALREADY costs ~2x resolves at deployment-faithful walls — speed converts
directly to resolutions; (2) resolve-recovery = the cleanest end-to-end ladder gate: as rungs land, tree
truncations must convert to attempts/passes toward native's band (~6-8/16 on this subset); failure to
recover once speed is fixed => THEN suspect behavioral/token issues; (3) per the no-wall-on-gates policy,
LOSSLESS gates treat wall-tripped as NA (right-censored) — but the fixed-wall resolve count is the honest
DEPLOYMENT metric and is now reported per arm alongside accept/CFWD/tps. WATCH: async's 6 ended-with-text
(vs tail6's 1) — final text without applied patch; classify during the async lossless gate.
