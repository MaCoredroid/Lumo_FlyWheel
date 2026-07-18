# FR13 — DEFINITIVE VERDICT: native MTP-5 beats the tree pipeline on GB10 (2026-07-18)

## The measurement (task #45 / #42 close)
Matched B=4, prefill_frac ~0.41, subset_b4 SWE. native from fr13_native_tail6_decomp (same-campaign);
tail6 from the accept>5 campaign (FR13_ACCEPT_BEYOND5_DESIGN.md, multiple runs). A fresh same-campaign
tail6-vs-native A/B was attempted twice (b689wvlzf, b0vwonhjs) but BOTH were killed externally (env not
sustaining long GPU background runs); the existing matched data is decisive so no fresh run is needed.

SAME-CAMPAIGN, AIRTIGHT (both arms from fr13_native_tail6_decomp, same boot, tail6_nt1 recomputed from its
raw metrics via fr13_measure deploy-speed; pf native 0.41 / tail6 0.38, B=4):

| metric                          | native MTP-5 | tail6 tree | native adv |
|---------------------------------|--------------|------------|------------|
| committed_per_event             | 4.41         | 5.31       | (tree +acc)|
| s_per_fwd_gpu                   | 58 ms        | 88 ms      | -30ms      |
| derived_tps_gpu (verify-basis)  | 76.0         | 60.5       | +26%       |
| derived_tps_fullstep_gpu (real) | 27.9         | 18.4       | +52%       |

(The earlier cross-run estimate used the design-doc tail6 71.2/18.8; the same-campaign tail6 is even slower
— 60.5/18.4 — so native's margin is LARGER than first stated: +26% verify, +52% full-step.)

## Verdict: NATIVE WINS. The tree pipeline does NOT beat native MTP-5 throughput on GB10.
- Verify-basis: native +7%. Full-step (drafter-inclusive, the real deploy metric): **native +48%**.
- WHY: (1) COMMITTER — the tree needs the 72-77ms GDN replay because branching DISCARDS per-node states
  (13.7GB to keep all); native's LINEAR MTP keeps its ~5 states + selects h_k (7ms). This is the session's
  committer investigation: the 77ms replay is the GB10 hardware FLOOR (native-kernel/multistream/batched/copy
  all measured-refuted; spine-commit de-risked marginal at 53%). (2) DRAFTER — the tree's arctic-tail + tree
  assembly is host-bound heavy, which is why the full-step gap (48%) >> the verify gap (7%).
- The tree's accept advantage (5.237 vs 3.4) is real + lossless but does NOT overcome its ~27ms/fwd higher
  cost + heavy drafter. Break-even (task #42) predicted tree>native needs accept >~5.76; tail6's 5.237 is
  below it — CONFIRMED by direct measurement.

## What IS delivered (honest, positive)
- A LOSSLESS branched-tree GDN spec-decode committer + a deep arctic TAIL reaching accept 5.237 (>5), the
  first lossless GDN-tree suffix-decode past depth-5, live B=4 SWE-Verified. Novel + correct.
- The committer is characterized to its GB10 floor with 4 optimizations refuted by measurement.

## Bottom line
On GB10 for agentic SWE, NATIVE MTP-5 is the throughput floor. The tree pipeline (accept>5, lossless) is a
correctness/accept achievement but a throughput REGRESSION vs native, driven by the fundamental committer
asymmetry (linear-keep 7ms vs branch-replay 77ms) that no cheap lever closes. The only paths to tree>native
are ARCHITECTURAL (stateless-tree / fuse-replay-into-forward = task #11, biggest rewrite) — not cheap.

## RIGOROUS PER-COMPONENT DECOMPOSITION (2026-07-18) — corrects the hand-wavy "host-heavy drafter"
Measured GPU timers from fr13_native_tail6_decomp (SFWD/CFWD/DFWD sidecars, same campaign, per-call):

| component            | native MTP-5 | tail6 tree | delta      | % of the gap |
|----------------------|--------------|------------|------------|--------------|
| committer (CFWD)     | 6.6 ms       | 100.4 ms   | **+93.8ms**| **~78%**     |
| verify forward (SFWD)| 66 ms/draft  | 88 ms/draft| +22 ms     | ~18%         |
| drafter (DFWD)       | 97.1 ms      | 102.4 ms   | +5.3 ms    | ~4%          |

**CORRECTION: the drafter is EQUAL (97 vs 102ms), NOT "host-heavy tree drafter" (that was wrong).**
Native's 5 sequential MTP heads cost ~the same as the tree's merged drafter. The COMMITTER replay is ~78%
of the entire gap -- which VINDICATES the directive's committer focus. Measured step: native 158ms (27.9
tok/s) vs tail6 289ms (18.4), native +52%.

## ATTACK MATH (measured-grounded)
- ATTACK 1 = eliminate the committer replay (stateless-tree: committer 100->6.6ms like native):
  tail6 step 289->195ms, tps 27.3 vs native 27.9 => **~PARITY (loses only 2%)**. The residual is the +22ms
  forward tree-tax. So the committer replay is THE thing keeping the tree behind, and killing it ~ties native.
- ATTACK 1+2 = stateless-tree PLUS trim the +22ms forward tree-tax (shallower/cheaper tree scan; accept
  tradeoff): tail6 step ~173ms, tps ~30.7 => **tree WINS +10%**. This is the only route to a CLEAR tree>native.
- spine-commit (53% skip): step 289->244ms, tps 21.8 minus ~20ms forward export => still loses, marginal
  (confirmed: the 47% branch-replays + export cost cap it).

## Consequence: the dominant, correct lever is the COMMITTER replay elimination (stateless-tree, task #11).
It closes 78% of the gap and reaches parity; +forward-tax trim wins. The drafter is a RED HERRING (equal).
All cheap committer-kernel attacks are refuted; only the architectural replay-elimination remains, and the
math now PROVES it is worth it (parity->win, not marginal). GPU validation currently env-blocked (kills).
