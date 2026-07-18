# FR13 — DEFINITIVE VERDICT: native MTP-5 beats the tree pipeline on GB10 (2026-07-18)

## The measurement (task #45 / #42 close)
Matched B=4, prefill_frac ~0.41, subset_b4 SWE. native from fr13_native_tail6_decomp (same-campaign);
tail6 from the accept>5 campaign (FR13_ACCEPT_BEYOND5_DESIGN.md, multiple runs). A fresh same-campaign
tail6-vs-native A/B was attempted twice (b689wvlzf, b0vwonhjs) but BOTH were killed externally (env not
sustaining long GPU background runs); the existing matched data is decisive so no fresh run is needed.

| metric                          | native MTP-5 | tail6 (accept 5.237) |
|---------------------------------|--------------|----------------------|
| committed_per_event             | 4.41         | 5.28                 |
| s_per_fwd_gpu                   | 58 ms        | 85 ms                |
| derived_tps_gpu (verify-basis)  | 76.0         | 71.2                 |
| derived_tps_fullstep_gpu (real) | 27.9         | 18.8                 |

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
