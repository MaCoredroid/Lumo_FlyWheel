# Q1.2b negative-loop closure review — 2026-09-28

**Bounded verdict: CLOSED.** The one remaining negative-power blocker from `q1-2b-post-consolidation-review.md` is repaired in the exact immutable snapshot below. Both A and B enforce the complete N1/N3/N4 verdict census, and N5's paired verdict remains diagnostic. No additional blocker was found in this narrow diff review. This is not a full-suite result, final freeze, or launch approval.

## Reviewed identities

Snapshot: `p0/monitor/review-response-20260927/negative-loop-fix-snapshot-20260928T001400Z/`.

| File | SHA-256 |
|---|---|
| `SNAPSHOT.json` | `fb30aa491b26baa6fdece27a98d8f9f179c5fda4f4eb72f39f9bb5b632586df0` |
| `tools/q1_2b_reduce_v2.py` | `f4286584f99d0157c7ca75ce0c67d27660ea5087b11a1438b24553ccebcbf67a` |
| `tools/tests/test_q1_2b_v2_pipeline.py` | `7c827fb91a1b9bdc9e0ae2b6c238d605f872230394a489508a7c7aa414c5d483` |
| `tools/q1_policy_evaluator_v2_1.py` | `2d25e4c9bb669a4c66a5a3caa5093b614e5aba136eb424efb0e410f0b0b61061` |

I independently verified **60/60 member hashes**. Compared against `repair-source-snapshot-20260928T000400Z`: exactly these three files changed, no members were added or removed. The other 57 files, including launcher, runner, policy and C2 preflight, are byte-identical. Previous findings/reproductions remain preserved in their earlier reports.

## Source closure

- Reducer **568–574** puts enforcement after the malformed-return block and explicitly loops `("A", "B")`. Expected N1/N4 counts equal the sum of instance counts; N3 equals two per fixture. Missing, short, extra, or any non-`FAIL` verdict is structural failure. Reducer **579–580** converts these findings into exit code 5.
- Reducer **575–576** records N5 paired verdict counts without making them a numerical-power gate. Its existing clean-C0 versus poisoned-C0 bitwise invariant is unchanged.
- Pipeline tests **237–249** parameterize N1/N3/N4, generate actual CPU-stub runner records in both processes, and call `reduce_run(..., "calibration", "all")`. They assert **rc=5**, a tag-specific **power-not-established** structural finding, **no malformed finding**, and all retained negative verdicts `PASS`. Thus these controls cannot succeed merely because raw records are absent or metrics fail recomputation.
- The test backend implements each no-op at the corruption operation: N1 uses the intended n14 path instead of the sibling path (60); N3 skips the swapped-state behavior (76); N4 replays the intended path instead of stale metadata (72). The runner still writes the ordinary retained tensors and computed arrays. This is appropriate for the missing-power regression, rather than rewriting declared metric arrays to manufacture a verdict.

## Independent local controls

Executed standard-library Python only, with `python3 -B` in the immutable snapshot. Parsed the exact reducer AST, extracted the final body beginning at line 566 into a temporary in-memory function, and supplied ordinary A/B aggregates `PASS`, empty existing findings, one fixture with two instances, and independently varied `negative_power`:

| Control | Result |
|---|---|
| Correct N1/N3/N4 counts, all `FAIL`, mixed N5 diagnostic `PASS`/`UNCOVERED` | rc=0 |
| One N1, N3, or N4 `PASS` in **A only**, B still correct | 3/3 rc=5, process/tag-specific structural finding |
| One N1, N3, or N4 `PASS` in **B only**, A still correct | 3/3 rc=5, process/tag-specific structural finding |
| Missing, short, extra, `UNCOVERED`, or `MALFORMED_EVIDENCE` N1 records, independently in each process | 10/10 rc=5 |

**17/17 exact-tail controls passed.** In particular the A-only controls close the prior risk that simply moving the old code would enforce only the last loop value B. The same isolated boundary returned rc=0 for nondetecting negatives in the preserved previous snapshot; its recorded counterexample is not overwritten.

The evaluator's only change is `_same_arr` plus its use for comparing already-bound reference arrays (new lines 46–57 and 367). The helper treats corresponding NaNs as the same recorded value while preserving finite-value, list-length, numeric-type, and signed-infinity distinctions. **8/8 isolated helper controls passed**: matching NaNs accepted; NaN versus finite, finite mismatch, opposite infinities, boolean versus number, wrong length and non-list refused; matching infinities accepted. This changes reference identity handling, not paired constants, eligibility thresholds, or the requirement that nonfinite references remain uncovered.

No Torch pipeline suite, remote command, container, GPU, model, or workload was executed by this reviewer. The worker's source-bound full CPU tests remain separate. The parent reports acceptance of the actual pinned-image C2 preflight for two fixtures; this narrow review neither reran it nor substitutes these AST controls for it. Final source/freeze reconciliation and authorization remain with the parent.
