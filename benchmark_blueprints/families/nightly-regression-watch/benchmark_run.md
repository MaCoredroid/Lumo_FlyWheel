# Benchmark Run: Nightly Regression Watch

## attempt_00 — baseline bundle-only calibration

The family started as a doc-only bundle. The prior child-agent run diagnosed the schema rollover, blocking contract, latest-of-day rule, and single-watch constraint, but it did not patch a concrete `ops_digest/` workspace or regenerate artifacts. That left the family in a valid low-20s design state but not Layer B ready.

## attempt_01 — family implementation and Layer B wiring

Shipped in this attempt:

- concrete five-variant `workspace_bundle/` with broken-but-runnable `ops_digest/` repos
- deterministic scorer at `verifiers/nightly-regression-watch/score_ranking.py`
- family declaration `family.yaml`
- verifier data with hidden tests, oracle repairs, manifests, and milestone scripts
- verification matrices for `v1-clean-baseline` and `v5-recovery-in-thread`

Baseline scores after regen:
- `v1-clean-baseline`: oracle `100/100`, empty `0/100`, shortcut `20/100`
- `v2-noisy-distractor`: oracle `100/100`, empty `0/100`, shortcut `20/100`
- `v3-dirty-state`: oracle `100/100`, empty `0/100`, shortcut `20/100`
- `v4-multi-corpus-objective`: oracle `100/100`, empty `0/100`, shortcut `20/100`
- `v5-recovery-in-thread`: oracle `100/100`, empty `0/100`, shortcut `20/100`

Verification matrix snapshots:

- `v1-clean-baseline`: `verification_matrix.md`
- `v5-recovery-in-thread`: `verification_matrix_v5.md`

Layer A status:

- Oracle / empty / shortcut baselines are wired and deterministic.
- Whole-family live probe is still pending. No `codex exec` calibration loop was launched in this turn.

Layer B status:

- dual-band scorer emitted (`P_benchmark`, `M_training`, schema `cnb55.verify_result.v3`)
- 5-slot milestones emitted plus milestone shell scripts
- integrity rules wired 1:1 in scorer and `family.yaml`
- verification matrices generated for V1 and stress variant V5
- manifest lock refreshed with current scorer/workspace hashes
