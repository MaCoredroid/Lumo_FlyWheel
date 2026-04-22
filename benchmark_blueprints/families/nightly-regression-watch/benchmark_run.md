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

## attempt_02 — live whole-family codex-exec probe

Exact command:

```bash
python3 benchmark_blueprints/families/nightly-regression-watch/tools/probe_family_live.py --attempt-id attempt_02_live_probe --n 1 --timeout-seconds 600
```

Artifacts:

- `benchmark_blueprints/families/nightly-regression-watch/report/attempt_02_live_probe_command.txt`
- `benchmark_blueprints/families/nightly-regression-watch/report/attempt_02_live_probe_probe_runs.jsonl`
- `benchmark_blueprints/families/nightly-regression-watch/report/attempt_02_live_probe_probe_report.txt`
- `benchmark_blueprints/families/nightly-regression-watch/report/attempt_02_live_probe_summary.json`
- `benchmark_blueprints/families/nightly-regression-watch/report/probe_run_logs/attempt_02_live_probe-*.log`

Per-variant live results:

- `v1-clean-baseline`: score `65`, `M_training=0.6500`, `codex_seconds=154.29`, `codex_exit=0`
- `v2-noisy-distractor`: score `65`, `M_training=0.6500`, `codex_seconds=178.32`, `codex_exit=0`
- `v3-dirty-state`: score `65`, `M_training=0.6500`, `codex_seconds=168.29`, `codex_exit=0`
- `v4-multi-corpus-objective`: score `20`, `M_training=0.2000`, `codex_seconds=230.42`, `codex_exit=0`
- `v5-recovery-in-thread`: score `65`, `M_training=0.6500`, `codex_seconds=244.48`, `codex_exit=0`

Common live failure pattern:

- `v1`, `v2`, `v3`, and `v5` repaired the parser/builder enough to pass visible tests but stopped short of the scorer's end-to-end contract, consistently missing the in-place automation-title/prompt alignment, runbook wording alignment, and hidden pytest expectations. Those runs landed at `65/100`.
- `v4` fell further to `20/100` because the generated digest artifact still drifted from the current code path on top of the same prompt/runbook/hidden-test misses.

Layer A gate values from the live attempt:

- `family_mean = 56.00`
- `max_variant_mean = 65.00`
- `min_variant_mean = 20.00`
- `monotonic_within_plus_3 = false`
- acceptance judgment: `red`

Acceptance verdict:

- Fail on `family_mean` (`56.00 > 25.00`)
- Fail on `max_variant_mean` (`65.00 > 40.00`)
- Fail on `min_variant_mean` (`20.00 > 10.00`)
- Fail monotonicity (`65, 65, 65, 20, 65`)

Calibration take-away:

- The current family is still much too easy for live `codex exec` on four of the five variants.
- The honest signal is now known: agents readily fix the parser and latest-of-day logic, but they often leave partial prompt/runbook contract drift. That is enough for a mid-60s score under the current scorer, which is far above the intended low-20s band.
- Next hardening should target the partial-repair plateau directly, most likely by increasing the visible/hidden weight on automation/runbook regeneration fidelity and by making the V4/V5 prompt-contract drift harder to bypass without fully aligning the artifact text.
