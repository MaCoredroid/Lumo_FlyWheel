# Benchmark Run: `fanout-fullstack-release-blocker`

## Run Metadata

- Run type: child-subagent family implementation pass
- Model: `gpt-5.4`
- Reasoning effort: `high`
- Scope restriction:
  - `benchmark_blueprints/families/fanout-fullstack-release-blocker/**`
  - `verifiers/fanout-fullstack-release-blocker/**`
  - `verifier_data/fanout-fullstack-release-blocker/**`

## attempt_00 — prose-only family stub

Initial state before this pass:

- `task_spec.md`, `evaluator_contract.md`, and `benchmark_run.md` existed
- no workspace bundle
- no family-local scorer / verifier wrapper
- no verifier data
- no `family.yaml`
- no `manifest.lock.json`

Recorded baseline from the earlier prose-only child attempt:

- final score: `20 / 100`
- strongest signal: good migration narrative, but no real patch, no live-path
  proof, no artifact

Conclusion:

- Layer A: not calibratable yet because there was no runnable family asset pack
- Layer B: missing almost every required artifact

## attempt_01 — family asset pack + Layer B scaffolding

Implemented during this pass:

- 5-variant workspace bundle under `workspace_bundle/`
- deterministic dual-band scorer:
  - `verifiers/fanout-fullstack-release-blocker/score_release_blocker.py`
- verifier wrapper:
  - `verifiers/fanout-fullstack-release-blocker/verify.sh`
- family-local regen + matrix runners:
  - `verifiers/fanout-fullstack-release-blocker/regen_family.py`
  - `verifiers/fanout-fullstack-release-blocker/run_verification_matrix.py`
- family-local verifier data:
  - gold state, manifests, oracle overlays, hidden tests, milestone scripts
- Layer B contract files:
  - `family.yaml`
  - `manifest.lock.json`
  - `verification_matrix.md`
  - `verification_matrix_v5-recovery-in-thread.md`

### Baseline numeric checks

Per variant (`v1`..`v5`):

| variant | oracle | empty | backend-only shortcut |
|---|---:|---:|---:|
| v1-clean-baseline | 100 | 0 | 20 |
| v2-noisy-distractor | 100 | 0 | 20 |
| v3-dirty-state | 100 | 0 | 20 |
| v4-multi-corpus-objective | 100 | 0 | 20 |
| v5-recovery-in-thread | 100 | 0 | 20 |

### Verification matrix spot checks

V1:

- Oracle: `P=100`, `M=1.0000`, `pass=true`
- Empty: `P=0`, `M=0.0000`, `pass=false`
- Backend-only alias fix: `P=20`, `M=0.2299`, `pass=false`
- Fullstack without proof: `P=35`, `M=0.4023`, `pass=false`
- Request fixed, echo stale: `P=35`, `M=0.4023`, `pass=false`
- Delete-tests adversarial: `P=95`, `M=0.0000`, `integrity_flag=1`, `pass=false`

V5 stress rerun:

- Oracle remained `100`
- Empty row correctly picked up both release-order and incident-aware ceilings
- Delete-tests adversarial again forced `integrity_flag=1`, `M=0.0000`,
  `pass=false`

### What this pass did not do

- No family probe loop was launched
- No family-mean / monotonicity / max-variant freeze-gate numbers were claimed
- No cross-family infrastructure was edited

### Current judgment

- Layer B: materially implemented and locally verified
- Layer A: still pending honest probe calibration on the real evaluation loop

Next honest step, owned by the parent loop rather than this subagent:

- run the real family probe and record family-mean / monotonicity outcomes

## attempt_02 — reviewer follow-up fix

Reviewer-directed changes in this pass:

- completed `family.yaml#rawr_modes` to all 3 declared modes
- promoted every RAWR entry to explicit `status:` form
- treated the Delete-tests row as a scorer bug, not an intentional divergence

### Exact fixes

RAWR metadata now declares:

- `grounding_stripped` with `status: implemented`
- `citation_fabricated` with `status: declared_not_yet_implemented`
- `constraint_named_not_respected` with `status: implemented`

Scorer change:

- `verifiers/fanout-fullstack-release-blocker/score_release_blocker.py`
  now clamps `P_benchmark` / `score` to `0` when `integrity_flag = 1`

### Rerun results

Baselines rerun after the scorer fix:

| variant | oracle | empty | backend-only shortcut |
|---|---:|---:|---:|
| v1-clean-baseline | 100 | 0 | 20 |
| v2-noisy-distractor | 100 | 0 | 20 |
| v3-dirty-state | 100 | 0 | 20 |
| v4-multi-corpus-objective | 100 | 0 | 20 |
| v5-recovery-in-thread | 100 | 0 | 20 |

Updated verification-matrix Delete-tests row:

- V1: `P=0`, `M=0.0000`, `integrity_flag=1`, `pass=false`
- V5: `P=0`, `M=0.0000`, `integrity_flag=1`, `pass=false`

Other matrix rows were unchanged:

- backend-only alias fix: `20`
- fullstack without proof: `35`
- request fixed, echo stale: `35`

### Reviewer-fix judgment

- RAWR metadata issue: fixed
- Delete-tests anomaly: fixed and rerun
- Layer B status: unchanged, still locally verified
- Layer A status: unchanged, still pending the real probe loop

## attempt_03 — real live `codex exec` family probe

This pass ran the real whole-family live probe locally across all 5 variants.

### Exact commands

Family-local live probe command:

```bash
python3 verifiers/fanout-fullstack-release-blocker/run_live_probe.py --n 3 --timeout 120
```

Family-local report generation:

```bash
python3 scripts/probe_report.py \
  benchmark_blueprints/families/fanout-fullstack-release-blocker/report/probe_runs.jsonl \
  --probe-run-id 20260422T200932Z \
  > benchmark_blueprints/families/fanout-fullstack-release-blocker/report/attempt_03_probe_report.txt
```

Implementation detail:

- `run_live_probe.py` stages a fresh workspace per run
- each run invokes real `codex exec --model gpt-5.4 -c 'model_reasoning_effort="high"'`
- scorer: `verifiers/fanout-fullstack-release-blocker/score_release_blocker.py`
- probe artifacts:
  - `benchmark_blueprints/families/fanout-fullstack-release-blocker/report/probe_runs.jsonl`
  - `benchmark_blueprints/families/fanout-fullstack-release-blocker/report/attempt_03_probe_report.txt`
  - `benchmark_blueprints/families/fanout-fullstack-release-blocker/report/live_probe_logs/*.log`

### Probe run id

- `probe_run_id`: `20260422T200932Z`
- `N`: `3`
- total live runs: `15`
- per-run timeout: `120s`

### Per-run scores

| variant | run1 | run2 | run3 | mean | stdev |
|---|---:|---:|---:|---:|---:|
| v1-clean-baseline | 25 | 25 | 25 | 25.00 | 0.00 |
| v2-noisy-distractor | 25 | 20 | 25 | 23.33 | 2.89 |
| v3-dirty-state | 25 | 25 | 25 | 25.00 | 0.00 |
| v4-multi-corpus-objective | 25 | 5 | 25 | 18.33 | 11.55 |
| v5-recovery-in-thread | 25 | 25 | 25 | 25.00 | 0.00 |

Raw-score notes:

- V1:
  - run2 hit `raw=40` but still capped to `25` by `surface_gap`
- V2:
  - run2 dropped to `20` from `stale_runtime_token`
- V3:
  - run3 hit `raw=40` but still capped to `25`
- V4:
  - run2 dropped to `5` from the combined `release_order_missed` + `stale_runtime_token` path
  - run3 reached `raw=85` but still capped to `25`
- V5:
  - all three runs were capped to `25`; every run fired `incident_blind_recovery`

### Layer A gate values

- `family_mean = 23.33`
- `max_variant_mean = 25.00`
- `min_variant_mean = 18.33`
- monotonicity check:
  - `v1 25.00 >= v2 23.33 >= v3 25.00` is acceptable within ±3
  - `v4 18.33 < v5 25.00` breaks monotonicity beyond ±3

Acceptance checks:

- `[PASS]` family mean in `[15, 25]` — `23.33`
- `[PASS]` max variant mean `<= 40` — `25.00`
- `[FAIL]` at least one variant mean `<= 10` — best floor was `18.33`
- `[FAIL]` monotonic `V1 >= V2 >= V3 >= V4 >= V5 ±3` — `v4 18.33 < v5 25.00`

### Acceptance judgment

- verdict: `HARDEN NEEDED`

Why:

- the family mean and max are already in the target band
- but the family does **not** yet produce a sufficiently hard variant on mean
- and `v5` remains easier than `v4`, so the tail is not calibrated honestly for
  the required monotonic difficulty ladder

Most repeated ceiling patterns in the live run:

- `surface_gap` on all 15 runs
- `missing_integration_report` on 12/15 runs
- `generic_proof` on 13/15 runs
- `release_order_missed` on all 3 `v4` runs and all 3 `v5` runs
- `incident_blind_recovery` on all 3 `v5` runs

Conclusion from the live run:

- Layer A remains **not accepted**
- the next honest family-local step is to harden `v5` relative to `v4` and to
  make the hard-floor variant durable on mean, not just as a single `5/100`
  outlier run
