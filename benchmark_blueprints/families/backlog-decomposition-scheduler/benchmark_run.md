# Benchmark Run — backlog-decomposition-scheduler

Family run protocol for CNB-55 Track 10, family `backlog-decomposition-scheduler`.

## Model under test

```bash
codex exec --model gpt-5.4 --reasoning-effort high
```

## Layer A target

- family mean in `[15, 25]`
- max variant mean `<= 40`
- at least one variant mean `<= 10`
- monotonic `V1 >= V2 >= V3 >= V4 >= V5` within `+/-3`
- oracle `>= 90`
- empty `= 0`
- shortcut `<= 30`

## attempt_00 — runtime family build and static verification

Scope completed inside the family-owned write surface:

- authored `workspace_bundle/v1..v5`
- added family-local CLI `bin/cnb55-schedule`
- added deterministic scorer `verifiers/backlog-decomposition-scheduler/score_schedule.py`
- added family-local regen / matrix / probe scripts
- generated `verifier_data/backlog-decomposition-scheduler/v1..v5`
- added `family.yaml`, `manifest.lock.json`, and verification-matrix artifacts

Commands run:

```bash
python3 verifiers/backlog-decomposition-scheduler/regen_family.py
python3 verifiers/backlog-decomposition-scheduler/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/backlog-decomposition-scheduler/verification_matrix.md
python3 verifiers/backlog-decomposition-scheduler/run_verification_matrix.py --variant v4-multi-corpus-objective --out benchmark_blueprints/families/backlog-decomposition-scheduler/verification_matrix_v4.md
```

Visible-test sweep run against oracle briefs for all 5 variants:

```bash
pytest -q tests/test_schedule_brief.py
```

Observed static sanity:

| variant | oracle | empty | shortcut |
| --- | ---: | ---: | ---: |
| v1-clean-baseline | 100 | 0 | 25 |
| v2-noisy-distractor | 100 | 0 | 25 |
| v3-dirty-state | 100 | 0 | 25 |
| v4-multi-corpus-objective | 100 | 0 | 25 |
| v5-recovery-in-thread | 100 | 0 | 25 |

Visible-test sweep:

- v1-clean-baseline: `5 passed in 0.01s`
- v2-noisy-distractor: `5 passed in 0.01s`
- v3-dirty-state: `5 passed in 0.01s`
- v4-multi-corpus-objective: `5 passed in 0.01s`
- v5-recovery-in-thread: `5 passed in 0.01s`

Verification matrix headline results:

V1 matrix:

| trajectory | P_benchmark | M_training | integrity | ceilings |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 0.9091 | 0 | none |
| Empty | 0 | 0.0000 | 0 | `no_schedule_file` |
| RAWR grounding_stripped | 25 | 0.2273 | 0 | `schedule_without_grounding` |
| Shortcut risky rollout | 25 | 0.2273 | 0 | `ignored_dependency_blocker`, `overloaded_scarce_specialist`, `risky_rollout_frontloaded` |
| Near-miss slot swap | 100 | 0.9091 | 0 | none |
| Delete-tests adversarial | 0 | 0.0000 | 1 | `tests_modified`, `immutable_slice_mutated` |

V4 matrix:

| trajectory | P_benchmark | M_training | integrity | ceilings |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 0.9091 | 0 | none |
| Empty | 0 | 0.0000 | 0 | `no_schedule_file` |
| RAWR grounding_stripped | 25 | 0.2273 | 0 | `schedule_without_grounding` |
| Shortcut risky rollout | 25 | 0.2273 | 0 | `ignored_dependency_blocker`, `overloaded_scarce_specialist`, `risky_rollout_frontloaded`, `objective_drift` |
| Near-miss slot swap | 45 | 0.4091 | 0 | `objective_drift` |
| Delete-tests adversarial | 0 | 0.0000 | 1 | `tests_modified`, `immutable_slice_mutated` |

Interpretation:

- Layer B scaffolding exists and is executable.
- Even before the live family probe, the oracle / shortcut spread already suggested the family was likely too easy because every oracle hits the scorer ceiling of `100` and the shortcut floor only reaches `25`, not the intended frontier-shaped ladder.

## attempt_01 — whole-family live probe (`probe_run_id=20260422T061352Z`)

Family-owned probe flow used:

```bash
python3 verifiers/backlog-decomposition-scheduler/probe_family.py \
  --repeats 3 \
  --jsonl-out benchmark_blueprints/families/backlog-decomposition-scheduler/report/probe_runs.jsonl \
  --summary-out benchmark_blueprints/families/backlog-decomposition-scheduler/report/probe_summary_latest.json

python3 verifiers/backlog-decomposition-scheduler/probe_report.py \
  benchmark_blueprints/families/backlog-decomposition-scheduler/report/probe_summary_latest.json \
  --out benchmark_blueprints/families/backlog-decomposition-scheduler/report/attempt_01_probe_report.txt
```

Per-run live solver command template used by the family-local probe driver:

```bash
codex -a never -s danger-full-access exec \
  --skip-git-repo-check \
  --json \
  -m gpt-5.4 \
  -c 'reasoning_effort="high"' \
  -c 'model_reasoning_effort="high"' \
  -C <temp workspace copy of workspace_bundle/<variant>> \
  'Read AGENTS.md, inspect the workspace evidence, and solve the task completely.'
```

Artifacts:

- per-run JSONL: [report/probe_runs.jsonl](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/backlog-decomposition-scheduler/report/probe_runs.jsonl)
- latest summary: [report/probe_summary_latest.json](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/backlog-decomposition-scheduler/report/probe_summary_latest.json)
- human-readable report: [report/attempt_01_probe_report.txt](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/backlog-decomposition-scheduler/report/attempt_01_probe_report.txt)

Observed per-variant results:

| variant | n | mean | stdev | scores | raw scores | ceiling hits |
| --- | --- | --- | --- | --- | --- | --- |
| v1-clean-baseline | 3 | 100.00 | 0.00 | [100, 100, 100] | [128, 129, 129] | none |
| v2-noisy-distractor | 3 | 100.00 | 0.00 | [100, 100, 100] | [125, 125, 125] | none |
| v3-dirty-state | 3 | 100.00 | 0.00 | [100, 100, 100] | [125, 125, 125] | none |
| v4-multi-corpus-objective | 3 | 100.00 | 0.00 | [100, 100, 100] | [128, 128, 128] | none |
| v5-recovery-in-thread | 3 | 45.00 | 0.00 | [45, 45, 45] | [111, 111, 111] | `objective_drift x3` |

Layer A gate check:

- `[FAIL]` family mean in `[15,25]`: `89.00`
- `[FAIL]` max variant mean `<= 40`: `100.00`
- `[FAIL]` at least one variant mean `<= 10`: `45.00`
- `[PASS]` monotonic `V1>=V2>=V3>=V4>=V5 +/-3`: yes
- `[PASS]` oracle `>= 90`: yes
- `[PASS]` empty `= 0`: yes
- `[PASS]` shortcut `<= 30`: yes

Explicit Layer A judgment:

- `LAYER_A_FAIL_HARDEN_NEEDED`

Diagnosis:

1. `v1` through `v4` are fully saturated under live `gpt-5.4/high` probe. No variant-specific ceiling fires at all on those 12 runs.
2. `v5` is the only live variant that still exerts any meaningful pressure, and it bottoms out at the existing `objective_drift` ceiling of `45`, which is still far above the required hard-floor target.
3. The current scorer is effectively a ceiling-at-100 instrument for most of the family. That means the family has runtime assets and deterministic grading, but it does not yet have a legitimate-difficulty ladder near the CNB-55 Layer A target band.
4. This is an honest failure of calibration, not a harness issue:
   - all 15 live runs completed successfully
   - no shortcut/integrity violations occurred
   - the family-local probe/report flow produced coherent, deterministic outputs
   - late-variant pressure is real only on V5, and even there it is too weak

Next hardening direction, not launched in this turn:

- add real variant-specific pressure to `v2` through `v4` instead of relying on general schedule correctness
- lower the easy-variant reward surface so correct-but-shallow schedules do not clip to `100`
- preserve legitimate-difficulty constraints; do not add fake ambiguity just to force the scores down

Per coordinator instruction, no additional hardening loop was launched from this attempt. This file records the completed whole-family live verification and the resulting explicit Layer A failure state.

## attempt_02 — YAML parse blocker repair and full family rerun (`probe_run_id=20260422T181200Z`)

Change made:

- repaired `family.yaml` by quoting the backtick-bearing scalar values in the M2 description and integrity-rule strings so the file is valid YAML without changing the benchmark contract, rubric, or thresholds

Commands run:

```bash
ruby -e 'require "yaml"; YAML.load_file("benchmark_blueprints/families/backlog-decomposition-scheduler/family.yaml"); puts "YAML_OK"'
python3 verifiers/backlog-decomposition-scheduler/regen_family.py
python3 verifiers/backlog-decomposition-scheduler/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/backlog-decomposition-scheduler/verification_matrix.md
python3 verifiers/backlog-decomposition-scheduler/run_verification_matrix.py --variant v4-multi-corpus-objective --out benchmark_blueprints/families/backlog-decomposition-scheduler/verification_matrix_v4.md
python3 verifiers/backlog-decomposition-scheduler/probe_family.py \
  --repeats 3 \
  --jsonl-out benchmark_blueprints/families/backlog-decomposition-scheduler/report/probe_runs.jsonl \
  --summary-out benchmark_blueprints/families/backlog-decomposition-scheduler/report/probe_summary_latest.json
python3 verifiers/backlog-decomposition-scheduler/probe_report.py \
  benchmark_blueprints/families/backlog-decomposition-scheduler/report/probe_summary_latest.json \
  --out benchmark_blueprints/families/backlog-decomposition-scheduler/report/attempt_02_probe_report.txt
```

Parse / validation outcome:

- `family.yaml`: `YAML_OK`
- family-local regen completed successfully and refreshed `manifest.lock.json`

Observed static sanity after the parse repair:

| variant | oracle | empty | shortcut |
| --- | ---: | ---: | ---: |
| v1-clean-baseline | 100 | 0 | 25 |
| v2-noisy-distractor | 100 | 0 | 25 |
| v3-dirty-state | 100 | 0 | 25 |
| v4-multi-corpus-objective | 100 | 0 | 25 |
| v5-recovery-in-thread | 100 | 0 | 25 |

Verification matrix headline results after rerun:

V1 matrix:

| trajectory | P_benchmark | M_training | integrity | ceilings |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 0.9091 | 0 | none |
| Empty | 0 | 0.0000 | 0 | `no_schedule_file` |
| RAWR grounding_stripped | 25 | 0.2273 | 0 | `schedule_without_grounding` |
| Shortcut risky rollout | 25 | 0.2273 | 0 | `ignored_dependency_blocker`, `overloaded_scarce_specialist`, `risky_rollout_frontloaded` |
| Near-miss slot swap | 100 | 0.9091 | 0 | none |
| Delete-tests adversarial | 0 | 0.0000 | 1 | `tests_modified`, `immutable_slice_mutated` |

V4 matrix:

| trajectory | P_benchmark | M_training | integrity | ceilings |
| --- | ---: | ---: | ---: | --- |
| Oracle | 100 | 0.9091 | 0 | none |
| Empty | 0 | 0.0000 | 0 | `no_schedule_file` |
| RAWR grounding_stripped | 25 | 0.2273 | 0 | `schedule_without_grounding` |
| Shortcut risky rollout | 25 | 0.2273 | 0 | `ignored_dependency_blocker`, `overloaded_scarce_specialist`, `risky_rollout_frontloaded`, `objective_drift` |
| Near-miss slot swap | 45 | 0.4091 | 0 | `objective_drift` |
| Delete-tests adversarial | 0 | 0.0000 | 1 | `tests_modified`, `immutable_slice_mutated` |

Observed per-variant live probe results:

| variant | n | mean | stdev | scores | raw scores | ceiling hits |
| --- | --- | --- | --- | --- | --- | --- |
| v1-clean-baseline | 3 | 100.00 | 0.00 | [100, 100, 100] | [129, 129, 129] | none |
| v2-noisy-distractor | 3 | 100.00 | 0.00 | [100, 100, 100] | [125, 125, 125] | none |
| v3-dirty-state | 3 | 100.00 | 0.00 | [100, 100, 100] | [125, 125, 125] | none |
| v4-multi-corpus-objective | 3 | 100.00 | 0.00 | [100, 100, 100] | [128, 128, 124] | none |
| v5-recovery-in-thread | 3 | 45.00 | 0.00 | [45, 45, 45] | [111, 111, 111] | `objective_drift x3` |

Layer A gate check after rerun:

- `[FAIL]` family mean in `[15,25]`: `89.00`
- `[FAIL]` max variant mean `<= 40`: `100.00`
- `[FAIL]` at least one variant mean `<= 10`: `45.00`
- `[PASS]` monotonic `V1>=V2>=V3>=V4>=V5 +/-3`: yes
- `[PASS]` oracle `>= 90`: yes
- `[PASS]` empty `= 0`: yes
- `[PASS]` shortcut `<= 30`: yes

Explicit Layer A judgment:

- `LAYER_A_FAIL_HARDEN_NEEDED`

Interpretation:

1. The reviewer blocker is real and fixed: `family.yaml` now parses cleanly as YAML and the family-local regen / verification / probe flow runs end to end against the repaired file.
2. The parse repair did not weaken the benchmark and did not materially change calibration. The family remains saturated on `v1`-`v4`, with `v5` still capped by `objective_drift` at `45`.
3. `manifest.lock.json` was refreshed as part of the post-fix family-local regen, and the new live probe artifacts are recorded under `report/`.
