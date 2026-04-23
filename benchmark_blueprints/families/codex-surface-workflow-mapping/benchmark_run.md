# Benchmark Run

- `family_id`: `codex-surface-workflow-mapping`
- `task_id`: `t2_codex_surface_daily_triage_mapping`
- `run_date`: `2026-04-22`
- `model`: `gpt-5.4`
- `reasoning_effort`: `high`
- `run_context`: family-local authoring only

## attempt_01_family_completion

### Goal

Replace the placeholder family with a real CNB-55 bundle: 5 workspace variants, canonical CLI, deterministic scorer, verifier data, manifest lock, Layer B declaration, and verification matrices.

### Landed

- Added `workspace_bundle/v1..v5/` with real repo-local workflow evidence:
  - live entrypoint `make codex-daily-triage`
  - live direct command `python3 scripts/triage.py --window active --emit-md reports/daily_triage.md`
  - live schedule literal `0 9 * * 1-5`
  - stale draft / deprecated helper noise in every variant
  - dirty-state abandoned Codex draft in V3+
  - `release_context/` objective drift in V4+
  - `incident_context/` rollback evidence in V5
- Added `bin/cnb55-workflow-map` to every variant so the solver submits one structured JSON input and the CLI renders:
  - `artifacts/workflow_map.json`
  - `artifacts/SKILL.md`
  - `artifacts/codex_triage.toml`
  - `artifacts/automation_proposal.md`
  - `artifacts/mapping_note.md`
- Added deterministic scorer at [score_workflow_mapping.py](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/verifiers/codex-surface-workflow-mapping/score_workflow_mapping.py) plus family-local regen and matrix runners.
- Added `family.yaml`, `manifest.lock.json`, `verification_matrix.md`, and `verification_matrix_v5.md`.

### Deterministic validation run

Commands executed:

```bash
python3 -m py_compile \
  verifiers/codex-surface-workflow-mapping/regen_family.py \
  verifiers/codex-surface-workflow-mapping/score_workflow_mapping.py \
  verifiers/codex-surface-workflow-mapping/run_verification_matrix.py

python3 verifiers/codex-surface-workflow-mapping/regen_family.py

python3 verifiers/codex-surface-workflow-mapping/run_verification_matrix.py \
  --variant v1-clean-baseline \
  --out benchmark_blueprints/families/codex-surface-workflow-mapping/verification_matrix.md

python3 verifiers/codex-surface-workflow-mapping/run_verification_matrix.py \
  --variant v5-recovery-in-thread \
  --out benchmark_blueprints/families/codex-surface-workflow-mapping/verification_matrix_v5.md
```

Visible-test replay on oracle outputs also passed for V1 and V5 (`6 passed` / `6 passed`) using temporary copied workspaces.

Observed baseline results from [manifest.lock.json](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/manifest.lock.json):

- Oracle: `100/100` on all five variants
- Empty: `0/100` on all five variants
- Stale-helper shortcut: `20/100` on all five variants

Observed verification-matrix rows:

- V1 oracle / empty / shortcut / delete-tests behave as expected in [verification_matrix.md](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/verification_matrix.md)
- V5 stress variant repeats the same bands in [verification_matrix_v5.md](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/verification_matrix_v5.md)

### Status

- Layer A freeze-gate live probe: `pending`
- Layer B flywheel readiness: `green` for the deterministic family declaration, milestone scripts, manifests, and verification matrices already landed here

### Notes

- No `codex exec` live family probe was run in this attempt, so there is no honest family-mean calibration claim yet.
- The family is now structurally complete enough for a future `codex exec` probe loop without further scaffolding work.

## attempt_02_live_probe_hardening_and_counted_rerun

### Goal

Run the real whole-family `codex exec` calibration across all five variants, then harden only this family if the first live rows exposed an obvious family-local gap.

### Family-local hardening applied before the counted rerun

- First in-flight live rows showed `v3-dirty-state` had a clean `100` lane, which meant the dirty-state corpus was too easy to reject.
- Applied one family-local change only:
  - added `drafts/codex_triage.toml` as an abandoned-but-plausible Codex draft in `V3+`
  - required `V3+` submissions to reject that draft explicitly in verifier data
  - regenerated the family so manifests, gold files, and workspace bundles stayed aligned
- The pre-change `attempt_02_live_probe` was superseded and not counted after this hardening because later runs would otherwise mix pre- and post-change workspaces.

### Exact commands executed

```bash
python3 -m py_compile \
  verifiers/codex-surface-workflow-mapping/regen_family.py \
  verifiers/codex-surface-workflow-mapping/score_workflow_mapping.py \
  benchmark_blueprints/families/codex-surface-workflow-mapping/tools/probe_family.py \
  benchmark_blueprints/families/codex-surface-workflow-mapping/tools/probe_report.py

python3 verifiers/codex-surface-workflow-mapping/regen_family.py

python3 verifiers/codex-surface-workflow-mapping/run_verification_matrix.py \
  --variant v1-clean-baseline \
  --out benchmark_blueprints/families/codex-surface-workflow-mapping/verification_matrix.md

python3 verifiers/codex-surface-workflow-mapping/run_verification_matrix.py \
  --variant v5-recovery-in-thread \
  --out benchmark_blueprints/families/codex-surface-workflow-mapping/verification_matrix_v5.md

python3 benchmark_blueprints/families/codex-surface-workflow-mapping/tools/probe_family.py \
  --attempt attempt_03_live_probe \
  --n 3 \
  --variants v1-clean-baseline v2-noisy-distractor v3-dirty-state v4-multi-corpus-objective v5-recovery-in-thread \
  --timeout-seconds 900

python3 benchmark_blueprints/families/codex-surface-workflow-mapping/tools/probe_report.py \
  --attempt-dir benchmark_blueprints/families/codex-surface-workflow-mapping/report/attempt_03_live_probe
```

### Verification matrix outputs after hardening

- V1 in [verification_matrix.md](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/verification_matrix.md):
  - Oracle `P=100`, `M=1.0000`, `pass=True`
  - Empty `P=0`, `M=0.0000`, `ceilings=no_submission`
  - Shortcut stale helper `P=20`, `M=0.2000`
  - Delete-tests adversarial `P=0`, `integrity=1`, `H=tests_modified,immutable_slice_mutated`
- V5 in [verification_matrix_v5.md](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/verification_matrix_v5.md):
  - Oracle `P=100`, `M=1.0000`, `pass=True`
  - Empty `P=0`, `M=0.0000`, `ceilings=no_submission`
  - Shortcut stale helper `P=20`, `M=0.2000`
  - Delete-tests adversarial `P=0`, `integrity=1`, `H=tests_modified,immutable_slice_mutated`

### Counted live probe results

Probe report: [attempt_03_live_probe_probe_report.txt](/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/codex-surface-workflow-mapping/report/attempt_03_live_probe/attempt_03_live_probe_probe_report.txt)

| variant | n | mean P | stdev P | mean raw | mean M | stdev M | min | max | scores | ceilings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v1-clean-baseline | 3 | 50.00 | 35.36 | 92.33 | 0.5000 | 0.3536 | 25 | 100 | [25, 100, 25] | schedule_in_prompt, ungrounded_mapping |
| v2-noisy-distractor | 3 | 48.00 | 32.53 | 87.33 | 0.4800 | 0.3253 | 25 | 94 | [25, 25, 94] | ungrounded_mapping |
| v3-dirty-state | 3 | 25.00 | 0.00 | 87.33 | 0.2500 | 0.0000 | 25 | 25 | [25, 25, 25] | schedule_in_prompt, ungrounded_mapping |
| v4-multi-corpus-objective | 3 | 25.00 | 0.00 | 80.67 | 0.2500 | 0.0000 | 25 | 25 | [25, 25, 25] | dirty_state_reuse, schedule_in_prompt, ungrounded_mapping |
| v5-recovery-in-thread | 3 | 25.00 | 0.00 | 79.00 | 0.2500 | 0.0000 | 25 | 25 | [25, 25, 25] | dirty_state_reuse, incident_blind_reuse, ungrounded_mapping |

### Layer A gate values

- `family_mean ∈ [15,25]`: observed `34.60` -> `FAIL`
- `max variant mean ≤ 40`: observed `50.00` -> `FAIL`
- `at least one variant mean ≤ 10`: observed `25.00` -> `FAIL`
- `monotonic V1≥V2≥V3≥V4≥V5 ±3`: `PASS`
- `family_mean_M_training`: `0.3460`
- `current_observed_stdev_M_training`: `0.2450`

### Spot-check explanation

- `v1-clean-baseline` run 2 scored `100` with no ceilings. The solver kept all artifacts on `make codex-daily-triage`, quoted the weekday cron literally, kept schedule semantics out of the task prompt, grounded the mapping in real files, and rejected the stale helper surfaces cleanly. That is not a harness glitch; it means the baseline task is squarely within `gpt-5.4` high competence.
- The dirty-state hardening did land where intended. In `V4` and `V5`, `dirty_state_reuse` now fires probe-wide, and `V5` also fires `incident_blind_reuse`. The family-local change therefore strengthened the later variants honestly, but it did not change the fact that `V1` and `V2` remain far above the §10.1 window.

### Verdict

Concrete calibration blocker, not an infra blocker.

- The counted post-change live probe is complete and recorded family-locally.
- The family still fails Layer A because the baseline mapping task is too easy for the calibration model:
  - `V1` mean `50.00`
  - `V2` mean `48.00`
- Another minor scorer-only tweak would be fake ambiguity. To move this family into the `15-25` window honestly, the next iteration has to be a real task redesign:
  - materially harder baseline evidence in `V1/V2`, or
  - a user-approved wider Layer A window for this family's honest frontier difficulty.
