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
