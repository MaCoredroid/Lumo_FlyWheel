# Candidate B Evaluation
- Worktree Path: `/tmp/v5-recovery-in-thread-candidate-b`

## Commands Run
- `python -m pytest -q tests/test_cli.py tests/test_service.py`
- `python - <<'PY'
from report_filters.service import compile_filters
print(compile_filters(['Ops---Latency__Summary']))
PY`

## Touched Files
- `src/report_filters/service.py`
- `tests/test_service.py`
- `docs/filter_contract.md`

## Observed Result
- `python -m pytest -q tests/test_cli.py tests/test_service.py` passes after moving normalization into the service layer.
- Direct caller smoke succeeds because the importer and saved-view repair job now share the same normalization owner.

## Evidence
- `repo_evidence/caller_matrix.md` shows the importer and repair job call `service.compile_filters(...)` directly.
- `artifacts/candidates/strategy_b_notes.md` keeps `cli.py` thin and avoids duplicated ownership.
- `release_context/importer_callers.md` makes the importer path the active release blocker.
- `incident_context/rollback_2026_07.md` explains why the service-layer repair is the recovery path.
