# Candidate A Evaluation
- Worktree Path: `/tmp/v3-dirty-state-candidate-a`

## Commands Run
- `python -m pytest -q tests/test_cli.py`
- `python - <<'PY'
from report_filters.service import compile_filters
print(compile_filters(['Ops---Latency__Summary']))
PY`

## Touched Files
- `src/report_filters/cli.py`

## Observed Result
- `python -m pytest -q tests/test_cli.py` would pass after the local CLI patch.
- Direct caller smoke still fails because `service.compile_filters(...)` never sees separator cleanup.

## Evidence
- `repo_evidence/caller_matrix.md` shows two direct service callers bypass the CLI.
- `artifacts/candidates/strategy_a_notes.md` documents the CLI-local shortcut.
- `artifacts/partial_work/cli_local_patch.diff` is abandoned partial work and not a head start.
