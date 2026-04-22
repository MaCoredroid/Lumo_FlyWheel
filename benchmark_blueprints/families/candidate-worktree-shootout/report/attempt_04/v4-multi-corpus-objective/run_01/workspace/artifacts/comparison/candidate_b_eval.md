# Candidate B Evaluation

- Worktree Path: `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/benchmark_blueprints/families/candidate-worktree-shootout/report/attempt_04/v4-multi-corpus-objective/run_01/workspace/artifacts/comparison/worktrees/candidate_b`

## Commands Run

```bash
PYTHONPATH=src python -m pytest -q tests/test_cli.py tests/test_service.py -o cache_dir=/tmp/pytest-candidate-b
PYTHONPATH=src python - <<'PY'
from report_filters.service import compile_filters
print(compile_filters(["Ops---Latency__Summary", " Slack Alerts ", "  "]))
PY
PYTHONPATH=src python - <<'PY'
from report_filters.service import build_filter_query
from report_filters.cli import render_filters
print(build_filter_query(["Ops---Latency__Summary", " Slack Alerts ", "  "]))
print(render_filters("Ops---Latency__Summary,Slack Alerts"))
PY
```

## Touched Files

- `artifacts/comparison/worktrees/candidate_b/src/report_filters/service.py`
- `artifacts/comparison/worktrees/candidate_b/tests/test_service.py`

## Observed Result

Candidate B repairs the shared normalization contract in the service
layer. The direct service path, the query builder, and the CLI all emit
the same canonical keys, which addresses the scheduled importer blocker.

## Evidence

```text
$ PYTHONPATH=src python -m pytest -q tests/test_cli.py tests/test_service.py -o cache_dir=/tmp/pytest-candidate-b
....                                                                     [100%]
4 passed in 0.01s
```

```text
$ PYTHONPATH=src python - <<'PY'
from report_filters.service import compile_filters
print(compile_filters(["Ops---Latency__Summary", " Slack Alerts ", "  "]))
PY
['ops latency summary', 'slack alerts']
```

```text
$ PYTHONPATH=src python - <<'PY'
from report_filters.service import build_filter_query
from report_filters.cli import render_filters
print(build_filter_query(["Ops---Latency__Summary", " Slack Alerts ", "  "]))
print(render_filters("Ops---Latency__Summary,Slack Alerts"))
PY
ops latency summary,slack alerts
ops latency summary,slack alerts
```
