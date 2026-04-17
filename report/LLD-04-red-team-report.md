# LLD-04 Red-Team Report

## Scope Checked

- `src/lumo_flywheel_serving/metrics.py`
- `src/lumo_flywheel_serving/task_orchestrator.py`
- `src/lumo_flywheel_serving/cli.py`
- `src/lumo_flywheel_serving/__init__.py`
- `tests/test_metrics.py`
- `tests/test_task_orchestrator.py`
- `tests/test_cli.py`
- `tests/test_telemetry.py`
- `report/LLD-04-implementation-report.md`
- Signed-off spec: `docs/LLD-04-Latency-Telemetry-Capture-v0_7.md`

## Gaps Found

1. The CLI smoke path indexed `schema["prefix_cache_hits"]`, but the landed telemetry resolver returns the logical key `cache_hits`. That was a runtime bug in `cmd_smoke_test()` and a mismatch with the signed-off schema contract.
2. `extract_turns()` was treating generic `message` events as assistant turns even when the trajectory included a non-assistant `role`. That was broader than the signed-off best-effort scope and could over-count turns.

## Fixes Applied

- Updated `cmd_smoke_test()` to read the cache-hit metric via `schema.get("cache_hits")` with a legacy fallback to `prefix_cache_hits`, and to fail clearly if no cache-hit key exists.
- Tightened `extract_turns()` so assistant-role filtering is applied before counting turn starts or assistant output.
- Added a parser regression test that verifies non-assistant message events do not create turns.
- Updated CLI tests to use the canonical `cache_hits` schema key.

## Tests Run

Command:

```bash
./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
./.venv/bin/python -m py_compile src/lumo_flywheel_serving/metrics.py src/lumo_flywheel_serving/task_orchestrator.py src/lumo_flywheel_serving/cli.py src/lumo_flywheel_serving/__init__.py tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
```

Outcome:

- `pytest`: 107 passed.
- `py_compile`: passed.

## Live Codex -> localvllm / vLLM Path Check

Attempted command:

```bash
./.venv/bin/python scripts/run_live_codex_long_task.py --family report-cli-markdown-evolution --variant inventory-ops --prompt "Read README.md and report the first heading. Do not edit files." --timeout-seconds 180 --json
```

Outcome:

- `countable: true`
- `infra_failure: false`
- `codex_result.returncode: 0`
- `grading_result.verify_result.pass: false`
- Hidden slice failures reported:
  - `inventory-ops hidden CLI slice did not pass`
  - `inventory-ops hidden renderer slice did not pass`
  - `inventory-ops follow-up/docs slice did not pass`
  - `Phase 2 pytest suite did not pass`

## Residual Risks

- `extract_turns()` remains best-effort by design. It now respects assistant-role filtering, but Codex event formats are still only partially specified.
- The live smoke probe completed successfully, but the benchmark harness still reported hidden slice failures, so end-to-end validation is not fully clean yet.
- `load_telemetry()` still drops malformed JSON lines silently; completeness is enforced downstream by `aggregate_by_model()` rather than at file parse time.
