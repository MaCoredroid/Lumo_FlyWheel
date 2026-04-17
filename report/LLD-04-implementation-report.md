# LLD-04 Implementation Report

## Summary

Implemented the LLD-04 latency telemetry pipeline in `src/lumo_flywheel_serving/metrics.py` and wired the orchestrator to the signed-off `snapshot_before(task_id, seed, attempt)` interface.

What changed:

- Added Prometheus parsing with histogram bucket filtering and label stripping.
- Added metric schema resolution for the required vLLM counters/histograms.
- Added `PendingSnapshot`, `TaskMetrics`, `SnapshotManager`, `TelemetryWriter`, `TelemetryConfig`, `LatencyCapture`, `LatencyRecord`, `ModelLatencySummary`, `TelemetryGapError`, `load_telemetry()`, `aggregate_by_model()`, and `extract_turns()`.
- Updated `TaskOrchestrator` to pass `seed` and `attempt` into `snapshot_before()`.
- Updated package exports in `src/lumo_flywheel_serving/__init__.py`.
- Updated CLI schema-variant detection to accept the new telemetry schema keys.
- Added and updated tests for parser/schema resolution, task metric computation, telemetry persistence/loading, aggregation, turn extraction, and the `LatencyCapture` lifecycle.

## Tests Run

Command:

```bash
./.venv/bin/python -m py_compile src/lumo_flywheel_serving/metrics.py src/lumo_flywheel_serving/task_orchestrator.py src/lumo_flywheel_serving/cli.py src/lumo_flywheel_serving/__init__.py tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
```

Outcome:

- `py_compile`: passed.
- `pytest`: 106 passed.

## Live Codex -> local vLLM Path Check

Command:

```bash
./.venv/bin/python scripts/run_live_codex_long_task.py --family report-cli-markdown-evolution --variant inventory-ops --prompt "Read README.md and report the first heading. Do not edit files." --timeout-seconds 180 --json
```

Outcome:

- `countable: true`
- `infra_failure: false`
- `pass: false`
- `codex_result.returncode: 0`
- `grading_result.verify_result.pass: false`
- Reported errors:
  - `inventory-ops hidden CLI slice did not pass`
  - `inventory-ops hidden renderer slice did not pass`
  - `inventory-ops follow-up/docs slice did not pass`
  - `Phase 2 pytest suite did not pass`

## Remaining Gaps / Risks

- `extract_turns()` is still best-effort and based on provisional Codex event types.
- The live smoke path completed but the underlying benchmark variant still failed its hidden verification slices, so the repo is not validated end-to-end beyond the unit tests.
- The telemetry compatibility layer still supports legacy schema call patterns for the local test suite; any future cleanup should remove that only after the downstream call sites are fully settled.
