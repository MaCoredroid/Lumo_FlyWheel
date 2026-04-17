# LLD-04 Red-Team Round 2 Verifier Report

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
- `report/LLD-04-red-team-report.md`
- Signed-off spec: `docs/LLD-04-Latency-Telemetry-Capture-v0_7.md`

## Checks Performed

- Verified the `LatencyCapture` lifecycle against the signed-off `snapshot_before(task_id, seed, attempt)` and `snapshot_after(task_id)` contract.
- Verified schema resolution, parser behavior, snapshot lifecycle, anomaly filtering, and aggregation behavior against the spec sections on telemetry capture, storage, and completeness enforcement.
- Checked the CLI smoke-test path for telemetry schema compatibility and cache-hit detection.
- Checked turn extraction behavior to ensure non-assistant message events do not create false turns.
- Confirmed `task_orchestrator.py` passes `seed` and `attempt` into telemetry capture at the call site required by the amendment.

## Gaps Found

No remaining LLD-04 gaps worth fixing were found in this round.

The earlier round-1 fixes are still present and validated:

- CLI smoke-test cache-hit lookup now uses the canonical `cache_hits` telemetry key with a legacy fallback.
- `extract_turns()` now ignores non-assistant message events.

## Tests Run

Command:

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
```

Outcome:

- `107 passed`

## Live Codex -> localvllm / vLLM Path Check

Command:

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python scripts/run_live_codex_long_task.py --family report-cli-markdown-evolution --variant inventory-ops --prompt "Read README.md and report the first heading. Do not edit files." --timeout-seconds 180 --json
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

## Exact Commands

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python scripts/run_live_codex_long_task.py --family report-cli-markdown-evolution --variant inventory-ops --prompt "Read README.md and report the first heading. Do not edit files." --timeout-seconds 180 --json
```

## Residual Risks

- `extract_turns()` remains best-effort because Codex event stream formats are not fully specified.
- The live smoke path succeeded at the infrastructure level, but the benchmark variant still failed hidden verification slices, so end-to-end benchmark validation is not clean.
- Telemetry completeness is still enforced downstream by `aggregate_by_model()`; missing telemetry remains a fail-closed condition.

