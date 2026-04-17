# LLD-04 Live Task Telemetry Report

Date: 2026-04-17

## What Changed

- Extended `scripts/run_live_codex_long_task.py` to reuse the existing LLD-04 `LatencyCapture` flow for real authored Codex-Long task runs against local vLLM.
- Added runtime config loading from `.codex/config.toml` so the live-task runner can derive the local model id and upstream metrics endpoint.
- Added per-run output directories and `result.json` emission for a machine-readable run artifact.
- Ensured telemetry is finalized with `snapshot_after()` before JSONL reload and aggregation, so the same real task run produces the row that is later validated and reported.
- Added a `command_success` flag plus `task_elapsed_seconds` and `end_to_end_elapsed_seconds` fields to the result payload.
- Added script coverage in `tests/test_smoke_codex_long_variant.py` for runtime config parsing and the live-task `main()` success path.

## Validation

Required pytest command:

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
```

Additional live-script coverage:

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_smoke_codex_long_variant.py
```

## Live Command Run

Runtime bootstrap used:

```bash
cd /home/mark/shared/lumoFlyWheel && ./scripts/bootstrap_local_runtime.sh
cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python -m lumo_flywheel_serving.cli --registry model_registry.yaml serve qwen3.5-27b --enable-request-logging
```

Live authored-task command:

```bash
cd /home/mark/shared/lumoFlyWheel && PYTHONPATH=src ./.venv/bin/python scripts/run_live_codex_long_task.py --repo-root /home/mark/shared/lumoFlyWheel --family report-cli-markdown-evolution --variant inventory-ops --json
```

## Outcome

- Variant run: `report-cli-markdown-evolution/inventory-ops`
- Command completed successfully: `true`
- Grading passed: `false`
- Shortcut detected: `false`
- Telemetry anomalies: `[]`
- Telemetry row emitted cleanly: `true`
- One-task elapsed time (`task_elapsed_seconds`): `69.23449842911214`
- End-to-end elapsed time (`end_to_end_elapsed_seconds`): `83.14308169204742`
- Result JSON: `/home/mark/shared/lumoFlyWheel/output/live_codex_long_task/report-cli-markdown-evolution/inventory-ops/20260417T225012Z/result.json`
- Telemetry JSONL: `/home/mark/shared/lumoFlyWheel/output/live_codex_long_task/report-cli-markdown-evolution/inventory-ops/20260417T225012Z/telemetry/latency_qwen3.5-27b_public_dev.jsonl`

The emitted JSONL row uses task id `report-cli-markdown-evolution/inventory-ops` and includes the expected LLD-04 fields, including `seed`, `attempt`, `ttft_ms`, `prefill_throughput_tps`, `decode_throughput_tps`, `cache_hit_rate_pct`, `wall_clock_s`, and an empty `anomalies` list.

## Remaining Gaps

- The chosen authored variant did not pass hidden grading; the implementation here proves the end-to-end run and telemetry capture path, not task-solving quality for that specific scenario.
- The live-task runner still assumes the upstream vLLM metrics port is `base_url.port - 1`; that matches the current local proxy layout but remains an implicit convention.
