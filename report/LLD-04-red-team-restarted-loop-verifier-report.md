# LLD-04 Restarted Loop Verifier Report

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
- `report/LLD-04-red-team-round-2-verifier-report.md`
- Signed-off spec: `docs/LLD-04-Latency-Telemetry-Capture-v0_7.md`

## Checks Performed

- Re-read the signed-off LLD-04 contract, with emphasis on `LatencyCapture`, anomaly handling, `load_telemetry()`, `aggregate_by_model()`, and the Sprint 1 validation checklist.
- Audited the landed implementation in `metrics.py`, `task_orchestrator.py`, `cli.py`, package exports, and the listed test/report files.
- Re-ran the mandated live smoke command once against the pre-fix implementation to confirm what it actually proved.
- Fixed the verifier gap in the smoke path so the command now exercises live `LatencyCapture.resolve_schema()`, `snapshot_before()`, `snapshot_after()`, JSONL persistence, `load_telemetry()`, and `aggregate_by_model()` completeness on a smoke record.
- Tightened the `LatencyCaptureProtocol.snapshot_after()` return type to match the signed-off interface.
- Added/updated CLI regression coverage for the telemetry-backed smoke path.

## Gap Found

The required `smoke-test` command previously proved transport-level request routing, prefix-cache movement, Responses follow-up ids, and the tool-call probe, but it did **not** prove the LLD-04 telemetry contract end to end. Specifically, it never instantiated `LatencyCapture`, never wrote a telemetry JSONL row, and never reloaded/aggregated that row through the fail-closed completeness path.

That was a real verifier gap against the signed-off contract and the user’s stated goal of proving the live telemetry path before reporting `FULL_SEQUENCE_GOOD`.

## Files Changed

- `src/lumo_flywheel_serving/cli.py`
- `src/lumo_flywheel_serving/metrics.py`
- `src/lumo_flywheel_serving/task_orchestrator.py`
- `tests/test_cli.py`

## Exact Commands

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python -m py_compile src/lumo_flywheel_serving/task_orchestrator.py src/lumo_flywheel_serving/cli.py src/lumo_flywheel_serving/metrics.py tests/test_cli.py
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python -m lumo_flywheel_serving.cli --registry model_registry.yaml smoke-test qwen3.5-27b --enable-request-logging
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python scripts/run_live_codex_long_task.py --family report-cli-markdown-evolution --variant inventory-ops --prompt "Read README.md and report the first heading. Do not edit files." --timeout-seconds 180 --json
```

## Tests Run

- `./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py`
  - Result: `107 passed`
- `./.venv/bin/python -m py_compile src/lumo_flywheel_serving/task_orchestrator.py src/lumo_flywheel_serving/cli.py src/lumo_flywheel_serving/metrics.py tests/test_cli.py`
  - Result: passed

## Live Telemetry Smoke

Plain result: **passed**.

Exact command:

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python -m lumo_flywheel_serving.cli --registry model_registry.yaml smoke-test qwen3.5-27b --enable-request-logging
```

Observed result:

- `direct_api_smoke_status: pass`
- `tool_call_probe_status: pass`
- `prefix_cache_hits_delta: 5488.0`
- `telemetry_task_id: smoke-test/qwen3.5-27b/1776455661`
- `telemetry_path: output/telemetry/latency_qwen3.5-27b_public_dev.jsonl`
- `telemetry_record.seed: 0`
- `telemetry_record.attempt: 1`
- `telemetry_record.ttft_ms: 4888.619041442871`
- `telemetry_record.prefill_throughput_tps: 298.7680095914004`
- `telemetry_record.decode_throughput_tps: 7.967028891072058`
- `telemetry_record.cache_hit_rate_pct: 43.02626421011368`
- `telemetry_record.prompt_tokens: 12755.0`
- `telemetry_record.kv_computed_tokens: 7267.0`
- `telemetry_record.gen_tokens: 141.0`
- `telemetry_record.ttft_count: 5`
- `telemetry_record.cache_queries: 12755.0`
- `telemetry_record.cache_hits: 5488.0`
- `telemetry_record.anomalies: []`
- `telemetry_summary.model_id: qwen3.5-27b`
- `telemetry_summary.pool_or_split: public_dev`
- `telemetry_summary.n_tasks: 1`

The appended telemetry JSONL row was present at the recorded path and matched the smoke output.

## Supplemental Live Codex Path Evidence

This supplemental command was attempted after the smoke:

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python scripts/run_live_codex_long_task.py --family report-cli-markdown-evolution --variant inventory-ops --prompt "Read README.md and report the first heading. Do not edit files." --timeout-seconds 180 --json
```

Observed result:

- `countable: false`
- `infra_failure: true`
- `pass: false`
- `excluded_reason: localvllm endpoint http://127.0.0.1:8001/v1 is unavailable and upstream port 8000 is not healthy`

This did not contradict the smoke result. The smoke command stops the serving stack when it exits, so the follow-on live Codex task found no healthy local endpoint.

## Outcome

- The verifier gap was real.
- The gap was fixed in repo code.
- The required live telemetry smoke now proves the LLD-04 telemetry path more directly: live schema resolution, before/after snapshotting, JSONL write, reload, and fail-closed aggregate selection all succeeded on the same smoke run.

## Residual Risks

- The smoke still validates the telemetry subsystem using live requests plus the Codex-compatible tool-call probe, not a full task-orchestrator campaign run. It is now much closer to the LLD-04 contract, but it is still a smoke rather than a full LLD-03/LLD-07/LLD-12 campaign.
- The supplemental `run_live_codex_long_task.py` path remains infra-sensitive because it expects a healthy long-lived local endpoint. Running it after `smoke-test` without keeping the server alive will continue to fail fast for that reason.
- `extract_turns()` remains best-effort, which is already acknowledged by the spec and was not the blocking issue in this pass.
