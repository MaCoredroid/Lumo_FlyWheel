# LLD-04 Post-Smoke-Fix Verifier Report

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
- `report/LLD-04-red-team-restarted-loop-verifier-report.md`
- Signed-off spec: `docs/LLD-04-Latency-Telemetry-Capture-v0_7.md`

## Checks Performed

- Re-read the signed-off LLD-04 contract and the LLD-03 telemetry sequencing text, with emphasis on the correctness invariants in LLD-04 §7.3 and the Sprint 1 validation checklist.
- Audited the current implementation and tests in the scoped files after commits `7a63823`, `d1addf2`, and `1db554b`.
- Re-ran the required repo regression slice before and after the smoke-path fix.
- Re-ran the required live telemetry smoke after the fix and captured the exact JSON result.

## Gap Found

The restarted-loop verifier fixed the larger smoke-path hole by exercising live `LatencyCapture` end to end, but the smoke command still under-validated the signed-off telemetry contract. It recorded `ttft_count`, `prefill_sum_s`, `decode_sum_s`, and `wall_clock_s`, yet it did not fail when:

- `ttft_count` disagreed with the known five smoke requests.
- `prefill_sum_s + decode_sum_s >= wall_clock_s`, which violates the signed-off LLD-04 correctness invariant.

That left a remaining path to report a passing live smoke even if the emitted telemetry was internally inconsistent.

## Fix Applied

- Updated `src/lumo_flywheel_serving/cli.py` so `smoke-test` now fails closed unless:
  - `ttft_count == 5` for the five smoke requests.
  - `prefill_sum_s + decode_sum_s < wall_clock_s`.
- Extended the smoke JSON output to include `prefill_sum_s`, `decode_sum_s`, and `wall_clock_s` for verifier evidence.
- Added CLI regressions covering the new pass/fail cases in `tests/test_cli.py`.

## Files Changed

- `src/lumo_flywheel_serving/cli.py`
- `tests/test_cli.py`

## Exact Commands

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python -m lumo_flywheel_serving.cli --registry model_registry.yaml smoke-test qwen3.5-27b --enable-request-logging
```

## Tests Run

- `./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py`
  - Result: `109 passed`

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
- `telemetry_task_id: smoke-test/qwen3.5-27b/1776456987`
- `telemetry_path: output/telemetry/latency_qwen3.5-27b_public_dev.jsonl`
- `telemetry_record.seed: 0`
- `telemetry_record.attempt: 1`
- `telemetry_record.ttft_ms: 4938.783264160156`
- `telemetry_record.prefill_throughput_tps: 296.0198050016656`
- `telemetry_record.decode_throughput_tps: 8.122390641014759`
- `telemetry_record.cache_hit_rate_pct: 43.02626421011368`
- `telemetry_record.prompt_tokens: 12755.0`
- `telemetry_record.kv_computed_tokens: 7267.0`
- `telemetry_record.gen_tokens: 87.0`
- `telemetry_record.prefill_sum_s: 24.54903312958777`
- `telemetry_record.decode_sum_s: 10.711132207885385`
- `telemetry_record.ttft_count: 5`
- `telemetry_record.cache_queries: 12755.0`
- `telemetry_record.cache_hits: 5488.0`
- `telemetry_record.wall_clock_s: 35.453251076862216`
- `telemetry_record.anomalies: []`
- `telemetry_summary.model_id: qwen3.5-27b`
- `telemetry_summary.pool_or_split: public_dev`
- `telemetry_summary.n_tasks: 1`

Signed-off invariant check from the live result:

- `ttft_count == 5`: passed
- `prefill_sum_s + decode_sum_s < wall_clock_s`: passed (`35.26016533747315 < 35.453251076862216`)
- `kv_computed_tokens <= prompt_tokens`: passed
- `cache_hits <= cache_queries`: passed

## Outcome

- A remaining LLD-04 verifier gap was present.
- The gap was fixed in repo code.
- The required live telemetry smoke passed after the fix, with the strengthened invariant checks now enforced by the smoke command itself.

## Residual Risks

- The required smoke now verifies the live telemetry path more rigorously, but it is still a smoke and not a full LLD-03/LLD-07/LLD-12 campaign run.
- `extract_turns()` remains best-effort and was not the blocking issue in this round.
