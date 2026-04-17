# LLD-04 Final Pre-FULL_SEQUENCE Verifier Report

## Scope

Audited the current LLD-04 implementation on `main` at `503a93003f97145d5940b9c76a0b55de0fea414d` against the signed-off spec `docs/LLD-04-Latency-Telemetry-Capture-v0_7.md`, with explicit review of:

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
- `report/LLD-04-post-smoke-fix-verifier-report.md`
- `report/LLD-04-final-restarted-loop-verifier-report.md`

## Checks Performed

- Re-read the signed-off LLD-04 contract with emphasis on:
  - parser validation requirements in §5.3
  - snapshot lifecycle in §6
  - delta/anomaly handling in §7 and §12
  - Sprint 1 checklist items in §15
- Re-audited the landed implementation and prior verifier reports to find anything still missing before `FULL_SEQUENCE_GOOD`.
- Verified that the live smoke path still exercises schema resolution, before/after snapshots, JSONL persistence, reload, and aggregation.
- Captured a fresh real vLLM `/metrics` sample from the pinned live server path and compared it against the parser/schema expectations from the spec.

## Remaining Gap Found

One real verifier gap remained.

The signed-off spec requires a committed real `/metrics` fixture plus a parser validation test against that fixture. The implementation and smoke-path fixes were already in place, but the repo still lacked:

- a committed raw `/metrics` fixture from the pinned vLLM setup
- a unit test proving `parse_prometheus_text()` and `resolve_metric_schema()` succeed against that real capture while excluding `_bucket` lines and keeping non-negative `_sum` / `_count` values

That left the Sprint 1 parser-validation checklist incomplete even though the live smoke was passing.

## Files Changed

- `tests/test_metrics.py`
  - Added a fixture-backed parser validation test against a real live vLLM `/metrics` capture.
- `tests/fixtures/vllm_metrics_qwen3.5-27b.prom`
  - Added the committed raw `/metrics` fixture captured from the pinned live server path.
- `report/LLD-04-final-pre-full-sequence-verifier-report.md`
  - Added this verifier report.

## Exact Commands

### Live fixture capture

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python -m lumo_flywheel_serving.cli --registry model_registry.yaml serve qwen3.5-27b --enable-request-logging
```

```bash
cd /home/mark/shared/lumoFlyWheel && curl -fsS -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8000/metrics > /tmp/lld04_vllm_metrics_fixture.prom
```

### Required repo tests

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
```

Result:

- `111 passed in 0.41s`

### Required live telemetry confirmation

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python -m lumo_flywheel_serving.cli --registry model_registry.yaml smoke-test qwen3.5-27b --enable-request-logging
```

Exact result:

```json
{
  "direct_api_smoke_status": "pass",
  "health": 200,
  "schema": {
    "prompt_tokens": "vllm:prompt_tokens_total",
    "generation_tokens": "vllm:generation_tokens_total",
    "kv_computed_tokens": "vllm:request_prefill_kv_computed_tokens",
    "cache_queries": "vllm:prefix_cache_queries_total",
    "cache_hits": "vllm:prefix_cache_hits_total",
    "ttft": "vllm:time_to_first_token_seconds",
    "prefill_time": "vllm:request_prefill_time_seconds",
    "decode_time": "vllm:request_decode_time_seconds",
    "itl": "vllm:inter_token_latency_seconds"
  },
  "tool_call_probe_status": "pass",
  "prefix_cache_hits_delta": 5488.0,
  "telemetry_task_id": "smoke-test/qwen3.5-27b/1776459117",
  "telemetry_path": "output/telemetry/latency_qwen3.5-27b_public_dev.jsonl",
  "telemetry_record": {
    "seed": 0,
    "attempt": 1,
    "ttft_ms": 5075.766468048096,
    "prefill_throughput_tps": 287.961002779361,
    "decode_throughput_tps": 7.9553490960409325,
    "cache_hit_rate_pct": 43.02626421011368,
    "prompt_tokens": 12755.0,
    "kv_computed_tokens": 7267.0,
    "gen_tokens": 122.0,
    "prefill_sum_s": 25.236056027933955,
    "decode_sum_s": 15.33559351414442,
    "ttft_count": 5,
    "cache_queries": 12755.0,
    "cache_hits": 5488.0,
    "wall_clock_s": 40.757438213564456,
    "anomalies": []
  },
  "telemetry_summary": {
    "model_id": "qwen3.5-27b",
    "pool_or_split": "public_dev",
    "n_tasks": 1
  }
}
```

Plain result:

- The required live LLD-04 telemetry confirmation passed.

## Outcome

- A real remaining gap was present.
- The gap was verifier coverage, not telemetry runtime logic.
- The gap is now fixed by landing the real `/metrics` fixture and the missing parser validation test.
- The required repo tests passed.
- The required live telemetry smoke passed.

## Residual Risks

- The live smoke is still a smoke, not a full multi-task LLD-03/LLD-07/LLD-12 campaign run.
- The committed `/metrics` fixture is intentionally tied to the currently pinned vLLM setup; if the pinned vLLM version changes, this fixture and its expected schema may need refresh.
