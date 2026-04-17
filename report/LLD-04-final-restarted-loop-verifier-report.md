# LLD-04 Final Restarted-Loop Verifier Report

Date: 2026-04-17

## Scope Audited

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
- Signed-off spec: `docs/LLD-04-Latency-Telemetry-Capture-v0_7.md`

## Checks Performed

- Re-read the signed-off LLD-04 contract, with emphasis on `load_telemetry()`, fail-closed completeness, anomaly handling, and the Sprint 1 validation checklist.
- Re-audited the current implementation state after commits `7a63823`, `d1addf2`, `1db554b`, and `c7e9ef6`.
- Re-ran the required repo test slice after the fix.
- Re-ran the required live vLLM Codex telemetry smoke on the patched tree and captured the exact result.

## Gap Found And Fixed

One remaining LLD-04 gap was present in `load_telemetry()`.

The signed-off spec shows fail-closed JSONL loading, but the implementation was still silently skipping malformed telemetry lines. That could hide telemetry artifact corruption whenever the broken line did not correspond to a currently reportable run, which is looser than the signed-off contract.

Fix landed:

- `src/lumo_flywheel_serving/metrics.py`
  `load_telemetry()` now raises immediately on malformed JSONL with file and line information instead of silently continuing.
- `tests/test_telemetry.py`
  Added a regression test proving malformed telemetry JSONL now fails closed.

## Exact Commands

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python -m lumo_flywheel_serving.cli --registry model_registry.yaml smoke-test qwen3.5-27b --enable-request-logging
```

## Tests Run

- `./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py`
  Result: `110 passed in 0.26s`

## Required Live Telemetry Smoke

The required live LLD-04 telemetry confirmation passed.

Exact command result:

```text
lumo-vllm
lumo-vllm
{
  "direct_api_smoke_status": "pass",
  "health": 200,
  "models": {
    "object": "list",
    "data": [
      {
        "id": "qwen3.5-27b",
        "object": "model",
        "created": 1776457899,
        "owned_by": "vllm",
        "root": "/models/qwen3.5-27b-fp8",
        "parent": null,
        "max_model_len": 131072,
        "permission": [
          {
            "id": "modelperm-91a1b8ef284b8764",
            "object": "model_permission",
            "created": 1776457899,
            "allow_create_engine": false,
            "allow_sampling": true,
            "allow_logprobs": true,
            "allow_search_indices": false,
            "allow_view": true,
            "allow_fine_tuning": false,
            "organization": "*",
            "group": null,
            "is_blocking": false
          }
        ]
      }
    ]
  },
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
  "chat_completion_ids": [
    "chatcmpl-b05b1564d56cddaa",
    "chatcmpl-b6657865012a4e2f"
  ],
  "responses_ids": [
    "resp_b4b6f590ea4c06fb",
    "resp_a94f98253215695d"
  ],
  "tool_call_probe_status": "pass",
  "reset_prefix_cache_status": 200,
  "prefix_cache_hits_delta": 5488.0,
  "telemetry_task_id": "smoke-test/qwen3.5-27b/1776457899",
  "telemetry_path": "output/telemetry/latency_qwen3.5-27b_public_dev.jsonl",
  "telemetry_record": {
    "seed": 0,
    "attempt": 1,
    "ttft_ms": 4906.560707092285,
    "prefill_throughput_tps": 298.498373590896,
    "decode_throughput_tps": 7.974081900309233,
    "cache_hit_rate_pct": 43.02626421011368,
    "prompt_tokens": 12755.0,
    "kv_computed_tokens": 7267.0,
    "gen_tokens": 131.0,
    "prefill_sum_s": 24.34519127383828,
    "decode_sum_s": 16.42822354193777,
    "ttft_count": 5,
    "cache_queries": 12755.0,
    "cache_hits": 5488.0,
    "wall_clock_s": 41.01054181996733,
    "anomalies": []
  },
  "telemetry_summary": {
    "model_id": "qwen3.5-27b",
    "pool_or_split": "public_dev",
    "n_tasks": 1
  }
}
```

## Files Changed

- `src/lumo_flywheel_serving/metrics.py`
- `tests/test_telemetry.py`
- `report/LLD-04-final-restarted-loop-verifier-report.md`

## Outcome

- Remaining gap found and fixed.
- Required repo tests passed.
- Required live telemetry smoke passed.
- LLD-04 is ready for `FULL_SEQUENCE_GOOD` after this commit.

## Residual Risks

- `load_telemetry()` now fails closed on malformed JSONL, but recovery for corrupted historical telemetry remains operator-driven because the original before/after window is not reconstructable after the fact.
- The required smoke is strong end-to-end validation for the live telemetry path, but it is still a smoke and not a full multi-task campaign run across the broader LLD-03/LLD-07/LLD-12 flow.
