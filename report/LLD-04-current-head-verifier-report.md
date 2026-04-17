# LLD-04 Current-Head Verifier Report

Date: 2026-04-17 UTC

## Baseline

- Repository: `/home/mark/shared/lumoFlyWheel`
- Branch: `main`
- Head commit: `f692638` (`Add LLD-04 full-sequence gate verifier report`)
- Signed-off spec used as the standard: `docs/LLD-04-Latency-Telemetry-Capture-v0_7.md`

## Scope Audited

- `src/lumo_flywheel_serving/metrics.py`
- `src/lumo_flywheel_serving/task_orchestrator.py`
- `src/lumo_flywheel_serving/cli.py`
- `src/lumo_flywheel_serving/__init__.py`
- `tests/test_metrics.py`
- `tests/test_task_orchestrator.py`
- `tests/test_cli.py`
- `tests/test_telemetry.py`
- `tests/fixtures/vllm_metrics_qwen3.5-27b.prom`
- `report/LLD-04-implementation-report.md`
- `report/LLD-04-red-team-report.md`
- `report/LLD-04-red-team-round-2-verifier-report.md`
- `report/LLD-04-red-team-restarted-loop-verifier-report.md`
- `report/LLD-04-post-smoke-fix-verifier-report.md`
- `report/LLD-04-final-restarted-loop-verifier-report.md`
- `report/LLD-04-final-pre-full-sequence-verifier-report.md`
- `report/LLD-04-full-sequence-gate-verifier-report.md`

## Checks Performed

1. Re-read the signed-off LLD-04 contract, with focus on:
   - schema resolution and required metrics
   - `snapshot_before(task_id, seed, attempt)` and `snapshot_after(task_id)` alignment with LLD-03
   - anomaly handling and default exclusions
   - fail-closed telemetry loading
   - `aggregate_by_model()` row-selection and completeness behavior
   - Sprint 1 validation invariants used by the smoke path
2. Re-audited the scoped implementation and tests against that contract.
3. Re-ran the required focused pytest suite on the current head.
4. Re-ran the required live vLLM Codex telemetry smoke on the current head and captured the exact result.

## Exact Commands

```bash
cd /home/mark/shared/lumoFlyWheel && git status --short --branch
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python -m lumo_flywheel_serving.cli --registry model_registry.yaml smoke-test qwen3.5-27b --enable-request-logging
```

## Tests Run

- `./.venv/bin/pytest tests/test_metrics.py tests/test_task_orchestrator.py tests/test_telemetry.py tests/test_cli.py`
  - Result: `111 passed in 0.36s`

## Required Live Telemetry Smoke

Command:

```bash
cd /home/mark/shared/lumoFlyWheel && ./.venv/bin/python -m lumo_flywheel_serving.cli --registry model_registry.yaml smoke-test qwen3.5-27b --enable-request-logging
```

Plain statement: the required live LLD-04 telemetry confirmation passed.

Observed result:

- `direct_api_smoke_status: pass`
- `health: 200`
- `tool_call_probe_status: pass`
- `reset_prefix_cache_status: 200`
- `telemetry_task_id: smoke-test/qwen3.5-27b/1776460994`
- `telemetry_path: output/telemetry/latency_qwen3.5-27b_public_dev.jsonl`
- `telemetry_record.seed: 0`
- `telemetry_record.attempt: 1`
- `telemetry_record.ttft_ms: 4974.453449249268`
- `telemetry_record.prefill_throughput_tps: 293.9014156403637`
- `telemetry_record.decode_throughput_tps: 7.98015693987972`
- `telemetry_record.cache_hit_rate_pct: 43.02626421011368`
- `telemetry_record.prompt_tokens: 12755.0`
- `telemetry_record.kv_computed_tokens: 7267.0`
- `telemetry_record.gen_tokens: 141.0`
- `telemetry_record.prefill_sum_s: 24.725978213362396`
- `telemetry_record.decode_sum_s: 17.66882544569671`
- `telemetry_record.ttft_count: 5`
- `telemetry_record.cache_queries: 12755.0`
- `telemetry_record.cache_hits: 5488.0`
- `telemetry_record.wall_clock_s: 42.58003091905266`
- `telemetry_record.anomalies: []`
- `telemetry_summary.model_id: qwen3.5-27b`
- `telemetry_summary.pool_or_split: public_dev`
- `telemetry_summary.n_tasks: 1`

Invariant spot-checks from the live result:

- `kv_computed_tokens <= prompt_tokens` held.
- `cache_hits <= cache_queries` held.
- `ttft_count == 5` held for the five smoke requests.
- `prefill_sum_s + decode_sum_s < wall_clock_s` held.
- reload plus `aggregate_by_model()` succeeded for exactly the smoke run.

## Files Changed

- `report/LLD-04-current-head-verifier-report.md`

Notes:

- The live smoke also appended a fresh row to `output/telemetry/latency_qwen3.5-27b_public_dev.jsonl`.
- That telemetry file was already dirty in the shared worktree before this verifier pass, and I did not revert it.
- No source or test files required changes.

## Outcome

I did not find a remaining LLD-04 correctness gap worth fixing on the current `main` head. The scoped implementation, tests, fixture-backed parser coverage, anomaly filtering, fail-closed loader behavior, orchestrator call alignment, and required live smoke all matched the signed-off contract closely enough for gate purposes.

Result: `FULL_SEQUENCE_GOOD`

## Residual Risks

- The required smoke is strong end-to-end validation for the live telemetry path, but it is still a smoke rather than a broader multi-task campaign run across the larger LLD-03/LLD-07/LLD-12 flow.
- Live startup remains infra-heavy for this model; most elapsed time in this pass was model load, compile, and warmup rather than telemetry logic.
- The append-only telemetry JSONL will continue to accumulate smoke rows unless an operator trims it intentionally; correctness is protected by the `(task_id, model_id, seed, attempt)` join and completeness rules.
