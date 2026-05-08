# Track B E2E — Proxy-Side Instrumentation Unblock

Generated: 2026-05-08

## Summary

Two of the four Track B E2E Round 0 blockers (Step A: Codex `--trace-out`; Step D: vLLM per-request metric join) are now unblocked **without requiring a vLLM container restart and without patching the Codex CLI**. The substrate is in the inference proxy at `127.0.0.1:8022` (`lumo_flywheel_serving.inference_proxy`), which sees every `/v1/responses` call Codex makes and now emits a per-request observability JSONL when `LUMO_TRACK_B_REQUEST_METRICS_OUT` is set. The runner consumes that JSONL and synthesizes a `lumo.track_b.codex_trace_correctness.v1`-conformant `codex_trace.jsonl` per task. The two remaining blockers (Step B DCGM, Step G NCU) are restart-gated and addressed via spec v2.

Round 0 readiness moves from `decision="round0_blocked"` with three deferred-by-user instrumentation gaps + one missing NCU step → `decision="round0_blocked"` with **only DCGM deferred + NCU missing**, and Step A/D now substantively complete pending the trace-correctness comparison artifact. The reduced-contract Round 0 baseline (`median_wallclock_s=95.023`, 13/13 trusted, measured under SuffixDecoding) is preserved as a frozen reference.

## Three corrections to the prior research preamble

The 2026-05-08 preamble overstated upstream availability on three fronts. Verified against the running system:

1. **Codex 0.128.0 binary has no OTEL telemetry.** `strings $(which codex)` returns 180 strings; zero match `otel`/`otlp`/`telemetry`/`trace_out`. Codex PR #2103 ("OpenTelemetry events", merged 2025-09-29) is **not** in the installed CLI. Step A unblock therefore cannot come from configuring an existing OTEL exporter; the proxy is the only feasible no-restart unblock surface.
2. **Live vLLM is already running SuffixDecoding (Technique 1).** `[VLLM-INIT] speculative_config={"method":"suffix","num_speculative_tokens":12,"suffix_decoding_max_cached_requests":1000,"suffix_decoding_max_spec_factor":2.0,"suffix_decoding_max_tree_depth":32,"suffix_decoding_min_token_prob":0.05}`. `arctic-inference==0.1.2` was installed via the `ModelServer` prelaunch hook. Live `/metrics` shows aggregate spec_decode acceptance ≈ **51.4%** (51351 accepted / 99855 draft tokens). The codex-harness-spec-decode plan's "020/025/028 ngram-PLD candidates" framing **does not match the running runtime** — Step 0d should run B-1/B-2/B-3 against the actual `method=suffix, k=12` config, not against ngram candidates.
3. **The `:8022` proxy is `lumo_flywheel_serving.inference_proxy`** routing Codex's `wire_api=responses` to vLLM at `:9950`. `:8022` is OUR Python code — every Codex turn passes through it. This is the choke point that unblocks Steps A and D simultaneously without touching vLLM.

## What changed (code)

Six existing files modified, two new files created.

### `src/lumo_flywheel_serving/inference_proxy.py`

Added env-var-gated per-request capture (~200 LoC delta). Default off — when `LUMO_TRACK_B_REQUEST_METRICS_OUT` is unset, behavior is unchanged.

- New module constants: `TRACK_B_REQUEST_METRICS_SCHEMA = "lumo.track_b.vllm_request_metrics.v1"`, `TRACK_B_REQUEST_METRICS_PRODUCER = "track_b_vllm_request_metrics_patch"` (matching what `run_track_b_e2e_task.py` already expected via `--vllm-request-metrics-jsonl`).
- New class `TrackBRequestMetricsCapture` with `from_env()` factory. Threadsafe (file-locked via `fcntl.flock`); never raises into the response path (capture failure is silent to preserve inference reliability).
- New helpers `_extract_response_metadata`, `_classify_regime`, `_build_request_metrics_row`. Regime heuristic: `tool-call` if any function-call SSE event observed → else `summary` (>=4096 text chars) → `reasoning` (>0) → `unknown`.
- `_write_chunked_stream` extended with optional `capture_state` kwarg; observes SSE blocks for usage, response_id, tool-call frames, text chars; records `ts_first_byte` on the first downstream chunk. Streaming behavior unchanged.
- `do_POST` records `metrics_before` (vLLM `/metrics` snapshot) on entry, `metrics_after` after stream completes, computes deltas for `prefill_sum_s`, `decode_sum_s`, `spec_decode_num_accepted_tokens`, `spec_decode_num_draft_tokens`, `spec_decode_num_drafts`, emits a JSONL row.
- `build_proxy_handler` accepts an optional `request_metrics_capture` parameter; auto-instantiates from env when not provided.

Per-request row (verified live):

```json
{
  "completion_tokens": 184,
  "decode_sum_s": 16.62,
  "first_byte_s": 17.00,
  "metrics_snapshot_collected": true,
  "model": "qwen3.5-27b",
  "prefill_sum_s": 0.29,
  "producer": "track_b_vllm_request_metrics_patch",
  "prompt_tokens": 11,
  "regime": "reasoning",
  "request_class": "eval",
  "request_id": "resp_87f1fc1ea357e37d",
  "request_path": "/v1/responses",
  "runtime_config_hash": "sha256:841fb0ea93184839dc7e85f93911f65ff385a3ed0fb9d9ff1250c4c510c4d542",
  "saw_response_completed": true,
  "schema": "lumo.track_b.vllm_request_metrics.v1",
  "spec_decode_num_accepted_tokens": 58.0,
  "spec_decode_num_draft_tokens": 264.0,
  "spec_decode_num_drafts": 86.0,
  "text_chars_observed": 660,
  "tool_call_observed": false,
  "ts_completed": "2026-05-08T18:27:36.271Z",
  "ts_first_byte": "2026-05-08T18:27:36.271Z",
  "ts_request_received": "2026-05-08T18:27:19.268Z",
  "upstream_status": 200,
  "wallclock_s": 17.00
}
```

### `scripts/run_track_b_e2e_task.py`

Three additions (~150 LoC delta):

1. **`_synthesize_codex_trace_from_proxy_rows`** — reads the captured proxy rows for this task's time window, emits `task_start` (with real `runtime_config_hash`, `task_id`, `ts`), per-turn `turn_start` + `turn_end` events with `vllm_request_id` (from upstream response id), `regime` (from proxy classifier), `completion_tokens`, plus enriched fields (`prompt_tokens`, `decode_sum_s`, `prefill_sum_s`, `spec_decode_num_accepted_tokens`, `spec_decode_num_draft_tokens`), and `task_end` with `exit_code`, `wallclock_s`. The output satisfies `lumo.track_b.codex_trace_correctness.v1` directly — no Codex source patch needed.
2. **`_gpu_mem_snapshot`** — best-effort GPU memory + utilization sample via `nvidia-smi` (configurable command via `LUMO_TRACK_B_GPU_NVSMI_CMD` so it can be routed into the NVIDIA container on DGX Spark where the host can't query GPU memory directly). Captured before and after each task; recorded as `gpu_mem_pre`/`gpu_mem_post` in `runner_metadata.json`. Never raises.
3. **`_write_vllm_per_turn_from_jsonl`** signature change — now returns `(summary, captured_rows)` so the runner can pass the rows directly to trace synthesis. Also relaxed: missing or empty source is no longer a hard error; emits a deferred `vllm_per_turn.json` with `deferred_reason="no_proxy_capture_rows_for_task"` so the schema remains valid even when Codex's intermittent 0-token bug fires.

`run_one` now:
- Snapshots GPU memory pre and post.
- Captures the proxy JSONL byte offset before invoking Codex; reads only the rows captured during this task's window after Codex completes.
- When proxy capture is wired (`--vllm-request-metrics-jsonl` set) and `--defer-codex-trace-out` is not set, synthesizes `codex_trace.jsonl` from the captured rows.

`_validate_codex_command_template` no longer requires `{trace_out}` placeholder. Trace emission is now produced by inference-proxy capture + runner-side synthesis, so the placeholder is decorative.

### `scripts/build_track_b_runtime_config_hash.py` (new)

Reads the `[VLLM-INIT]` log lines written by `ModelServer`'s prelaunch hook (`/tmp/lumo-l0c-fp8-cutlass-run30-logs/vllm_qwen3.5-27b.log` by default), parses both compact `key=value` and JSON-shaped values, and emits `sha256:<hex>` over a canonical-JSON dict of fourteen load-bearing fields: `model_id`, `served_model_name`, `vllm_version`, `git_hash`, `quantization`, `kv_cache_dtype`, `max_model_len`, `gpu_memory_utilization`, `enforce_eager`, `tuned_config_id`, `weight_version_id`, `kernel_runtime_activation`, `speculative_config`, `wire_api`. Replaces the `sha256:aaaa...aaaa` placeholder used in the deferred Round 0 sweep.

Live-system value: `sha256:841fb0ea93184839dc7e85f93911f65ff385a3ed0fb9d9ff1250c4c510c4d542`.

### `scripts/preflight_track_b_e2e.py`

`codex_trace_out_supported` is now satisfied by **either** `--trace-out` in Codex CLI help **or** proxy-side trace synthesis being available (i.e., the JSONL side-channel coverage check `vllm_request_metrics_side_channel.ok=True`). The check now exposes both signals separately so the readiness manifest can audit which substrate produced the trace.

### `scripts/build_track_b_e2e_readiness_manifest.py`

Step A's evidence dict now includes `proxy_trace_synthesis_available`, `codex_native_trace_out_flag`, and `trace_substrate_ok` (= patch OR proxy synthesis). The Step A `ok` predicate is now `trace_substrate_ok AND trace_correctness_verified AND codex_trace_out_supported`. Description text updated to reflect the dual-substrate model.

### Tests

- `tests/test_inference_proxy.py` — six new tests covering regime classifier, response metadata extraction, env-driven from_env, delta computation including missing/negative cases, JSONL record schema/producer enforcement, end-to-end streaming proxy capture with the complete schema. Plus a "no env var = no capture row" assertion.
- `tests/test_track_b_e2e_runner.py` — replaced the trace-out-required validation test with one that documents the new optional placeholder rule. Added `_gpu_mem_snapshot` mock to existing tests that monkeypatch `subprocess.run`. Updated `_write_vllm_per_turn_from_jsonl` test calls to unpack the new tuple return.
- `tests/test_build_track_b_runtime_config_hash.py` — new file. Three tests: compact init log parsing, canonical payload field selection, deterministic hash that's order-independent.

**Test outcome:** 87 passed across `test_inference_proxy*`, `test_track_b_e2e_runner`, `test_track_b_e2e_readiness_manifest`, `test_track_b_e2e_round_driver`, `test_build_track_b_runtime_config_hash`. No regressions.

## Live artifacts produced

- `/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl` — proxy capture, currently 4–7 rows from smoke + repro runs. Schema-conformant; consumable by the runner's `--vllm-request-metrics-jsonl`.
- `/tmp/track_b_runtime_config_hash.json` — live runtime config payload + hash.
- `/tmp/track_b_capture_smoke/round_0/release-note-to-plan-translation__v1-clean-baseline/run_01/` — end-to-end smoke artifacts:
  - `codex_trace.jsonl` — 6 events (task_start + 2× {turn_start, turn_end} + task_end), all with real `runtime_config_hash`, valid `vllm_request_id`, `regime` populated.
  - `vllm_per_turn.json` — populated `requests` keyed by upstream response id with `prompt_tokens`, `completion_tokens`, `decode_sum_s`, `prefill_sum_s`, `spec_decode_*` counters, `decode_tps`, `accepted_per_draft_token`.
  - `vllm_request_metrics.jsonl` — 2 raw proxy rows.
  - `runner_metadata.json` — includes `codex_trace_synthesis={"synthesized": true, "turn_count": 2, "source_rows": 2}`, `vllm_request_metrics_capture` summary, `gpu_mem_pre`/`gpu_mem_post`, real `runtime_config_hash`.
- `/tmp/track_b_preflight_with_proxy.json` — refreshed preflight: `round0_may_run=true`, `blocking_reasons=[]`, `deferred_reasons=["dcgm_profile_fields_available"]`, `codex_trace_out_supported.ok=true`, `vllm_request_metrics_join_available.ok=true`.
- `/tmp/track_b_readiness_with_proxy.json` — refreshed readiness: Step D **complete**, Step A **missing-only-the-3-task-byte-equality-artifact**, Step B deferred (DCGM, restart-gated), Step G missing (NCU; drop in spec v2).

## Process state

- **Inference proxy at `:8022`:** restarted with the new code, PID `4190011`, env: `LUMO_TRACK_B_REQUEST_METRICS_OUT=/tmp/track_b_e2e_proxy_capture/request_metrics.jsonl`, `LUMO_TRACK_B_RUNTIME_CONFIG_HASH=sha256:841fb0ea93184839dc7e85f93911f65ff385a3ed0fb9d9ff1250c4c510c4d542`. Listening on `127.0.0.1:8022` upstream `http://127.0.0.1:9950`. PID file at `/tmp/track_b_e2e_proxy_8022.pid`, log at `/tmp/track_b_e2e_proxy_8022.log`, stderr at `/tmp/track_b_e2e_proxy_8022.stderr`.
- **vLLM container `lumo-vllm-l0c-fp8-cutlass-run30`:** unchanged (Up 34h+). Same `speculative_config` as recorded above. Aggregate spec_decode acceptance now ~51% from `/metrics`.

## Codex 0.128.0 0-token quirk surfaced (not introduced)

Roughly 1 in 3 Codex `exec` invocations against the local-proxy `responses` provider returns `turn.completed` with `usage: 0 tokens` despite making a real `/v1/responses` call (visible in proxy capture). The trigger correlates with a 403 on `/v1/models?client_version=0.128.0` (the proxy blocks non-inference paths) and looks intermittent within a session. The runner now handles this gracefully: deferred `vllm_per_turn.json` with `deferred_reason="no_proxy_capture_rows_for_task"`, and a degenerate `codex_trace.jsonl` with `task_start`+`task_end` only. For the eventual round_0 re-collection sweep, this should be mitigated by the existing `--discard-cold-attempt-exit` cold-attempt-warmup pattern plus a small retry loop on zero-token completions.

## Hard gates after refresh

| Gate | Before | After |
|---|---|---|
| `preflight_round0_may_run` | true (with 3 deferred) | true (1 deferred: DCGM) |
| `round0_summary_verified` | true (reduced contract) | true |
| `round_proposal_prompt_verified` | true | true |
| `trace_correctness_verified` | false | false (artifact pending; substrate now in place) |
| `ncu_profiles_verified` | false | false (drop in spec v2) |
| `all_implementation_steps_complete` | false | false |

Step status:

| Step | Before | After | Notes |
|---|---|---|---|
| A. Codex trace emission | deferred | **substrate complete; artifact missing** | Need 3-task × {capture-on,capture-off} byte-equality artifact at `output/track_b_e2e/codex_trace_emitter_correctness.json` |
| B. DCGM/NVML 100 Hz sampler | deferred | deferred | Restart-gated. Kineto pivot via `VLLM_TORCH_PROFILER_DIR` + vLLM `/start_profile` per-task wrapper. |
| C. E2E task runner | complete | complete | |
| D. vLLM per-turn metrics | deferred | **complete** | Side-channel JSONL produced by proxy is read by `--vllm-request-metrics-jsonl`. |
| E. Summary join + diagnosis | complete | complete | |
| F. Round proposal prompt | complete | complete | |
| G. Round 0 dry run / NCU | missing | missing | Drop in spec v2; replace with Kineto-derived per-archetype profile or one-time Nsight Systems gated on vLLM relaunch. |

## What remains

In priority order, with budget estimates:

1. **Trace-correctness artifact (Step A's last gate)** — 3 tasks × {capture-on, capture-off}, byte-equality assertion on `model_outputs` / `tool_call_sequence` / `milestone_scores`. The proxy is read-only on the response stream so equality is structural; the verifier still expects two real runs to compare. ~6–10 minutes.
2. **Round_0 re-collection under full instrumentation** — 13 tasks with proxy capture + real runtime hash + GPU mem snapshots, replacing the reduced-contract artifacts that have the `aaaa...` placeholder hash. ~20 minutes.
3. **Spec v2 doc** — `track-b-e2e-agentic-saturation-plan-20260507-v2.md`. Drop NCU. Replace DCGM with Kineto via `VLLM_TORCH_PROFILER_DIR` + `/start_profile`. Note SuffixDecoding (Technique 1) has shipped; reframe Step 0d as B-1/B-2/B-3 against `method=suffix, k=12`. ~30 minutes.
4. **Step 0d execution** — define and run B-1/B-2/B-3 correctness gates against the live SuffixDecoding config on tool-call-inclusive tasks (the round_0 13-task workload already qualifies). ~60–90 minutes including analysis.
5. **Stage Kineto patch** — diff for the prelaunch hook that adds `VLLM_TORCH_PROFILER_DIR=/tmp/track_b_kineto_traces` to the vLLM container env. Not applied until next vLLM relaunch (operator-gated).

## What I will NOT do without further authorization

- Restart the vLLM container `lumo-vllm-l0c-fp8-cutlass-run30`. The Kineto pivot for Step B and the dropping of NCU for Step G both require operator approval to relaunch with `VLLM_TORCH_PROFILER_DIR` set.
- Delete the existing reduced-contract `output/track_b_e2e/round_0/round_summary.json`. It is the frozen baseline reference; the re-collected round_0 will land at a sibling path.

## Files touched, ready to commit

```
modified:   scripts/build_track_b_e2e_readiness_manifest.py
modified:   scripts/preflight_track_b_e2e.py
modified:   scripts/run_track_b_e2e_task.py
modified:   src/lumo_flywheel_serving/inference_proxy.py
modified:   tests/test_inference_proxy.py
modified:   tests/test_track_b_e2e_runner.py
new file:   scripts/build_track_b_runtime_config_hash.py
new file:   tests/test_build_track_b_runtime_config_hash.py
new file:   docs/reports/auto_research/track-b-e2e-proxy-side-instrumentation-20260508.md
```
