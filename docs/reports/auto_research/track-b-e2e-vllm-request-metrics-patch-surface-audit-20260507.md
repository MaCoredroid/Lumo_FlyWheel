# Track B E2E vLLM Request Metrics Patch Surface Audit

Generated: 2026-05-07

## Summary

Round 0 remains blocked on vLLM request correlation. The active vLLM server exposes aggregate Prometheus metrics for prompt tokens, generation tokens, request success, latency histograms, and spec decode counters, but those series are not labeled by `request_id` or `vllm_request_id`. The Track B parser can preserve request-id labels when they exist, but the live metrics endpoint does not emit them.

This means §5.1 of `track-b-e2e-agentic-saturation-plan-20260507.md` cannot be satisfied by configuration alone in the inspected source snapshot. A vLLM patch or a separate per-request JSONL side-channel is required before `vllm_per_turn.json` can be truthfully keyed to Codex turns.

## Live Environment Evidence

- vLLM `/health` is available on `http://127.0.0.1:9950/health`.
- Live `/metrics` includes `vllm:prompt_tokens_total`, `vllm:generation_tokens_total`, `vllm:request_success_total`, and `vllm:time_to_first_token_seconds`.
- Those live series carry labels like `engine="0"`, `model_name="qwen3.5-27b"`, `source="local_compute"`, and `finished_reason="stop"`.
- No live line matched `request_id`, `vllm_request_id`, `request=`, or `X-Request-Id`.

## Source Snapshot Inspected

Path:

`output/auto_research/qwen3.5-27b-responses-sdk-adapter-cutover-heavy-l0c-mutation-fp8_gemm-20260503T191447Z/cutlass_source_base/vllm-source`

Git commit:

`2a69949bdadf0e8942b7a1619b229cb475beef20`

## Findings

1. vLLM OpenAI serving can preserve a caller-provided request id.

   `vllm/entrypoints/openai/engine/serving.py` reads `X-Request-Id` in `_base_request_id()` and otherwise generates a random id. Chat completion serving prefixes that id into `chatcmpl-...` and passes sub-request ids into `engine_client.generate()`.

2. The optional response-header middleware is not metric correlation.

   `vllm/entrypoints/openai/cli_args.py` defines `enable_request_id_headers`, and `vllm/entrypoints/openai/api_server.py` installs `XRequestIdMiddleware` when enabled. That middleware only adds `X-Request-Id` to responses. It does not add request labels to Prometheus metrics.

3. Core Prometheus labels are aggregate labels only.

   `vllm/v1/metrics/loggers.py` builds the base metric label list as `["model_name", "engine"]`. Some metrics extend it with low-cardinality dimensions such as `source`, `finished_reason`, `sleep_state`, or `position`; the inspected metric construction does not include request id.

4. Finished request stats do not carry a request id into the Prometheus logger.

   `vllm/v1/metrics/stats.py` defines `FinishedRequestStats` with latency, prompt-token, generation-token, max-token, cache, and finish-reason fields. It omits `request_id`. `PrometheusStatLogger.record()` iterates `iteration_stats.finished_requests` and observes aggregate counters/histograms by engine, so the request id is already gone at that layer.

5. Spec decode metrics are aggregate per engine, not per request.

   `vllm/v1/spec_decode/metrics.py` constructs `vllm:spec_decode_num_drafts`, `vllm:spec_decode_num_draft_tokens`, `vllm:spec_decode_num_accepted_tokens`, and per-position accepted-token counters with the same caller-provided aggregate labels. No request label exists there either.

## Required Patch Surface

The least invasive truthful path is not to add high-cardinality `request_id` labels to public Prometheus metrics. Instead:

1. Add `request_id` to the internal finished-request metric record, or introduce a sibling per-request stats record before aggregation drops the id.
2. Emit a bounded local JSONL side-channel, for example `--track-b-request-metrics-out PATH`, from the vLLM API/engine process with one line per completed request:

   ```json
   {"request_id":"chatcmpl-trackb-...","ts_start":"...","ts_end":"...","prompt_tokens":1234,"generation_tokens":56,"ttft_s":0.42,"mean_tpot_s":0.03,"prefill_s":0.31,"decode_s":1.65,"spec_decode_num_drafts":120,"spec_decode_num_draft_tokens":1440,"spec_decode_num_accepted_tokens":310}
   ```

3. Gate the Track B runner on exact id overlap between:
   - Codex `turn_start.vllm_request_id`
   - vLLM per-request JSONL `request_id`
   - request-window timestamps used to slice DCGM samples

If Prometheus request labels are still preferred, they should be behind an explicit disabled-by-default flag with a warning about high cardinality. For this 13-task offline loop, a local JSONL artifact is a better fit.

## Readiness Impact

`vllm_request_metrics_join_available=false` is a hard Round 0 blocker. The gate may pass through either request-labeled Prometheus metrics or a bounded request-keyed JSONL side-channel, but the Track B summary code must not infer per-turn vLLM metrics from aggregate deltas while multiple turns or tasks can contribute to the same process-level counters.

The existing Track B parser work remains useful: it will consume request-labeled metrics or a request-keyed JSON artifact once vLLM emits one. It is not sufficient by itself to make the live server ready.
