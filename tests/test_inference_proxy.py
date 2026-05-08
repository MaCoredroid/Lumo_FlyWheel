from __future__ import annotations

import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

from lumo_flywheel_serving.inference_proxy import (
    TRACK_B_REQUEST_METRICS_PRODUCER,
    TRACK_B_REQUEST_METRICS_SCHEMA,
    TrackBRequestMetricsCapture,
    _classify_regime,
    _extract_response_metadata,
    _normalize_request_shaping_policy,
    _write_chunked_stream,
    is_inference_path,
    normalize_responses_request_payload,
    normalize_responses_response_payload,
    normalize_responses_sse_block,
    build_proxy_handler,
)
from lumo_flywheel_serving.tuned_config import RuntimeStateStore, make_tuned_config_bundle, persist_tuned_config_bundle


def test_normalize_responses_request_payload_flattens_nested_function_tools() -> None:
    payload = {
        "model": "qwen3.5-27b",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": "Run a shell command.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }

    normalized = normalize_responses_request_payload(payload)

    assert normalized["tools"] == [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Run a shell command.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]


def test_normalize_responses_request_payload_preserves_existing_tool_shapes() -> None:
    payload = {
        "model": "qwen3.5-27b",
        "tools": [
            {"type": "function", "name": "exec_command", "parameters": {"type": "object"}},
            {"type": "web_search"},
        ],
    }

    normalized = normalize_responses_request_payload(payload)

    assert normalized == payload


def test_normalize_responses_request_payload_adds_reasoning_status() -> None:
    payload = {
        "model": "qwen3.5-27b",
        "input": [
            {
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "thinking"}],
            }
        ],
    }

    normalized = normalize_responses_request_payload(payload)

    assert normalized["input"][0]["status"] == "completed"
    assert normalized["input"][0]["id"].startswith("rs_")


def test_normalize_responses_response_payload_adds_reasoning_status() -> None:
    payload = {
        "type": "response.completed",
        "response": {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "thinking"}],
                }
            ]
        },
    }

    normalized = normalize_responses_response_payload(payload)

    assert normalized["response"]["output"][0]["status"] == "completed"
    assert normalized["response"]["output"][0]["id"].startswith("rs_")


def test_normalize_responses_response_payload_trims_function_call_argument_trailing_text() -> None:
    payload = {
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "name": "exec_command",
            "arguments": '{"cmd": "pwd"} trailing text',
        },
    }

    normalized = normalize_responses_response_payload(payload)

    assert normalized["item"]["arguments"] == '{"cmd":"pwd"}'


def test_normalize_responses_sse_block_adds_reasoning_status() -> None:
    block = (
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"output":[{"type":"reasoning","summary":[]}]}}\n'
    )

    normalized = normalize_responses_sse_block(block)

    assert b'"status":"completed"' in normalized
    assert b'"id":"rs_' in normalized
    assert normalized.endswith(b"\n\n")


def test_normalize_responses_sse_block_trims_function_call_argument_trailing_text() -> None:
    block = (
        b"event: response.output_item.done\n"
        b'data: {"type":"response.output_item.done","item":{"type":"function_call","name":"exec_command",'
        b'"arguments":"{\\"cmd\\": \\"pwd\\"} trailing text"}}\n'
    )

    normalized = normalize_responses_sse_block(block)

    assert b'"arguments":"{\\"cmd\\":\\"pwd\\"}"' in normalized
    assert b"trailing text" not in normalized


def test_proxy_get_v1_models_403s_so_codex_skips_model_refresh(tmp_path: Path) -> None:
    """Verify GET /v1/models stays 403 — Codex 0.128.0 misbehaves more on a
    partial-shape response than on an outright 403 (softer fallback path)."""

    proxy, proxy_thread, proxy_url = _start_server(
        build_proxy_handler("http://upstream.invalid", state_root=tmp_path / "state")
    )
    try:
        response = requests.request("GET", f"{proxy_url}/v1/models?client_version=0.128.0", timeout=5)
        assert response.status_code == 403
        assert "inference paths only" in response.text
    finally:
        proxy.shutdown()
        proxy_thread.join(timeout=5)
        proxy.server_close()


def test_is_inference_path_only_allows_inference_endpoints() -> None:
    assert is_inference_path("/v1/responses") is True
    assert is_inference_path("/v1/chat/completions") is True
    assert is_inference_path("/metrics") is False


def test_write_chunked_stream_tolerates_broken_pipe() -> None:
    class _BrokenPipeWriter(io.BytesIO):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def write(self, data: bytes) -> int:
            self.calls += 1
            if self.calls == 2:
                raise BrokenPipeError
            return super().write(data)

    class _Handler:
        def __init__(self) -> None:
            self.wfile = _BrokenPipeWriter()

    class _Upstream:
        def __init__(self) -> None:
            self.closed = False

        def iter_content(self, chunk_size: int):
            assert chunk_size == 8192
            yield b"hello"
            yield b"world"

        def close(self) -> None:
            self.closed = True

    handler = _Handler()
    upstream = _Upstream()

    _write_chunked_stream(handler, upstream)

    assert upstream.closed is True


def test_write_chunked_stream_closes_cleanly_on_upstream_chunk_error() -> None:
    class _Handler:
        def __init__(self) -> None:
            self.wfile = io.BytesIO()

    class _Upstream:
        def __init__(self) -> None:
            self.closed = False

        def iter_content(self, chunk_size: int):
            assert chunk_size == 8192
            yield b"event: response.output_text.delta\ndata: {}\n\n"
            raise requests.exceptions.ChunkedEncodingError("ended early")

        def close(self) -> None:
            self.closed = True

    handler = _Handler()
    upstream = _Upstream()

    _write_chunked_stream(handler, upstream)

    output = handler.wfile.getvalue()
    assert b"response.output_text.delta" in output
    assert b"response.completed" in output
    assert b'"id":"resp_proxy_synthetic"' in output
    assert output.endswith(b"0\r\n\r\n")
    assert upstream.closed is True


def test_write_chunked_stream_emits_error_when_upstream_fails_before_events() -> None:
    class _Handler:
        def __init__(self) -> None:
            self.wfile = io.BytesIO()

    class _Upstream:
        def __init__(self) -> None:
            self.closed = False

        def iter_content(self, chunk_size: int):
            assert chunk_size == 8192
            raise requests.exceptions.ChunkedEncodingError("ended early")
            yield b""

        def close(self) -> None:
            self.closed = True

    handler = _Handler()
    upstream = _Upstream()

    _write_chunked_stream(handler, upstream)

    output = handler.wfile.getvalue()
    assert b"upstream_stream_error" in output
    assert output.endswith(b"0\r\n\r\n")
    assert upstream.closed is True


def test_write_chunked_stream_synthesizes_completion_when_upstream_omits_it() -> None:
    class _Handler:
        def __init__(self) -> None:
            self.wfile = io.BytesIO()

    class _Upstream:
        def __init__(self) -> None:
            self.closed = False

        def iter_content(self, chunk_size: int):
            assert chunk_size == 8192
            yield b"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n"

        def close(self) -> None:
            self.closed = True

    handler = _Handler()
    upstream = _Upstream()

    _write_chunked_stream(handler, upstream)

    output = handler.wfile.getvalue()
    assert b"response.output_text.delta" in output
    assert b"response.completed" in output
    assert b'"id":"resp_proxy_synthetic"' in output
    assert output.endswith(b"0\r\n\r\n")
    assert upstream.closed is True


def test_write_chunked_stream_synthesizes_completion_with_observed_response_id() -> None:
    class _Handler:
        def __init__(self) -> None:
            self.wfile = io.BytesIO()

    class _Upstream:
        def __init__(self) -> None:
            self.closed = False

        def iter_content(self, chunk_size: int):
            assert chunk_size == 8192
            yield (
                b"event: response.created\n"
                b'data: {"type":"response.created","response":{"id":"resp_real_123","model":"qwen3.5-27b",'
                b'"created_at":1770000000,"status":"in_progress","output":[]}}\n\n'
            )
            yield (
                b"event: response.output_text.delta\n"
                b'data: {"type":"response.output_text.delta","response_id":"resp_real_123","delta":"ok"}\n\n'
            )

        def close(self) -> None:
            self.closed = True

    handler = _Handler()
    upstream = _Upstream()

    _write_chunked_stream(handler, upstream)

    output = handler.wfile.getvalue()
    assert b"response.completed" in output
    assert b'"id":"resp_real_123"' in output
    assert b'"model":"qwen3.5-27b"' in output
    assert b'"created_at":1770000000' in output
    assert output.endswith(b"0\r\n\r\n")
    assert upstream.closed is True


def test_proxy_streams_sse_without_buffering_upstream_content(monkeypatch, tmp_path: Path) -> None:
    class _UpstreamResponse:
        status_code = 200
        headers = {"Content-Type": "text/event-stream"}

        @property
        def content(self) -> bytes:
            raise AssertionError("SSE responses must not be buffered through .content")

        def iter_content(self, chunk_size: int):
            assert chunk_size == 8192
            yield b'event: response.completed\ndata: {"type":"response.completed","response":{"output":[]}}\n\n'

        def close(self) -> None:
            return

    def fake_post(*args: object, **kwargs: object) -> _UpstreamResponse:
        assert kwargs["stream"] is True
        return _UpstreamResponse()

    monkeypatch.setattr("lumo_flywheel_serving.inference_proxy.requests.post", fake_post)
    proxy, proxy_thread, proxy_url = _start_server(
        build_proxy_handler("http://upstream.invalid", state_root=tmp_path / "state")
    )
    try:
        response = requests.request(
            "POST",
            f"{proxy_url}/v1/responses",
            json={"model": "qwen3.5-27b", "input": "stream"},
            timeout=10,
        )

        assert response.status_code == 200
        assert b"response.completed" in response.content
    finally:
        proxy.shutdown()
        proxy_thread.join(timeout=5)
        proxy.server_close()


def _activate_request_shaping_bundle(
    *,
    state_root: Path,
    bundle_root: Path,
    request_shaping: dict[str, object],
) -> None:
    bundle = make_tuned_config_bundle(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        weight_version_id="2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        workload_distribution_id="prmj-v1-live",
        vllm_config={
            "max_num_seqs": 2,
            "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 131072,
            "kv_cache_dtype": "fp8_e5m2",
        },
        request_shaping=request_shaping,
        objective={"metric": "eval_throughput", "value": 1.0},
        measurement_trace_ref="measurement.json",
        search_trace_ref="search.json",
        baseline_bundle_id=None,
        regression_guard={},
        safety_rails={},
    )
    bundle_path = persist_tuned_config_bundle(bundle, bundle_root)
    RuntimeStateStore(state_root).activate_bundle(bundle_path, bundle)


def test_proxy_accepts_legacy_target_concurrency_request_shaping(tmp_path: Path) -> None:
    bundle = make_tuned_config_bundle(
        model_id="qwen3.5-27b",
        family_id="proposal-ranking-manager-judgment",
        weight_version_id="2e1b21350ce589fcaafbb3c7d7eac526a7aed582",
        workload_distribution_id="prmj-v1-live",
        vllm_config={
            "max_num_seqs": 4,
            "max_num_batched_tokens": 8192,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.90,
            "max_model_len": 131072,
            "kv_cache_dtype": "fp8_e5m2",
        },
        request_shaping={"target_concurrency": 3},
        objective={"metric": "eval_throughput", "value": 1.0},
        measurement_trace_ref="measurement.json",
        search_trace_ref="search.json",
        baseline_bundle_id=None,
        regression_guard={},
        safety_rails={},
    )
    bundle_path = persist_tuned_config_bundle(bundle, tmp_path / "bundles")

    policy = _normalize_request_shaping_policy(bundle_path, bundle)

    assert policy is not None
    assert policy.concurrency_cap_eval == 3
    assert policy.concurrency_cap_rollout == 0
    assert policy.admission_queue_depth_max == 0


def _start_server(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def test_proxy_enforces_eval_cap_with_queue_full_429(tmp_path: Path) -> None:
    first_upstream_started = threading.Event()

    class _UpstreamHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            first_upstream_started.set()
            time.sleep(0.25)
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    state_root = tmp_path / "state"
    _activate_request_shaping_bundle(
        state_root=state_root,
        bundle_root=tmp_path / "bundles",
        request_shaping={
            "concurrency_cap_eval": 1,
            "concurrency_cap_rollout": 1,
            "admission_queue_depth_max": 0,
            "per_request_kv_budget": 65536,
            "priority_preemption": "strict",
        },
    )
    upstream, upstream_thread, upstream_url = _start_server(_UpstreamHandler)
    proxy, proxy_thread, proxy_url = _start_server(
        build_proxy_handler(upstream_url, state_root=state_root)
    )
    first_response: dict[str, requests.Response] = {}
    first_thread = threading.Thread(
        target=lambda: first_response.setdefault(
            "response",
            requests.post(
                f"{proxy_url}/v1/responses",
                headers={"X-Lumo-Request-Class": "eval"},
                json={"model": "qwen3.5-27b", "input": "first"},
                timeout=10,
            ),
        )
    )
    first_thread.start()
    try:
        assert first_upstream_started.wait(timeout=5)
        second = requests.post(
            f"{proxy_url}/v1/responses",
            headers={"X-Lumo-Request-Class": "eval"},
            json={"model": "qwen3.5-27b", "input": "second"},
            timeout=10,
        )
        first_thread.join(timeout=5)
        assert first_response["response"].status_code == 200
        assert second.status_code == 429
        assert second.headers["Retry-After"] == "1"
        assert second.json()["error"]["code"] == "queue_full"
    finally:
        proxy.shutdown()
        upstream.shutdown()
        first_thread.join(timeout=5)
        proxy_thread.join(timeout=5)
        upstream_thread.join(timeout=5)
        proxy.server_close()
        upstream.server_close()


def test_proxy_routes_eval_and_rollout_to_separate_caps(tmp_path: Path) -> None:
    upstream_started = threading.Barrier(2)

    class _UpstreamHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            upstream_started.wait(timeout=5)
            time.sleep(0.05)
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    state_root = tmp_path / "state"
    _activate_request_shaping_bundle(
        state_root=state_root,
        bundle_root=tmp_path / "bundles",
        request_shaping={
            "concurrency_cap_eval": 1,
            "concurrency_cap_rollout": 1,
            "admission_queue_depth_max": 0,
            "per_request_kv_budget": 65536,
            "priority_preemption": "graceful",
        },
    )
    upstream, upstream_thread, upstream_url = _start_server(_UpstreamHandler)
    proxy, proxy_thread, proxy_url = _start_server(
        build_proxy_handler(upstream_url, state_root=state_root)
    )
    try:
        responses: list[requests.Response] = []

        def post_request(request_class: str) -> None:
            responses.append(
                requests.post(
                    f"{proxy_url}/v1/responses",
                    headers={"X-Lumo-Request-Class": request_class},
                    json={"model": "qwen3.5-27b", "input": request_class},
                    timeout=10,
                )
            )

        eval_thread = threading.Thread(target=post_request, args=("eval",))
        rollout_thread = threading.Thread(target=post_request, args=("rollout",))
        eval_thread.start()
        rollout_thread.start()
        eval_thread.join(timeout=5)
        rollout_thread.join(timeout=5)

        assert sorted(response.status_code for response in responses) == [200, 200]
    finally:
        proxy.shutdown()
        upstream.shutdown()
        proxy_thread.join(timeout=5)
        upstream_thread.join(timeout=5)
        proxy.server_close()
        upstream.server_close()


def test_proxy_records_advisory_fields_without_enforcing_output_cap_or_preemption(tmp_path: Path) -> None:
    captured_payloads: list[dict[str, object]] = []

    class _UpstreamHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
            captured_payloads.append(payload)
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    state_root = tmp_path / "state"
    _activate_request_shaping_bundle(
        state_root=state_root,
        bundle_root=tmp_path / "bundles",
        request_shaping={
            "concurrency_cap_eval": 1,
            "concurrency_cap_rollout": 1,
            "admission_queue_depth_max": 0,
            "per_request_kv_budget": 32768,
            "priority_preemption": "strict",
        },
    )
    upstream, upstream_thread, upstream_url = _start_server(_UpstreamHandler)
    proxy, proxy_thread, proxy_url = _start_server(
        build_proxy_handler(upstream_url, state_root=state_root)
    )
    try:
        response = requests.post(
            f"{proxy_url}/v1/responses",
            headers={"X-Lumo-Request-Class": "rollout"},
            json={"model": "qwen3.5-27b", "input": "rollout", "max_output_tokens": 40000},
            timeout=10,
        )

        assert response.status_code == 200
        assert captured_payloads == [{"model": "qwen3.5-27b", "input": "rollout", "max_output_tokens": 40000}]
    finally:
        proxy.shutdown()
        upstream.shutdown()
        proxy_thread.join(timeout=5)
        upstream_thread.join(timeout=5)
        proxy.server_close()
        upstream.server_close()


# --- Track B per-request capture ---


def test_classify_regime_buckets() -> None:
    assert _classify_regime({"has_tool_call": True}) == "tool-call"
    assert _classify_regime({"text_chars": 4096}) == "summary"
    assert _classify_regime({"text_chars": 100}) == "reasoning"
    assert _classify_regime({}) == "unknown"


def test_extract_response_metadata_captures_usage_tool_call_and_text() -> None:
    ctx: dict[str, object] = {}
    _extract_response_metadata({"type": "response.output_text.delta", "delta": "hello "}, ctx)
    _extract_response_metadata({"type": "response.output_text.delta", "delta": "world"}, ctx)
    _extract_response_metadata(
        {"type": "response.output_item.added", "item": {"type": "function_call"}},
        ctx,
    )
    _extract_response_metadata(
        {"type": "response.completed", "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}},
        ctx,
    )

    assert ctx["text_chars"] == len("hello world")
    assert ctx["has_tool_call"] is True
    assert ctx["usage"] == {"prompt_tokens": 100, "completion_tokens": 50}


def test_track_b_capture_from_env_returns_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("LUMO_TRACK_B_REQUEST_METRICS_OUT", raising=False)
    monkeypatch.delenv("LUMO_TRACK_B_RUNTIME_CONFIG_HASH", raising=False)
    assert TrackBRequestMetricsCapture.from_env("http://upstream.invalid") is None


def test_track_b_capture_from_env_returns_instance(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "track_b_request_metrics.jsonl"
    monkeypatch.setenv("LUMO_TRACK_B_REQUEST_METRICS_OUT", str(out))
    monkeypatch.setenv("LUMO_TRACK_B_RUNTIME_CONFIG_HASH", "sha256:" + "0" * 64)
    capture = TrackBRequestMetricsCapture.from_env("http://upstream.invalid")
    assert capture is not None


def test_track_b_capture_compute_deltas_handles_missing_and_negative() -> None:
    capture = TrackBRequestMetricsCapture(Path("/tmp/unused.jsonl"), "http://upstream.invalid/metrics")
    before = {
        "vllm:spec_decode_num_accepted_tokens_total": 100.0,
        "vllm:spec_decode_num_draft_tokens_total": 200.0,
        "vllm:request_decode_time_seconds_sum": 5.0,
        "vllm:request_prefill_time_seconds_sum": 1.0,
    }
    after = {
        "vllm:spec_decode_num_accepted_tokens_total": 110.0,
        "vllm:spec_decode_num_draft_tokens_total": 220.0,
        "vllm:request_decode_time_seconds_sum": 5.5,
        "vllm:request_prefill_time_seconds_sum": 1.2,
    }
    deltas = capture.compute_deltas(before, after)
    assert deltas["spec_decode_num_accepted_tokens"] == 10.0
    assert deltas["spec_decode_num_draft_tokens"] == 20.0
    assert deltas["decode_sum_s"] == pytest.approx(0.5)
    assert deltas["prefill_sum_s"] == pytest.approx(0.2)

    # Negative delta (counter reset) → None
    after_reset = {
        "vllm:spec_decode_num_accepted_tokens_total": 5.0,
        "vllm:spec_decode_num_draft_tokens_total": 220.0,
        "vllm:request_decode_time_seconds_sum": 5.5,
        "vllm:request_prefill_time_seconds_sum": 1.2,
    }
    deltas_reset = capture.compute_deltas(before, after_reset)
    assert deltas_reset["spec_decode_num_accepted_tokens"] is None

    # Empty before/after → None
    deltas_empty = capture.compute_deltas({}, {})
    assert all(value is None for value in deltas_empty.values())


def test_track_b_capture_record_writes_schema_and_producer(tmp_path: Path) -> None:
    out = tmp_path / "rows.jsonl"
    capture = TrackBRequestMetricsCapture(out, "http://upstream.invalid/metrics", runtime_config_hash="sha256:" + "1" * 64)
    capture.record({"request_id": "resp_abc", "prompt_tokens": 10, "completion_tokens": 5})
    capture.record({"request_id": "resp_def", "prompt_tokens": 11, "completion_tokens": 6})

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    for row in rows:
        assert row["schema"] == TRACK_B_REQUEST_METRICS_SCHEMA
        assert row["producer"] == TRACK_B_REQUEST_METRICS_PRODUCER
        assert row["runtime_config_hash"] == "sha256:" + "1" * 64
    assert rows[0]["request_id"] == "resp_abc"
    assert rows[1]["request_id"] == "resp_def"


def test_proxy_emits_track_b_capture_row_for_streaming_response(monkeypatch, tmp_path: Path) -> None:
    out = tmp_path / "track_b_request_metrics.jsonl"
    monkeypatch.setenv("LUMO_TRACK_B_REQUEST_METRICS_OUT", str(out))
    monkeypatch.setenv("LUMO_TRACK_B_RUNTIME_CONFIG_HASH", "sha256:" + "a" * 64)

    metrics_seq = iter(
        [
            (
                "# HELP vllm:spec_decode_num_accepted_tokens_total foo\n"
                "# TYPE vllm:spec_decode_num_accepted_tokens_total counter\n"
                'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 100.0\n'
                "# HELP vllm:spec_decode_num_draft_tokens_total foo\n"
                "# TYPE vllm:spec_decode_num_draft_tokens_total counter\n"
                'vllm:spec_decode_num_draft_tokens_total{engine="0"} 200.0\n'
                "# HELP vllm:request_decode_time_seconds_sum foo\n"
                "# TYPE vllm:request_decode_time_seconds_sum counter\n"
                'vllm:request_decode_time_seconds_sum{engine="0"} 5.0\n'
                "# HELP vllm:request_prefill_time_seconds_sum foo\n"
                "# TYPE vllm:request_prefill_time_seconds_sum counter\n"
                'vllm:request_prefill_time_seconds_sum{engine="0"} 1.0\n'
            ),
            (
                'vllm:spec_decode_num_accepted_tokens_total{engine="0"} 130.0\n'
                'vllm:spec_decode_num_draft_tokens_total{engine="0"} 250.0\n'
                'vllm:request_decode_time_seconds_sum{engine="0"} 5.6\n'
                'vllm:request_prefill_time_seconds_sum{engine="0"} 1.3\n'
            ),
        ]
    )

    class _MetricsResp:
        status_code = 200

        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return

    class _UpstreamResp:
        status_code = 200
        headers = {"Content-Type": "text/event-stream"}

        def iter_content(self, chunk_size: int):
            yield (
                b'event: response.created\n'
                b'data: {"type":"response.created","response":{"id":"resp_test_123","model":"qwen3.5-27b"}}\n\n'
                b'event: response.output_text.delta\n'
                b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
                b'event: response.completed\n'
                b'data: {"type":"response.completed","response":{"id":"resp_test_123","model":"qwen3.5-27b","usage":{"prompt_tokens":42,"completion_tokens":7}}}\n\n'
            )

        def close(self) -> None:
            return

    def fake_get(url: str, *args: object, **kwargs: object):
        return _MetricsResp(next(metrics_seq))

    def fake_post(*args: object, **kwargs: object) -> _UpstreamResp:
        return _UpstreamResp()

    monkeypatch.setattr("lumo_flywheel_serving.inference_proxy.requests.get", fake_get)
    monkeypatch.setattr("lumo_flywheel_serving.inference_proxy.requests.post", fake_post)

    proxy, proxy_thread, proxy_url = _start_server(
        build_proxy_handler("http://upstream.invalid", state_root=tmp_path / "state")
    )
    try:
        response = requests.request(
            "POST",
            f"{proxy_url}/v1/responses",
            json={"model": "qwen3.5-27b", "input": "ping"},
            timeout=10,
        )
        assert response.status_code == 200
        # Allow a brief moment for the capture to flush
        for _ in range(20):
            if out.is_file() and out.stat().st_size > 0:
                break
            time.sleep(0.05)
    finally:
        proxy.shutdown()
        proxy_thread.join(timeout=5)
        proxy.server_close()

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == TRACK_B_REQUEST_METRICS_SCHEMA
    assert row["producer"] == TRACK_B_REQUEST_METRICS_PRODUCER
    assert row["request_id"] == "resp_test_123"
    assert row["model"] == "qwen3.5-27b"
    assert row["prompt_tokens"] == 42
    assert row["completion_tokens"] == 7
    assert row["spec_decode_num_accepted_tokens"] == 30.0
    assert row["spec_decode_num_draft_tokens"] == 50.0
    assert row["decode_sum_s"] == pytest.approx(0.6)
    assert row["prefill_sum_s"] == pytest.approx(0.3)
    assert row["runtime_config_hash"] == "sha256:" + "a" * 64
    assert row["regime"] in {"reasoning", "unknown"}
    assert row["saw_response_completed"] is True


def test_proxy_does_not_emit_capture_when_env_unset(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LUMO_TRACK_B_REQUEST_METRICS_OUT", raising=False)

    class _UpstreamResp:
        status_code = 200
        headers = {"Content-Type": "text/event-stream"}

        def iter_content(self, chunk_size: int):
            yield b'event: response.completed\ndata: {"type":"response.completed","response":{"id":"x","output":[]}}\n\n'

        def close(self) -> None:
            return

    def fake_post(*args: object, **kwargs: object) -> _UpstreamResp:
        return _UpstreamResp()

    monkeypatch.setattr("lumo_flywheel_serving.inference_proxy.requests.post", fake_post)

    proxy, proxy_thread, proxy_url = _start_server(
        build_proxy_handler("http://upstream.invalid", state_root=tmp_path / "state")
    )
    try:
        response = requests.request(
            "POST",
            f"{proxy_url}/v1/responses",
            json={"model": "qwen3.5-27b", "input": "ping"},
            timeout=10,
        )
        assert response.status_code == 200
    finally:
        proxy.shutdown()
        proxy_thread.join(timeout=5)
        proxy.server_close()
