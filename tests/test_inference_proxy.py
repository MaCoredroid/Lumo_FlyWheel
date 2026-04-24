from __future__ import annotations

import io
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from lumo_flywheel_serving.inference_proxy import (
    _write_chunked_stream,
    is_inference_path,
    normalize_responses_request_payload,
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
