from __future__ import annotations

import io

from lumo_flywheel_serving.inference_proxy import (
    _write_chunked_stream,
    is_inference_path,
    normalize_responses_request_payload,
)


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
