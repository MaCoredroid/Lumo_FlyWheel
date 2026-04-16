from __future__ import annotations

from lumo_flywheel_serving.inference_proxy import (
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
