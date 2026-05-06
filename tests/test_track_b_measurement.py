from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import requests


def _load_measure_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "measure_track_b_real_workload.py"
    spec = importlib.util.spec_from_file_location("measure_track_b_real_workload", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_http_error_preserves_response_body() -> None:
    measure = _load_measure_module()
    response = requests.Response()
    response.status_code = 500
    response.reason = "Internal Server Error"
    response.url = "http://127.0.0.1:9950/v1/responses"
    response._content = b'{"error":"spec decode failed"}'

    with pytest.raises(requests.HTTPError) as exc_info:
        measure._raise_for_status_with_body(response)

    message = str(exc_info.value)
    assert "500 Server Error" in message
    assert "response_body=" in message
    assert "spec decode failed" in message


def test_metric_summary_records_wall_decode_throughput() -> None:
    measure = _load_measure_module()
    before = {
        "vllm:prompt_tokens_total": 0.0,
        "vllm:generation_tokens_total": 0.0,
        "vllm:request_prefill_kv_computed_tokens_sum": 0.0,
        "vllm:time_to_first_token_seconds_sum": 0.0,
        "vllm:time_to_first_token_seconds_count": 0.0,
        "vllm:request_prefill_time_seconds_sum": 0.0,
        "vllm:request_decode_time_seconds_sum": 0.0,
        "vllm:inter_token_latency_seconds_sum": 0.0,
        "vllm:prefix_cache_queries_total": 0.0,
        "vllm:prefix_cache_hits_total": 0.0,
    }
    after = {
        **before,
        "vllm:prompt_tokens_total": 100.0,
        "vllm:generation_tokens_total": 40.0,
        "vllm:request_prefill_kv_computed_tokens_sum": 100.0,
        "vllm:request_prefill_time_seconds_sum": 2.0,
        "vllm:request_decode_time_seconds_sum": 8.0,
    }

    summary = measure._metric_summary(before, after, request_count=4, elapsed_s=2.0)

    assert summary["step_consumption"]["decode_tokens_per_s"] == 5.0
    assert summary["step_consumption"]["wall_decode_tokens_per_s"] == 20.0
