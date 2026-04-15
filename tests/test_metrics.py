from lumo_flywheel_serving.metrics import compute_task_metrics, parse_prometheus_text, resolve_metric_schema


def test_parse_prometheus_text_and_schema_resolution() -> None:
    metrics = parse_prometheus_text(
        """
# HELP vllm:prompt_tokens Prompt tokens
vllm:prompt_tokens 10
vllm:generation_tokens_total 4
vllm:prefix_cache_queries_total 2
vllm:prefix_cache_hits 1
vllm:request_prefill_kv_computed_tokens_sum 6
vllm:time_to_first_token_seconds_sum 1.5
vllm:time_to_first_token_seconds_count 2
vllm:request_prefill_time_seconds_sum 2.5
vllm:request_decode_time_seconds_sum 1.0
"""
    )
    assert metrics["vllm:prompt_tokens"] == 10
    assert resolve_metric_schema(metrics) == {
        "prompt_tokens": "vllm:prompt_tokens",
        "generation_tokens": "vllm:generation_tokens_total",
        "prefix_cache_queries": "vllm:prefix_cache_queries_total",
        "prefix_cache_hits": "vllm:prefix_cache_hits",
        "kv_computed_tokens_sum": "vllm:request_prefill_kv_computed_tokens_sum",
        "ttft_seconds_sum": "vllm:time_to_first_token_seconds_sum",
        "ttft_seconds_count": "vllm:time_to_first_token_seconds_count",
        "prefill_seconds_sum": "vllm:request_prefill_time_seconds_sum",
        "decode_seconds_sum": "vllm:request_decode_time_seconds_sum",
    }


def test_parse_prometheus_text_strips_labels_and_accumulates_series() -> None:
    metrics = parse_prometheus_text(
        """
vllm:prompt_tokens_total{model_name="qwen3.5-27b",engine="0"} 10
vllm:generation_tokens_total{model_name="qwen3.5-27b",engine="0"} 4
vllm:prefix_cache_queries_total{model_name="qwen3.5-27b",engine="0"} 3
vllm:prefix_cache_hits_total{model_name="qwen3.5-27b",engine="0"} 1
vllm:request_prefill_kv_computed_tokens_sum{model_name="qwen3.5-27b",engine="0"} 11
vllm:time_to_first_token_seconds_sum{model_name="qwen3.5-27b",engine="0"} 1.8
vllm:time_to_first_token_seconds_count{model_name="qwen3.5-27b",engine="0"} 2
vllm:request_prefill_time_seconds_sum{model_name="qwen3.5-27b",engine="0"} 2.5
vllm:request_prefill_time_seconds_count{model_name="qwen3.5-27b",engine="0"} 3
vllm:request_decode_time_seconds_sum{model_name="qwen3.5-27b",engine="0"} 1.2
"""
    )
    assert metrics["vllm:prompt_tokens_total"] == 10
    assert metrics["vllm:generation_tokens_total"] == 4
    assert metrics["vllm:prefix_cache_queries_total"] == 3
    assert metrics["vllm:prefix_cache_hits_total"] == 1
    assert metrics["vllm:request_prefill_kv_computed_tokens_sum"] == 11
    assert metrics["vllm:time_to_first_token_seconds_sum"] == 1.8
    assert metrics["vllm:time_to_first_token_seconds_count"] == 2
    assert metrics["vllm:request_prefill_time_seconds_sum"] == 2.5
    assert metrics["vllm:request_prefill_time_seconds_count"] == 3
    assert metrics["vllm:request_decode_time_seconds_sum"] == 1.2
    assert resolve_metric_schema(metrics) == {
        "prompt_tokens": "vllm:prompt_tokens_total",
        "generation_tokens": "vllm:generation_tokens_total",
        "prefix_cache_queries": "vllm:prefix_cache_queries_total",
        "prefix_cache_hits": "vllm:prefix_cache_hits_total",
        "kv_computed_tokens_sum": "vllm:request_prefill_kv_computed_tokens_sum",
        "ttft_seconds_sum": "vllm:time_to_first_token_seconds_sum",
        "ttft_seconds_count": "vllm:time_to_first_token_seconds_count",
        "prefill_seconds_sum": "vllm:request_prefill_time_seconds_sum",
        "decode_seconds_sum": "vllm:request_decode_time_seconds_sum",
    }


def test_compute_task_metrics_uses_sum_deltas() -> None:
    before = {
        "vllm:prompt_tokens_total": 100.0,
        "vllm:generation_tokens_total": 40.0,
        "vllm:request_prefill_kv_computed_tokens_sum": 80.0,
        "vllm:prefix_cache_queries": 10.0,
        "vllm:prefix_cache_hits": 5.0,
        "vllm:time_to_first_token_seconds_sum": 2.0,
        "vllm:time_to_first_token_seconds_count": 4.0,
        "vllm:request_prefill_time_seconds_sum": 3.0,
        "vllm:request_decode_time_seconds_sum": 2.0,
    }
    after = {
        "vllm:prompt_tokens_total": 160.0,
        "vllm:generation_tokens_total": 58.0,
        "vllm:request_prefill_kv_computed_tokens_sum": 104.0,
        "vllm:prefix_cache_queries": 16.0,
        "vllm:prefix_cache_hits": 10.0,
        "vllm:time_to_first_token_seconds_sum": 3.5,
        "vllm:time_to_first_token_seconds_count": 7.0,
        "vllm:request_prefill_time_seconds_sum": 5.0,
        "vllm:request_decode_time_seconds_sum": 3.0,
    }
    metrics = compute_task_metrics(
        before=before,
        after=after,
        schema={
            "prompt_tokens": "vllm:prompt_tokens_total",
            "generation_tokens": "vllm:generation_tokens_total",
            "prefix_cache_queries": "vllm:prefix_cache_queries",
            "prefix_cache_hits": "vllm:prefix_cache_hits",
            "kv_computed_tokens_sum": "vllm:request_prefill_kv_computed_tokens_sum",
            "ttft_seconds_sum": "vllm:time_to_first_token_seconds_sum",
            "ttft_seconds_count": "vllm:time_to_first_token_seconds_count",
            "prefill_seconds_sum": "vllm:request_prefill_time_seconds_sum",
            "decode_seconds_sum": "vllm:request_decode_time_seconds_sum",
        },
    )
    assert metrics["prompt_tokens"] == 60.0
    assert metrics["kv_computed_tokens"] == 24.0
    assert metrics["gen_tokens"] == 18.0
    assert metrics["ttft_ms"] == 500.0
    assert metrics["prefill_throughput_tps"] == 12.0
    assert metrics["decode_throughput_tps"] == 18.0
    assert round(metrics["cache_hit_rate_pct"] or 0.0, 2) == round(5.0 / 6.0 * 100, 2)
