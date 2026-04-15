from lumo_flywheel_serving.metrics import compute_task_metrics, parse_prometheus_text, resolve_metric_schema


def test_parse_prometheus_text_and_schema_resolution() -> None:
    metrics = parse_prometheus_text(
        """
# HELP vllm:prompt_tokens Prompt tokens
vllm:prompt_tokens 10
vllm:generation_tokens_total 4
vllm:prefix_cache_hits 1
"""
    )
    assert metrics["vllm:prompt_tokens"] == 10
    assert resolve_metric_schema(metrics) == {
        "prompt_tokens": "vllm:prompt_tokens",
        "generation_tokens": "vllm:generation_tokens_total",
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
        },
    )
    assert metrics["prompt_tokens"] == 60.0
    assert metrics["kv_computed_tokens"] == 24.0
    assert metrics["gen_tokens"] == 18.0
    assert metrics["ttft_ms"] == 500.0
    assert metrics["prefill_throughput_tps"] == 12.0
    assert metrics["decode_throughput_tps"] == 18.0
    assert round(metrics["cache_hit_rate_pct"] or 0.0, 2) == round(5.0 / 6.0 * 100, 2)
