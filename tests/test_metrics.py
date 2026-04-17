from __future__ import annotations

from pathlib import Path

from lumo_flywheel_serving.metrics import (
    REQUIRED_METRIC_VARIANTS,
    PendingSnapshot,
    compute_task_metrics,
    parse_prometheus_text,
    resolve_metric_schema,
)


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
vllm:inter_token_latency_seconds_sum 0.5
vllm:request_decode_time_seconds_bucket{le="1"} 7
"""
    )
    assert metrics["vllm:prompt_tokens"] == 10
    assert "vllm:request_decode_time_seconds_bucket" not in metrics
    assert resolve_metric_schema(metrics) == {
        "prompt_tokens": "vllm:prompt_tokens",
        "generation_tokens": "vllm:generation_tokens_total",
        "cache_queries": "vllm:prefix_cache_queries_total",
        "cache_hits": "vllm:prefix_cache_hits",
        "kv_computed_tokens": "vllm:request_prefill_kv_computed_tokens",
        "ttft": "vllm:time_to_first_token_seconds",
        "prefill_time": "vllm:request_prefill_time_seconds",
        "decode_time": "vllm:request_decode_time_seconds",
        "itl": "vllm:inter_token_latency_seconds",
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
vllm:inter_token_latency_seconds_sum{model_name="qwen3.5-27b",engine="0"} 0.4
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
        "cache_queries": "vllm:prefix_cache_queries_total",
        "cache_hits": "vllm:prefix_cache_hits_total",
        "kv_computed_tokens": "vllm:request_prefill_kv_computed_tokens",
        "ttft": "vllm:time_to_first_token_seconds",
        "prefill_time": "vllm:request_prefill_time_seconds",
        "decode_time": "vllm:request_decode_time_seconds",
        "itl": "vllm:inter_token_latency_seconds",
    }


def test_parse_prometheus_text_against_live_vllm_fixture() -> None:
    raw = (Path(__file__).parent / "fixtures" / "vllm_metrics_qwen3.5-27b.prom").read_text(encoding="utf-8")

    # The fixture is a real /metrics capture, so bucket lines must be present
    # in the raw text and absent from the parsed output.
    assert "_bucket" in raw

    metrics = parse_prometheus_text(raw)
    schema = resolve_metric_schema(metrics)

    assert not any(key.endswith("_bucket") for key in metrics)

    for candidates in REQUIRED_METRIC_VARIANTS.values():
        assert any(
            candidate in metrics
            or f"{candidate}_sum" in metrics
            or f"{candidate}_count" in metrics
            for candidate in candidates
        )

    for key, value in metrics.items():
        if key.startswith("vllm:") and (key.endswith("_sum") or key.endswith("_count")):
            assert value >= 0

    histogram_keys = {"kv_computed_tokens", "ttft", "prefill_time", "decode_time", "itl"}
    for logical_name, base in schema.items():
        if logical_name in histogram_keys:
            assert f"{base}_sum" in metrics
            assert f"{base}_count" in metrics
        else:
            assert base in metrics


def test_compute_task_metrics_uses_pending_snapshot_contract() -> None:
    pending = PendingSnapshot(
        task_id="family-a/v1",
        seed=7,
        attempt=2,
        snapshot={
            "vllm:prompt_tokens_total": 100.0,
            "vllm:generation_tokens_total": 40.0,
            "vllm:request_prefill_kv_computed_tokens_sum": 80.0,
            "vllm:prefix_cache_queries_total": 10.0,
            "vllm:prefix_cache_hits_total": 5.0,
            "vllm:time_to_first_token_seconds_sum": 2.0,
            "vllm:time_to_first_token_seconds_count": 4.0,
            "vllm:request_prefill_time_seconds_sum": 3.0,
            "vllm:request_decode_time_seconds_sum": 2.0,
        },
        timestamp=10.0,
    )
    after = {
        "vllm:prompt_tokens_total": 160.0,
        "vllm:generation_tokens_total": 58.0,
        "vllm:request_prefill_kv_computed_tokens_sum": 104.0,
        "vllm:prefix_cache_queries_total": 16.0,
        "vllm:prefix_cache_hits_total": 10.0,
        "vllm:time_to_first_token_seconds_sum": 3.5,
        "vllm:time_to_first_token_seconds_count": 7.0,
        "vllm:request_prefill_time_seconds_sum": 5.0,
        "vllm:request_decode_time_seconds_sum": 3.0,
    }
    metrics = compute_task_metrics(
        pending,
        after,
        14.5,
        {
            "prompt_tokens": "vllm:prompt_tokens_total",
            "generation_tokens": "vllm:generation_tokens_total",
            "cache_queries": "vllm:prefix_cache_queries_total",
            "cache_hits": "vllm:prefix_cache_hits_total",
            "kv_computed_tokens": "vllm:request_prefill_kv_computed_tokens",
            "ttft": "vllm:time_to_first_token_seconds",
            "prefill_time": "vllm:request_prefill_time_seconds",
            "decode_time": "vllm:request_decode_time_seconds",
            "itl": "vllm:inter_token_latency_seconds",
        },
        model_id="qwen3.5-27b",
        track="codex_long",
        pool_or_split="train_long",
    )
    assert metrics.task_id == "family-a/v1"
    assert metrics.seed == 7
    assert metrics.attempt == 2
    assert metrics.prompt_tokens == 60.0
    assert metrics.kv_computed_tokens == 24.0
    assert metrics.gen_tokens == 18.0
    assert metrics.ttft_ms == 500.0
    assert metrics.prefill_throughput_tps == 12.0
    assert metrics.decode_throughput_tps == 18.0
    assert round(metrics.cache_hit_rate_pct or 0.0, 2) == round(5.0 / 6.0 * 100, 2)
    assert metrics.wall_clock_s == 4.5
    assert metrics.anomalies == []


def test_compute_task_metrics_rejects_counter_resets() -> None:
    before = {
        "vllm:prompt_tokens_total": 100.0,
        "vllm:generation_tokens_total": 40.0,
        "vllm:request_prefill_kv_computed_tokens_sum": 80.0,
        "vllm:prefix_cache_queries_total": 10.0,
        "vllm:prefix_cache_hits_total": 5.0,
        "vllm:time_to_first_token_seconds_sum": 2.0,
        "vllm:time_to_first_token_seconds_count": 4.0,
        "vllm:request_prefill_time_seconds_sum": 3.0,
        "vllm:request_decode_time_seconds_sum": 2.0,
    }
    after = dict(before)
    after["vllm:request_decode_time_seconds_sum"] = 1.0

    try:
        compute_task_metrics(
            before=before,
            after=after,
            schema={
                "prompt_tokens": "vllm:prompt_tokens_total",
                "generation_tokens": "vllm:generation_tokens_total",
                "cache_queries": "vllm:prefix_cache_queries_total",
                "cache_hits": "vllm:prefix_cache_hits_total",
                "kv_computed_tokens": "vllm:request_prefill_kv_computed_tokens",
                "ttft": "vllm:time_to_first_token_seconds",
                "prefill_time": "vllm:request_prefill_time_seconds",
                "decode_time": "vllm:request_decode_time_seconds",
                "itl": "vllm:inter_token_latency_seconds",
            },
        )
    except RuntimeError as exc:
        assert "decreased between /metrics snapshots" in str(exc)
    else:
        raise AssertionError("Expected compute_task_metrics() to reject counter resets")
