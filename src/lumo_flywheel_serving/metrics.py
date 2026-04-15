from __future__ import annotations

COUNTER_VARIANTS = {
    "prompt_tokens": ["vllm:prompt_tokens_total", "vllm:prompt_tokens"],
    "generation_tokens": ["vllm:generation_tokens_total", "vllm:generation_tokens"],
    "prefix_cache_queries": ["vllm:prefix_cache_queries_total", "vllm:prefix_cache_queries"],
    "prefix_cache_hits": ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_hits"],
}


def _normalize_metric_key(key: str) -> str:
    return key.split("{", 1)[0]


def parse_prometheus_text(raw: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        key = parts[0]
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        normalized_key = _normalize_metric_key(key)
        metrics[normalized_key] = metrics.get(normalized_key, 0.0) + value
    return metrics


def resolve_metric_schema(metrics_snapshot: dict[str, float]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for logical_name, candidates in COUNTER_VARIANTS.items():
        found = next((candidate for candidate in candidates if candidate in metrics_snapshot), None)
        if found is None:
            raise RuntimeError(
                f"vLLM /metrics does not expose any of {candidates}. Update metric constants for the pinned vLLM version."
            )
        resolved[logical_name] = found
    return resolved


def _required_delta(before: dict[str, float], after: dict[str, float], key: str) -> float:
    if key not in before or key not in after:
        raise RuntimeError(f"Expected metric '{key}' not found in both /metrics snapshots")
    return after[key] - before[key]


def compute_task_metrics(
    before: dict[str, float], after: dict[str, float], schema: dict[str, str]
) -> dict[str, float | None]:
    prompt_tokens = _required_delta(before, after, schema["prompt_tokens"])
    kv_computed_tokens = _required_delta(before, after, "vllm:request_prefill_kv_computed_tokens_sum")
    gen_tokens = _required_delta(before, after, schema["generation_tokens"])
    cache_queries = _required_delta(before, after, schema["prefix_cache_queries"])
    cache_hits = _required_delta(before, after, schema["prefix_cache_hits"])
    ttft_sum = _required_delta(before, after, "vllm:time_to_first_token_seconds_sum")
    ttft_count = _required_delta(before, after, "vllm:time_to_first_token_seconds_count")
    prefill_sum_s = _required_delta(before, after, "vllm:request_prefill_time_seconds_sum")
    decode_sum_s = _required_delta(before, after, "vllm:request_decode_time_seconds_sum")

    return {
        "ttft_ms": (ttft_sum / ttft_count * 1000) if ttft_count > 0 else None,
        "prefill_throughput_tps": (kv_computed_tokens / prefill_sum_s) if prefill_sum_s > 0 else None,
        "decode_throughput_tps": (gen_tokens / decode_sum_s) if decode_sum_s > 0 else None,
        "cache_hit_rate_pct": (cache_hits / cache_queries * 100) if cache_queries > 0 else None,
        "prompt_tokens": prompt_tokens,
        "kv_computed_tokens": kv_computed_tokens,
        "gen_tokens": gen_tokens,
        "prefill_sum_s": prefill_sum_s,
        "decode_sum_s": decode_sum_s,
    }
