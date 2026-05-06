#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumo_flywheel_serving.metrics import compute_task_metrics, parse_prometheus_text, resolve_metric_schema  # noqa: E402


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _raise_for_status_with_body(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text.strip().replace("\n", "\\n")
        if len(body) > 1200:
            body = body[:1200] + "...<truncated>"
        detail = f"{exc}; response_body={body or '<empty>'}"
        raise requests.HTTPError(detail, response=response) from exc


def _load_workload(path: Path) -> tuple[dict[str, Any], Path]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"workload file must be a YAML mapping: {path}")
    seed_ref = str(payload.get("seed_trace_ref") or "")
    if not seed_ref:
        raise RuntimeError(f"workload file is missing seed_trace_ref: {path}")
    seed_path = Path(seed_ref)
    if not seed_path.is_absolute():
        seed_path = path.parent / seed_path
    if not seed_path.is_file():
        raise RuntimeError(f"seed trace is missing: {seed_path}")
    return payload, seed_path


def _load_seed(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError(f"seed row must be a JSON object: {line}")
        rows.append(payload)
    if not rows:
        raise RuntimeError(f"seed trace is empty: {path}")
    return rows


def _metrics(metrics_url: str) -> dict[str, float]:
    response = requests.get(metrics_url, timeout=20)
    _raise_for_status_with_body(response)
    return parse_prometheus_text(response.text)


def _metric_summary(before: dict[str, float], after: dict[str, float], *, request_count: int, elapsed_s: float) -> dict[str, Any]:
    try:
        schema = resolve_metric_schema(after)
        metrics = compute_task_metrics(before=before, after=after, schema=schema)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"metrics_unavailable: {exc}",
            "request_count": request_count,
            "elapsed_s": round(elapsed_s, 6),
        }
    gen_tokens = float(metrics.get("gen_tokens") or 0.0)
    decode_sum_s = float(metrics.get("decode_sum_s") or 0.0)
    prefill_sum_s = float(metrics.get("prefill_sum_s") or 0.0)
    prompt_tokens = float(metrics.get("prompt_tokens") or 0.0)
    kv_tokens = float(metrics.get("kv_computed_tokens") or 0.0)
    return {
        "available": True,
        "request_count": request_count,
        "elapsed_s": round(elapsed_s, 6),
        "metrics_delta": {
            "prompt_tokens": prompt_tokens,
            "kv_computed_tokens": kv_tokens,
            "generation_tokens": gen_tokens,
            "prefill_sum_s": prefill_sum_s,
            "decode_sum_s": decode_sum_s,
            "ttft_sum_s": float(metrics.get("ttft_sum_s") or 0.0),
            "ttft_count": int(metrics.get("ttft_count") or 0),
            "cache_queries": float(metrics.get("cache_queries") or 0.0),
            "cache_hits": float(metrics.get("cache_hits") or 0.0),
        },
        "step_consumption": {
            "prompt_tokens_per_request": round(prompt_tokens / max(request_count, 1), 3),
            "generation_tokens_per_request": round(gen_tokens / max(request_count, 1), 3),
            "prefill_ms_per_kv_token": round(prefill_sum_s * 1000.0 / kv_tokens, 6) if kv_tokens > 0 else None,
            "decode_ms_per_generated_token": round(decode_sum_s * 1000.0 / gen_tokens, 6) if gen_tokens > 0 else None,
            "decode_tokens_per_s": round(gen_tokens / decode_sum_s, 6) if decode_sum_s > 0 else None,
            "cache_hit_rate_pct": metrics.get("cache_hit_rate_pct"),
        },
        "bottleneck_hint": "decode" if decode_sum_s >= prefill_sum_s else "prefill",
    }


def _aggregate_metric_summaries(summaries: list[dict[str, Any]], *, request_count: int, elapsed_s: float) -> dict[str, Any]:
    if not summaries or not all(bool(row.get("available")) for row in summaries):
        return {
            "available": False,
            "reason": "one_or_more_window_metric_summaries_unavailable",
            "request_count": request_count,
            "elapsed_s": round(elapsed_s, 6),
        }
    totals = {
        "prompt_tokens": 0.0,
        "kv_computed_tokens": 0.0,
        "generation_tokens": 0.0,
        "prefill_sum_s": 0.0,
        "decode_sum_s": 0.0,
        "ttft_sum_s": 0.0,
        "ttft_count": 0.0,
        "cache_queries": 0.0,
        "cache_hits": 0.0,
    }
    for summary in summaries:
        delta = summary.get("metrics_delta") if isinstance(summary.get("metrics_delta"), dict) else {}
        for key in totals:
            totals[key] += float(delta.get(key) or 0.0)
    gen_tokens = totals["generation_tokens"]
    decode_sum_s = totals["decode_sum_s"]
    prefill_sum_s = totals["prefill_sum_s"]
    kv_tokens = totals["kv_computed_tokens"]
    cache_queries = totals["cache_queries"]
    return {
        "available": True,
        "request_count": request_count,
        "elapsed_s": round(elapsed_s, 6),
        "metrics_delta": totals,
        "step_consumption": {
            "prompt_tokens_per_request": round(totals["prompt_tokens"] / max(request_count, 1), 3),
            "generation_tokens_per_request": round(gen_tokens / max(request_count, 1), 3),
            "prefill_ms_per_kv_token": round(prefill_sum_s * 1000.0 / kv_tokens, 6) if kv_tokens > 0 else None,
            "decode_ms_per_generated_token": round(decode_sum_s * 1000.0 / gen_tokens, 6) if gen_tokens > 0 else None,
            "decode_tokens_per_s": round(gen_tokens / decode_sum_s, 6) if decode_sum_s > 0 else None,
            "cache_hit_rate_pct": (totals["cache_hits"] / cache_queries * 100.0) if cache_queries > 0 else None,
        },
        "bottleneck_hint": "decode" if decode_sum_s >= prefill_sum_s else "prefill",
    }


def _post_response(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    entry: dict[str, Any],
    request_id: str,
    prompt_token_cap: int,
    max_output_token_cap: int | None,
) -> dict[str, Any]:
    prompt_tokens = int(entry.get("prompt_tokens") or 1)
    if prompt_token_cap > 0:
        prompt_tokens = min(prompt_tokens, prompt_token_cap)
    requested_output = int(
        entry.get("request_max_output_tokens")
        or entry.get("output_tokens")
        or entry.get("thinking_tokens")
        or 1
    )
    output_tokens = max(1, requested_output)
    if max_output_token_cap is not None:
        output_tokens = min(output_tokens, max_output_token_cap)
    prompt = " ".join(["token"] * max(prompt_tokens, 1))
    started = time.monotonic()
    response = requests.post(
        f"{endpoint.rstrip('/')}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Lumo-Request-Class": str(entry.get("class") or entry.get("request_class") or "eval"),
        },
        json={
            "model": model,
            "input": prompt,
            "max_output_tokens": output_tokens,
        },
        timeout=max(60, output_tokens * 3),
    )
    wall_s = time.monotonic() - started
    _raise_for_status_with_body(response)
    payload = response.json()
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    return {
        "request_id": request_id,
        "prompt_tokens_requested": prompt_tokens,
        "max_output_tokens": output_tokens,
        "wall_s": round(wall_s, 6),
        "usage": usage,
        "status": payload.get("status") if isinstance(payload, dict) else None,
    }


def _run_warm_batch(
    *,
    entries: list[dict[str, Any]],
    endpoint: str,
    api_key: str,
    model: str,
    window_id: int,
    warm_concurrency: int,
    prompt_token_cap: int,
    max_output_token_cap: int | None,
) -> list[dict[str, Any]]:
    max_workers = max(1, min(warm_concurrency, len(entries)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _post_response,
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                entry=entry,
                request_id=f"window-{window_id:03d}-warm-{index + 1:03d}",
                prompt_token_cap=prompt_token_cap,
                max_output_token_cap=max_output_token_cap,
            ): index
            for index, entry in enumerate(entries)
        }
        rows: list[dict[str, Any]] = []
        for future in as_completed(futures):
            row = future.result()
            row["warm_index"] = futures[future] + 1
            rows.append(row)
    return sorted(rows, key=lambda row: int(row["warm_index"]))


def measure(args: argparse.Namespace) -> dict[str, Any]:
    workload, seed_path = _load_workload(args.workload_file)
    seed_rows = _load_seed(seed_path)
    health = requests.get(args.health_url, timeout=20)
    _raise_for_status_with_body(health)
    if args.reset_prefix_cache:
        reset = requests.post(args.reset_prefix_cache_url, headers={"Authorization": f"Bearer {args.api_key}"}, timeout=30)
        _raise_for_status_with_body(reset)
    max_output_cap = args.max_output_token_cap if args.max_output_token_cap > 0 else None
    warm_per_window = args.completions_per_task - args.cold_completions
    if warm_per_window < 1:
        raise RuntimeError("completions_per_task minus cold_completions must leave at least one warm completion")
    windows: list[dict[str, Any]] = []
    warm_started = time.monotonic()
    for window_index in range(args.task_count):
        base = window_index * args.completions_per_task
        cold_rows: list[dict[str, Any]] = []
        for cold_index in range(args.cold_completions):
            entry = seed_rows[(base + cold_index) % len(seed_rows)]
            cold_rows.append(
                _post_response(
                    endpoint=args.endpoint,
                    api_key=args.api_key,
                    model=args.model,
                    entry=entry,
                    request_id=f"window-{window_index + 1:03d}-cold-{cold_index + 1:03d}",
                    prompt_token_cap=args.prompt_token_cap,
                    max_output_token_cap=max_output_cap,
                )
            )
        before_window_warm = _metrics(args.metrics_url)
        warm_entries = [
            seed_rows[(base + args.cold_completions + offset) % len(seed_rows)]
            for offset in range(warm_per_window)
        ]
        window_warm_started = time.monotonic()
        warm_rows = _run_warm_batch(
            entries=warm_entries,
            endpoint=args.endpoint,
            api_key=args.api_key,
            model=args.model,
            window_id=window_index + 1,
            warm_concurrency=args.warm_concurrency,
            prompt_token_cap=args.prompt_token_cap,
            max_output_token_cap=max_output_cap,
        )
        window_warm_elapsed = time.monotonic() - window_warm_started
        after_window_warm = _metrics(args.metrics_url)
        warm_summary = _metric_summary(
            before_window_warm,
            after_window_warm,
            request_count=len(warm_rows),
            elapsed_s=window_warm_elapsed,
        )
        windows.append(
            {
                "window_index": window_index + 1,
                "cold_completions": cold_rows,
                "warm_completions": warm_rows,
                "warm_metrics_consumption": warm_summary,
            }
        )
    warm_elapsed = time.monotonic() - warm_started
    aggregate = _aggregate_metric_summaries(
        [
            row["warm_metrics_consumption"]
            for row in windows
            if isinstance(row.get("warm_metrics_consumption"), dict)
        ],
        request_count=args.task_count * warm_per_window,
        elapsed_s=warm_elapsed,
    )
    decode_tps = None
    if aggregate.get("available"):
        step = aggregate.get("step_consumption") if isinstance(aggregate.get("step_consumption"), dict) else {}
        decode_tps = step.get("decode_tokens_per_s")
    speedup = float(decode_tps) / args.baseline_decode_tps if decode_tps is not None else None
    target_tps = args.baseline_decode_tps * args.target_multiplier
    return {
        "schema": "lumo.track_b.real_workload_first_five.v1",
        "measured_at": _now(),
        "source_reports": {
            "track_b": "docs/reports/auto_research/l0-warm-decode-quality-bounded-track-20260505.md",
            "l0c_cutlass": "docs/reports/auto_research/l0c-cutlass-round-20260505T204655Z.md",
        },
        "endpoint": args.endpoint,
        "metrics_url": args.metrics_url,
        "model": args.model,
        "workload_file": str(args.workload_file),
        "seed_trace": str(seed_path),
        "workload_distribution_id": workload.get("workload_distribution_id"),
        "task_count": args.task_count,
        "completions_per_task": args.completions_per_task,
        "cold_completions_discarded": args.cold_completions,
        "warm_completions_measured": warm_per_window,
        "warm_concurrency": args.warm_concurrency,
        "prompt_token_cap": args.prompt_token_cap,
        "max_output_token_cap": max_output_cap,
        "baseline_decode_tps": args.baseline_decode_tps,
        "target_multiplier": args.target_multiplier,
        "target_decode_tps": target_tps,
        "decode_tps": decode_tps,
        "warm_decode_tps": decode_tps,
        "speedup_vs_baseline": speedup,
        "pass": bool(decode_tps is not None and float(decode_tps) >= target_tps),
        "aggregate_warm_metrics_consumption": aggregate,
        "windows": windows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Track B on the real L0c workload: first 5 completions, last 4 warm.")
    parser.add_argument("--workload-file", type=Path, default=REPO_ROOT / "benchmark_blueprints" / "workloads" / "responses-sdk-adapter-cutover-heavy" / "workload.yaml")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--reset-prefix-cache-url", default="http://127.0.0.1:9950/reset_prefix_cache")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--completions-per-task", type=int, default=5)
    parser.add_argument("--cold-completions", type=int, default=1)
    parser.add_argument("--warm-concurrency", type=int, default=4)
    parser.add_argument("--prompt-token-cap", type=int, default=0)
    parser.add_argument("--max-output-token-cap", type=int, default=0)
    parser.add_argument("--baseline-decode-tps", type=float, default=7.5)
    parser.add_argument("--target-multiplier", type=float, default=5.0)
    parser.add_argument("--reset-prefix-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.task_count < 1:
        raise RuntimeError("--task-count must be >= 1")
    if args.completions_per_task < 2:
        raise RuntimeError("--completions-per-task must be >= 2")
    if args.cold_completions < 0:
        raise RuntimeError("--cold-completions must be >= 0")
    if args.warm_concurrency < 1:
        raise RuntimeError("--warm-concurrency must be >= 1")
    if args.baseline_decode_tps <= 0:
        raise RuntimeError("--baseline-decode-tps must be > 0")
    if args.target_multiplier <= 1:
        raise RuntimeError("--target-multiplier must be > 1")
    result = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("pass") else 2


if __name__ == "__main__":
    raise SystemExit(main())
