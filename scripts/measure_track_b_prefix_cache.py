#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


def _metrics(port: int) -> dict[str, float]:
    text = requests.get(f"http://127.0.0.1:{port}/metrics", timeout=20).text
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in {
            "vllm:prefix_cache_queries_total",
            "vllm:prefix_cache_hits_total",
            "vllm:generation_tokens_total",
            "vllm:prompt_tokens_total",
        }:
            continue
        try:
            values[name] = values.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return values


def _delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return float(after.get(key, 0.0) - before.get(key, 0.0))


def _completion(
    *,
    port: int,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": 0,
        },
        timeout=180,
    )
    wall_s = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    completion_tokens = int(usage.get("completion_tokens") or 0)
    return {
        "wall_s": wall_s,
        "completion_tokens": completion_tokens,
        "wall_output_tps": completion_tokens / wall_s if wall_s > 0 and completion_tokens else 0.0,
        "usage": usage,
    }


def measure(args: argparse.Namespace) -> dict[str, Any]:
    health = requests.get(f"http://127.0.0.1:{args.port}/health", timeout=20)
    health.raise_for_status()
    if args.reset_prefix_cache:
        requests.post(
            f"http://127.0.0.1:{args.port}/reset_prefix_cache",
            headers={"Authorization": f"Bearer {args.api_key}"},
            timeout=20,
        ).raise_for_status()
    shared_prefix = " ".join(["trackb-prefix-cache"] * args.prefix_words)
    before = _metrics(args.port)
    requests_payloads: list[dict[str, Any]] = []
    serial_turns = max(1, args.turns)
    for index in range(serial_turns):
        prompt = _prompt(shared_prefix=shared_prefix, index=index)
        requests_payloads.append(
            _completion(
                port=args.port,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
        )
    batch_payloads: list[dict[str, Any]] = []
    batch_wall_s: float | None = None
    if args.concurrent_requests > 0:
        batch_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
            futures = {
                executor.submit(
                    _completion,
                    port=args.port,
                    api_key=args.api_key,
                    model=args.model,
                    prompt=_prompt(
                        shared_prefix=shared_prefix,
                        index=serial_turns + batch_index,
                    ),
                    max_tokens=args.max_tokens,
                ): batch_index
                for batch_index in range(args.concurrent_requests)
            }
            for future in as_completed(futures):
                payload = future.result()
                payload["batch_index"] = futures[future]
                batch_payloads.append(payload)
        batch_wall_s = time.perf_counter() - batch_started
        batch_payloads.sort(key=lambda row: int(row["batch_index"]))
    after = _metrics(args.port)
    queries = _delta(after, before, "vllm:prefix_cache_queries_total")
    hits = _delta(after, before, "vllm:prefix_cache_hits_total")
    warm = requests_payloads[1:] if len(requests_payloads) > 1 else requests_payloads
    warm_tokens = sum(float(row["completion_tokens"]) for row in warm)
    warm_wall = sum(float(row["wall_s"]) for row in warm)
    batch_tokens = sum(float(row["completion_tokens"]) for row in batch_payloads)
    batch_decode_tps = (
        batch_tokens / batch_wall_s
        if batch_wall_s is not None and batch_wall_s > 0 and batch_tokens
        else None
    )
    serial_decode_tps = warm_tokens / warm_wall if warm_wall > 0 and warm_tokens else None
    best_decode_tps = max(
        [value for value in (serial_decode_tps, batch_decode_tps) if value is not None],
        default=None,
    )
    return {
        "schema": "lumo.track_b.prefix_cache_measurement.v1",
        "measured_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model": args.model,
        "port": args.port,
        "turns": args.turns,
        "prefix_words": args.prefix_words,
        "max_tokens": args.max_tokens,
        "concurrent_requests": args.concurrent_requests,
        "requests": requests_payloads,
        "batch_requests": batch_payloads,
        "batch_wall_s": batch_wall_s,
        "metrics_before": before,
        "metrics_after": after,
        "metric_deltas": {
            "prefix_cache_queries": queries,
            "prefix_cache_hits": hits,
            "prefix_cache_hit_rate": hits / queries if queries > 0 else None,
            "generation_tokens": _delta(after, before, "vllm:generation_tokens_total"),
            "prompt_tokens": _delta(after, before, "vllm:prompt_tokens_total"),
        },
        "serial_decode_tps": serial_decode_tps,
        "batch_decode_tps": batch_decode_tps,
        "decode_tps": best_decode_tps,
        "candidate_status": "measured",
    }


def _prompt(*, shared_prefix: str, index: int) -> str:
    return (
        "Use this shared context without repeating it.\n"
        f"{shared_prefix}\n\n"
        f"Turn {index}: answer with one short sentence about prefix caching."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Track B Round 0 prefix-cache behavior on a live vLLM endpoint.")
    parser.add_argument("--port", type=int, default=9950)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=0,
        help="Optional warm concurrent batch after serial cache warmup; decode_tps records the best serial or aggregate batch rate.",
    )
    parser.add_argument("--prefix-words", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--reset-prefix-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.turns < 1:
        raise RuntimeError("--turns must be >= 1")
    if args.concurrent_requests < 0:
        raise RuntimeError("--concurrent-requests must be >= 0")
    result = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
