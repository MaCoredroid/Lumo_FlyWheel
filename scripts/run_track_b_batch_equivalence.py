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
    choices = payload.get("choices") if isinstance(payload, dict) else []
    first = choices[0] if isinstance(choices, list) and choices else {}
    text = first.get("text") if isinstance(first, dict) else ""
    usage = payload.get("usage") if isinstance(payload, dict) else {}
    return {
        "text": text if isinstance(text, str) else "",
        "usage": usage if isinstance(usage, dict) else {},
        "wall_s": wall_s,
    }


def _prompt(index: int, shared_prefix: str) -> str:
    return (
        "Context marker follows. Do not continue the marker text.\n"
        f"{shared_prefix}\n\n"
        f"Probe {index}. Complete the next line with exactly the word OK.\nAnswer: OK"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    requests.get(f"http://127.0.0.1:{args.port}/health", timeout=20).raise_for_status()
    if args.reset_prefix_cache:
        requests.post(
            f"http://127.0.0.1:{args.port}/reset_prefix_cache",
            headers={"Authorization": f"Bearer {args.api_key}"},
            timeout=20,
        ).raise_for_status()
    shared_prefix = " ".join(["trackb-equivalence"] * args.prefix_words)
    prompts = [_prompt(index, shared_prefix) for index in range(args.prompt_count)]

    serial = [
        _completion(
            port=args.port,
            api_key=args.api_key,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
        )
        for prompt in prompts
    ]
    with ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
        futures = {
            executor.submit(
                _completion,
                port=args.port,
                api_key=args.api_key,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
            ): index
            for index, prompt in enumerate(prompts)
        }
        concurrent: list[dict[str, Any] | None] = [None] * len(prompts)
        for future in as_completed(futures):
            concurrent[futures[future]] = future.result()
    concurrent_complete = [row or {"text": "", "usage": {}, "wall_s": None} for row in concurrent]
    comparisons = []
    for index, (serial_row, concurrent_row) in enumerate(zip(serial, concurrent_complete, strict=True)):
        comparisons.append(
            {
                "index": index,
                "match": serial_row["text"] == concurrent_row["text"],
                "serial_text": serial_row["text"],
                "concurrent_text": concurrent_row["text"],
                "serial_usage": serial_row["usage"],
                "concurrent_usage": concurrent_row["usage"],
            }
        )
    match_count = sum(1 for row in comparisons if row["match"])
    return {
        "schema": "lumo.track_b.batch_equivalence_gate.v1",
        "measured_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model": args.model,
        "port": args.port,
        "prompt_count": args.prompt_count,
        "concurrent_requests": args.concurrent_requests,
        "match_count": match_count,
        "match_rate": match_count / len(comparisons) if comparisons else 0.0,
        "pass": bool(comparisons) and match_count == len(comparisons),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic serial-vs-batched equivalence gate for Track B batching.")
    parser.add_argument("--port", type=int, default=9950)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--concurrent-requests", type=int, default=4)
    parser.add_argument("--prefix-words", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--reset-prefix-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prompt_count < 1:
        raise RuntimeError("--prompt-count must be >= 1")
    if args.concurrent_requests < 1:
        raise RuntimeError("--concurrent-requests must be >= 1")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
