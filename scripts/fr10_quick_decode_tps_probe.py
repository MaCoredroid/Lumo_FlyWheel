#!/usr/bin/env python3
"""Quick FR10 B4 decode-TPS probe for homogeneous decode modes.

This is the cheap starting-point speed read, not the SWE-bench verdict. It
expects a live shared FR10 cu130 server and runs one mode at a time so the
per-request decode-mode isolation invariant is preserved.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:9950"
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_PROMPTS = [
    "Write a Python function to reverse a singly linked list, then explain it.",
    "Explain what a hash table is and how collisions are handled, in detail.",
    "Implement binary search over a sorted list in Python with comments.",
    "What is the time complexity of merge sort and why? Walk through it.",
    "Write a Python function to check if a string is a palindrome, with tests.",
    "Describe the difference between a process and a thread, with examples.",
    "Implement a Python decorator that times function execution and logs it.",
    "Explain how a B-tree differs from a binary search tree and when to use each.",
]
METRIC_NAMES = {
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:request_decode_time_seconds_sum": "decode_seconds",
    "vllm:request_prefill_time_seconds_sum": "prefill_seconds",
    "vllm:iteration_tokens_total_sum": "iteration_tokens_sum",
    "vllm:iteration_tokens_total_count": "iteration_tokens_count",
    "vllm:spec_decode_num_accepted_tokens_total": "spec_accepted_tokens",
    "vllm:spec_decode_num_draft_tokens_total": "spec_draft_tokens",
    "vllm:spec_decode_num_drafts_total": "spec_drafts",
}


def _headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def _post_json(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else None


def _get_text(endpoint: str, path: str, timeout: float) -> str:
    with urllib.request.urlopen(endpoint.rstrip("/") + path, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _wait_health(endpoint: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            _get_text(endpoint, "/health", timeout=5)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s: {last_exc}")


def _scrape_metrics(endpoint: str) -> dict[str, float]:
    text = _get_text(endpoint, "/metrics", timeout=10)
    out = {value: 0.0 for value in METRIC_NAMES.values()}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        key = METRIC_NAMES.get(name)
        if key is None:
            continue
        try:
            out[key] += float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
    return out


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: float(after.get(key, 0.0) - before.get(key, 0.0)) for key in after}


def _read_prompts(path: str | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts


def _reset_prefix_cache(endpoint: str) -> str | None:
    try:
        _post_json(endpoint, "/reset_prefix_cache", {}, timeout=30)
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def _run_requests(
    *,
    endpoint: str,
    model: str,
    prompts: list[str],
    mode: str,
    samples_per_prompt: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    request_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for prompt_id, prompt in enumerate(prompts):
        next_sample = 0
        while next_sample < samples_per_prompt:
            n = min(batch_size, samples_per_prompt - next_sample)
            payload: dict[str, Any] = {
                "model": model,
                "prompt": [prompt] * n if n > 1 else prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "return_token_ids": True,
                "vllm_xargs": {"fr10_decode_mode": mode},
            }
            req_t0 = time.time()
            ts_request_received = datetime.now(timezone.utc).isoformat()
            data = _post_json(endpoint, "/v1/completions", payload, timeout=timeout)
            ts_completed = datetime.now(timezone.utc).isoformat()
            req_elapsed = time.time() - req_t0
            req_completion_tokens = 0
            for choice in data["choices"]:
                token_ids = choice.get("token_ids") or []
                req_completion_tokens += len(token_ids)
                records.append(
                    {
                        "mode": mode,
                        "prompt_id": prompt_id,
                        "sample_index": next_sample + int(choice["index"]),
                        "local_choice_index": int(choice["index"]),
                        "finish_reason": choice.get("finish_reason"),
                        "token_count": len(token_ids),
                        "request_elapsed_s": req_elapsed,
                    }
                )
            request_rows.append(
                {
                    "ts_request_received": ts_request_received,
                    "ts_completed": ts_completed,
                    "oracle_session_id": f"fr10_quick_{mode}",
                    "oracle_run_anchor": f"{mode}_prompt{prompt_id}_sample{next_sample}",
                    "mode": mode,
                    "prompt_id": prompt_id,
                    "batch_size": n,
                    "completion_tokens": req_completion_tokens,
                    "decode_sum_s": req_elapsed,
                    "num_requests_running_before": n,
                    "num_requests_running_after": 0,
                }
            )
            next_sample += n
    return records, request_rows, time.time() - t0


def _summarize_mode(
    *,
    mode: str,
    records: list[dict[str, Any]],
    wall_s: float,
    metric_delta: dict[str, float],
    reset_error: str | None,
) -> dict[str, Any]:
    returned_tokens = int(sum(int(row["token_count"]) for row in records))
    decode_seconds = metric_delta.get("decode_seconds", 0.0)
    gen_tokens = metric_delta.get("generation_tokens", 0.0)
    spec_draft = metric_delta.get("spec_draft_tokens", 0.0)
    spec_acc = metric_delta.get("spec_accepted_tokens", 0.0)
    spec_drafts = metric_delta.get("spec_drafts", 0.0)
    return {
        "mode": mode,
        "records": len(records),
        "returned_tokens": returned_tokens,
        "wall_s": wall_s,
        "returned_tokens_per_wall_s": returned_tokens / wall_s if wall_s > 0 else None,
        "metrics_generation_tokens": gen_tokens,
        "metrics_decode_seconds": decode_seconds,
        "warm_decode_tps": gen_tokens / decode_seconds if decode_seconds > 0 else None,
        "returned_tokens_per_decode_s": (
            returned_tokens / decode_seconds if decode_seconds > 0 else None
        ),
        "spec_accepted_tokens": spec_acc,
        "spec_draft_tokens": spec_draft,
        "spec_drafts": spec_drafts,
        "accepted_per_draft_token": spec_acc / spec_draft if spec_draft > 0 else None,
        "accepted_per_draft_event": spec_acc / spec_drafts if spec_drafts > 0 else None,
        "metric_delta": metric_delta,
        "reset_prefix_cache_error": reset_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts-file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", nargs="+", default=["naive_mtp", "tree_mtp"])
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--wait-health", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--request-metrics-out")
    args = parser.parse_args()

    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    prompts = _read_prompts(args.prompts_file)
    result: dict[str, Any] = {
        "schema": "fr10.quick_decode_tps.v1",
        "endpoint": args.endpoint,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "batch_size": args.batch_size,
        "samples_per_prompt": args.samples_per_prompt,
        "max_tokens": args.max_tokens,
        "prompts": prompts,
        "modes": {},
    }
    all_request_rows: list[dict[str, Any]] = []
    for mode in args.modes:
        reset_error = _reset_prefix_cache(args.endpoint)
        if args.warmup_samples:
            _run_requests(
                endpoint=args.endpoint,
                model=args.model,
                prompts=prompts[:1],
                mode=mode,
                samples_per_prompt=args.warmup_samples,
                batch_size=min(args.batch_size, args.warmup_samples),
                max_tokens=min(args.max_tokens, 16),
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.request_timeout,
            )
        before = _scrape_metrics(args.endpoint)
        records, request_rows, wall_s = _run_requests(
            endpoint=args.endpoint,
            model=args.model,
            prompts=prompts,
            mode=mode,
            samples_per_prompt=args.samples_per_prompt,
            batch_size=args.batch_size,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.request_timeout,
        )
        all_request_rows.extend(request_rows)
        after = _scrape_metrics(args.endpoint)
        result["modes"][mode] = _summarize_mode(
            mode=mode,
            records=records,
            wall_s=wall_s,
            metric_delta=_delta(after, before),
            reset_error=reset_error,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.request_metrics_out:
        metrics_out = Path(args.request_metrics_out)
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with metrics_out.open("w", encoding="utf-8") as fh:
            for row in all_request_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(result["modes"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
