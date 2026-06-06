#!/usr/bin/env python3
"""Quick FR10 B4 decode-TPS probe for homogeneous decode modes.

This is the cheap starting-point speed read, not the SWE-bench verdict. It
expects a live shared FR10 cu130 server and runs one mode at a time so the
per-request decode-mode isolation invariant is preserved.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from collections import Counter
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
    seed: int | None,
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
            if seed is not None:
                payload["seed"] = int(seed)
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
    request_rows: list[dict[str, Any]],
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
    per_request_tps = [
        float(row["completion_tokens"]) / float(row["decode_sum_s"])
        for row in request_rows
        if float(row.get("decode_sum_s") or 0.0) > 0
    ]
    per_request_tps_sorted = sorted(per_request_tps)
    return {
        "mode": mode,
        "records": len(records),
        "request_count": len(request_rows),
        "returned_tokens": returned_tokens,
        "wall_s": wall_s,
        "returned_tokens_per_wall_s": returned_tokens / wall_s if wall_s > 0 else None,
        "per_request_decode_tps_mean": (
            statistics.fmean(per_request_tps) if per_request_tps else None
        ),
        "per_request_decode_tps_median": (
            statistics.median(per_request_tps) if per_request_tps else None
        ),
        "per_request_decode_tps_min": per_request_tps_sorted[0]
        if per_request_tps_sorted
        else None,
        "per_request_decode_tps_max": per_request_tps_sorted[-1]
        if per_request_tps_sorted
        else None,
        "metrics_generation_tokens": gen_tokens,
        "metrics_decode_seconds": decode_seconds,
        "metrics_sum_decode_tps_concurrency_deflated": (
            gen_tokens / decode_seconds if decode_seconds > 0 else None
        ),
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _assert_tree_engagement(
    *,
    sampler_debug_path: Path,
    tree_accept_path: Path,
    expected_draft_count: int,
) -> dict[str, Any]:
    debug_rows = _load_jsonl(sampler_debug_path)
    accept_rows = _load_jsonl(tree_accept_path)
    gpu_rows = [row for row in debug_rows if row.get("event") == "gpu_tree_metadata"]

    def _active_draft_counts(row: dict[str, Any]) -> list[int]:
        return [
            int(x)
            for x in (row.get("num_draft_tokens") or [])
            if int(x) > 0
        ]

    ok_rows = [
        row
        for row in gpu_rows
        if row.get("reason") == "ok"
        and row.get("has_tree_parent_indices") is True
        and _active_draft_counts(row)
        and all(x == expected_draft_count for x in _active_draft_counts(row))
    ]
    reasons = Counter(str(row.get("reason")) for row in gpu_rows)
    draft_counts = Counter(
        ",".join(str(int(x)) for x in (row.get("num_draft_tokens") or []))
        for row in gpu_rows
    )
    tree_accept_rows = [
        row for row in accept_rows if row.get("event") in {"tree_sample_accept", "tree_path_lcp_max"}
    ]
    summary = {
        "sampler_debug": str(sampler_debug_path),
        "tree_accept_trace": str(tree_accept_path),
        "expected_draft_count": expected_draft_count,
        "gpu_tree_metadata_rows": len(gpu_rows),
        "gpu_tree_metadata_ok_rows": len(ok_rows),
        "gpu_tree_metadata_reasons": dict(reasons),
        "gpu_tree_metadata_draft_counts": dict(draft_counts),
        "tree_accept_rows": len(tree_accept_rows),
        "engaged": bool(gpu_rows and len(ok_rows) == len(gpu_rows) and tree_accept_rows),
    }
    if not summary["engaged"]:
        raise RuntimeError(
            "tree_mtp engagement assertion failed before reporting metrics: "
            + json.dumps(summary, sort_keys=True)
        )
    return summary


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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wait-health", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--request-metrics-out")
    parser.add_argument("--require-tree-engagement", action="store_true")
    parser.add_argument("--tree-sampler-debug-log", type=Path)
    parser.add_argument("--tree-accept-log", type=Path)
    parser.add_argument("--expected-draft-count", type=int, default=9)
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
        "seed": args.seed,
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
                seed=args.seed,
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
            seed=args.seed,
            timeout=args.request_timeout,
        )
        all_request_rows.extend(request_rows)
        after = _scrape_metrics(args.endpoint)
        result["modes"][mode] = _summarize_mode(
            mode=mode,
            records=records,
            request_rows=request_rows,
            wall_s=wall_s,
            metric_delta=_delta(after, before),
            reset_error=reset_error,
        )
    if args.require_tree_engagement:
        if args.tree_sampler_debug_log is None or args.tree_accept_log is None:
            raise RuntimeError(
                "--require-tree-engagement needs --tree-sampler-debug-log and --tree-accept-log"
            )
        result["tree_engagement"] = _assert_tree_engagement(
            sampler_debug_path=args.tree_sampler_debug_log,
            tree_accept_path=args.tree_accept_log,
            expected_draft_count=args.expected_draft_count,
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
