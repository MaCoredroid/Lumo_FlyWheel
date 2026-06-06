#!/usr/bin/env python3
"""FR12 deliverable B=4 SWE-4 sampling/speed probe.

This is intentionally an output-level deliverable probe, not a sub-kernel
parity harness.  It saves token ids for distribution-floor comparison and
scrapes vLLM speculation metrics around the measured window.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


METRIC_NAMES = {
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:request_decode_time_seconds_sum": "decode_seconds",
    "vllm:request_prefill_time_seconds_sum": "prefill_seconds",
    "vllm:spec_decode_num_accepted_tokens_total": "spec_accepted_tokens",
    "vllm:spec_decode_num_draft_tokens_total": "spec_draft_tokens",
    "vllm:spec_decode_num_drafts_total": "spec_drafts",
}


def _post_json(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else None


def _get_text(endpoint: str, path: str, timeout: float) -> str:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        headers={"Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def _cached_prompt_for(instance_id: str) -> str | None:
    matches = sorted(Path("output").glob(f"fr10_b4temp06_top095_tree_n11_swe4*/**/per_task/{instance_id}/prompt.md"))
    for match in matches:
        text = match.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            return text
    return None


def _load_swe4_prompts(path: Path, *, max_prompt_chars: int) -> list[dict[str, Any]]:
    subset = json.loads(path.read_text(encoding="utf-8"))
    instance_ids = [str(x) for x in subset.get("instance_ids") or []]
    if not instance_ids:
        raise RuntimeError(f"subset has no instance_ids: {path}")
    try:
        from datasets import load_dataset
    except ModuleNotFoundError:
        prompts: list[dict[str, Any]] = []
        for instance_id in instance_ids:
            prompt = _cached_prompt_for(instance_id)
            if prompt is None:
                raise RuntimeError(
                    "datasets is not installed and no cached prompt.md exists for "
                    f"{instance_id}"
                )
            if max_prompt_chars > 0:
                prompt = prompt[:max_prompt_chars]
            prompts.append({"prompt_id": len(prompts), "instance_id": instance_id, "prompt": prompt})
        return prompts

    ds = load_dataset(
        str(subset.get("dataset_name") or "princeton-nlp/SWE-bench_Verified"),
        split=str(subset.get("split") or "test"),
    )
    by_id = {str(row["instance_id"]): row for row in ds if str(row["instance_id"]) in set(instance_ids)}
    missing = sorted(set(instance_ids) - set(by_id))
    if missing:
        raise RuntimeError(f"subset instances missing from dataset: {missing}")
    prompts: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        row = by_id[instance_id]
        problem = str(row.get("problem_statement") or "").strip()
        if max_prompt_chars > 0:
            problem = problem[:max_prompt_chars]
        prompt = (
            "You are working on a SWE-bench Verified task.\n"
            f"Instance: {instance_id}\n"
            f"Repository: {row.get('repo') or 'unknown'}\n"
            f"Base commit: {row.get('base_commit') or ''}\n\n"
            "Problem statement:\n"
            f"{problem}\n\n"
            "Reason about the fix. Do not use tools; provide the likely code-change plan and key files."
        )
        prompts.append({"prompt_id": len(prompts), "instance_id": instance_id, "prompt": prompt})
    return prompts


def _reset_prefix_cache(endpoint: str) -> str | None:
    try:
        _post_json(endpoint, "/reset_prefix_cache", {}, timeout=30)
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def _one_request(
    *,
    endpoint: str,
    model: str,
    mode: str,
    prompt_row: dict[str, Any],
    sample_index: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    timeout: float,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_row["prompt"],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "return_token_ids": True,
        "vllm_xargs": {"fr10_decode_mode": mode},
    }
    if seed is not None:
        payload["seed"] = int(seed) + int(prompt_row["prompt_id"]) * 1000 + sample_index
    t0 = time.time()
    data = _post_json(endpoint, "/v1/completions", payload, timeout=timeout)
    elapsed = time.time() - t0
    choice = data["choices"][0]
    token_ids = [int(x) for x in (choice.get("token_ids") or [])]
    return {
        "prompt_id": int(prompt_row["prompt_id"]),
        "sample_index": int(sample_index),
        "choice_index": 0,
        "instance_id": prompt_row["instance_id"],
        "prompt": prompt_row["prompt"],
        "token_ids": token_ids,
        "text": choice.get("text", ""),
        "finish_reason": choice.get("finish_reason"),
        "token_count": len(token_ids),
        "request_elapsed_s": elapsed,
    }


def _run_window(
    *,
    endpoint: str,
    model: str,
    mode: str,
    prompts: list[dict[str, Any]],
    samples_per_prompt: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    jobs = [
        (prompt, sample)
        for sample in range(samples_per_prompt)
        for prompt in prompts
    ]
    records: list[dict[str, Any]] = []
    t0 = time.time()
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start : start + batch_size]
        with cf.ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(
                    _one_request,
                    endpoint=endpoint,
                    model=model,
                    mode=mode,
                    prompt_row=prompt,
                    sample_index=sample,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    timeout=timeout,
                )
                for prompt, sample in batch
            ]
            for future in cf.as_completed(futures):
                records.append(future.result())
    records.sort(key=lambda row: (int(row["sample_index"]), int(row["prompt_id"])))
    return records, time.time() - t0


def _summarize(records: list[dict[str, Any]], metric_delta: dict[str, float], wall_s: float) -> dict[str, Any]:
    token_counts = [int(row["token_count"]) for row in records]
    req_tps = [
        int(row["token_count"]) / float(row["request_elapsed_s"])
        for row in records
        if float(row["request_elapsed_s"]) > 0
    ]
    spec_acc = metric_delta.get("spec_accepted_tokens", 0.0)
    spec_draft = metric_delta.get("spec_draft_tokens", 0.0)
    spec_drafts = metric_delta.get("spec_drafts", 0.0)
    gen_tokens = metric_delta.get("generation_tokens", 0.0)
    decode_seconds = metric_delta.get("decode_seconds", 0.0)
    return {
        "records": len(records),
        "returned_tokens": int(sum(token_counts)),
        "token_count_min": min(token_counts) if token_counts else None,
        "token_count_max": max(token_counts) if token_counts else None,
        "wall_s": wall_s,
        "returned_tokens_per_wall_s": sum(token_counts) / wall_s if wall_s > 0 else None,
        "request_tps_mean": statistics.fmean(req_tps) if req_tps else None,
        "request_tps_median": statistics.median(req_tps) if req_tps else None,
        "metrics_generation_tokens": gen_tokens,
        "metrics_decode_seconds": decode_seconds,
        "warm_decode_tps": gen_tokens / decode_seconds if decode_seconds > 0 else None,
        "spec_accepted_tokens": spec_acc,
        "spec_draft_tokens": spec_draft,
        "spec_drafts": spec_drafts,
        "accepted_per_draft_token": spec_acc / spec_draft if spec_draft > 0 else None,
        "accepted_per_draft_event": spec_acc / spec_drafts if spec_drafts > 0 else None,
        "metric_delta": metric_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950")
    parser.add_argument("--model", default="qwen3.6-27b")
    parser.add_argument("--subset", type=Path, default=Path("docs/reports/auto_research/swe-bench-agentic-b4-four-verified-20260530.json"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", default="tree_mtp")
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wait-health", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--max-prompt-chars", type=int, default=0)
    args = parser.parse_args()

    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    prompts = _load_swe4_prompts(args.subset, max_prompt_chars=args.max_prompt_chars)
    reset_error = _reset_prefix_cache(args.endpoint)
    if args.warmup:
        _run_window(
            endpoint=args.endpoint,
            model=args.model,
            mode=args.mode,
            prompts=prompts[:1],
            samples_per_prompt=1,
            batch_size=1,
            max_tokens=min(16, args.max_tokens),
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            timeout=args.request_timeout,
        )
    before = _scrape_metrics(args.endpoint)
    records, wall_s = _run_window(
        endpoint=args.endpoint,
        model=args.model,
        mode=args.mode,
        prompts=prompts,
        samples_per_prompt=args.samples_per_prompt,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        timeout=args.request_timeout,
    )
    after = _scrape_metrics(args.endpoint)
    payload = {
        "schema": "fr12.deliverable_swe4_probe.v1",
        "label": args.label,
        "measured_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "endpoint": args.endpoint,
        "model": args.model,
        "mode": args.mode,
        "subset": str(args.subset),
        "samples_per_prompt": args.samples_per_prompt,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "reset_prefix_cache_error": reset_error,
        "prompts": prompts,
        "records": records,
        "summary": _summarize(records, _delta(after, before), wall_s),
        "token_first_position_counts": dict(Counter(row["token_ids"][0] for row in records if row["token_ids"])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
