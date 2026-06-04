#!/usr/bin/env python3
"""Collect and compare FR9 Gate B greedy exact-token evidence.

This probe talks to the live vLLM OpenAI-compatible server.  It does not prove
the full Gate B by itself; it records exact generated token IDs for fixed prompts
and batch shapes so arms can be compared honestly.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ENDPOINT = "http://127.0.0.1:9950"
DEFAULT_MODEL = "qwen3.6-27b"
DEFAULT_CONTAINER = "lumo-vllm-track-b-suffix"
DEFAULT_PROMPTS = [
    "Q: Count from one to five. A:",
    "Q: What is 7 plus 8? A:",
    "Q: Write three lowercase letters in alphabetical order. A:",
    "Q: Name the color of clear daytime sky in one word. A:",
    "Q: Complete the pattern: red, orange, yellow, A:",
    "Q: State one Unix command that lists files. A:",
    "Q: In Python, what boolean value is 1 == 1? A:",
    "Q: Give one word that rhymes with time. A:",
]


def _load_api_key(args: argparse.Namespace) -> str | None:
    if args.api_key:
        return args.api_key
    if args.api_key_env:
        import os

        value = os.environ.get(args.api_key_env)
        if value:
            return value
    if args.api_key_from_container:
        return subprocess.check_output(
            [
                "docker",
                "exec",
                args.api_key_from_container,
                "sh",
                "-lc",
                "printenv VLLM_API_KEY",
            ],
            text=True,
        ).strip()
    return None


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _post_json(
    endpoint: str,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout: float,
) -> Any:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    if not body:
        return None
    return json.loads(body)


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
        except Exception as exc:  # noqa: BLE001 - report final health failure.
            last_exc = exc
            time.sleep(2)
    raise RuntimeError(f"server did not become healthy within {timeout_s}s: {last_exc}")


def _read_prompts(path: str | None) -> list[str]:
    if not path:
        return list(DEFAULT_PROMPTS)
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def collect(args: argparse.Namespace) -> int:
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)

    api_key = _load_api_key(args)
    headers = _headers(api_key)
    prompts = _read_prompts(args.prompts_file)

    reset_error = None
    if not args.skip_reset_prefix_cache:
        try:
            _post_json(args.endpoint, "/reset_prefix_cache", headers, {}, timeout=30)
        except Exception as exc:  # noqa: BLE001 - endpoint may return empty/non-JSON.
            reset_error = f"{type(exc).__name__}: {exc}"

    records: list[dict[str, Any]] = []
    all_batch_shapes: list[tuple[str, list[tuple[int, str]]]] = [
        ("b1", [(prompt_id, prompt) for prompt_id, prompt in enumerate(prompts)]),
        ("b4", [(prompt_id, prompt) for prompt_id, prompt in enumerate(prompts)]),
    ]
    batch_shapes = [
        item for item in all_batch_shapes if args.batch_shape in ("both", item[0])
    ]
    for batch_name, prompt_items in batch_shapes:
        batch_prompts = [prompt for _, prompt in prompt_items]
        payload: dict[str, Any] = {
            "model": args.model,
            "prompt": batch_prompts if len(batch_prompts) > 1 else batch_prompts[0],
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "logprobs": 1,
            "return_token_ids": True,
        }
        if args.fr10_decode_mode:
            payload["vllm_xargs"] = {"fr10_decode_mode": args.fr10_decode_mode}
        if args.seed is not None:
            payload["seed"] = args.seed
        data = _post_json(
            args.endpoint,
            "/v1/completions",
            headers,
            payload,
            timeout=args.request_timeout,
        )
        for choice in data["choices"]:
            local_choice_index = int(choice["index"])
            prompt_id, prompt = prompt_items[local_choice_index]
            logprobs = choice.get("logprobs") or {}
            records.append(
                {
                    "batch": batch_name,
                    "choice_index": prompt_id,
                    "local_choice_index": local_choice_index,
                    "prompt": prompt,
                    "finish_reason": choice.get("finish_reason"),
                    "text": choice.get("text"),
                    "token_ids": choice.get("token_ids"),
                    "logprob_tokens": logprobs.get("tokens"),
                }
            )

    metrics_error = None
    spec_decode_metric_present = False
    try:
        metrics = _get_text(args.endpoint, "/metrics", timeout=10)
        spec_decode_metric_present = (
            "vllm:spec_decode_num_draft_tokens_total" in metrics
        )
    except Exception as exc:  # noqa: BLE001 - metrics is helpful, not required.
        metrics_error = f"{type(exc).__name__}: {exc}"

    artifact = {
        "arm": args.arm,
        "endpoint": args.endpoint,
        "model": args.model,
        "temperature": 0,
        "seed": args.seed,
        "fr10_decode_mode": args.fr10_decode_mode,
        "batch_shape": args.batch_shape,
        "max_tokens": args.max_tokens,
        "ts": time.time(),
        "prompts": prompts,
        "reset_prefix_cache_error": reset_error,
        "metrics_error": metrics_error,
        "spec_decode_metric_present": spec_decode_metric_present,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "out": str(out),
                "records": len(records),
                "reset_prefix_cache_error": reset_error,
            },
            sort_keys=True,
        )
    )
    return 0


def sample_temp(args: argparse.Namespace) -> int:
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)

    api_key = _load_api_key(args)
    headers = _headers(api_key)

    reset_error = None
    if not args.skip_reset_prefix_cache:
        try:
            _post_json(args.endpoint, "/reset_prefix_cache", headers, {}, timeout=30)
        except Exception as exc:  # noqa: BLE001 - endpoint may return empty/non-JSON.
            reset_error = f"{type(exc).__name__}: {exc}"

    records: list[dict[str, Any]] = []
    next_sample = 0
    while next_sample < args.samples:
        batch_size = min(args.batch_size, args.samples - next_sample)
        payload: dict[str, Any] = {
            "model": args.model,
            "prompt": [args.prompt] * batch_size if batch_size > 1 else args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "logprobs": args.logprobs,
            "return_token_ids": True,
        }
        if args.fr10_decode_mode:
            payload["vllm_xargs"] = {"fr10_decode_mode": args.fr10_decode_mode}
        data = _post_json(
            args.endpoint,
            "/v1/completions",
            headers,
            payload,
            timeout=args.request_timeout,
        )
        for choice in data["choices"]:
            records.append(
                {
                    "sample_index": next_sample + int(choice["index"]),
                    "local_choice_index": int(choice["index"]),
                    "finish_reason": choice.get("finish_reason"),
                    "text": choice.get("text"),
                    "token_ids": choice.get("token_ids") or [],
                }
            )
        next_sample += batch_size

    artifact = {
        "arm": args.arm,
        "endpoint": args.endpoint,
        "model": args.model,
        "temperature": args.temperature,
        "fr10_decode_mode": args.fr10_decode_mode,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "samples": args.samples,
        "batch_size": args.batch_size,
        "ts": time.time(),
        "reset_prefix_cache_error": reset_error,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "out": str(out),
                "samples": len(records),
                "temperature": args.temperature,
                "reset_prefix_cache_error": reset_error,
            },
            sort_keys=True,
        )
    )
    return 0


def sample_prompts_temp(args: argparse.Namespace) -> int:
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)

    api_key = _load_api_key(args)
    headers = _headers(api_key)
    prompts = _read_prompts(args.prompts_file)

    reset_error = None
    if not args.skip_reset_prefix_cache:
        try:
            _post_json(args.endpoint, "/reset_prefix_cache", headers, {}, timeout=30)
        except Exception as exc:  # noqa: BLE001 - endpoint may return empty/non-JSON.
            reset_error = f"{type(exc).__name__}: {exc}"

    records: list[dict[str, Any]] = []
    for prompt_id, prompt in enumerate(prompts):
        next_sample = 0
        while next_sample < args.samples_per_prompt:
            batch_size = min(args.batch_size, args.samples_per_prompt - next_sample)
            payload: dict[str, Any] = {
                "model": args.model,
                "prompt": [prompt] * batch_size if batch_size > 1 else prompt,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "logprobs": args.logprobs,
                "return_token_ids": True,
            }
            if args.fr10_decode_mode:
                payload["vllm_xargs"] = {"fr10_decode_mode": args.fr10_decode_mode}
            if args.seed is not None:
                payload["seed"] = args.seed + prompt_id * args.samples_per_prompt + next_sample
            data = _post_json(
                args.endpoint,
                "/v1/completions",
                headers,
                payload,
                timeout=args.request_timeout,
            )
            for choice in data["choices"]:
                local_choice_index = int(choice["index"])
                records.append(
                    {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "sample_index": next_sample + local_choice_index,
                        "local_choice_index": local_choice_index,
                        "finish_reason": choice.get("finish_reason"),
                        "text": choice.get("text"),
                        "token_ids": choice.get("token_ids") or [],
                    }
                )
            next_sample += batch_size

    artifact = {
        "arm": args.arm,
        "endpoint": args.endpoint,
        "model": args.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "fr10_decode_mode": args.fr10_decode_mode,
        "max_tokens": args.max_tokens,
        "samples_per_prompt": args.samples_per_prompt,
        "batch_size": args.batch_size,
        "ts": time.time(),
        "prompts": prompts,
        "reset_prefix_cache_error": reset_error,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "out": str(out),
                "records": len(records),
                "prompts": len(prompts),
                "samples_per_prompt": args.samples_per_prompt,
                "temperature": args.temperature,
                "reset_prefix_cache_error": reset_error,
            },
            sort_keys=True,
        )
    )
    return 0


def collect_logprobs(args: argparse.Namespace) -> int:
    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)

    api_key = _load_api_key(args)
    headers = _headers(api_key)
    prompts = _read_prompts(args.prompts_file)

    reset_error = None
    if not args.skip_reset_prefix_cache:
        try:
            _post_json(args.endpoint, "/reset_prefix_cache", headers, {}, timeout=30)
        except Exception as exc:  # noqa: BLE001 - endpoint may return empty/non-JSON.
            reset_error = f"{type(exc).__name__}: {exc}"

    records: list[dict[str, Any]] = []
    batch_shapes: list[tuple[str, list[tuple[int, str]]]] = [
        ("b1", [(prompt_id, prompt) for prompt_id, prompt in enumerate(prompts)]),
        ("b4", [(prompt_id, prompt) for prompt_id, prompt in enumerate(prompts)]),
    ]
    for batch_name, prompt_items in batch_shapes:
        batch_prompts = [prompt for _, prompt in prompt_items]
        payload: dict[str, Any] = {
            "model": args.model,
            "prompt": batch_prompts if len(batch_prompts) > 1 else batch_prompts[0],
            "max_tokens": 1,
            "temperature": args.temperature,
            "logprobs": args.top_k,
            "return_token_ids": True,
        }
        if args.fr10_decode_mode:
            payload["vllm_xargs"] = {"fr10_decode_mode": args.fr10_decode_mode}
        data = _post_json(
            args.endpoint,
            "/v1/completions",
            headers,
            payload,
            timeout=args.request_timeout,
        )
        for choice in data["choices"]:
            local_choice_index = int(choice["index"])
            prompt_id, prompt = prompt_items[local_choice_index]
            logprobs = choice.get("logprobs") or {}
            top_logprobs = (logprobs.get("top_logprobs") or [{}])[0] or {}
            records.append(
                {
                    "batch": batch_name,
                    "choice_index": prompt_id,
                    "local_choice_index": local_choice_index,
                    "prompt": prompt,
                    "sampled_token_ids": choice.get("token_ids") or [],
                    "sampled_tokens": logprobs.get("tokens") or [],
                    "token_logprobs": logprobs.get("token_logprobs") or [],
                    "top_logprobs": top_logprobs,
                }
            )

    artifact = {
        "arm": args.arm,
        "endpoint": args.endpoint,
        "model": args.model,
        "temperature": args.temperature,
        "fr10_decode_mode": args.fr10_decode_mode,
        "top_k": args.top_k,
        "ts": time.time(),
        "prompts": prompts,
        "reset_prefix_cache_error": reset_error,
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "arm": args.arm,
                "out": str(out),
                "records": len(records),
                "temperature": args.temperature,
                "top_k": args.top_k,
                "reset_prefix_cache_error": reset_error,
            },
            sort_keys=True,
        )
    )
    return 0


def _record_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (record["batch"], int(record["choice_index"]), record["prompt"])


def _first_diff(left: list[int], right: list[int]) -> int | None:
    for idx, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    left_records = {_record_key(record): record for record in left["records"]}
    right_records = {_record_key(record): record for record in right["records"]}

    mismatches: list[dict[str, Any]] = []
    missing_left = sorted(set(right_records) - set(left_records))
    missing_right = sorted(set(left_records) - set(right_records))
    for key in sorted(set(left_records) & set(right_records)):
        left_ids = left_records[key].get("token_ids") or []
        right_ids = right_records[key].get("token_ids") or []
        diff = _first_diff(left_ids, right_ids)
        if diff is not None:
            mismatches.append(
                {
                    "batch": key[0],
                    "choice_index": key[1],
                    "prompt": key[2],
                    "first_diff_index": diff,
                    "left_token": left_ids[diff] if diff < len(left_ids) else None,
                    "right_token": right_ids[diff] if diff < len(right_ids) else None,
                    "left_token_ids": left_ids,
                    "right_token_ids": right_ids,
                }
            )

    result = {
        "left_arm": left.get("arm"),
        "right_arm": right.get("arm"),
        "left": args.left,
        "right": args.right,
        "matched_records": len(set(left_records) & set(right_records)),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "mismatches": mismatches,
        "exact_match": not missing_left and not missing_right and not mismatches,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["exact_match"] else 1


def compare_batches(args: argparse.Namespace) -> int:
    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    left_records = {
        (int(record["choice_index"]), record["prompt"]): record
        for record in artifact["records"]
        if record["batch"] == args.left_batch
    }
    right_records = {
        (int(record["choice_index"]), record["prompt"]): record
        for record in artifact["records"]
        if record["batch"] == args.right_batch
    }

    mismatches: list[dict[str, Any]] = []
    missing_left = sorted(set(right_records) - set(left_records))
    missing_right = sorted(set(left_records) - set(right_records))
    for key in sorted(set(left_records) & set(right_records)):
        left_ids = left_records[key].get("token_ids") or []
        right_ids = right_records[key].get("token_ids") or []
        diff = _first_diff(left_ids, right_ids)
        if diff is not None:
            mismatches.append(
                {
                    "choice_index": key[0],
                    "prompt": key[1],
                    "first_diff_index": diff,
                    "left_token": left_ids[diff] if diff < len(left_ids) else None,
                    "right_token": right_ids[diff] if diff < len(right_ids) else None,
                    "left_token_ids": left_ids,
                    "right_token_ids": right_ids,
                }
            )

    result = {
        "arm": artifact.get("arm"),
        "artifact": args.artifact,
        "left_batch": args.left_batch,
        "right_batch": args.right_batch,
        "matched_records": len(set(left_records) & set(right_records)),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "mismatches": mismatches,
        "exact_match": not missing_left and not missing_right and not mismatches,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["exact_match"] else 1


def _position_counts(records: list[dict[str, Any]], position: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        token_ids = record.get("token_ids") or []
        token = str(token_ids[position]) if position < len(token_ids) else "<missing>"
        counts[token] += 1
    return counts


def _tv(left_counts: Counter[str], right_counts: Counter[str]) -> float:
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    keys = set(left_counts) | set(right_counts)
    return 0.5 * sum(
        abs(left_counts.get(key, 0) / left_total - right_counts.get(key, 0) / right_total)
        for key in keys
    )


def _chi_square_2sample(left_counts: Counter[str], right_counts: Counter[str]) -> tuple[float, int]:
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    total = left_total + right_total
    stat = 0.0
    nonzero_columns = 0
    for key in set(left_counts) | set(right_counts):
        column = left_counts.get(key, 0) + right_counts.get(key, 0)
        if column == 0:
            continue
        nonzero_columns += 1
        expected_left = left_total * column / total
        expected_right = right_total * column / total
        if expected_left > 0:
            stat += (left_counts.get(key, 0) - expected_left) ** 2 / expected_left
        if expected_right > 0:
            stat += (right_counts.get(key, 0) - expected_right) ** 2 / expected_right
    return stat, max(0, nonzero_columns - 1)


def _top_logprob_distribution(
    top_logprobs: dict[str, float],
    support: set[str],
) -> dict[str, float]:
    probs = {token: math.exp(float(top_logprobs[token])) for token in support if token in top_logprobs}
    residual = max(0.0, 1.0 - sum(math.exp(float(value)) for value in top_logprobs.values()))
    probs["<other>"] = residual
    return probs


def _distribution_tv(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def compare_logprobs(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    left_records = {_record_key(record): record for record in left["records"]}
    right_records = {_record_key(record): record for record in right["records"]}

    rows = []
    missing_left = sorted(set(right_records) - set(left_records))
    missing_right = sorted(set(left_records) - set(right_records))
    for key in sorted(set(left_records) & set(right_records)):
        left_top = left_records[key].get("top_logprobs") or {}
        right_top = right_records[key].get("top_logprobs") or {}
        support = set(left_top) | set(right_top)
        left_dist = _top_logprob_distribution(left_top, support)
        right_dist = _top_logprob_distribution(right_top, support)
        tv = _distribution_tv(left_dist, right_dist)
        left_top_items = sorted(left_top.items(), key=lambda item: item[1], reverse=True)
        right_top_items = sorted(right_top.items(), key=lambda item: item[1], reverse=True)
        rows.append(
            {
                "batch": key[0],
                "choice_index": key[1],
                "prompt": key[2],
                "tv_with_other_bucket": tv,
                "left_top": left_top_items[: args.top],
                "right_top": right_top_items[: args.top],
                "left_top_mass": sum(math.exp(float(value)) for value in left_top.values()),
                "right_top_mass": sum(math.exp(float(value)) for value in right_top.values()),
                "support_size": len(support),
            }
        )

    max_tv = max((row["tv_with_other_bucket"] for row in rows), default=0.0)
    result = {
        "left_arm": left.get("arm"),
        "right_arm": right.get("arm"),
        "left": args.left,
        "right": args.right,
        "temperature": [left.get("temperature"), right.get("temperature")],
        "top_k": [left.get("top_k"), right.get("top_k")],
        "matched_records": len(set(left_records) & set(right_records)),
        "missing_left": missing_left,
        "missing_right": missing_right,
        "max_tv_with_other_bucket": max_tv,
        "rows": rows,
        "passes_tv_threshold": max_tv <= args.tv_threshold,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passes_tv_threshold"] else 1


def compare_sampling(args: argparse.Namespace) -> int:
    left = json.loads(Path(args.left).read_text(encoding="utf-8"))
    right = json.loads(Path(args.right).read_text(encoding="utf-8"))
    left_records = left["records"]
    right_records = right["records"]
    max_len = max(
        [len(record.get("token_ids") or []) for record in left_records + right_records]
        or [0]
    )
    max_positions = min(max_len, args.positions)
    positions = []
    for position in range(max_positions):
        left_counts = _position_counts(left_records, position)
        right_counts = _position_counts(right_records, position)
        stat, df = _chi_square_2sample(left_counts, right_counts)
        tv = _tv(left_counts, right_counts)
        positions.append(
            {
                "position": position,
                "tv": tv,
                "chi_square": stat,
                "df": df,
                "chi_square_per_df": stat / df if df else math.inf if stat else 0.0,
                "left_top": left_counts.most_common(args.top),
                "right_top": right_counts.most_common(args.top),
            }
        )

    sequence_left = Counter(json.dumps(record.get("token_ids") or []) for record in left_records)
    sequence_right = Counter(json.dumps(record.get("token_ids") or []) for record in right_records)
    seq_stat, seq_df = _chi_square_2sample(sequence_left, sequence_right)
    seq_tv = _tv(sequence_left, sequence_right)
    max_position_tv = max((row["tv"] for row in positions), default=0.0)
    result = {
        "left_arm": left.get("arm"),
        "right_arm": right.get("arm"),
        "left": args.left,
        "right": args.right,
        "left_samples": len(left_records),
        "right_samples": len(right_records),
        "temperature": [left.get("temperature"), right.get("temperature")],
        "max_position_tv": max_position_tv,
        "sequence_tv": seq_tv,
        "sequence_chi_square": seq_stat,
        "sequence_df": seq_df,
        "positions": positions,
        "passes_tv_threshold": max_position_tv <= args.tv_threshold,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passes_tv_threshold"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--arm", required=True)
    collect_parser.add_argument("--out", required=True)
    collect_parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    collect_parser.add_argument("--model", default=DEFAULT_MODEL)
    collect_parser.add_argument("--max-tokens", type=int, default=24)
    collect_parser.add_argument("--seed", type=int)
    collect_parser.add_argument("--batch-shape", choices=["both", "b1", "b4"], default="both")
    collect_parser.add_argument("--prompts-file")
    collect_parser.add_argument("--request-timeout", type=float, default=180)
    collect_parser.add_argument("--wait-health", type=float, default=0)
    collect_parser.add_argument("--skip-reset-prefix-cache", action="store_true")
    collect_parser.add_argument("--api-key")
    collect_parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    collect_parser.add_argument(
        "--fr10-decode-mode",
        choices=["naive_mtp", "non_mtp", "tree_mtp"],
        help="Pass FR10 per-request decode mode through vLLM vllm_xargs.",
    )
    collect_parser.add_argument(
        "--api-key-from-container",
        default=DEFAULT_CONTAINER,
        help="Docker container to read VLLM_API_KEY from; use '' to disable.",
    )
    collect_parser.set_defaults(func=collect)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    compare_parser.set_defaults(func=compare)

    compare_batches_parser = subparsers.add_parser("compare-batches")
    compare_batches_parser.add_argument("artifact")
    compare_batches_parser.add_argument("--left-batch", default="b1")
    compare_batches_parser.add_argument("--right-batch", default="b4")
    compare_batches_parser.set_defaults(func=compare_batches)

    sample_parser = subparsers.add_parser("sample-temp")
    sample_parser.add_argument("--arm", required=True)
    sample_parser.add_argument("--out", required=True)
    sample_parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    sample_parser.add_argument("--model", default=DEFAULT_MODEL)
    sample_parser.add_argument("--prompt", default=DEFAULT_PROMPTS[2])
    sample_parser.add_argument("--samples", type=int, default=256)
    sample_parser.add_argument("--batch-size", type=int, default=4)
    sample_parser.add_argument("--max-tokens", type=int, default=12)
    sample_parser.add_argument("--temperature", type=float, default=0.6)
    sample_parser.add_argument("--logprobs", type=int, default=1)
    sample_parser.add_argument("--request-timeout", type=float, default=180)
    sample_parser.add_argument("--wait-health", type=float, default=0)
    sample_parser.add_argument("--skip-reset-prefix-cache", action="store_true")
    sample_parser.add_argument("--api-key")
    sample_parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    sample_parser.add_argument("--api-key-from-container", default=DEFAULT_CONTAINER)
    sample_parser.add_argument(
        "--fr10-decode-mode",
        choices=["naive_mtp", "non_mtp", "tree_mtp"],
        help="Pass FR10 per-request decode mode through vLLM vllm_xargs.",
    )
    sample_parser.set_defaults(func=sample_temp)

    sample_prompts_parser = subparsers.add_parser("sample-prompts-temp")
    sample_prompts_parser.add_argument("--arm", required=True)
    sample_prompts_parser.add_argument("--out", required=True)
    sample_prompts_parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    sample_prompts_parser.add_argument("--model", default=DEFAULT_MODEL)
    sample_prompts_parser.add_argument("--prompts-file")
    sample_prompts_parser.add_argument("--samples-per-prompt", type=int, default=32)
    sample_prompts_parser.add_argument("--batch-size", type=int, default=4)
    sample_prompts_parser.add_argument("--max-tokens", type=int, default=32)
    sample_prompts_parser.add_argument("--temperature", type=float, default=0.6)
    sample_prompts_parser.add_argument("--logprobs", type=int, default=1)
    sample_prompts_parser.add_argument("--seed", type=int)
    sample_prompts_parser.add_argument("--request-timeout", type=float, default=180)
    sample_prompts_parser.add_argument("--wait-health", type=float, default=0)
    sample_prompts_parser.add_argument("--skip-reset-prefix-cache", action="store_true")
    sample_prompts_parser.add_argument("--api-key")
    sample_prompts_parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    sample_prompts_parser.add_argument("--api-key-from-container", default=DEFAULT_CONTAINER)
    sample_prompts_parser.add_argument(
        "--fr10-decode-mode",
        choices=["naive_mtp", "non_mtp", "tree_mtp"],
        help="Pass FR10 per-request decode mode through vLLM vllm_xargs.",
    )
    sample_prompts_parser.set_defaults(func=sample_prompts_temp)

    compare_sampling_parser = subparsers.add_parser("compare-sampling")
    compare_sampling_parser.add_argument("left")
    compare_sampling_parser.add_argument("right")
    compare_sampling_parser.add_argument("--positions", type=int, default=12)
    compare_sampling_parser.add_argument("--top", type=int, default=8)
    compare_sampling_parser.add_argument("--tv-threshold", type=float, default=0.05)
    compare_sampling_parser.set_defaults(func=compare_sampling)

    logprobs_parser = subparsers.add_parser("collect-logprobs")
    logprobs_parser.add_argument("--arm", required=True)
    logprobs_parser.add_argument("--out", required=True)
    logprobs_parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    logprobs_parser.add_argument("--model", default=DEFAULT_MODEL)
    logprobs_parser.add_argument("--prompts-file")
    logprobs_parser.add_argument("--temperature", type=float, default=0.6)
    logprobs_parser.add_argument("--top-k", type=int, default=20)
    logprobs_parser.add_argument("--request-timeout", type=float, default=180)
    logprobs_parser.add_argument("--wait-health", type=float, default=0)
    logprobs_parser.add_argument("--skip-reset-prefix-cache", action="store_true")
    logprobs_parser.add_argument("--api-key")
    logprobs_parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    logprobs_parser.add_argument("--api-key-from-container", default=DEFAULT_CONTAINER)
    logprobs_parser.add_argument(
        "--fr10-decode-mode",
        choices=["naive_mtp", "non_mtp", "tree_mtp"],
        help="Pass FR10 per-request decode mode through vLLM vllm_xargs.",
    )
    logprobs_parser.set_defaults(func=collect_logprobs)

    compare_logprobs_parser = subparsers.add_parser("compare-logprobs")
    compare_logprobs_parser.add_argument("left")
    compare_logprobs_parser.add_argument("right")
    compare_logprobs_parser.add_argument("--top", type=int, default=8)
    compare_logprobs_parser.add_argument("--tv-threshold", type=float, default=0.01)
    compare_logprobs_parser.set_defaults(func=compare_logprobs)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "api_key_from_container", None) == "":
        args.api_key_from_container = None
    try:
        return args.func(args)
    except (RuntimeError, ValueError, urllib.error.URLError, subprocess.CalledProcessError) as exc:
        print(f"fr9_gate_b_greedy_probe.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
