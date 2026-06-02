#!/usr/bin/env python3
"""Collect and compare FR9 Gate B greedy exact-token evidence.

This probe talks to the live vLLM OpenAI-compatible server.  It does not prove
the full Gate B by itself; it records exact generated token IDs for fixed prompts
and batch shapes so arms can be compared honestly.
"""

from __future__ import annotations

import argparse
import json
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
    batch_shapes: list[tuple[str, list[tuple[int, str]]]] = [
        ("b1", [(prompt_id, prompt) for prompt_id, prompt in enumerate(prompts)]),
        ("b4", [(prompt_id, prompt) for prompt_id, prompt in enumerate(prompts)]),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--arm", required=True)
    collect_parser.add_argument("--out", required=True)
    collect_parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    collect_parser.add_argument("--model", default=DEFAULT_MODEL)
    collect_parser.add_argument("--max-tokens", type=int, default=24)
    collect_parser.add_argument("--prompts-file")
    collect_parser.add_argument("--request-timeout", type=float, default=180)
    collect_parser.add_argument("--wait-health", type=float, default=0)
    collect_parser.add_argument("--skip-reset-prefix-cache", action="store_true")
    collect_parser.add_argument("--api-key")
    collect_parser.add_argument("--api-key-env", default="VLLM_API_KEY")
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
