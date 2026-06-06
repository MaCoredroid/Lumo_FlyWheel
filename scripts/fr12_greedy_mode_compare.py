#!/usr/bin/env python3
"""FR12 greedy mode comparison against a live patched vLLM server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from fr10_quick_decode_tps_probe import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    _assert_tree_engagement,
    _headers,
    _read_prompts,
    _reset_prefix_cache,
    _wait_health,
)


def _post_json(endpoint: str, path: str, payload: dict[str, Any], timeout: float) -> Any:
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else None


def _run_mode(
    *,
    endpoint: str,
    model: str,
    prompts: list[str],
    mode: str,
    max_tokens: int,
    batch_size: int,
    timeout: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        payload: dict[str, Any] = {
            "model": model,
            "prompt": batch if len(batch) > 1 else batch[0],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "return_token_ids": True,
            "vllm_xargs": {"fr10_decode_mode": mode},
        }
        data = _post_json(endpoint, "/v1/completions", payload, timeout=timeout)
        for choice in data["choices"]:
            local = int(choice["index"])
            prompt_id = start + local
            token_ids = [int(x) for x in (choice.get("token_ids") or [])]
            rows.append(
                {
                    "mode": mode,
                    "prompt_id": prompt_id,
                    "sample_index": 0,
                    "local_choice_index": local,
                    "prompt": prompts[prompt_id],
                    "token_ids": token_ids,
                    "text": choice.get("text", ""),
                    "finish_reason": choice.get("finish_reason"),
                    "token_count": len(token_ids),
                }
            )
    return sorted(rows, key=lambda row: int(row["prompt_id"]))


def _first_diff(left: list[int], right: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _compare(base: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    by_base = {int(row["prompt_id"]): row for row in base}
    by_cand = {int(row["prompt_id"]): row for row in candidate}
    rows = []
    for prompt_id in sorted(set(by_base) | set(by_cand)):
        left = by_base.get(prompt_id, {}).get("token_ids") or []
        right = by_cand.get(prompt_id, {}).get("token_ids") or []
        diff = _first_diff(left, right)
        rows.append(
            {
                "prompt_id": prompt_id,
                "base_token_count": len(left),
                "candidate_token_count": len(right),
                "first_diff": diff,
                "base_token": left[diff] if diff is not None and diff < len(left) else None,
                "candidate_token": right[diff]
                if diff is not None and diff < len(right)
                else None,
                "matched": diff is None,
            }
        )
    return {
        "records": len(rows),
        "matched_records": sum(1 for row in rows if row["matched"]),
        "first_mismatch": next((row for row in rows if not row["matched"]), None),
        "all_match": all(row["matched"] for row in rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompts-file")
    parser.add_argument("--out", required=True)
    parser.add_argument("--modes", nargs="+", default=["non_mtp", "tree_mtp"])
    parser.add_argument("--base-mode", default="non_mtp")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--wait-health", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--require-tree-engagement", action="store_true")
    parser.add_argument("--tree-sampler-debug-log", type=Path)
    parser.add_argument("--tree-accept-log", type=Path)
    parser.add_argument("--expected-draft-count", type=int, default=9)
    args = parser.parse_args()

    if args.wait_health:
        _wait_health(args.endpoint, args.wait_health)
    prompts = _read_prompts(args.prompts_file) if args.prompts_file else list(DEFAULT_PROMPTS)
    result: dict[str, Any] = {
        "schema": "fr12.greedy_mode_compare.v1",
        "endpoint": args.endpoint,
        "model": args.model,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
        "base_mode": args.base_mode,
        "modes": {},
        "comparisons": {},
    }
    for mode in args.modes:
        _reset_prefix_cache(args.endpoint)
        t0 = time.time()
        rows = _run_mode(
            endpoint=args.endpoint,
            model=args.model,
            prompts=prompts,
            mode=mode,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            timeout=args.request_timeout,
        )
        result["modes"][mode] = {
            "records": rows,
            "wall_s": time.time() - t0,
            "returned_tokens": sum(int(row["token_count"]) for row in rows),
        }

    base = result["modes"][args.base_mode]["records"]
    for mode in args.modes:
        if mode == args.base_mode:
            continue
        result["comparisons"][f"{mode}_vs_{args.base_mode}"] = _compare(
            base, result["modes"][mode]["records"]
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
    print(json.dumps(result["comparisons"], indent=2, sort_keys=True))
    ok = all(comp["all_match"] for comp in result["comparisons"].values())
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
