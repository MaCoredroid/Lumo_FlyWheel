#!/usr/bin/env python3
"""Warmup-pass executor for Track B Round 4a.

POSTs the captured Codex CLI static system prompt (instructions + tools) plus a
tiny user message to vLLM /v1/responses, so that the prefix cache contains the
~64K-token task-agnostic prefix before per-task measurement begins. This eliminates
the cold turn-1 prefill cost (~90s) that dominates v3 wallclock.

Per spec §5.3 protocol:
  1. POST /reset_prefix_cache (caller's responsibility — runner does this first).
  2. Call this script — first invocation primes the cache.
  3. Call this script again — second invocation verifies cache hit rate ≥ 0.95.

The verifier captures the prefix_cache_queries/hits delta for the second call
and asserts it meets the rule-17 threshold.

Inputs:
  --system-prompt-json <path>   codex_system_prompt.json artifact (from
                                build_track_b_codex_system_prompt_decomposition.py
                                or its companion writer)
  --endpoint <url>              upstream /v1 base (e.g. http://127.0.0.1:9950/v1)
  --metrics-url <url>           Prometheus metrics endpoint (cache-rate verify)
  --api-key <str>               OpenAI-compat API key
  --model <str>                 model name
  --mode prime|verify|both      prime = single warmup; verify = two warmups +
                                second-call hit-rate ≥ threshold; both = same
                                as verify (default)
  --hit-rate-threshold <float>  default 0.95
  --out <path>                  per-attempt warmup_pass artifact
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

WARMUP_USER_MESSAGE = "ok"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_metric_value(metrics_text: str, name: str) -> float | None:
    """Return the first matching sample value for the given metric name (any labels)."""
    pat = re.compile(rf"^{re.escape(name)}\{{[^}}]*\}}\s+([0-9.eE+-]+)\s*$", re.MULTILINE)
    m = pat.search(metrics_text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _fetch_metrics(metrics_url: str, *, timeout: float = 10.0) -> str:
    r = requests.get(metrics_url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _post_warmup(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    instructions: str,
    tools: list[Any],
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    payload = {
        "model": model,
        "instructions": instructions,
        "tools": tools,
        # tool_choice="auto" matches what Codex sends, ensuring vLLM renders the
        # full tool inventory into the prompt (with "none", vLLM strips tools
        # and the warmup payload tokenizes to only ~4.5K — defeating the point).
        "tool_choice": "auto",
        "input": [{"role": "user", "content": WARMUP_USER_MESSAGE, "type": "message"}],
        "stream": False,
        "store": False,
        "max_output_tokens": 8,
    }
    r = requests.post(
        f"{endpoint.rstrip('/')}/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    wall_s = time.monotonic() - started
    body_text = r.text
    try:
        body_json = r.json() if r.status_code < 500 else None
    except ValueError:
        body_json = None
    usage = (body_json or {}).get("usage") if isinstance(body_json, dict) else None
    return {
        "status_code": r.status_code,
        "wall_s": round(wall_s, 6),
        "usage": usage,
        "ok": r.status_code < 400,
        "body_excerpt": body_text[:500] if r.status_code >= 400 else None,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--system-prompt-json", required=True)
    p.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    p.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--model", default="qwen3.5-27b")
    p.add_argument("--mode", choices=["prime", "verify", "both"], default="both")
    p.add_argument("--hit-rate-threshold", type=float, default=0.95)
    p.add_argument("--out", default="")
    p.add_argument("--timeout-s", type=float, default=300.0)
    args = p.parse_args()

    sp = json.loads(Path(args.system_prompt_json).read_text(encoding="utf-8"))
    if sp.get("schema") != "lumo.track_b.codex_system_prompt.v1":
        print(f"warning: unexpected system-prompt schema: {sp.get('schema')!r}", file=sys.stderr)
    instructions = sp.get("instructions") or ""
    tools = sp.get("tools") or []
    static_hash = sp.get("static_content_hash") or ""

    record: dict[str, Any] = {
        "schema": "lumo.track_b.codex_warmup_pass.v1",
        "ts": _now(),
        "endpoint": args.endpoint,
        "model": args.model,
        "system_prompt_content_hash": static_hash,
        "instructions_tokens": sp.get("instructions_tokens"),
        "tools_tokens": sp.get("tools_tokens"),
        "mode": args.mode,
        "hit_rate_threshold": args.hit_rate_threshold,
    }

    # First call (prime)
    metrics_pre1 = _fetch_metrics(args.metrics_url)
    q_pre1 = _parse_metric_value(metrics_pre1, "vllm:prefix_cache_queries_total") or 0.0
    h_pre1 = _parse_metric_value(metrics_pre1, "vllm:prefix_cache_hits_total") or 0.0
    call1 = _post_warmup(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        instructions=instructions,
        tools=tools,
        timeout=args.timeout_s,
    )
    metrics_post1 = _fetch_metrics(args.metrics_url)
    q_post1 = _parse_metric_value(metrics_post1, "vllm:prefix_cache_queries_total") or 0.0
    h_post1 = _parse_metric_value(metrics_post1, "vllm:prefix_cache_hits_total") or 0.0
    dq1 = q_post1 - q_pre1
    dh1 = h_post1 - h_pre1
    record["prime"] = {
        **call1,
        "prefix_cache_queries_delta": dq1,
        "prefix_cache_hits_delta": dh1,
        "hit_rate": (dh1 / dq1) if dq1 > 0 else None,
    }
    if not call1["ok"]:
        record["passed"] = False
        record["fail_reason"] = f"prime_call_status_{call1['status_code']}"
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"passed": False, "fail_reason": record["fail_reason"], "prime": record["prime"]}, indent=2))
        return 2

    if args.mode == "prime":
        record["passed"] = True
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"passed": True, "mode": "prime", "prime": record["prime"]}, indent=2))
        return 0

    # Second call (verify)
    metrics_pre2 = _fetch_metrics(args.metrics_url)
    q_pre2 = _parse_metric_value(metrics_pre2, "vllm:prefix_cache_queries_total") or 0.0
    h_pre2 = _parse_metric_value(metrics_pre2, "vllm:prefix_cache_hits_total") or 0.0
    call2 = _post_warmup(
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        instructions=instructions,
        tools=tools,
        timeout=args.timeout_s,
    )
    metrics_post2 = _fetch_metrics(args.metrics_url)
    q_post2 = _parse_metric_value(metrics_post2, "vllm:prefix_cache_queries_total") or 0.0
    h_post2 = _parse_metric_value(metrics_post2, "vllm:prefix_cache_hits_total") or 0.0
    dq2 = q_post2 - q_pre2
    dh2 = h_post2 - h_pre2
    hit_rate2 = (dh2 / dq2) if dq2 > 0 else 0.0
    record["verify"] = {
        **call2,
        "prefix_cache_queries_delta": dq2,
        "prefix_cache_hits_delta": dh2,
        "hit_rate": hit_rate2,
    }

    passed = call2["ok"] and hit_rate2 >= args.hit_rate_threshold
    record["passed"] = passed
    if not passed:
        record["fail_reason"] = (
            f"verify_call_status_{call2['status_code']}" if not call2["ok"]
            else f"verify_hit_rate_{hit_rate2:.3f}_below_{args.hit_rate_threshold}"
        )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "passed": passed,
        "system_prompt_content_hash": static_hash,
        "prime_hit_rate": record["prime"]["hit_rate"],
        "prime_wall_s": record["prime"]["wall_s"],
        "verify_hit_rate": hit_rate2,
        "verify_wall_s": call2["wall_s"],
        "fail_reason": record.get("fail_reason"),
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
