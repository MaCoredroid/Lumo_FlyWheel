#!/usr/bin/env python3
"""Step 0d: B-1 / B-2 / B-3 correctness gate against the live spec_decode config.

Drives ``run_track_b_tool_call_gate.py --suite {b1,b2,b3}`` against the
running vLLM + proxy and aggregates a single Step 0d artifact. Resets prefix
cache between suites so each gate measures the same cold/warm shape. Records
which suites pass/fail plus the live ``runtime_config_hash`` so downstream
diagnosis can attribute findings to a specific spec_decode config.

Live runtime is currently ``method=suffix, num_speculative_tokens=12,
suffix_decoding_max_tree_depth=32`` per the prelaunch hook. v1 of the
codex-harness-spec-decode plan called for B-1/B-2/B-3 against ngram-PLD
candidates 020/025/028; v2 (this driver) targets the actually-shipped
SuffixDecoding config, with the candidate ngram configs as a Round 1
fallback only if the live config fails the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _run_suite(args: argparse.Namespace, suite: str, out_dir: Path) -> dict[str, Any]:
    out_path = out_dir / f"{suite}.json"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_track_b_tool_call_gate.py"),
        "--suite",
        suite,
        "--endpoint",
        args.endpoint,
        "--health-url",
        args.health_url,
        "--metrics-url",
        args.metrics_url,
        "--reset-prefix-cache-url",
        args.reset_prefix_cache_url,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--benchmark-family",
        args.benchmark_family,
        "--variant",
        args.variant,
        "--probe-count",
        str(args.probe_count),
        "--concurrent-requests",
        str(args.concurrent_requests),
        "--min-pass-rate",
        str(args.min_pass_rate),
        "--reset-prefix-cache",
        "--output",
        str(out_path),
    ]
    if not args.exact_arguments:
        cmd.append("--no-exact-arguments")
    if args.measure_throughput:
        cmd.append("--measure-throughput")
        cmd.extend(["--target-decode-tps", str(args.target_decode_tps)])

    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_ROOT)
    payload: dict[str, Any] = {
        "suite": suite,
        "command": cmd,
        "rc": result.returncode,
        "stdout_tail": result.stdout[-1500:],
        "stderr_tail": result.stderr[-1500:],
        "out_path": str(out_path),
    }
    if out_path.is_file():
        try:
            payload["report"] = json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            payload["report_error"] = str(exc)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 0d: B-1/B-2/B-3 correctness gate against live spec_decode config.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "output" / "track_b_step_0d_live_suffix")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8022/v1")
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--reset-prefix-cache-url", default="http://127.0.0.1:9950/reset_prefix_cache")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--benchmark-family", default="release-note-to-plan-translation")
    parser.add_argument("--variant", default="v1-clean-baseline")
    parser.add_argument("--probe-count", type=int, default=4)
    parser.add_argument("--concurrent-requests", type=int, default=4)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--measure-throughput", action="store_true")
    parser.add_argument(
        "--exact-arguments",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Default True for byte-exact match. Pass --no-exact-arguments to "
            "use structural matching (parsed argument shape + required-contains "
            "checks) -- needed when the model has legitimate output "
            "nondeterminism (e.g., apply_patch path variants under "
            "SuffixDecoding) and the bug under test is parser-level, not "
            "tokenizer-level."
        ),
    )
    parser.add_argument("--target-decode-tps", type=float, default=9.0)
    parser.add_argument("--runtime-config-hash", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runtime_config_hash = args.runtime_config_hash or os.environ.get("LUMO_TRACK_B_RUNTIME_CONFIG_HASH", "")
    if not runtime_config_hash:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "build_track_b_runtime_config_hash.py")],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            runtime_config_hash = result.stdout.strip()

    suites = ("b1", "b2", "b3")
    suite_results: dict[str, dict[str, Any]] = {}
    for suite in suites:
        print(f"running step 0d / {suite}...", file=sys.stderr)
        suite_results[suite] = _run_suite(args, suite, args.out_dir)

    aggregated = {
        "schema": "lumo.track_b.step_0d_correctness_gate.v1",
        "measured_at": _now(),
        "runtime_config_hash": runtime_config_hash,
        "live_speculative_method": "suffix",
        "live_num_speculative_tokens": 12,
        "benchmark_family": args.benchmark_family,
        "variant": args.variant,
        "model": args.model,
        "probe_count": args.probe_count,
        "concurrent_requests": args.concurrent_requests,
        "min_pass_rate": args.min_pass_rate,
        "target_decode_tps": args.target_decode_tps if args.measure_throughput else None,
        "suites": suite_results,
        "gate_pass": all(
            isinstance(suite.get("report"), dict) and suite["report"].get("pass") is True
            for suite in suite_results.values()
        ),
        "per_suite_pass": {
            suite: bool(isinstance(payload.get("report"), dict) and payload["report"].get("pass") is True)
            for suite, payload in suite_results.items()
        },
        "per_suite_pass_rate": {
            suite: payload["report"].get("pass_rate") if isinstance(payload.get("report"), dict) else None
            for suite, payload in suite_results.items()
        },
    }
    aggregated_path = args.out_dir / "step_0d_correctness_gate.json"
    aggregated_path.write_text(json.dumps(aggregated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"step 0d aggregated -> {aggregated_path}", file=sys.stderr)
    print(json.dumps({k: aggregated[k] for k in ("gate_pass", "per_suite_pass", "per_suite_pass_rate")}, indent=2))
    return 0 if aggregated["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
