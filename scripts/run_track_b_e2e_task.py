#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lumo_flywheel_serving.metrics import (  # noqa: E402
    compute_vllm_per_request_metrics,
    parse_prometheus_samples,
    parse_prometheus_text,
    resolve_metric_schema,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _request(method: str, url: str, *, api_key: str | None = None, timeout: float = 20.0) -> requests.Response:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    response = requests.request(method, url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def _write_prompt(workspace: Path, prompt_path: Path) -> None:
    pieces: list[str] = []
    for rel in ["AGENTS.md", ".scenario_variant"]:
        path = workspace / rel
        if path.is_file():
            pieces.append(f"## {rel}\n\n{path.read_text(encoding='utf-8', errors='replace').strip()}\n")
    prompt_path.write_text("\n".join(pieces).strip() + "\n", encoding="utf-8")


def _format_command(template: str, mapping: dict[str, str]) -> list[str]:
    rendered = template.format(**mapping)
    return shlex.split(rendered)


def _metrics_text(metrics_url: str) -> str:
    return _request("GET", metrics_url).text


def _write_vllm_per_turn(task_dir: Path, before_raw: str, after_raw: str) -> None:
    schema = resolve_metric_schema(parse_prometheus_text(after_raw))
    per_request = compute_vllm_per_request_metrics(
        parse_prometheus_samples(before_raw),
        parse_prometheus_samples(after_raw),
        schema,
    )
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps({"requests": per_request}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_sampler(args: argparse.Namespace, task_dir: Path) -> subprocess.Popen[str] | None:
    if args.no_dcgm:
        return None
    sampler_out = task_dir / "dcgm_samples.jsonl"
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "sample_dcgm_during_task.py"),
        "--out",
        str(sampler_out),
        "--gpu",
        str(args.gpu),
        "--interval-s",
        str(args.dcgm_interval_s),
    ]
    return subprocess.Popen(command, cwd=REPO_ROOT, text=True)


def _stop_sampler(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def run_one(args: argparse.Namespace, family: str, variant: str) -> int:
    workspace = REPO_ROOT / "benchmark_blueprints" / "families" / family / "workspace_bundle" / variant
    if not workspace.is_dir():
        raise RuntimeError(f"workspace bundle missing: {workspace}")
    task_dir = Path(args.out_root) / f"round_{args.round}" / f"{family}__{variant}" / f"run_{args.attempt:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    trace_out = task_dir / "codex_trace.jsonl"
    prompt_path = task_dir / "prompt.md"
    _write_prompt(workspace, prompt_path)

    _request("GET", args.health_url)
    if args.reset_prefix_cache_url:
        _request("POST", args.reset_prefix_cache_url, api_key=args.api_key, timeout=30)
    metrics_pre = _metrics_text(args.metrics_url)
    (task_dir / "vllm_metrics_pre.txt").write_text(metrics_pre, encoding="utf-8")

    sampler = _run_sampler(args, task_dir)
    started = time.monotonic()
    command = _format_command(
        args.codex_command_template,
        {
            "trace_out": str(trace_out),
            "workspace": str(workspace),
            "prompt_file": str(prompt_path),
            "model": args.model,
            "endpoint": args.endpoint,
            "api_key": args.api_key,
            "task_dir": str(task_dir),
        },
    )
    result: subprocess.CompletedProcess[str] | None = None
    try:
        result = subprocess.run(
            command,
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout_s,
            env={**os.environ, "OPENAI_API_KEY": args.api_key, "OPENAI_BASE_URL": args.endpoint},
        )
    finally:
        _stop_sampler(sampler)
    elapsed_s = time.monotonic() - started
    (task_dir / "codex_stdout.log").write_text(result.stdout if result else "", encoding="utf-8")
    (task_dir / "codex_stderr.log").write_text(result.stderr if result else "", encoding="utf-8")
    metrics_post = _metrics_text(args.metrics_url)
    (task_dir / "vllm_metrics_post.txt").write_text(metrics_post, encoding="utf-8")
    _write_vllm_per_turn(task_dir, metrics_pre, metrics_post)
    metadata = {
        "schema": "lumo.track_b.e2e_runner_metadata.v1",
        "recorded_at": _now(),
        "family": family,
        "variant": variant,
        "round": args.round,
        "attempt": args.attempt,
        "workspace": str(workspace.relative_to(REPO_ROOT)),
        "trace_out": str(trace_out),
        "elapsed_s": elapsed_s,
        "codex_command_template": args.codex_command_template,
        "codex_exit_code": result.returncode if result else None,
    }
    (task_dir / "runner_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not trace_out.is_file():
        raise RuntimeError("patched Codex did not create codex_trace.jsonl; Round 0 preflight must fail")
    return int(result.returncode if result else 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Track B E2E task with truthful artifact capture.")
    parser.add_argument("family", nargs="?", help="Benchmark family id, or omit with --tasks all.")
    parser.add_argument("variant", nargs="?", default="v1-clean-baseline")
    parser.add_argument("--tasks", choices=["one", "all"], default="one")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1, help="Number of independent attempts to run per task.")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "output" / "track_b_e2e"))
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--reset-prefix-cache-url", default="http://127.0.0.1:9950/reset_prefix_cache")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dcgm-interval-s", type=float, default=0.01)
    parser.add_argument("--no-dcgm", action="store_true")
    parser.add_argument(
        "--codex-command-template",
        required=True,
        help=(
            "Command template for the patched Codex binary. Available placeholders: "
            "{trace_out}, {workspace}, {prompt_file}, {model}, {endpoint}, {api_key}, {task_dir}."
        ),
    )
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")

    if args.tasks == "all":
        from build_track_b_e2e_summary import TRACK_B_E2E_TASKS

        failures = 0
        for task_id in TRACK_B_E2E_TASKS:
            family, variant = task_id.split("/", 1)
            base_attempt = args.attempt
            for offset in range(args.repeat):
                args.attempt = base_attempt + offset
                try:
                    failures += 1 if run_one(args, family, variant) != 0 else 0
                except Exception as exc:
                    print(f"{task_id} attempt {args.attempt}: {exc}", file=sys.stderr)
                    failures += 1
            args.attempt = base_attempt
        return 1 if failures else 0
    if not args.family:
        parser.error("family is required unless --tasks all is used")
    try:
        failures = 0
        base_attempt = args.attempt
        for offset in range(args.repeat):
            args.attempt = base_attempt + offset
            failures += 1 if run_one(args, args.family, args.variant) != 0 else 0
        args.attempt = base_attempt
        return 1 if failures else 0
    except Exception as exc:
        print(f"run_track_b_e2e_task.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
