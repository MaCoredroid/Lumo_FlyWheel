#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNEL_ID = "::regex:.*linear.*|.*attention.*|.*sample.*|.*spec.*:"
NCU_REQUIRED_METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_memory_ld_pred_on.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "tpc__warps_active.avg.pct_of_peak_sustained_active",
)
ARCHETYPE_TASKS = {
    "long-text": "sqlalchemy-2-session-modernization/v1-clean-baseline",
    "tool-call-frame": "policy-aware-request-resolution/v1-clean-baseline",
    "pure-investigation": "dead-flag-reachability-audit/v1-clean-baseline",
    "multimodal-prefill": "responsive-checkout-visual-regression/v1-clean-baseline",
    "subagent-orchestration": "fanout-fullstack-release-blocker/v1-clean-baseline",
}


def _default_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def _validate_codex_command_template(template: str) -> None:
    if "{trace_out}" not in template:
        raise ValueError("codex command template must include {trace_out}")


def _validate_runtime_config_hash(runtime_config_hash: str) -> None:
    if not runtime_config_hash.startswith("sha256:") or not runtime_config_hash.removeprefix("sha256:"):
        raise ValueError("--runtime-config-hash must be a non-empty sha256:<digest> value")


def _profile_path(out_root: Path, archetype: str) -> Path:
    return out_root / f"ncu_{archetype}.csv"


def _metadata_path(out_root: Path, archetype: str) -> Path:
    return out_root / f"ncu_{archetype}.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_profile(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"NCU profile missing or empty: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    missing = [metric for metric in NCU_REQUIRED_METRICS if metric not in text]
    if missing:
        raise RuntimeError(f"NCU profile {path} is missing required metrics: {', '.join(missing)}")


def _ncu_command(args: argparse.Namespace, archetype: str) -> list[str]:
    out_root = Path(args.out_root)
    task_out_root = Path(args.task_out_root)
    task_id = ARCHETYPE_TASKS[archetype]
    command = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--kernel-id",
        KERNEL_ID,
        "--launch-skip-before-match",
        str(args.launch_skip_before_match),
        "--launch-count",
        str(args.launch_count),
        "--metrics",
        ",".join(NCU_REQUIRED_METRICS),
        "--csv",
        "--log-file",
        str(_profile_path(out_root, archetype)),
        "--",
        args.python,
        str(REPO_ROOT / "scripts" / "run_track_b_e2e_task.py"),
        task_id,
        "--round",
        str(args.round),
        "--attempt",
        "1",
        "--repeat",
        "1",
        "--out-root",
        str(task_out_root),
        "--health-url",
        args.health_url,
        "--metrics-url",
        args.metrics_url,
        "--endpoint",
        args.endpoint,
        "--api-key",
        args.api_key,
        "--model",
        args.model,
        "--runtime-config-hash",
        args.runtime_config_hash,
        "--timeout-s",
        str(args.timeout_s),
        "--no-dcgm",
        "--ncu-mode",
        "--codex-command-template",
        args.codex_command_template,
    ]
    if args.vllm_request_metrics_jsonl:
        command.extend(["--vllm-request-metrics-jsonl", args.vllm_request_metrics_jsonl])
    return command


def _write_profile_metadata(args: argparse.Namespace, archetype: str, command: list[str]) -> None:
    out_root = Path(args.out_root)
    profile_path = _profile_path(out_root, archetype)
    metadata = {
        "schema": "lumo.track_b.ncu_archetype_profile.v1",
        "recorded_at": _now(),
        "round": args.round,
        "archetype": archetype,
        "task_id": ARCHETYPE_TASKS[archetype],
        "runtime_config_hash": args.runtime_config_hash,
        "profile_csv": str(profile_path.relative_to(REPO_ROOT)) if profile_path.is_relative_to(REPO_ROOT) else str(profile_path),
        "required_metrics": list(NCU_REQUIRED_METRICS),
        "ncu_command": command,
    }
    _metadata_path(out_root, archetype).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )


def run_profiles(args: argparse.Namespace) -> int:
    _validate_codex_command_template(args.codex_command_template)
    _validate_runtime_config_hash(args.runtime_config_hash)
    if shutil.which(args.ncu_bin) is None:
        raise RuntimeError(f"ncu binary not found: {args.ncu_bin}")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    Path(args.task_out_root).mkdir(parents=True, exist_ok=True)
    archetypes = list(ARCHETYPE_TASKS) if args.archetype == "all" else [args.archetype]
    for archetype in archetypes:
        command = _ncu_command(args, archetype)
        result = _run(command)
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr, end="")
            return result.returncode
        _validate_profile(_profile_path(out_root, archetype))
        _write_profile_metadata(args, archetype, command)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Track B E2E one-shot NCU archetype profiles.")
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--archetype", choices=["all", *ARCHETYPE_TASKS.keys()], default="all")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "output" / "track_b_e2e"))
    parser.add_argument(
        "--task-out-root",
        default=str(REPO_ROOT / "output" / "track_b_e2e" / "ncu_task_runs"),
        help="Separate root for profiled task artifacts so NCU runs do not contaminate measurement rounds.",
    )
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--ncu-bin", default="ncu")
    parser.add_argument("--launch-skip-before-match", type=int, default=200)
    parser.add_argument("--launch-count", type=int, default=16)
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--runtime-config-hash", required=True)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--vllm-request-metrics-jsonl", default="")
    parser.add_argument("--codex-command-template", required=True)
    args = parser.parse_args(argv)
    try:
        return run_profiles(args)
    except Exception as exc:
        print(f"run_track_b_e2e_ncu_profiles.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
