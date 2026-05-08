#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
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


def _validate_codex_command_template(template: str, *, require_trace_out: bool = True) -> None:
    if require_trace_out and "{trace_out}" not in template:
        raise ValueError("codex command template must include {trace_out}")


def _validate_runtime_config_hash(runtime_config_hash: str) -> None:
    digest = runtime_config_hash.removeprefix("sha256:")
    if (
        not runtime_config_hash.startswith("sha256:")
        or len(digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in digest)
    ):
        raise ValueError("--runtime-config-hash must be a sha256:<64-hex-digest> value")


def _validate_server_launch_args(args: argparse.Namespace) -> None:
    if args.profile_target not in ("server-launch", "container-server-launch"):
        return
    if not args.server_launch_command:
        raise ValueError(f"--server-launch-command is required with --profile-target {args.profile_target}")
    if not args.server_ready_url:
        raise ValueError(f"--server-ready-url is required with --profile-target {args.profile_target}")
    if args.profile_target == "container-server-launch":
        if not args.container_name:
            raise ValueError("--container-name is required with --profile-target container-server-launch")
        if not args.container_profile_csv:
            raise ValueError("--container-profile-csv is required with --profile-target container-server-launch")


def _profile_path(out_root: Path, archetype: str) -> Path:
    return out_root / f"ncu_{archetype}.csv"


def _metadata_path(out_root: Path, archetype: str) -> Path:
    return out_root / f"ncu_{archetype}.json"


def _server_stdout_path(out_root: Path, archetype: str) -> Path:
    return out_root / f"ncu_{archetype}_server_stdout.log"


def _server_stderr_path(out_root: Path, archetype: str) -> Path:
    return out_root / f"ncu_{archetype}_server_stderr.log"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_csv_number(value: str) -> float | None:
    cleaned = value.strip().strip('"').replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    if not cleaned:
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _metric_values(text: str) -> dict[str, list[float]]:
    values = {metric: [] for metric in NCU_REQUIRED_METRICS}
    for row in csv.reader(text.splitlines()):
        matched = next((metric for metric in NCU_REQUIRED_METRICS if metric in row), None)
        if matched is None:
            continue
        values[matched].extend(
            parsed
            for cell in row
            if cell != matched
            for parsed in [_finite_csv_number(cell)]
            if parsed is not None
        )
    return values


def _validate_profile(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"NCU profile missing or empty: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if "No kernels were profiled" in text:
        raise RuntimeError(f"NCU profile {path} contains no profiled kernels")
    values = _metric_values(text)
    missing = [metric for metric in NCU_REQUIRED_METRICS if metric not in text]
    if missing:
        raise RuntimeError(f"NCU profile {path} is missing required metrics: {', '.join(missing)}")
    nonfinite = [metric for metric, metric_values in values.items() if not metric_values]
    if nonfinite:
        raise RuntimeError(f"NCU profile {path} has no finite values for required metrics: {', '.join(nonfinite)}")


def _ncu_prefix(args: argparse.Namespace, archetype: str, *, log_file: str | Path | None = None) -> list[str]:
    out_root = Path(args.out_root)
    profile_log_file = str(log_file) if log_file is not None else str(_profile_path(out_root, archetype))
    return [
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
        profile_log_file,
    ]


def _task_command(args: argparse.Namespace, archetype: str) -> list[str]:
    task_out_root = Path(args.task_out_root)
    task_id = ARCHETYPE_TASKS[archetype]
    command = [
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
        "--reset-prefix-cache-url",
        args.reset_prefix_cache_url,
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
    if args.defer_codex_trace_out:
        command.append("--defer-codex-trace-out")
    if args.defer_vllm_request_metrics_join:
        command.append("--defer-vllm-request-metrics-join")
    if args.defer_dcgm_profile_fields:
        command.append("--defer-dcgm-profile-fields")
    return command


def _ncu_command(args: argparse.Namespace, archetype: str) -> list[str]:
    return [*_ncu_prefix(args, archetype), "--", *_task_command(args, archetype)]


def _format_server_launch_command(template: str, args: argparse.Namespace, archetype: str) -> str:
    replacements = {
        "{archetype}": archetype,
        "{model}": args.model,
        "{runtime_config_hash}": args.runtime_config_hash,
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def _server_ncu_command(args: argparse.Namespace, archetype: str) -> list[str]:
    server_command = _format_server_launch_command(args.server_launch_command, args, archetype)
    return [*_ncu_prefix(args, archetype), "--", "bash", "-lc", server_command]


def _container_profile_csv(args: argparse.Namespace, archetype: str) -> str:
    return _format_server_launch_command(args.container_profile_csv, args, archetype)


def _container_ncu_command(args: argparse.Namespace, archetype: str) -> list[str]:
    server_command = _format_server_launch_command(args.server_launch_command, args, archetype)
    container_csv = _container_profile_csv(args, archetype)
    quoted_ncu_command = " ".join(
        shlex.quote(part)
        for part in [
            "rm",
            "-f",
            container_csv,
            ";",
            *_ncu_prefix(args, archetype, log_file=container_csv),
            "--",
            "bash",
            "-lc",
            server_command,
        ]
    )
    # Keep the separator outside shell quoting.
    quoted_ncu_command = quoted_ncu_command.replace(shlex.quote(";"), ";")
    return ["docker", "exec", args.container_name, "bash", "-lc", quoted_ncu_command]


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
        "profile_target": args.profile_target,
        "server_launch_command": args.server_launch_command if args.profile_target in ("server-launch", "container-server-launch") else "",
        "server_ready_url": args.server_ready_url if args.profile_target in ("server-launch", "container-server-launch") else "",
        "container_name": args.container_name if args.profile_target == "container-server-launch" else "",
        "container_profile_csv": _container_profile_csv(args, archetype) if args.profile_target == "container-server-launch" else "",
        "container_server_stop_command": args.container_server_stop_command if args.profile_target == "container-server-launch" else "",
        "deferred_instrumentation_checks": sorted(
            check
            for check, enabled in {
                "codex_trace_out_supported": args.defer_codex_trace_out,
                "vllm_request_metrics_join_available": args.defer_vllm_request_metrics_join,
                "dcgm_profile_fields_available": args.defer_dcgm_profile_fields,
            }.items()
            if enabled
        ),
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


def _popen(command: list[str], *, stdout_path: Path, stderr_path: Path) -> subprocess.Popen[bytes]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("wb")
    stderr_handle = stderr_path.open("wb")
    try:
        return subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=os.environ.copy(),
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise


def _wait_for_ready(url: str, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"server did not become ready at {url}: {last_error}")


def _wait_for_server_ready(
    url: str,
    process: subprocess.Popen[bytes],
    *,
    stderr_path: Path,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:] if stderr_path.is_file() else ""
            detail = f": {stderr_tail}" if stderr_tail else ""
            raise RuntimeError(f"server exited before ready at {url} with return code {returncode}{detail}")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"server did not become ready at {url}: {last_error}")


def _terminate_server(process: subprocess.Popen[bytes], *, timeout_s: float) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_s)


def _stop_container_server(args: argparse.Namespace) -> None:
    if not args.container_server_stop_command:
        return
    result = _run(["docker", "exec", args.container_name, "bash", "-lc", args.container_server_stop_command])
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr, end="")


def _copy_container_profile(args: argparse.Namespace, archetype: str) -> None:
    result = _run(
        [
            "docker",
            "cp",
            f"{args.container_name}:{_container_profile_csv(args, archetype)}",
            str(_profile_path(Path(args.out_root), archetype)),
        ]
    )
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr, end="")
        raise RuntimeError(f"failed to copy container NCU profile for {archetype}")


def _server_returncode_ok(returncode: int | None) -> bool:
    return returncode in (0, -15, 143)


def _run_server_profile(args: argparse.Namespace, archetype: str) -> int:
    out_root = Path(args.out_root)
    server_command = _server_ncu_command(args, archetype)
    server = _popen(
        server_command,
        stdout_path=_server_stdout_path(out_root, archetype),
        stderr_path=_server_stderr_path(out_root, archetype),
    )
    task_result: subprocess.CompletedProcess[str] | None = None
    try:
        _wait_for_server_ready(
            args.server_ready_url,
            server,
            stderr_path=_server_stderr_path(out_root, archetype),
            timeout_s=args.server_ready_timeout_s,
        )
        task_result = _run(_task_command(args, archetype))
        if task_result.returncode != 0:
            print(task_result.stderr, file=sys.stderr, end="")
            return task_result.returncode
    finally:
        _terminate_server(server, timeout_s=args.server_shutdown_timeout_s)
    if not _server_returncode_ok(server.returncode):
        server_stderr = _server_stderr_path(out_root, archetype).read_text(
            encoding="utf-8",
            errors="replace",
        )
        print(server_stderr, file=sys.stderr, end="")
        return int(server.returncode or 2)
    _validate_profile(_profile_path(out_root, archetype))
    _write_profile_metadata(args, archetype, server_command)
    return 0


def _run_container_server_profile(args: argparse.Namespace, archetype: str) -> int:
    out_root = Path(args.out_root)
    server_command = _container_ncu_command(args, archetype)
    server = _popen(
        server_command,
        stdout_path=_server_stdout_path(out_root, archetype),
        stderr_path=_server_stderr_path(out_root, archetype),
    )
    try:
        _wait_for_server_ready(
            args.server_ready_url,
            server,
            stderr_path=_server_stderr_path(out_root, archetype),
            timeout_s=args.server_ready_timeout_s,
        )
        task_result = _run(_task_command(args, archetype))
        if task_result.returncode != 0:
            print(task_result.stderr, file=sys.stderr, end="")
            return task_result.returncode
    finally:
        _stop_container_server(args)
        _terminate_server(server, timeout_s=args.server_shutdown_timeout_s)
    if not _server_returncode_ok(server.returncode):
        server_stderr = _server_stderr_path(out_root, archetype).read_text(
            encoding="utf-8",
            errors="replace",
        )
        print(server_stderr, file=sys.stderr, end="")
        return int(server.returncode or 2)
    _copy_container_profile(args, archetype)
    _validate_profile(_profile_path(out_root, archetype))
    _write_profile_metadata(args, archetype, server_command)
    return 0


def run_profiles(args: argparse.Namespace) -> int:
    _validate_codex_command_template(args.codex_command_template, require_trace_out=not args.defer_codex_trace_out)
    _validate_runtime_config_hash(args.runtime_config_hash)
    _validate_server_launch_args(args)
    if args.profile_target == "container-server-launch" and shutil.which("docker") is None:
        raise RuntimeError("docker binary not found")
    if args.profile_target != "container-server-launch" and shutil.which(args.ncu_bin) is None:
        raise RuntimeError(f"ncu binary not found: {args.ncu_bin}")
    out_root = Path(args.out_root).resolve()
    task_out_root = Path(args.task_out_root).resolve()
    args.out_root = str(out_root)
    args.task_out_root = str(task_out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    task_out_root.mkdir(parents=True, exist_ok=True)
    archetypes = list(ARCHETYPE_TASKS) if args.archetype == "all" else [args.archetype]
    for archetype in archetypes:
        if args.profile_target == "server-launch":
            rc = _run_server_profile(args, archetype)
            if rc != 0:
                return rc
            continue
        if args.profile_target == "container-server-launch":
            rc = _run_container_server_profile(args, archetype)
            if rc != 0:
                return rc
            continue
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
    parser.add_argument(
        "--profile-target",
        choices=["task-wrapper", "server-launch", "container-server-launch"],
        default="task-wrapper",
    )
    parser.add_argument("--launch-skip-before-match", type=int, default=200)
    parser.add_argument("--launch-count", type=int, default=16)
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--reset-prefix-cache-url", default="http://127.0.0.1:9950/reset_prefix_cache")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--runtime-config-hash", required=True)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--vllm-request-metrics-jsonl", default="")
    parser.add_argument("--defer-codex-trace-out", action="store_true")
    parser.add_argument("--defer-vllm-request-metrics-join", action="store_true")
    parser.add_argument("--defer-dcgm-profile-fields", action="store_true")
    parser.add_argument(
        "--server-launch-command",
        default="",
        help=(
            "Shell command to launch the profiled server under NCU when "
            "--profile-target server-launch is used. Supports {archetype}, "
            "{model}, and {runtime_config_hash} placeholders."
        ),
    )
    parser.add_argument("--server-ready-url", default="")
    parser.add_argument("--server-ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--server-shutdown-timeout-s", type=float, default=30.0)
    parser.add_argument("--container-name", default="")
    parser.add_argument(
        "--container-profile-csv",
        default="/tmp/track_b_ncu_{archetype}.csv",
        help="Path inside the container where NCU writes CSV before docker cp.",
    )
    parser.add_argument(
        "--container-server-stop-command",
        default="",
        help="Optional shell command executed inside the container to stop the profiled server after the task.",
    )
    parser.add_argument("--codex-command-template", required=True)
    args = parser.parse_args(argv)
    try:
        return run_profiles(args)
    except Exception as exc:
        print(f"run_track_b_e2e_ncu_profiles.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
