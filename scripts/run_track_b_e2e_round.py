#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def _validate_codex_command_template(template: str) -> None:
    if "{trace_out}" not in template:
        raise ValueError("codex command template must include {trace_out}")


def _tasks() -> list[str]:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_track_b_e2e_summary import TRACK_B_E2E_TASKS  # noqa: PLC0415

    return list(TRACK_B_E2E_TASKS)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _attempt_dir(out_root: Path, round_index: int, task_id: str, attempt: int) -> Path:
    family, variant = task_id.split("/", 1)
    return out_root / f"round_{round_index}" / f"{family}__{variant}" / f"run_{attempt:02d}"


def _attempt_wallclocks(out_root: Path, round_index: int, task_id: str, repeat: int) -> list[float]:
    wallclocks: list[float] = []
    for attempt in range(1, repeat + 1):
        metadata_path = _attempt_dir(out_root, round_index, task_id, attempt) / "runner_metadata.json"
        payload = _read_json(metadata_path)
        elapsed_s = payload.get("elapsed_s")
        if not isinstance(elapsed_s, (int, float)):
            raise RuntimeError(f"missing numeric elapsed_s in {metadata_path}")
        wallclocks.append(float(elapsed_s))
    return wallclocks


def _preflight_command(args: argparse.Namespace, preflight_out: Path) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "preflight_track_b_e2e.py"),
        "--out",
        str(preflight_out),
        "--health-url",
        args.health_url,
        "--metrics-url",
        args.metrics_url,
    ]
    if args.vllm_request_metrics_jsonl:
        command.extend(["--vllm-request-metrics-jsonl", args.vllm_request_metrics_jsonl])
    return command


def _runner_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "run_track_b_e2e_task.py"),
        "--round",
        str(args.round),
        "--tasks",
        "all",
        "--repeat",
        str(args.repeat),
        "--out-root",
        args.out_root,
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
        "--timeout-s",
        str(args.timeout_s),
        "--codex-command-template",
        args.codex_command_template,
    ]
    if args.vllm_request_metrics_jsonl:
        command.extend(["--vllm-request-metrics-jsonl", args.vllm_request_metrics_jsonl])
    return command


def _task_summary_command(
    args: argparse.Namespace,
    task_id: str,
    task_dir: Path,
    wallclocks: list[float],
) -> list[str]:
    family, variant = task_id.split("/", 1)
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "build_track_b_e2e_summary.py"),
        "task",
        "--round",
        str(args.round),
        "--task-dir",
        str(task_dir),
        "--family",
        family,
        "--variant",
        variant,
        "--runtime-config-hash",
        args.runtime_config_hash,
        "--run-wallclocks-json",
        json.dumps(wallclocks),
        "--clock-skew-ms-p99",
        str(args.clock_skew_ms_p99),
        "--trace-emitter-correctness-verified-at",
        args.trace_emitter_correctness_verified_at,
        "--cold-completion-discarded",
        "--cache-reset-verified",
        "--protocol-hash-match",
        "--generation-volume-within-band",
        "--sample-hash-match",
    ]
    if args.write_untrusted_diagnostic:
        command.append("--write-untrusted-diagnostic")
    return command


def _round_summary_command(args: argparse.Namespace, round_dir: Path) -> list[str]:
    command = [
        args.python,
        str(REPO_ROOT / "scripts" / "build_track_b_e2e_summary.py"),
        "round",
        "--round",
        str(args.round),
        "--round-dir",
        str(round_dir),
        "--runtime-config-hash",
        args.runtime_config_hash,
        "--hypothesis",
        args.hypothesis,
        "--config-delta-vs-prior-round",
        args.config_delta_vs_prior_round,
        "--auto-research-agent-recommendation",
        args.auto_research_agent_recommendation,
        "--next-round-proposal",
        args.next_round_proposal,
    ]
    if args.write_untrusted_diagnostic:
        command.append("--write-untrusted-diagnostic")
    return command


def _read_blockers(preflight_out: Path) -> str:
    if not preflight_out.is_file():
        return "preflight did not write an audit artifact"
    payload = _read_json(preflight_out)
    blockers = payload.get("blocking_reasons")
    if isinstance(blockers, list) and blockers:
        return ", ".join(str(blocker) for blocker in blockers)
    return "preflight failed"


def run_round(args: argparse.Namespace) -> int:
    _validate_codex_command_template(args.codex_command_template)
    if args.repeat < 3:
        raise ValueError("--repeat must be >= 3 for truthful median measurements")

    out_root = Path(args.out_root)
    round_dir = out_root / f"round_{args.round}"
    round_dir.mkdir(parents=True, exist_ok=True)
    preflight_out = round_dir / "preflight_audit.json"

    preflight = _run(_preflight_command(args, preflight_out))
    if preflight.returncode != 0:
        print(f"Track B E2E round blocked by preflight: {_read_blockers(preflight_out)}", file=sys.stderr)
        return 1

    runner = _run(_runner_command(args))
    if runner.returncode != 0:
        print(runner.stderr, file=sys.stderr, end="")
        return runner.returncode

    for task_id in _tasks():
        task_dir = _attempt_dir(out_root, args.round, task_id, 1)
        wallclocks = _attempt_wallclocks(out_root, args.round, task_id, args.repeat)
        summary = _run(_task_summary_command(args, task_id, task_dir, wallclocks))
        if summary.returncode != 0:
            print(summary.stderr, file=sys.stderr, end="")
            return summary.returncode

    round_summary = _run(_round_summary_command(args, round_dir))
    if round_summary.returncode != 0:
        print(round_summary.stderr, file=sys.stderr, end="")
    return round_summary.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a hard-gated Track B E2E measurement round.")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--runtime-config-hash", required=True)
    parser.add_argument("--codex-command-template", required=True)
    parser.add_argument("--python", default=_default_python())
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--out-root", default=str(REPO_ROOT / "output" / "track_b_e2e"))
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--vllm-request-metrics-jsonl", default="")
    parser.add_argument("--clock-skew-ms-p99", type=float, required=True)
    parser.add_argument("--trace-emitter-correctness-verified-at", required=True)
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--config-delta-vs-prior-round", default="")
    parser.add_argument("--auto-research-agent-recommendation", default="")
    parser.add_argument("--next-round-proposal", default="")
    parser.add_argument("--write-untrusted-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run_round(args)
    except Exception as exc:
        print(f"run_track_b_e2e_round.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
