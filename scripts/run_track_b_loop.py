#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"YAML must be a mapping: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_tsv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _candidate_ids(round_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in (round_dir / "candidates").glob("[0-9][0-9][0-9]"):
        if path.is_dir():
            ids.append(int(path.name))
    return sorted(ids)


def _next_candidate_id(round_dir: Path) -> str:
    ids = _candidate_ids(round_dir)
    return f"{(max(ids) + 1) if ids else 0:03d}"


def _render_agent_prompt(round_dir: Path, candidate_dir: Path, candidate_id: str) -> str:
    spec = _load_yaml(round_dir / "round_spec.yaml")
    strategy = (round_dir / "strategy_brief.md").read_text(encoding="utf-8")
    prior = (round_dir / "prior_cutlass_memory.md").read_text(encoding="utf-8")
    audit = _load_json(round_dir / "completion_audit.json") or {}
    quality_history = ""
    quality_history_path = round_dir / "quality_gate_history.tsv"
    if quality_history_path.is_file():
        quality_history = "\n".join(quality_history_path.read_text(encoding="utf-8").splitlines()[-12:])
    branch_log = _load_json(round_dir / "branch_log.json")
    return "\n".join(
        [
            "# Track B Auto-Research Candidate Authoring",
            "",
            "You are a fresh implementation worker inside a Karpathy-style auto-research loop.",
            "The controller owns measurement, gates, keep/discard, and ledgers. Your job is to author exactly one candidate artifact.",
            "",
            "## Hard Rules",
            "",
            f"- Candidate id: `{candidate_id}`",
            f"- Candidate directory: `{candidate_dir}`",
            "- Write only inside that candidate directory.",
            "- Do not edit source files, tests, quality fixtures, prior memory, or round ledgers.",
            "- Do not run expensive live benchmarks; the controller runs gates after you exit.",
            "- Preserve target model weights and sampling behavior.",
            "- Build on prior CUTLASS negative memory; do not propose another tile/schedule/stage mutation unless your config changes the available serving surface.",
            "",
            "## Required Files",
            "",
            "Write these files before exiting:",
            "",
            "1. `candidate_analysis.md` with these bullets:",
            "   - speed_thesis",
            "   - expected_affected_counter",
            "   - quality_risk",
            "   - why_not_prior_failure",
            "",
            "2. `serve_config.yaml` with one of these supported controller surfaces:",
            "   - `request_shaping.target_concurrency: <1-8>` for batching/concurrency experiments",
            "   - `prefix_cache` settings for prefix-cache experiments",
            "   - `spec_decode` settings only if the current runtime actually exposes such flags",
            "",
            "3. Optional `notes.md` with any blocker or measurement caveat.",
            "",
            "## Current Objective",
            "",
            f"- Baseline decode: `{spec.get('baseline_decode_tps')}` tok/s",
            f"- Target decode: `{spec.get('target_decode_tps')}` tok/s",
            f"- Best audit so far: `{audit.get('best_decode_tps')}` tok/s",
            "",
            "## Recent Controller Outcomes",
            "",
            "Quality gate history tail:",
            "",
            "```tsv",
            quality_history,
            "```",
            "",
            "Branch log summary:",
            "",
            "```json",
            json.dumps(branch_log, indent=2, sort_keys=True)[:6000] if branch_log is not None else "[]",
            "```",
            "",
            "## Strategy Brief",
            "",
            strategy,
            "",
            "## Prior CUTLASS Memory",
            "",
            prior,
        ]
    )


def _spawn_codex(round_dir: Path, candidate_dir: Path, prompt: str, timeout_s: int) -> dict[str, Any]:
    last_message = candidate_dir / "agent_last_message.txt"
    transcript = candidate_dir / "agent_session.jsonl"
    prompt_path = candidate_dir / "iteration_brief.md"
    prompt_path.write_text(prompt, encoding="utf-8")
    argv = [
        "codex",
        "-c",
        'model="gpt-5.5"',
        "-c",
        'model_reasoning_effort="high"',
        "exec",
        "--cd",
        str(round_dir),
        "--json",
        "--output-last-message",
        str(last_message),
        "--skip-git-repo-check",
        "-",
    ]
    started = time.monotonic()
    with tempfile_file() as stdout_file, tempfile_file() as stderr_file:
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            return {"ok": False, "error": f"agent_binary_missing: {exc}"}
        assert proc.stdin is not None
        proc.stdin.write(prompt.encode("utf-8"))
        proc.stdin.close()
        while proc.poll() is None:
            if timeout_s > 0 and time.monotonic() - started >= timeout_s:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                break
            time.sleep(1.0)
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout_bytes = stdout_file.read()
        stderr_bytes = stderr_file.read()
    transcript.write_bytes(stdout_bytes)
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": f"agent_exit_{proc.returncode}: {stderr_bytes.decode('utf-8', errors='replace')[:4000]}",
            "transcript": str(transcript),
        }
    return {"ok": True, "transcript": str(transcript), "last_message": str(last_message)}


class tempfile_file:
    def __enter__(self):
        import tempfile

        self._handle = tempfile.TemporaryFile()
        return self._handle

    def __exit__(self, exc_type, exc, tb):
        self._handle.close()


def _parse_target_concurrency(candidate_dir: Path) -> int | None:
    config_path = candidate_dir / "serve_config.yaml"
    if not config_path.is_file():
        return None
    try:
        config = _load_yaml(config_path)
    except Exception:
        return None
    request_shaping = config.get("request_shaping")
    if isinstance(request_shaping, dict):
        value = request_shaping.get("target_concurrency") or request_shaping.get("concurrent_requests")
        if value is not None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return None
            if 1 <= parsed <= 8:
                return parsed
    return None


def _run_cmd(argv: list[str], *, cwd: Path, output_path: Path | None = None) -> tuple[int, str]:
    completed = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    text = completed.stdout + completed.stderr
    if output_path is not None:
        output_path.write_text(text, encoding="utf-8")
    return completed.returncode, text


def _evaluate_candidate(args: argparse.Namespace, round_dir: Path, candidate_dir: Path, candidate_id: str) -> dict[str, Any]:
    analysis_path = candidate_dir / "candidate_analysis.md"
    if not analysis_path.is_file():
        return {"status": "rejected", "reason": "candidate_analysis_missing"}
    concurrency = _parse_target_concurrency(candidate_dir)
    if concurrency is None:
        return {"status": "rejected", "reason": "unsupported_or_missing_serve_config"}
    throughput_path = candidate_dir / "throughput.json"
    cmd = [
        str(REPO_ROOT / "scripts" / "measure_track_b_prefix_cache.py"),
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--turns",
        "2",
        "--concurrent-requests",
        str(concurrency),
        "--prefix-words",
        str(args.prefix_words),
        "--max-tokens",
        str(args.max_tokens),
        "--reset-prefix-cache",
        "--output",
        str(throughput_path),
    ]
    code, text = _run_cmd(cmd, cwd=REPO_ROOT, output_path=candidate_dir / "throughput_command.log")
    if code != 0:
        return {"status": "rejected", "reason": "throughput_measure_failed", "detail": text[-2000:]}
    throughput = _load_json(throughput_path) or {}
    spec = _load_yaml(round_dir / "round_spec.yaml")
    target_tps = float(spec["success_criteria"]["decode_speed_at_least_tps"])
    decode_tps = float(throughput.get("decode_tps") or 0.0)
    if decode_tps < target_tps:
        return {
            "status": "rejected",
            "reason": "speed_below_target",
            "decode_tps": decode_tps,
            "target_decode_tps": target_tps,
        }
    b1_path = candidate_dir / "b1_result.json"
    cmd = [
        str(REPO_ROOT / "scripts" / "run_track_b_batch_equivalence.py"),
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--prompt-count",
        str(max(4, concurrency)),
        "--concurrent-requests",
        str(concurrency),
        "--prefix-words",
        str(min(args.prefix_words, 1024)),
        "--max-tokens",
        "8",
        "--reset-prefix-cache",
        "--output",
        str(b1_path),
    ]
    code, text = _run_cmd(cmd, cwd=REPO_ROOT, output_path=candidate_dir / "b1_command.log")
    b1 = _load_json(b1_path) or {}
    if code != 0 or not b1.get("pass"):
        return {
            "status": "rejected",
            "reason": "b1_equivalence_failed",
            "decode_tps": decode_tps,
            "target_decode_tps": target_tps,
            "b1": b1,
        }
    trace_file = _load_yaml(round_dir / "round_spec.yaml").get("workload_trace")
    if not trace_file:
        return {
            "status": "accepted_for_speed_not_promoted",
            "reason": "workload_trace_missing_for_b2_b3",
            "decode_tps": decode_tps,
            "target_decode_tps": target_tps,
            "concurrency": concurrency,
            "throughput_ref": str(throughput_path.relative_to(round_dir)),
            "b1_ref": str(b1_path.relative_to(round_dir)),
        }
    trace_path = Path(str(trace_file))
    b2_path = candidate_dir / "b2_result.json"
    cmd = [
        str(REPO_ROOT / "scripts" / "run_track_b_workload_equivalence.py"),
        "--suite",
        "b2",
        "--trace-file",
        str(trace_path),
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--probe-count",
        "4",
        "--concurrent-requests",
        str(concurrency),
        "--prefix-words",
        "512",
        "--max-tokens",
        "8",
        "--reset-prefix-cache",
        "--output",
        str(b2_path),
    ]
    code, text = _run_cmd(cmd, cwd=REPO_ROOT, output_path=candidate_dir / "b2_command.log")
    b2 = _load_json(b2_path) or {}
    if code != 0 or not b2.get("pass"):
        return {
            "status": "rejected",
            "reason": "b2_workload_equivalence_failed",
            "decode_tps": decode_tps,
            "target_decode_tps": target_tps,
            "concurrency": concurrency,
            "throughput_ref": str(throughput_path.relative_to(round_dir)),
            "b1_ref": str(b1_path.relative_to(round_dir)),
            "b2_ref": str(b2_path.relative_to(round_dir)),
            "b2": b2,
        }
    b3_path = candidate_dir / "b3_result.json"
    cmd = [
        str(REPO_ROOT / "scripts" / "run_track_b_workload_equivalence.py"),
        "--suite",
        "b3",
        "--trace-file",
        str(trace_path),
        "--port",
        str(args.port),
        "--model",
        args.model,
        "--probe-count",
        "8",
        "--concurrent-requests",
        str(concurrency),
        "--prefix-words",
        "1024",
        "--max-tokens",
        "8",
        "--reset-prefix-cache",
        "--output",
        str(b3_path),
    ]
    code, text = _run_cmd(cmd, cwd=REPO_ROOT, output_path=candidate_dir / "b3_command.log")
    b3 = _load_json(b3_path) or {}
    if code != 0 or not b3.get("pass"):
        return {
            "status": "rejected",
            "reason": "b3_workload_equivalence_failed",
            "decode_tps": decode_tps,
            "target_decode_tps": target_tps,
            "concurrency": concurrency,
            "throughput_ref": str(throughput_path.relative_to(round_dir)),
            "b1_ref": str(b1_path.relative_to(round_dir)),
            "b2_ref": str(b2_path.relative_to(round_dir)),
            "b3_ref": str(b3_path.relative_to(round_dir)),
            "b3": b3,
        }
    return {
        "status": "accepted_promoted",
        "decode_tps": decode_tps,
        "target_decode_tps": target_tps,
        "concurrency": concurrency,
        "throughput_ref": str(throughput_path.relative_to(round_dir)),
        "b1_ref": str(b1_path.relative_to(round_dir)),
        "b2_ref": str(b2_path.relative_to(round_dir)),
        "b3_ref": str(b3_path.relative_to(round_dir)),
    }


def _update_ledgers(round_dir: Path, candidate_id: str, result: dict[str, Any]) -> None:
    branch_log_path = round_dir / "branch_log.json"
    branch_log = json.loads(branch_log_path.read_text(encoding="utf-8")) if branch_log_path.is_file() else []
    if not isinstance(branch_log, list):
        branch_log = []
    branch_log.append({"candidate_id": candidate_id, **result, "recorded_at": _now()})
    _write_json(branch_log_path, branch_log)
    if result["status"].startswith("accepted"):
        _append_tsv(
            round_dir / "quality_gate_history.tsv",
            {
                "candidate_id": candidate_id,
                "tier": "speed",
                "status": "pass",
                "score_json": {"decode_tps": result.get("decode_tps"), "target_decode_tps": result.get("target_decode_tps")},
                "artifact_ref": result.get("throughput_ref", ""),
                "recorded_at": _now(),
            },
            ["candidate_id", "tier", "status", "score_json", "artifact_ref", "recorded_at"],
        )
        _append_tsv(
            round_dir / "quality_gate_history.tsv",
            {
                "candidate_id": candidate_id,
                "tier": "b1_strong_equivalence",
                "status": "pass",
                "score_json": {"concurrency": result.get("concurrency")},
                "artifact_ref": result.get("b1_ref", ""),
                "recorded_at": _now(),
            },
            ["candidate_id", "tier", "status", "score_json", "artifact_ref", "recorded_at"],
        )
        if result.get("b2_ref"):
            _append_tsv(
                round_dir / "quality_gate_history.tsv",
                {
                    "candidate_id": candidate_id,
                    "tier": "b2_workload_equivalence",
                    "status": "pass",
                    "score_json": {"concurrency": result.get("concurrency")},
                    "artifact_ref": result.get("b2_ref", ""),
                    "recorded_at": _now(),
                },
                ["candidate_id", "tier", "status", "score_json", "artifact_ref", "recorded_at"],
            )
        if result.get("b3_ref"):
            _append_tsv(
                round_dir / "quality_gate_history.tsv",
                {
                    "candidate_id": candidate_id,
                    "tier": "b3_workload_equivalence",
                    "status": "pass",
                    "score_json": {"concurrency": result.get("concurrency")},
                    "artifact_ref": result.get("b3_ref", ""),
                    "recorded_at": _now(),
                },
                ["candidate_id", "tier", "status", "score_json", "artifact_ref", "recorded_at"],
            )
    else:
        _append_tsv(
            round_dir / "mutations_rejected.tsv",
            {
                "candidate_id": candidate_id,
                "tier": "controller",
                "cost_bucket": result.get("reason", "rejected"),
                "reason": result.get("reason", "rejected"),
                "first_failing_metric": "",
                "recorded_at": _now(),
            },
            ["candidate_id", "tier", "cost_bucket", "reason", "first_failing_metric", "recorded_at"],
        )


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    round_dir = args.round_dir.resolve()
    if not (round_dir / "round_spec.yaml").is_file():
        raise RuntimeError(f"round_spec.yaml missing: {round_dir}")
    monitor_rows: list[dict[str, Any]] = []
    for _ in range(args.max_attempts):
        candidate_id = _next_candidate_id(round_dir)
        candidate_dir = round_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True)
        prompt = _render_agent_prompt(round_dir, candidate_dir, candidate_id)
        spawn = _spawn_codex(round_dir, candidate_dir, prompt, args.agent_timeout_s)
        (candidate_dir / "spawn_result.json").write_text(
            json.dumps(spawn, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not spawn.get("ok"):
            result = {"status": "rejected", "reason": "agent_spawn_failed", "spawn": spawn}
        else:
            result = _evaluate_candidate(args, round_dir, candidate_dir, candidate_id)
        (candidate_dir / "controller_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _update_ledgers(round_dir, candidate_id, result)
        monitor_rows.append({"candidate_id": candidate_id, **result})
        if result.get("status") == "accepted_promoted" and not args.keep_searching_after_accept:
            break
    summary = {
        "schema": "lumo.track_b.loop_run.v1",
        "round_dir": str(round_dir),
        "attempts": monitor_rows,
        "completed_at": _now(),
    }
    _write_json(round_dir / "loop_monitor_latest.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Track B Karpathy-style auto-research controller loop.")
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--agent-timeout-s", type=int, default=900)
    parser.add_argument("--port", type=int, default=9950)
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--prefix-words", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--keep-searching-after-accept", action="store_true")
    args = parser.parse_args()
    result = run_loop(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
