#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_TASKS = 3


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_bytes(path: Path) -> bytes:
    if not path.is_file():
        return b""
    return path.read_bytes()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve(base_dir: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base_dir / path


def _codex_version() -> str:
    result = subprocess.run(
        ["codex", "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _codex_trace_out_supported() -> bool:
    result = subprocess.run(
        ["codex", "exec", "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    return result.returncode == 0 and "--trace-out" in result.stdout


def _task_result(base_dir: Path, raw: dict[str, Any]) -> dict[str, Any]:
    enabled_model = _read_bytes(_resolve(base_dir, raw.get("trace_out_enabled_model_outputs", "")))
    disabled_model = _read_bytes(_resolve(base_dir, raw.get("trace_out_disabled_model_outputs", "")))
    enabled_tools = _read_bytes(_resolve(base_dir, raw.get("trace_out_enabled_tool_call_sequence", "")))
    disabled_tools = _read_bytes(_resolve(base_dir, raw.get("trace_out_disabled_tool_call_sequence", "")))
    enabled_scores_path = _resolve(base_dir, raw.get("trace_out_enabled_milestone_scores", ""))
    disabled_scores_path = _resolve(base_dir, raw.get("trace_out_disabled_milestone_scores", ""))
    enabled_scores = _load_json(enabled_scores_path) if enabled_scores_path.is_file() else None
    disabled_scores = _load_json(disabled_scores_path) if disabled_scores_path.is_file() else None
    return {
        "task_id": str(raw.get("task_id", "")),
        "trace_out_enabled_exit_code": raw.get("trace_out_enabled_exit_code"),
        "trace_out_disabled_exit_code": raw.get("trace_out_disabled_exit_code"),
        "model_outputs_byte_identical": enabled_model == disabled_model and bool(enabled_model),
        "tool_call_sequences_byte_identical": enabled_tools == disabled_tools and bool(enabled_tools),
        "milestone_scores_identical": enabled_scores == disabled_scores and enabled_scores is not None,
        "evidence_sha256": {
            "trace_out_enabled_model_outputs": _sha256(enabled_model),
            "trace_out_disabled_model_outputs": _sha256(disabled_model),
            "trace_out_enabled_tool_call_sequence": _sha256(enabled_tools),
            "trace_out_disabled_tool_call_sequence": _sha256(disabled_tools),
        },
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.comparison_manifest)
    base_dir = Path(args.base_dir) if args.base_dir else manifest_path.parent
    manifest = _load_json(manifest_path)
    raw_tasks = manifest.get("tasks") if isinstance(manifest, dict) else []
    if not isinstance(raw_tasks, list):
        raw_tasks = []
    tasks = [_task_result(base_dir, task if isinstance(task, dict) else {}) for task in raw_tasks]
    trace_out_supported = bool(manifest.get("trace_out_supported", _codex_trace_out_supported()))
    payload = {
        "schema": "lumo.track_b.codex_trace_correctness.v1",
        "verified_at": _now(),
        "codex_version": str(manifest.get("codex_version") or _codex_version()),
        "trace_out_supported": trace_out_supported,
        "comparison_manifest": str(manifest_path),
        "tasks": tasks,
    }
    ok = trace_out_supported and len(tasks) >= MIN_TASKS and all(
        task.get("trace_out_enabled_exit_code") == 0
        and task.get("trace_out_disabled_exit_code") == 0
        and task.get("model_outputs_byte_identical") is True
        and task.get("tool_call_sequences_byte_identical") is True
        and task.get("milestone_scores_identical") is True
        for task in tasks
    )
    payload["ok"] = ok
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the Track B Codex trace-emitter correctness artifact from enabled/disabled run evidence."
    )
    parser.add_argument("--comparison-manifest", required=True)
    parser.add_argument("--base-dir", default="")
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "output" / "track_b_e2e" / "codex_trace_emitter_correctness.json"),
    )
    args = parser.parse_args()
    try:
        payload = verify(args)
    except Exception as exc:
        print(f"verify_track_b_codex_trace_correctness.py: {exc}", file=sys.stderr)
        return 2
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
