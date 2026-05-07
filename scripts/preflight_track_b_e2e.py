#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _command(command: list[str], timeout_s: float = 10.0) -> dict[str, Any]:
    path = shutil.which(command[0])
    if path is None:
        return {"ok": False, "reason": "not_found", "command": command}
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "timeout", "command": command}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _get(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
    try:
        response = requests.get(url, timeout=timeout_s)
    except requests.RequestException as exc:
        return {"ok": False, "reason": type(exc).__name__, "error": str(exc)}
    return {"ok": 200 <= response.status_code < 300, "status_code": response.status_code, "text": response.text}


def _has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    codex_help = _command(["codex", "exec", "--help"])
    codex_version = _command(["codex", "--version"])
    health = _get(args.health_url)
    metrics = _get(args.metrics_url)
    metrics_text = str(metrics.get("text") or "")
    pynvml_check = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('pynvml') else 1)",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    checks = {
        "vllm_health": {"ok": bool(health.get("ok")), "status_code": health.get("status_code")},
        "spec_decode_metrics_exposed": {
            "ok": _has_any(metrics_text, ["vllm:spec_decode_num_drafts_total"])
            and _has_any(metrics_text, ["vllm:spec_decode_num_draft_tokens_total"])
            and _has_any(metrics_text, ["vllm:spec_decode_num_accepted_tokens_total"])
        },
        "vllm_request_id_labels_exposed": {
            "ok": _has_any(metrics_text, ["request_id=", "vllm_request_id=", "request="])
        },
        "codex_installed": {"ok": codex_version.get("ok"), "version": str(codex_version.get("stdout") or "").strip()},
        "codex_trace_out_supported": {"ok": "--trace-out" in str(codex_help.get("stdout") or "")},
        "codex_json_events_supported": {"ok": "--json" in str(codex_help.get("stdout") or "")},
        "nvidia_smi_available": {"ok": shutil.which("nvidia-smi") is not None},
        "ncu_available": {"ok": shutil.which("ncu") is not None},
        "pynvml_available": {
            "ok": pynvml_check.returncode == 0,
            "stderr": pynvml_check.stderr,
        },
    }
    blockers = [name for name, check in checks.items() if name in args.required_checks and not check["ok"]]
    return {
        "schema": "lumo.track_b.e2e_preflight_audit.v1",
        "recorded_at": _now(),
        "health_url": args.health_url,
        "metrics_url": args.metrics_url,
        "checks": checks,
        "blocking_reasons": blockers,
        "round0_may_run": not blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether Track B E2E Round 0 may truthfully run.")
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--required-checks",
        nargs="*",
        default=[
            "vllm_health",
            "spec_decode_metrics_exposed",
            "vllm_request_id_labels_exposed",
            "codex_trace_out_supported",
            "nvidia_smi_available",
            "ncu_available",
            "pynvml_available",
        ],
    )
    args = parser.parse_args()
    payload = audit(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if payload["round0_may_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
