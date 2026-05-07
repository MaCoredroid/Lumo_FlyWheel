#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]


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


REQUEST_ID_LABELS = ("request_id", "vllm_request_id", "request")
REQUEST_JOIN_REQUIRED_METRICS = (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
REQUEST_JOIN_REQUIRED_JSONL_FIELDS = (
    "prompt_tokens",
    "generation_tokens",
    "spec_decode_num_draft_tokens",
    "spec_decode_num_accepted_tokens",
)
REQUEST_JOIN_JSONL_FIELD_ALIASES = {
    "generation_tokens": ("generation_tokens", "completion_tokens"),
}
DCGM_PROFILE_FIELDS = (
    "dram_active_pct",
    "sm_active_pct",
    "sm_occupancy_pct",
    "pipe_tensor_active_pct",
    "pipe_fp16_active_pct",
)


def _metric_label_names(line: str) -> set[str]:
    if "{" not in line or "}" not in line:
        return set()
    label_block = line.split("{", 1)[1].split("}", 1)[0]
    names: set[str] = set()
    for item in label_block.split(","):
        name = item.split("=", 1)[0].strip()
        if name:
            names.add(name)
    return names


def _request_labeled_metric_coverage(metrics_text: str) -> dict[str, bool]:
    coverage = {metric: False for metric in REQUEST_JOIN_REQUIRED_METRICS}
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(None, 1)[0]
        if metric_name not in coverage:
            continue
        labels = _metric_label_names(line)
        if any(label in labels for label in REQUEST_ID_LABELS):
            coverage[metric_name] = True
    return coverage


def _request_metrics_jsonl_coverage(path_text: str) -> dict[str, Any]:
    if not path_text:
        return {
            "ok": False,
            "reason": "not_configured",
            "path": "",
            "sample_count": 0,
            "required_field_coverage": {field: False for field in REQUEST_JOIN_REQUIRED_JSONL_FIELDS},
        }
    path = Path(path_text)
    if not path.is_file():
        return {
            "ok": False,
            "reason": "not_found",
            "path": str(path),
            "sample_count": 0,
            "required_field_coverage": {field: False for field in REQUEST_JOIN_REQUIRED_JSONL_FIELDS},
        }
    rows = []
    field_coverage = {field: False for field in REQUEST_JOIN_REQUIRED_JSONL_FIELDS}
    request_id_seen = False
    valid_row_count = 0
    invalid_row_count = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError:
            invalid_row_count += 1
            continue
        if not isinstance(payload, dict):
            invalid_row_count += 1
            continue
        rows.append(payload)
        row_has_request_id = any(payload.get(key) for key in ("request_id", "vllm_request_id", "id"))
        request_id_seen = request_id_seen or row_has_request_id
        row_has_required_fields = True
        for field in REQUEST_JOIN_REQUIRED_JSONL_FIELDS:
            aliases = REQUEST_JOIN_JSONL_FIELD_ALIASES.get(field, (field,))
            if any(isinstance(payload.get(alias), (int, float)) for alias in aliases):
                field_coverage[field] = True
            else:
                row_has_required_fields = False
        if row_has_request_id and row_has_required_fields:
            valid_row_count += 1
        else:
            invalid_row_count += 1
    ok = valid_row_count > 0
    return {
        "ok": ok,
        "reason": "" if ok else "missing_required_request_metrics",
        "path": str(path),
        "sample_count": len(rows),
        "valid_request_metric_row_count": valid_row_count,
        "invalid_request_metric_row_count": invalid_row_count,
        "request_id_seen": request_id_seen,
        "required_field_coverage": field_coverage,
        "accepted_field_aliases": REQUEST_JOIN_JSONL_FIELD_ALIASES,
    }


def _sampler_smoke(measurement_python: Path, duration_s: float) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(prefix="track_b_dcgm_", suffix=".jsonl") as handle:
        result = subprocess.run(
            [
                str(measurement_python),
                str(REPO_ROOT / "scripts" / "sample_dcgm_during_task.py"),
                "--out",
                handle.name,
                "--duration-s",
                str(duration_s),
                "--interval-s",
                "0.01",
                "--allow-unstamped-smoke",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(5.0, duration_s + 5.0),
        )
        rows: list[dict[str, Any]] = []
        for line in Path(handle.name).read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    profile_fields_present = any(
        all(isinstance(row.get(field), (int, float)) for field in DCGM_PROFILE_FIELDS)
        for row in rows
    )
    observed_numeric_fields = sorted(
        {
            field
            for row in rows
            for field in DCGM_PROFILE_FIELDS
            if isinstance(row.get(field), (int, float))
        }
    )
    telemetry_sources = sorted(
        {str(row.get("telemetry_source")) for row in rows if row.get("telemetry_source")}
    )
    return {
        "ok": result.returncode == 0 and bool(rows),
        "returncode": result.returncode,
        "sample_count": len(rows),
        "profile_fields_present": profile_fields_present,
        "observed_numeric_profile_fields": observed_numeric_fields,
        "missing_profile_fields": [
            field for field in DCGM_PROFILE_FIELDS if field not in observed_numeric_fields
        ],
        "telemetry_sources": telemetry_sources,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    codex_help = _command(["codex", "exec", "--help"])
    codex_version = _command(["codex", "--version"])
    health = _get(args.health_url)
    metrics = _get(args.metrics_url)
    metrics_text = str(metrics.get("text") or "")
    request_label_coverage = _request_labeled_metric_coverage(metrics_text)
    side_channel_coverage = _request_metrics_jsonl_coverage(getattr(args, "vllm_request_metrics_jsonl", ""))
    measurement_python = Path(args.python)
    sampler_smoke = _sampler_smoke(measurement_python, args.sampler_smoke_duration_s)
    pynvml_check = subprocess.run(
        [
            str(measurement_python),
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
            "ok": all(request_label_coverage.values()),
            "required_metric_coverage": request_label_coverage,
            "accepted_label_names": list(REQUEST_ID_LABELS),
        },
        "vllm_request_metrics_side_channel": side_channel_coverage,
        "vllm_request_metrics_join_available": {
            "ok": all(request_label_coverage.values()) or side_channel_coverage["ok"],
            "prometheus_request_labels_ok": all(request_label_coverage.values()),
            "jsonl_side_channel_ok": side_channel_coverage["ok"],
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
        "dcgm_sampler_runs": {
            "ok": sampler_smoke["ok"],
            "sample_count": sampler_smoke["sample_count"],
            "telemetry_sources": sampler_smoke["telemetry_sources"],
            "stderr": sampler_smoke["stderr"],
        },
        "dcgm_profile_fields_available": {
            "ok": sampler_smoke["profile_fields_present"],
            "sample_count": sampler_smoke["sample_count"],
            "observed_numeric_profile_fields": sampler_smoke["observed_numeric_profile_fields"],
            "missing_profile_fields": sampler_smoke["missing_profile_fields"],
        },
    }
    blockers = [name for name, check in checks.items() if name in args.required_checks and not check["ok"]]
    return {
        "schema": "lumo.track_b.e2e_preflight_audit.v1",
        "recorded_at": _now(),
        "measurement_python": str(measurement_python),
        "health_url": args.health_url,
        "metrics_url": args.metrics_url,
        "checks": checks,
        "blocking_reasons": blockers,
        "round0_may_run": not blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether Track B E2E Round 0 may truthfully run.")
    default_python = REPO_ROOT / ".venv" / "bin" / "python"
    parser.add_argument("--python", default=str(default_python if default_python.exists() else Path(sys.executable)))
    parser.add_argument("--sampler-smoke-duration-s", type=float, default=0.05)
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--vllm-request-metrics-jsonl", default="")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--required-checks",
        nargs="*",
        default=[
            "vllm_health",
            "spec_decode_metrics_exposed",
            "vllm_request_metrics_join_available",
            "codex_trace_out_supported",
            "nvidia_smi_available",
            "ncu_available",
            "pynvml_available",
            "dcgm_sampler_runs",
            "dcgm_profile_fields_available",
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
