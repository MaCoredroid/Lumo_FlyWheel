#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANT = "v1-clean-baseline"
REQUEST_JOIN_JSONL_SCHEMA = "lumo.track_b.vllm_request_metrics.v1"
REQUEST_JOIN_JSONL_PRODUCER = "track_b_vllm_request_metrics_patch"
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


def _finite_nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0


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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _prepare_attempt_workspace(source_workspace: Path, task_dir: Path) -> Path:
    attempt_workspace = task_dir / "workspace"
    if attempt_workspace.exists():
        raise RuntimeError(f"attempt workspace already exists: {attempt_workspace}")
    shutil.copytree(
        source_workspace,
        attempt_workspace,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    return attempt_workspace


def _validate_codex_command_template(template: str, *, require_trace_out: bool = True) -> None:
    if not require_trace_out:
        return
    # No-op: the {trace_out} placeholder is no longer required because trace
    # emission is now produced by the inference proxy + runner-side synthesis,
    # not by a Codex CLI flag. Templates that include it remain valid (the
    # placeholder is still substituted by the runner) but absence is allowed.
    return


def _validate_runtime_config_hash(runtime_config_hash: str) -> None:
    digest = runtime_config_hash.removeprefix("sha256:")
    if (
        not runtime_config_hash.startswith("sha256:")
        or len(digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in digest)
    ):
        raise ValueError("--runtime-config-hash must be a sha256:<64-hex-digest> value")


def _resolve_task_args(family: str, variant: str) -> tuple[str, str]:
    if "/" not in family:
        return family, variant
    parsed_family, parsed_variant = family.split("/", 1)
    if not parsed_family or not parsed_variant:
        raise ValueError("task id must be formatted as family/variant")
    if variant != DEFAULT_VARIANT and variant != parsed_variant:
        raise ValueError("variant must not conflict with the family/variant task id")
    return parsed_family, parsed_variant


def _metrics_text(metrics_url: str) -> str:
    return _request("GET", metrics_url).text


def _write_vllm_per_turn(task_dir: Path, before_raw: str, after_raw: str, *, runtime_config_hash: str = "") -> None:
    schema = resolve_metric_schema(parse_prometheus_text(after_raw))
    per_request = compute_vllm_per_request_metrics(
        parse_prometheus_samples(before_raw),
        parse_prometheus_samples(after_raw),
        schema,
    )
    if not per_request:
        raise RuntimeError("Prometheus metrics did not produce any request-keyed vLLM rows")
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps({"requests": per_request, "runtime_config_hash": runtime_config_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_deferred_vllm_per_turn(task_dir: Path, *, runtime_config_hash: str, reason: str) -> None:
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps(
            {
                "requests": {},
                "runtime_config_hash": runtime_config_hash,
                "deferred": True,
                "deferred_reason": reason,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _normalize_vllm_request_metrics(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    request_id = row.get("request_id") or row.get("vllm_request_id") or row.get("id")
    if not request_id:
        return None
    if row.get("schema") != REQUEST_JOIN_JSONL_SCHEMA or row.get("producer") != REQUEST_JOIN_JSONL_PRODUCER:
        raise RuntimeError(
            f"vLLM request metrics row for {request_id} is missing Track B producer metadata"
        )
    prompt_tokens = row.get("prompt_tokens")
    completion_tokens = row.get("completion_tokens", row.get("generation_tokens"))
    prefill_sum_s = row.get("prefill_sum_s", row.get("prefill_s"))
    decode_sum_s = row.get("decode_sum_s", row.get("decode_s"))
    accepted = row.get("spec_decode_num_accepted_tokens")
    draft_tokens = row.get("spec_decode_num_draft_tokens")
    required = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "spec_decode_num_accepted_tokens": accepted,
        "spec_decode_num_draft_tokens": draft_tokens,
    }
    missing = [key for key, value in required.items() if not _finite_nonnegative_number(value)]
    if missing:
        raise RuntimeError(f"vLLM request metrics row for {request_id} is missing numeric fields: {', '.join(missing)}")
    normalized = dict(row)
    normalized.update(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prefill_sum_s": prefill_sum_s,
            "decode_sum_s": decode_sum_s,
            "spec_decode_num_accepted_tokens": accepted,
            "spec_decode_num_draft_tokens": draft_tokens,
        }
    )
    if (
        _finite_nonnegative_number(completion_tokens)
        and _finite_nonnegative_number(decode_sum_s)
        and float(decode_sum_s) > 0
        and "decode_tps" not in normalized
    ):
        normalized["decode_tps"] = completion_tokens / decode_sum_s
    if (
        _finite_nonnegative_number(accepted)
        and _finite_nonnegative_number(draft_tokens)
        and float(draft_tokens) > 0
        and "accepted_per_draft_token" not in normalized
    ):
        normalized["accepted_per_draft_token"] = accepted / draft_tokens
    return str(request_id), normalized


def _write_vllm_per_turn_from_jsonl(
    task_dir: Path,
    source: Path,
    *,
    start_offset: int = 0,
    runtime_config_hash: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requests_by_id: dict[str, dict[str, Any]] = {}
    captured_rows: list[dict[str, Any]] = []
    if source.is_file():
        with source.open("r", encoding="utf-8") as handle:
            handle.seek(start_offset)
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Malformed vLLM request metrics JSONL at {source}:{line_number}") from exc
                if not isinstance(row, dict):
                    continue
                if runtime_config_hash:
                    row = dict(row)
                    row["runtime_config_hash"] = runtime_config_hash
                captured_rows.append(row)
                normalized = _normalize_vllm_request_metrics(row)
                if normalized is None:
                    continue
                request_id, metrics = normalized
                requests_by_id[request_id] = metrics
    if not requests_by_id:
        # Codex may produce zero per-turn requests (e.g. 0.128.0's intermittent
        # turn.completed-with-0-tokens path). Emit a deferred per-turn record so
        # the runner can still synthesize a schema-valid codex_trace.jsonl.
        (task_dir / "vllm_request_metrics.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in captured_rows),
            encoding="utf-8",
        )
        _write_deferred_vllm_per_turn(
            task_dir,
            runtime_config_hash=runtime_config_hash,
            reason="no_proxy_capture_rows_for_task",
        )
        return (
            {
                "source": str(source),
                "start_offset": start_offset,
                "end_offset": source.stat().st_size if source.is_file() else 0,
                "captured_row_count": len(captured_rows),
                "normalized_request_count": 0,
                "request_ids": [],
                "deferred": True,
                "deferred_reason": "no_proxy_capture_rows_for_task",
            },
            captured_rows,
        )
    (task_dir / "vllm_request_metrics.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in captured_rows),
        encoding="utf-8",
    )
    (task_dir / "vllm_per_turn.json").write_text(
        json.dumps({"requests": requests_by_id, "runtime_config_hash": runtime_config_hash}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source": str(source),
        "start_offset": start_offset,
        "end_offset": source.stat().st_size,
        "captured_row_count": len(captured_rows),
        "normalized_request_count": len(requests_by_id),
        "request_ids": sorted(requests_by_id),
    }
    return summary, captured_rows


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
    if args.runtime_config_hash:
        command.extend(["--runtime-config-hash", args.runtime_config_hash])
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


def _deferred_instrumentation_checks(args: argparse.Namespace) -> list[str]:
    return sorted(
        check
        for check, enabled in {
            "codex_trace_out_supported": args.defer_codex_trace_out,
            "vllm_request_metrics_join_available": args.defer_vllm_request_metrics_join,
            "dcgm_profile_fields_available": args.defer_dcgm_profile_fields,
        }.items()
        if enabled
    )


def _gpu_mem_snapshot() -> dict[str, Any] | None:
    """Best-effort GPU memory snapshot via nvidia-smi; never raises.

    On the DGX Spark host, nvidia-smi from outside the NVIDIA container reports
    [N/A] for memory.used/free/total because the GPU is only addressable from
    inside the container. Set ``LUMO_TRACK_B_GPU_NVSMI_CMD`` to e.g.
    ``docker exec lumo-vllm-l0c-fp8-cutlass-run30 nvidia-smi`` to route the
    query into the container.
    """

    raw_cmd = os.environ.get("LUMO_TRACK_B_GPU_NVSMI_CMD", "nvidia-smi").strip() or "nvidia-smi"
    cmd = shlex.split(raw_cmd) + [
        "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return {"recorded_at": _now(), "gpus": [], "stderr": result.stderr.strip()}

    def _maybe_int(value: str) -> int | None:
        try:
            return int(value)
        except ValueError:
            return None

    rows: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [piece.strip() for piece in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": _maybe_int(parts[0]),
                "memory_used_mb": _maybe_int(parts[1]),
                "memory_total_mb": _maybe_int(parts[2]),
                "memory_free_mb": _maybe_int(parts[3]),
                "gpu_utilization_pct": _maybe_int(parts[4]),
                "raw": line.strip(),
            }
        )
    if not rows:
        return None
    return {"recorded_at": _now(), "gpus": rows, "command": raw_cmd}


def _synthesize_codex_trace_from_proxy_rows(
    path: Path,
    *,
    family: str,
    variant: str,
    runtime_config_hash: str,
    started_at: datetime,
    ended_at: datetime,
    exit_code: int | None,
    captured_rows: list[dict[str, Any]],
    task_score: float | None = None,
    wallclock_s: float | None = None,
) -> dict[str, Any]:
    """Build a codex_trace.jsonl satisfying lumo.track_b.codex_trace_correctness.v1.

    Source: per-request rows captured by the inference proxy when
    ``LUMO_TRACK_B_REQUEST_METRICS_OUT`` is set. Each row corresponds to one
    Codex turn (one /v1/responses call). The runner brackets the per-turn
    events with ``task_start`` + ``task_end`` so the verifier can validate the
    schema without a Codex source patch.
    """

    task_id = f"{family}/{variant}"
    if wallclock_s is None:
        wallclock_s = max(0.0, (ended_at - started_at).total_seconds())

    def _ts(value: Any, fallback: datetime) -> str:
        if isinstance(value, str) and value:
            return value
        return fallback.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    rows: list[dict[str, Any]] = [
        {
            "event": "task_start",
            "task_id": task_id,
            "family": family,
            "variant": variant,
            "runtime_config_hash": runtime_config_hash,
            "ts": _ts(None, started_at),
        }
    ]
    sorted_rows = sorted(
        (row for row in captured_rows if isinstance(row, dict) and row.get("request_id")),
        key=lambda row: row.get("ts_request_received") or "",
    )
    for index, row in enumerate(sorted_rows):
        regime = row.get("regime") or "unknown"
        request_id = str(row.get("request_id") or "")
        completion_tokens = row.get("completion_tokens")
        rows.append(
            {
                "event": "turn_start",
                "task_id": task_id,
                "turn": index,
                "regime": regime,
                "vllm_request_id": request_id,
                "runtime_config_hash": runtime_config_hash,
                "ts": _ts(row.get("ts_request_received"), started_at),
            }
        )
        rows.append(
            {
                "event": "turn_end",
                "task_id": task_id,
                "turn": index,
                "vllm_request_id": request_id,
                "runtime_config_hash": runtime_config_hash,
                "ts": _ts(row.get("ts_completed"), ended_at),
                "completion_tokens": completion_tokens if completion_tokens is not None else 0,
                "prompt_tokens": row.get("prompt_tokens"),
                "decode_sum_s": row.get("decode_sum_s"),
                "prefill_sum_s": row.get("prefill_sum_s"),
                "spec_decode_num_accepted_tokens": row.get("spec_decode_num_accepted_tokens"),
                "spec_decode_num_draft_tokens": row.get("spec_decode_num_draft_tokens"),
            }
        )
    rows.append(
        {
            "event": "task_end",
            "task_id": task_id,
            "runtime_config_hash": runtime_config_hash,
            "ts": _ts(None, ended_at),
            "exit_code": exit_code if exit_code is not None else 0,
            "wallclock_s": wallclock_s,
            "task_score": task_score,
        }
    )
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {
        "synthesized": True,
        "turn_count": len(sorted_rows),
        "source_rows": len(captured_rows),
    }


def _write_deferred_trace(
    path: Path,
    *,
    family: str,
    variant: str,
    runtime_config_hash: str,
    started_at: datetime,
    ended_at: datetime,
    exit_code: int | None,
) -> None:
    task_id = f"{family}/{variant}"
    wallclock_s = max(0.0, (ended_at - started_at).total_seconds())
    rows = [
        {
            "event": "task_start",
            "task_id": task_id,
            "family": family,
            "variant": variant,
            "runtime_config_hash": runtime_config_hash,
            "ts": started_at.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "trace_deferred": True,
            "deferred_reason": "codex_trace_out_supported",
        },
        {
            "event": "task_end",
            "task_id": task_id,
            "runtime_config_hash": runtime_config_hash,
            "ts": ended_at.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "exit_code": exit_code,
            "wallclock_s": wallclock_s,
            "task_score": None,
            "trace_deferred": True,
            "deferred_reason": "codex_trace_out_supported",
        },
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_missing_workspace_diagnostic(args: argparse.Namespace, family: str, variant: str, workspace: Path) -> None:
    task_dir = Path(args.out_root) / f"round_{args.round}" / f"{family}__{variant}" / f"run_{args.attempt:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    trace_out = task_dir / "codex_trace.jsonl"
    started_at = datetime.now(UTC)
    ended_at = started_at + timedelta(milliseconds=1)
    (task_dir / "prompt.md").write_text("", encoding="utf-8")
    (task_dir / "codex_stdout.log").write_text("", encoding="utf-8")
    (task_dir / "codex_stderr.log").write_text(f"workspace bundle missing: {workspace}\n", encoding="utf-8")
    (task_dir / "vllm_metrics_pre.txt").write_text("", encoding="utf-8")
    (task_dir / "vllm_metrics_post.txt").write_text("", encoding="utf-8")
    _write_deferred_vllm_per_turn(
        task_dir,
        runtime_config_hash=args.runtime_config_hash,
        reason="vllm_request_metrics_join_available",
    )
    _write_deferred_trace(
        trace_out,
        family=family,
        variant=variant,
        runtime_config_hash=args.runtime_config_hash,
        started_at=started_at,
        ended_at=ended_at,
        exit_code=None,
    )
    metadata = {
        "schema": "lumo.track_b.e2e_runner_metadata.v1",
        "recorded_at": _now(),
        "family": family,
        "variant": variant,
        "round": args.round,
        "attempt": args.attempt,
        "workspace": str(workspace),
        "workspace_missing": True,
        "trace_out": str(trace_out),
        "elapsed_s": 0.0,
        "runtime_config_hash": args.runtime_config_hash,
        "codex_command_template": args.codex_command_template,
        "vllm_request_metrics_jsonl": args.vllm_request_metrics_jsonl,
        "vllm_request_metrics_capture": None,
        "ncu_mode": bool(args.ncu_mode),
        "codex_exit_code": None,
        "deferred_instrumentation_checks": _deferred_instrumentation_checks(args),
    }
    (task_dir / "runner_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_one(args: argparse.Namespace, family: str, variant: str) -> int:
    _validate_runtime_config_hash(args.runtime_config_hash)
    source_workspace = REPO_ROOT / "benchmark_blueprints" / "families" / family / "workspace_bundle" / variant
    if not source_workspace.is_dir():
        if args.allow_missing_workspace_diagnostic:
            _write_missing_workspace_diagnostic(args, family, variant, source_workspace)
            return 0
        raise RuntimeError(f"workspace bundle missing: {source_workspace}")
    task_dir = Path(args.out_root) / f"round_{args.round}" / f"{family}__{variant}" / f"run_{args.attempt:02d}"
    task_dir = task_dir.resolve()
    task_dir.mkdir(parents=True, exist_ok=True)
    workspace = _prepare_attempt_workspace(source_workspace, task_dir)
    trace_out = task_dir / "codex_trace.jsonl"
    prompt_path = task_dir / "prompt.md"
    _write_prompt(workspace, prompt_path)

    _request("GET", args.health_url)
    if args.reset_prefix_cache_url:
        _request("POST", args.reset_prefix_cache_url, api_key=args.api_key, timeout=30)
    gpu_mem_pre = _gpu_mem_snapshot()
    vllm_request_metrics_jsonl = Path(args.vllm_request_metrics_jsonl) if args.vllm_request_metrics_jsonl else None
    vllm_request_metrics_start_offset = (
        vllm_request_metrics_jsonl.stat().st_size
        if vllm_request_metrics_jsonl is not None and vllm_request_metrics_jsonl.is_file()
        else 0
    )
    metrics_pre = _metrics_text(args.metrics_url)
    (task_dir / "vllm_metrics_pre.txt").write_text(metrics_pre, encoding="utf-8")

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
    sampler = _run_sampler(args, task_dir)
    started_at = datetime.now(UTC)
    started = time.monotonic()
    result: subprocess.CompletedProcess[str] | None = None
    zero_token_retry_count = 0
    max_retries = max(0, int(getattr(args, "zero_token_retries", 0) or 0))
    try:
        for retry_index in range(max_retries + 1):
            offset_before = (
                vllm_request_metrics_jsonl.stat().st_size
                if vllm_request_metrics_jsonl is not None and vllm_request_metrics_jsonl.is_file()
                else 0
            )
            result = subprocess.run(
                command,
                cwd=workspace,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=args.timeout_s,
                env={**os.environ, "OPENAI_API_KEY": args.api_key, "OPENAI_BASE_URL": args.endpoint},
            )
            # Detect Codex 0.128.0 zero-token quirk: codex exits 0 but never
            # made a /v1/responses call (proxy capture file gained no bytes).
            offset_after = (
                vllm_request_metrics_jsonl.stat().st_size
                if vllm_request_metrics_jsonl is not None and vllm_request_metrics_jsonl.is_file()
                else 0
            )
            if vllm_request_metrics_jsonl is None or offset_after > offset_before:
                break
            if retry_index >= max_retries:
                break
            zero_token_retry_count += 1
            print(
                f"run_track_b_e2e_task.py: zero-token codex completion (attempt {retry_index + 1}); "
                f"retrying ({retry_index + 1}/{max_retries})",
                file=sys.stderr,
            )
    finally:
        _stop_sampler(sampler)
    elapsed_s = time.monotonic() - started
    ended_at = datetime.now(UTC)
    (task_dir / "codex_stdout.log").write_text(result.stdout if result else "", encoding="utf-8")
    (task_dir / "codex_stderr.log").write_text(result.stderr if result else "", encoding="utf-8")
    metrics_post = _metrics_text(args.metrics_url)
    (task_dir / "vllm_metrics_post.txt").write_text(metrics_post, encoding="utf-8")
    gpu_mem_post = _gpu_mem_snapshot()
    vllm_request_metrics_capture: dict[str, Any] | None = None
    captured_proxy_rows: list[dict[str, Any]] = []
    if vllm_request_metrics_jsonl is not None:
        vllm_request_metrics_capture, captured_proxy_rows = _write_vllm_per_turn_from_jsonl(
            task_dir,
            vllm_request_metrics_jsonl,
            start_offset=vllm_request_metrics_start_offset,
            runtime_config_hash=args.runtime_config_hash,
        )
    elif args.defer_vllm_request_metrics_join:
        _write_deferred_vllm_per_turn(
            task_dir,
            runtime_config_hash=args.runtime_config_hash,
            reason="vllm_request_metrics_join_available",
        )
    else:
        _write_vllm_per_turn(task_dir, metrics_pre, metrics_post, runtime_config_hash=args.runtime_config_hash)
    codex_trace_synthesis: dict[str, Any] | None = None
    proxy_capture_configured = vllm_request_metrics_jsonl is not None
    if (
        proxy_capture_configured
        and not args.defer_codex_trace_out
        and not trace_out.is_file()
    ):
        codex_trace_synthesis = _synthesize_codex_trace_from_proxy_rows(
            trace_out,
            family=family,
            variant=variant,
            runtime_config_hash=args.runtime_config_hash,
            started_at=started_at,
            ended_at=ended_at,
            exit_code=result.returncode if result else None,
            captured_rows=captured_proxy_rows,
            wallclock_s=elapsed_s,
        )
    metadata = {
        "schema": "lumo.track_b.e2e_runner_metadata.v1",
        "recorded_at": _now(),
        "family": family,
        "variant": variant,
        "round": args.round,
        "attempt": args.attempt,
        "workspace": _display_path(workspace),
        "source_workspace": _display_path(source_workspace),
        "workspace_isolated": True,
        "trace_out": str(trace_out),
        "elapsed_s": elapsed_s,
        "runtime_config_hash": args.runtime_config_hash,
        "codex_command_template": args.codex_command_template,
        "vllm_request_metrics_jsonl": args.vllm_request_metrics_jsonl,
        "vllm_request_metrics_capture": vllm_request_metrics_capture,
        "codex_trace_synthesis": codex_trace_synthesis,
        "gpu_mem_pre": gpu_mem_pre,
        "gpu_mem_post": gpu_mem_post,
        "ncu_mode": bool(args.ncu_mode),
        "codex_exit_code": result.returncode if result else None,
        "zero_token_retry_count": zero_token_retry_count,
        "zero_token_retry_cohort": zero_token_retry_count > 0,
        "deferred_instrumentation_checks": _deferred_instrumentation_checks(args),
    }
    (task_dir / "runner_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not trace_out.is_file():
        if not args.defer_codex_trace_out:
            raise RuntimeError(
                "no codex_trace.jsonl was produced; supply --vllm-request-metrics-jsonl pointing "
                "at the inference proxy capture file (LUMO_TRACK_B_REQUEST_METRICS_OUT) or set --defer-codex-trace-out"
            )
        _write_deferred_trace(
            trace_out,
            family=family,
            variant=variant,
            runtime_config_hash=args.runtime_config_hash,
            started_at=started_at,
            ended_at=ended_at,
            exit_code=result.returncode if result else None,
        )
    return int(result.returncode if result else 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Track B E2E task with truthful artifact capture.")
    parser.add_argument("family", nargs="?", help="Benchmark family id, or omit with --tasks all.")
    parser.add_argument("variant", nargs="?", default="v1-clean-baseline")
    parser.add_argument("--tasks", choices=["one", "all"], default="one")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1, help="Number of independent attempts to run per task.")
    parser.add_argument(
        "--discard-cold-attempt-exit",
        action="store_true",
        help="Record run_01 artifacts but do not fail the process on its exit code; measured repeats still fail.",
    )
    parser.add_argument("--out-root", default=str(REPO_ROOT / "output" / "track_b_e2e"))
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--reset-prefix-cache-url", default="http://127.0.0.1:9950/reset_prefix_cache")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--runtime-config-hash", default="")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dcgm-interval-s", type=float, default=0.01)
    parser.add_argument("--no-dcgm", action="store_true")
    parser.add_argument(
        "--ncu-mode",
        action="store_true",
        help="Record that this run is an isolated NCU archetype profile run.",
    )
    parser.add_argument("--defer-codex-trace-out", action="store_true")
    parser.add_argument("--defer-vllm-request-metrics-join", action="store_true")
    parser.add_argument("--defer-dcgm-profile-fields", action="store_true")
    parser.add_argument("--allow-missing-workspace-diagnostic", action="store_true")
    parser.add_argument(
        "--zero-token-retries",
        type=int,
        default=0,
        help=(
            "Maximum retries when Codex 0.128.0's zero-token quirk fires "
            "(codex exits 0 but never made a /v1/responses call). Detected "
            "via no new bytes in --vllm-request-metrics-jsonl. Default 0 "
            "(no retry); set to 3 for round-level sweeps where this quirk "
            "would dominate variance. Tasks needing retry are tagged "
            "zero_token_retry_cohort=true so the round summary can keep "
            "them in a separate cohort if desired."
        ),
    )
    parser.add_argument(
        "--vllm-request-metrics-jsonl",
        default="",
        help=(
            "Optional vLLM per-request JSONL side-channel. When supplied, "
            "the runner normalizes it into vllm_per_turn.json instead of "
            "requiring request_id labels in Prometheus."
        ),
    )
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
    try:
        _validate_codex_command_template(args.codex_command_template, require_trace_out=not args.defer_codex_trace_out)
        _validate_runtime_config_hash(args.runtime_config_hash)
    except ValueError as exc:
        parser.error(str(exc))

    if args.tasks == "all":
        from build_track_b_e2e_summary import TRACK_B_E2E_TASKS

        failures = 0
        for task_id in TRACK_B_E2E_TASKS:
            family, variant = task_id.split("/", 1)
            base_attempt = args.attempt
            for offset in range(args.repeat):
                args.attempt = base_attempt + offset
                cold_discarded = args.discard_cold_attempt_exit and offset == 0
                try:
                    rc = run_one(args, family, variant)
                    failures += 1 if rc != 0 and not cold_discarded else 0
                except Exception as exc:
                    print(f"{task_id} attempt {args.attempt}: {exc}", file=sys.stderr)
                    failures += 0 if cold_discarded else 1
            args.attempt = base_attempt
        return 1 if failures else 0
    if not args.family:
        parser.error("family is required unless --tasks all is used")
    try:
        family, variant = _resolve_task_args(args.family, args.variant)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        failures = 0
        base_attempt = args.attempt
        for offset in range(args.repeat):
            args.attempt = base_attempt + offset
            rc = run_one(args, family, variant)
            cold_discarded = args.discard_cold_attempt_exit and offset == 0
            failures += 1 if rc != 0 and not cold_discarded else 0
        args.attempt = base_attempt
        return 1 if failures else 0
    except Exception as exc:
        print(f"run_track_b_e2e_task.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
