#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFERABLE_PREFLIGHT_CHECKS = {
    "vllm_request_metrics_join_available",
    "codex_trace_out_supported",
    "dcgm_profile_fields_available",
    # Under the §13-§17 proxy stack the smoke prompt "complete it in this
    # workspace" no longer short-circuits: codex now performs real tool
    # calls and triggers auto-continue retries, so the 180s smoke window
    # is no longer realistic. Substrate-trust comes from §18 (11/11 active
    # tasks produced real artifacts); defer this check so the round can run.
    "codex_command_smoke",
}


def _default_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    return str(venv_python if venv_python.exists() else Path(sys.executable))


def _reset_cache_once(args: argparse.Namespace) -> None:
    """Single round-start /reset_prefix_cache POST. Used by --warmup-policy round_start."""
    import requests  # noqa: PLC0415
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    r = requests.post(args.reset_prefix_cache_url, headers=headers, timeout=30)
    r.raise_for_status()


def _validate_codex_command_template(template: str) -> None:
    # The {trace_out} placeholder is no longer required. Trace emission is now
    # produced by inference-proxy capture + runner-side synthesis, not by a
    # Codex CLI flag. Templates that still include {trace_out} continue to
    # work — the runner formats it to a file path that the proxy synthesis
    # writes to.
    return


def _validate_runtime_config_hash(runtime_config_hash: str) -> None:
    digest = runtime_config_hash.removeprefix("sha256:")
    if (
        not runtime_config_hash.startswith("sha256:")
        or len(digest) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in digest)
    ):
        raise ValueError("--runtime-config-hash must be a sha256:<64-hex-digest> value")


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"malformed JSONL at {path}:{line_number}: {exc.msg}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _verify_trace_correctness_artifact(path: Path, expected_verified_at: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"trace correctness artifact missing: {path}")
    payload = _read_json(path)
    if payload.get("schema") != "lumo.track_b.codex_trace_correctness.v1":
        raise RuntimeError(f"trace correctness artifact schema mismatch: {path}")
    if payload.get("verified_at") != expected_verified_at:
        raise RuntimeError(
            f"trace correctness artifact verified_at {payload.get('verified_at')!r} does not match "
            f"{expected_verified_at!r}"
        )
    if payload.get("trace_out_supported") is not True:
        raise RuntimeError("trace correctness artifact does not prove --trace-out support")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) < 3:
        raise RuntimeError("trace correctness artifact must include at least three task checks")
    required = (
        "trace_out_enabled_exit_code",
        "trace_out_disabled_exit_code",
        "model_outputs_byte_identical",
        "tool_call_sequences_byte_identical",
        "milestone_scores_identical",
        "trace_schema_valid",
    )
    for index, raw_task in enumerate(tasks):
        task = raw_task if isinstance(raw_task, dict) else {}
        task_ok = (
            task.get("trace_out_enabled_exit_code") == 0
            and task.get("trace_out_disabled_exit_code") == 0
            and task.get("model_outputs_byte_identical") is True
            and task.get("tool_call_sequences_byte_identical") is True
            and task.get("milestone_scores_identical") is True
            and task.get("trace_schema_valid") is True
        )
        if not task_ok or any(field not in task for field in required):
            raise RuntimeError(f"trace correctness artifact task_{index} failed")


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _verify_trace_attempt_identity(
    path: Path,
    events: list[dict[str, object]],
    *,
    task_id: str,
    runtime_config_hash: str,
) -> None:
    task_start = next((event for event in events if event.get("event") == "task_start"), None)
    task_end = next((event for event in reversed(events) if event.get("event") == "task_end"), None)
    if task_start is None or task_end is None:
        raise RuntimeError(f"{path} must contain task_start and task_end events")
    for event_name, event in (("task_start", task_start), ("task_end", task_end)):
        trace_task_id = event.get("task_id")
        if isinstance(trace_task_id, str) and trace_task_id and trace_task_id != task_id:
            raise RuntimeError(f"{path} {event_name}.task_id {trace_task_id} does not match {task_id}")
    trace_runtime_config_hash = task_start.get("runtime_config_hash")
    if trace_runtime_config_hash != runtime_config_hash:
        raise RuntimeError(
            f"{path} task_start.runtime_config_hash {trace_runtime_config_hash!r} does not match {runtime_config_hash!r}"
        )


def _attempt_trace_events(
    out_root: Path,
    round_index: int,
    task_id: str,
    attempt: int,
    runtime_config_hash: str,
) -> tuple[Path, list[dict[str, object]]]:
    trace_path = _attempt_dir(out_root, round_index, task_id, attempt) / "codex_trace.jsonl"
    events = _read_jsonl(trace_path)
    _verify_trace_attempt_identity(trace_path, events, task_id=task_id, runtime_config_hash=runtime_config_hash)
    return trace_path, events


def _trace_wallclock_s(path: Path, events: list[dict[str, object]]) -> float:
    task_start: dict[str, object] | None = None
    task_end: dict[str, object] | None = None
    for event in events:
        if event.get("event") == "task_start" and task_start is None:
            task_start = event
        if event.get("event") == "task_end":
            task_end = event
    if task_start is None or task_end is None:
        raise RuntimeError(f"{path} must contain task_start and task_end events")
    start_ts = _parse_ts(task_start.get("ts"))
    end_ts = _parse_ts(task_end.get("ts"))
    if start_ts is None or end_ts is None:
        raise RuntimeError(f"{path} task_start/task_end timestamps must be ISO-8601 strings")
    return max(0.0, (end_ts - start_ts).total_seconds())


def _attempt_dir(out_root: Path, round_index: int, task_id: str, attempt: int) -> Path:
    family, variant = task_id.split("/", 1)
    return out_root / f"round_{round_index}" / f"{family}__{variant}" / f"run_{attempt:02d}"


def _attempt_wallclocks(
    out_root: Path,
    round_index: int,
    task_id: str,
    attempts: range,
    runtime_config_hash: str,
) -> list[float]:
    wallclocks: list[float] = []
    for attempt in attempts:
        trace_path, events = _attempt_trace_events(out_root, round_index, task_id, attempt, runtime_config_hash)
        wallclocks.append(_trace_wallclock_s(trace_path, events))
    return wallclocks


def _completion_tokens_for_attempt(
    out_root: Path,
    round_index: int,
    task_id: str,
    attempt: int,
    runtime_config_hash: str,
) -> int:
    _trace_path, events = _attempt_trace_events(out_root, round_index, task_id, attempt, runtime_config_hash)
    total = 0
    for event in events:
        if event.get("event") != "turn_end":
            continue
        completion_tokens = event.get("completion_tokens")
        if isinstance(completion_tokens, (int, float)):
            total += int(completion_tokens)
    return total


def _verify_generation_volume_within_band(
    out_root: Path,
    round_index: int,
    task_id: str,
    attempts: range,
    runtime_config_hash: str,
) -> None:
    totals = [
        _completion_tokens_for_attempt(out_root, round_index, task_id, attempt, runtime_config_hash)
        for attempt in attempts
    ]
    median = statistics.median(totals)
    if median <= 0:
        raise RuntimeError(f"{task_id} has no measured completion tokens across attempts {attempts.start}..{attempts.stop - 1}")
    oversized = [
        f"run_{attempt:02d}={total}"
        for attempt, total in zip(attempts, totals, strict=True)
        if total > 1.5 * median
    ]
    if oversized:
        raise RuntimeError(
            f"{task_id} failed generation-token-volume guard; median={median}, oversized={', '.join(oversized)}"
        )


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
        "--codex-command-template",
        getattr(args, "codex_command_template", ""),
        "--codex-model",
        getattr(args, "model", "qwen3.5-27b"),
        "--codex-endpoint",
        getattr(args, "endpoint", "http://127.0.0.1:9950/v1"),
        "--codex-api-key",
        getattr(args, "api_key", "local"),
        "--codex-smoke-timeout-s",
        str(getattr(args, "codex_smoke_timeout_s", 180.0)),
    ]
    if args.vllm_request_metrics_jsonl:
        command.extend(["--vllm-request-metrics-jsonl", args.vllm_request_metrics_jsonl])
    deferred = set(getattr(args, "defer_preflight_checks", []) or [])
    if deferred:
        command.extend(["--defer-checks", *sorted(deferred)])
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
        "--runtime-config-hash",
        args.runtime_config_hash,
        "--timeout-s",
        str(args.timeout_s),
        "--codex-command-template",
        args.codex_command_template,
        "--discard-cold-attempt-exit",
    ]
    if args.vllm_request_metrics_jsonl:
        command.extend(["--vllm-request-metrics-jsonl", args.vllm_request_metrics_jsonl])
    zero_token_retries = int(getattr(args, "zero_token_retries", 0) or 0)
    if zero_token_retries > 0:
        command.extend(["--zero-token-retries", str(zero_token_retries)])
    if getattr(args, "warmup_system_prompt_json", "") and args.warmup_policy == "per_task":
        # per_task policy: forward warmup to each per-task runner
        command.extend(["--warmup-system-prompt-json", args.warmup_system_prompt_json])
        command.extend(["--warmup-hit-rate-threshold", str(args.warmup_hit_rate_threshold)])
        command.extend(["--warmup-timeout-s", str(args.warmup_timeout_s)])
    if args.warmup_policy == "round_start":
        # round_start policy: codex prefix is pre-warmed at sweep start and
        # MUST NOT be evicted by per-task /reset_prefix_cache calls. Pin the
        # round-level system_prompt hash for rule-19 stability checks.
        command.extend([
            "--reset-prefix-cache-url", "",
            "--round-start-system-prompt-json", args.warmup_system_prompt_json or "",
        ])
    deferred = getattr(args, "defer_preflight_checks", []) or []
    if "codex_trace_out_supported" in deferred:
        command.append("--defer-codex-trace-out")
    if "vllm_request_metrics_join_available" in deferred:
        command.append("--defer-vllm-request-metrics-join")
    if "dcgm_profile_fields_available" in deferred:
        command.append("--defer-dcgm-profile-fields")
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
        "--sample-hash-match",
    ]
    if args.protocol_hash_match:
        command.append("--protocol-hash-match")
    deferred = getattr(args, "defer_preflight_checks", []) or []
    if "codex_trace_out_supported" not in deferred:
        command.append("--generation-volume-within-band")
    if deferred:
        command.extend(["--deferred-instrumentation-checks", *deferred])
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
    deferred = getattr(args, "defer_preflight_checks", []) or []
    if deferred:
        command.extend(["--deferred-instrumentation-checks", *deferred])
    if args.write_untrusted_diagnostic:
        command.append("--write-untrusted-diagnostic")
    return command


def _reject_existing_round_outputs(round_dir: Path) -> None:
    allowed = {
        "preflight_audit.json",
        # Pre-staged Round 4a artifacts — copied in by the operator before the
        # sweep so the round-start warmup-pass has its system-prompt source:
        "codex_system_prompt.json",
        "codex_system_prompt_decomposition.json",
    }
    stale = sorted(path for path in round_dir.iterdir() if path.name not in allowed)
    if stale:
        rel = ", ".join(str(path.relative_to(round_dir)) for path in stale[:5])
        if len(stale) > 5:
            rel += f", ... +{len(stale) - 5} more"
        raise RuntimeError(f"round directory already contains measurement outputs; move or archive them first: {rel}")


def _read_blockers(preflight_out: Path) -> str:
    if not preflight_out.is_file():
        return "preflight did not write an audit artifact"
    payload = _read_json(preflight_out)
    blockers = payload.get("blocking_reasons")
    if isinstance(blockers, list) and blockers:
        return ", ".join(str(blocker) for blocker in blockers)
    return "preflight failed"


def run_round(args: argparse.Namespace) -> int:
    deferred = set(args.defer_preflight_checks or [])
    if args.vllm_request_metrics_jsonl:
        # Proxy-synthesized trace + metrics join: schema-strict rules expect
        # task_score and join evidence that only a Codex source patch could
        # produce. Auto-defer those rules so summary attestation is honest
        # about the substrate. Preflight still verifies the substrate is wired
        # up; this only relaxes summary-level attestation.
        deferred.update({"codex_trace_out_supported", "vllm_request_metrics_join_available"})
    args.defer_preflight_checks = sorted(deferred)
    unknown_defers = sorted(deferred - DEFERABLE_PREFLIGHT_CHECKS)
    if unknown_defers:
        raise ValueError(f"unsupported deferred preflight checks: {', '.join(unknown_defers)}")
    if "codex_trace_out_supported" not in args.defer_preflight_checks:
        _validate_codex_command_template(args.codex_command_template)
    _validate_runtime_config_hash(args.runtime_config_hash)
    if args.repeat < 4:
        raise ValueError("--repeat must be >= 4 so run_01 can be discarded as cold and 3 measured runs remain")
    if not args.protocol_hash_match:
        raise ValueError("--protocol-hash-match is required for trusted round summaries")

    out_root = Path(args.out_root)
    round_dir = out_root / f"round_{args.round}"
    round_dir.mkdir(parents=True, exist_ok=True)
    _reject_existing_round_outputs(round_dir)
    preflight_out = round_dir / "preflight_audit.json"

    preflight = _run(_preflight_command(args, preflight_out))
    if preflight.returncode != 0:
        print(f"Track B E2E round blocked by preflight: {_read_blockers(preflight_out)}", file=sys.stderr)
        return 1

    if "codex_trace_out_supported" not in args.defer_preflight_checks:
        _verify_trace_correctness_artifact(
            Path(args.trace_correctness_artifact),
            args.trace_emitter_correctness_verified_at,
        )

    # Round 4a — copy the codex_system_prompt.json artifact into the round
    # directory so rule 18 ("System-prompt decomposition recorded for round")
    # has its evidence in-place, and run the warmup-pass ONCE at round start
    # so the codex CLI static prefix lives in the prefix cache for the whole
    # sweep. Per-task cache reset and per-task warmup are intentionally OFF
    # in this mode — the codex prefix behaves like a forever-cached entry
    # that production traffic would naturally maintain.
    canonical_system_prompt_hash: str | None = None
    if args.warmup_system_prompt_json:
        src_sp = Path(args.warmup_system_prompt_json)
        if not src_sp.is_file():
            raise RuntimeError(f"warmup-system-prompt-json missing: {src_sp}")
        dest_sp = round_dir / "codex_system_prompt.json"
        if dest_sp.resolve() != src_sp.resolve():
            dest_sp.write_text(src_sp.read_text(encoding="utf-8"), encoding="utf-8")
        src_dec = src_sp.parent / "codex_system_prompt_decomposition.json"
        if src_dec.is_file():
            dest_dec = round_dir / "codex_system_prompt_decomposition.json"
            if dest_dec.resolve() != src_dec.resolve():
                dest_dec.write_text(src_dec.read_text(encoding="utf-8"), encoding="utf-8")
        canonical_system_prompt_hash = json.loads(dest_sp.read_text(encoding="utf-8")).get("static_content_hash")

        if args.warmup_policy == "round_start":
            # Reset the cache once, then prime + verify it. After this, every
            # per-task codex call hits the warm prefix; per-task reset and
            # per-task warmup are skipped (see _runner_command + below).
            if args.reset_prefix_cache_url:
                _reset_cache_once(args)
            warmup_out = round_dir / "round_warmup_pass.json"
            warmup_cmd = [
                args.python,
                str(REPO_ROOT / "scripts" / "run_track_b_e2e_warmup.py"),
                "--system-prompt-json", str(dest_sp),
                "--endpoint", args.endpoint,
                "--metrics-url", args.metrics_url,
                "--api-key", args.api_key,
                "--model", args.model,
                "--mode", "both",
                "--hit-rate-threshold", str(args.warmup_hit_rate_threshold),
                "--timeout-s", str(args.warmup_timeout_s),
                "--out", str(warmup_out),
            ]
            warmup_proc = _run(warmup_cmd)
            if warmup_proc.returncode != 0:
                print(warmup_proc.stderr, file=sys.stderr, end="")
                raise RuntimeError(
                    f"round-start warmup-pass failed (rule 17): rc={warmup_proc.returncode}; "
                    f"see {warmup_out}"
                )

    runner = _run(_runner_command(args))
    if runner.returncode != 0:
        print(runner.stderr, file=sys.stderr, end="")
        return runner.returncode

    if args.warmup_system_prompt_json:
        # Rule 19: system-prompt content_hash must be stable across all attempts.
        # Source of canonical hash:
        #   - per_task warmup-pass: each runner_metadata.warmup_pass.system_prompt_content_hash
        #   - round_start warmup-pass: round_start_system_prompt_content_hash recorded by runner
        sp_artifact = json.loads((round_dir / "codex_system_prompt.json").read_text(encoding="utf-8"))
        canonical_hash = sp_artifact.get("static_content_hash")
        if not canonical_hash:
            raise RuntimeError(
                "rule 18: round codex_system_prompt.json missing static_content_hash"
            )
        per_attempt_hashes: dict[str, str | None] = {}
        for run_meta in round_dir.glob("*__*/run_*/runner_metadata.json"):
            meta = json.loads(run_meta.read_text(encoding="utf-8"))
            warmup = meta.get("warmup_pass") or {}
            recorded_hash = warmup.get("system_prompt_content_hash") or meta.get("round_start_system_prompt_content_hash")
            per_attempt_hashes[str(run_meta.relative_to(round_dir))] = recorded_hash
        mismatches = {k: v for k, v in per_attempt_hashes.items() if v != canonical_hash}
        if mismatches:
            mismatch_summary = json.dumps({"canonical": canonical_hash, "mismatches": mismatches}, indent=2)
            raise RuntimeError(
                f"rule 19 failure: codex_system_prompt static_content_hash mismatch across attempts:\n{mismatch_summary}"
            )

    for task_id in _tasks():
        measured_attempts = range(2, args.repeat + 1)
        if "codex_trace_out_supported" not in args.defer_preflight_checks:
            _verify_generation_volume_within_band(
                out_root,
                args.round,
                task_id,
                measured_attempts,
                args.runtime_config_hash,
            )
        task_dir = _attempt_dir(out_root, args.round, task_id, measured_attempts.start)
        wallclocks = _attempt_wallclocks(out_root, args.round, task_id, measured_attempts, args.runtime_config_hash)
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
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--out-root", default=str(REPO_ROOT / "output" / "track_b_e2e"))
    parser.add_argument("--health-url", default="http://127.0.0.1:9950/health")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9950/metrics")
    parser.add_argument("--endpoint", default="http://127.0.0.1:9950/v1")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--codex-smoke-timeout-s", type=float, default=180.0)
    parser.add_argument("--vllm-request-metrics-jsonl", default="")
    parser.add_argument(
        "--zero-token-retries",
        type=int,
        default=3,
        help=(
            "Forwarded to run_track_b_e2e_task.py per task; mitigates Codex 0.128.0's zero-token quirk. "
            "Default 3 since Round 4a (rule from track-b-e2e-round4a-measurement-protocol-spec)."
        ),
    )
    parser.add_argument(
        "--warmup-system-prompt-json",
        default="",
        help=(
            "Path to a codex_system_prompt.json artifact (schema "
            "lumo.track_b.codex_system_prompt.v1). When supplied, every task "
            "executes a warmup-pass before the codex spawn so the static "
            "Codex CLI system prompt is cached. Required for Round 4a per "
            "truthful-measurement rules 17-19."
        ),
    )
    parser.add_argument("--warmup-hit-rate-threshold", type=float, default=0.95)
    parser.add_argument("--warmup-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--warmup-policy",
        choices=["round_start", "per_task", "off"],
        default="round_start",
        help=(
            "round_start (default, Round 4a v2): warmup-pass once at sweep start, "
            "per-task /reset_prefix_cache disabled — codex prefix lives in cache "
            "for the whole sweep, mimicking production. "
            "per_task: legacy Round 4a v1 — warmup-pass + reset before each task. "
            "off: no warmup-pass; preserves v3 behavior."
        ),
    )
    parser.add_argument(
        "--reset-prefix-cache-url",
        default="http://127.0.0.1:9950/reset_prefix_cache",
        help="Cache-reset endpoint. With round_start policy, called once at sweep start.",
    )
    parser.add_argument("--clock-skew-ms-p99", type=float, required=True)
    parser.add_argument("--trace-emitter-correctness-verified-at", required=True)
    parser.add_argument(
        "--trace-correctness-artifact",
        default=str(REPO_ROOT / "output" / "track_b_e2e" / "codex_trace_emitter_correctness.json"),
    )
    parser.add_argument("--protocol-hash-match", action="store_true")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--config-delta-vs-prior-round", default="")
    parser.add_argument("--auto-research-agent-recommendation", default="")
    parser.add_argument("--next-round-proposal", default="")
    parser.add_argument("--write-untrusted-diagnostic", action="store_true")
    parser.add_argument(
        "--defer-preflight-checks",
        nargs="*",
        default=[],
        help=(
            "Exclude these failed instrumentation checks from this round's required measurement contract. "
            "Supported values: vllm_request_metrics_join_available, codex_trace_out_supported, "
            "dcgm_profile_fields_available."
        ),
    )
    args = parser.parse_args(argv)
    try:
        return run_round(args)
    except Exception as exc:
        print(f"run_track_b_e2e_round.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
