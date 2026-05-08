#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE_CORRECTNESS_MIN_TASKS = 3
TRACE_CORRECTNESS_REQUIRED_TASK_FIELDS = (
    "task_id",
    "trace_out_enabled_exit_code",
    "trace_out_disabled_exit_code",
    "model_outputs_byte_identical",
    "tool_call_sequences_byte_identical",
    "milestone_scores_identical",
    "trace_schema_valid",
)
ROUND0_MIN_TRUSTED_TASKS = 12
NCU_ARCHETYPES = (
    "long-text",
    "tool-call-frame",
    "pure-investigation",
    "multimodal-prefill",
    "subagent-orchestration",
)
NCU_REQUIRED_METRICS = (
    "gpu__time_duration.sum",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_memory_ld_pred_on.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "tpc__warps_active.avg.pct_of_peak_sustained_active",
)
NCU_ARCHETYPE_TASKS = {
    "long-text": "sqlalchemy-2-session-modernization/v1-clean-baseline",
    "tool-call-frame": "policy-aware-request-resolution/v1-clean-baseline",
    "pure-investigation": "dead-flag-reachability-audit/v1-clean-baseline",
    "multimodal-prefill": "responsive-checkout-visual-regression/v1-clean-baseline",
    "subagent-orchestration": "fanout-fullstack-release-blocker/v1-clean-baseline",
}
ROUND_PROPOSAL_PROMPT = "prompts/track_b_e2e_round_proposal.md"
ROUND_PROPOSAL_REQUIRED_TEXT = {
    "uses_hard_gated_round_driver": "scripts/run_track_b_e2e_round.py",
    "passes_runtime_config_hash": "--runtime-config-hash {{runtime_config_hash}}",
    "passes_protocol_hash_gate": "--protocol-hash-match",
    "passes_trace_correctness_artifact": "--trace-correctness-artifact output/track_b_e2e/codex_trace_emitter_correctness.json",
    "uses_preflight_script": "scripts/preflight_track_b_e2e.py",
    "checks_all_spec_decode_counters": "spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total",
}
ROUND_PROPOSAL_FORBIDDEN_TEXT = {
    "direct_repeat3_task_measurement": "run_track_b_e2e_task.py --round {{round}} --tasks all --repeat 3",
}
TRACE_PATCH_CANDIDATES = (
    "vendor/codex-cli/patches/trace_emitter.patch",
    "patches/codex/trace_emitter.patch",
)
TRACE_PATCH_REQUIRED_TEXT = {
    "unified_diff": "diff --git",
    "codex_rust_surface": "codex-rs/",
    "trace_out_flag": "--trace-out",
    "task_start_event": "task_start",
    "turn_start_event": "turn_start",
    "turn_end_event": "turn_end",
    "tool_call_event": "tool_call",
    "task_end_event": "task_end",
    "runtime_hash_stamp": "runtime_config_hash",
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _exists(rel: str) -> bool:
    return (REPO_ROOT / rel).exists()


def _contains(rel: str, needle: str) -> bool:
    path = REPO_ROOT / rel
    if not path.is_file():
        return False
    return needle in path.read_text(encoding="utf-8", errors="replace")


def _runtime_config_hash_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdefABCDEF" for char in digest)


def _finite_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0


def _round_proposal_prompt_verification() -> dict[str, Any]:
    path = REPO_ROOT / ROUND_PROPOSAL_PROMPT
    exists = path.is_file()
    text = path.read_text(encoding="utf-8", errors="replace") if exists else ""
    required = {name: needle in text for name, needle in ROUND_PROPOSAL_REQUIRED_TEXT.items()}
    forbidden = {name: needle in text for name, needle in ROUND_PROPOSAL_FORBIDDEN_TEXT.items()}
    missing_required = [name for name, present in required.items() if not present]
    present_forbidden = [name for name, present in forbidden.items() if present]
    reasons: list[str] = []
    if not exists:
        reasons.append("prompt_missing")
    reasons.extend(f"missing_{name}" for name in missing_required)
    reasons.extend(f"forbidden_{name}_present" for name in present_forbidden)
    return {
        "ok": exists and not missing_required and not present_forbidden,
        "prompt": ROUND_PROPOSAL_PROMPT,
        "exists": exists,
        "required": required,
        "forbidden": forbidden,
        "reasons": reasons,
    }


def _trace_patch_verification() -> dict[str, Any]:
    patches: list[dict[str, Any]] = []
    for rel in TRACE_PATCH_CANDIDATES:
        path = REPO_ROOT / rel
        exists = path.is_file()
        text = path.read_text(encoding="utf-8", errors="replace") if exists else ""
        required = {name: needle in text for name, needle in TRACE_PATCH_REQUIRED_TEXT.items()}
        missing_required = [name for name, present in required.items() if not present]
        patches.append(
            {
                "path": rel,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "required": required,
                "missing_required": missing_required,
                "ok": exists and bool(text.strip()) and not missing_required,
            }
        )
    ok = any(patch["ok"] for patch in patches)
    reasons: list[str] = []
    if not any(patch["exists"] for patch in patches):
        reasons.append("trace_patch_missing")
    elif not ok:
        reasons.append("trace_patch_content_invalid")
    return {
        "ok": ok,
        "candidates": list(TRACE_PATCH_CANDIDATES),
        "required_text": TRACE_PATCH_REQUIRED_TEXT,
        "reasons": reasons,
        "patches": patches,
    }


def _trace_correctness_verification(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    tasks = payload.get("tasks") if isinstance(payload.get("tasks"), list) else []
    reasons: list[str] = []
    if not payload:
        reasons.append("artifact_missing_or_invalid_json")
    if payload.get("schema") != "lumo.track_b.codex_trace_correctness.v1":
        reasons.append("schema_mismatch")
    if not isinstance(payload.get("verified_at"), str) or not payload.get("verified_at"):
        reasons.append("verified_at_missing")
    if payload.get("trace_out_supported") is not True:
        reasons.append("trace_out_supported_not_true")
    if len(tasks) < TRACE_CORRECTNESS_MIN_TASKS:
        reasons.append("too_few_tasks")

    task_results: list[dict[str, Any]] = []
    for index, raw_task in enumerate(tasks):
        task = raw_task if isinstance(raw_task, dict) else {}
        missing = [field for field in TRACE_CORRECTNESS_REQUIRED_TASK_FIELDS if field not in task]
        task_ok = (
            not missing
            and task.get("trace_out_enabled_exit_code") == 0
            and task.get("trace_out_disabled_exit_code") == 0
            and task.get("model_outputs_byte_identical") is True
            and task.get("tool_call_sequences_byte_identical") is True
            and task.get("milestone_scores_identical") is True
            and task.get("trace_schema_valid") is True
        )
        if not task_ok:
            reasons.append(f"task_{index}_failed")
        task_results.append(
            {
                "task_id": task.get("task_id", f"task_{index}"),
                "ok": task_ok,
                "missing_fields": missing,
            }
        )

    return {
        "ok": not reasons,
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "schema": payload.get("schema"),
        "verified_at": payload.get("verified_at"),
        "task_count": len(tasks),
        "min_task_count": TRACE_CORRECTNESS_MIN_TASKS,
        "reasons": reasons,
        "tasks": task_results,
    }


def _round0_summary_verification(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    reasons: list[str] = []
    if not payload:
        reasons.append("summary_missing_or_invalid_json")
    if payload.get("schema") != "lumo.track_b.e2e_round_summary.v1":
        reasons.append("schema_mismatch")
    if payload.get("round") != 0:
        reasons.append("round_not_zero")
    if not isinstance(payload.get("runtime_config_hash"), str) or not payload.get("runtime_config_hash"):
        reasons.append("runtime_config_hash_missing")
    elif not _runtime_config_hash_valid(payload.get("runtime_config_hash")):
        reasons.append("runtime_config_hash_invalid")
    if payload.get("sample_hash") is None:
        reasons.append("sample_hash_missing")

    trusted_task_count = payload.get("trusted_task_count")
    trusted_unique_task_count = payload.get("trusted_unique_task_count")
    tasks_completed = payload.get("tasks_completed")
    tasks_correctness_passed = payload.get("tasks_correctness_passed")
    if not isinstance(trusted_task_count, int) or trusted_task_count < ROUND0_MIN_TRUSTED_TASKS:
        reasons.append("too_few_trusted_tasks")
    if not isinstance(trusted_unique_task_count, int) or trusted_unique_task_count < ROUND0_MIN_TRUSTED_TASKS:
        reasons.append("too_few_unique_trusted_tasks")
    if not isinstance(tasks_completed, int) or tasks_completed < ROUND0_MIN_TRUSTED_TASKS:
        reasons.append("too_few_completed_tasks")
    if not isinstance(tasks_correctness_passed, int) or tasks_correctness_passed < ROUND0_MIN_TRUSTED_TASKS:
        reasons.append("too_few_correct_tasks")
    if payload.get("duplicate_trusted_task_ids") not in ([], None):
        reasons.append("duplicate_trusted_task_ids_present")
    if payload.get("unexpected_trusted_task_ids") not in ([], None):
        reasons.append("unexpected_trusted_task_ids_present")
    if payload.get("sample_hash_mismatch_count") != 0:
        reasons.append("sample_hash_mismatch")
    if payload.get("runtime_config_hash_mismatch_count") != 0:
        reasons.append("runtime_config_hash_mismatch")
    if payload.get("task_summary_schema_mismatch_count") != 0:
        reasons.append("task_summary_schema_mismatch")
    if payload.get("task_summary_round_mismatch_count") != 0:
        reasons.append("task_summary_round_mismatch")
    if not _finite_positive_number(payload.get("median_wallclock_s")):
        reasons.append("median_wallclock_missing")
    if not _finite_positive_number(payload.get("aggregate_wallclock_s")):
        reasons.append("aggregate_wallclock_missing")
    if not isinstance(payload.get("diagnosis_distribution"), dict) or not payload.get("diagnosis_distribution"):
        reasons.append("diagnosis_distribution_missing")

    return {
        "ok": not reasons,
        "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "schema": payload.get("schema"),
        "round": payload.get("round"),
        "trusted_task_count": trusted_task_count,
        "trusted_unique_task_count": trusted_unique_task_count,
        "min_trusted_task_count": ROUND0_MIN_TRUSTED_TASKS,
        "tasks_completed": tasks_completed,
        "tasks_correctness_passed": tasks_correctness_passed,
        "runtime_config_hash_mismatch_count": payload.get("runtime_config_hash_mismatch_count"),
        "task_summary_schema_mismatch_count": payload.get("task_summary_schema_mismatch_count"),
        "task_summary_round_mismatch_count": payload.get("task_summary_round_mismatch_count"),
        "reasons": reasons,
    }


def _ncu_profile_verification(output_dir: Path, *, expected_runtime_config_hash: str = "") -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    reasons: list[str] = []
    for archetype in NCU_ARCHETYPES:
        path = output_dir / f"ncu_{archetype}.csv"
        metadata_path = output_dir / f"ncu_{archetype}.json"
        exists = path.is_file()
        metadata = _load_json(metadata_path)
        size_bytes = path.stat().st_size if exists else 0
        text = path.read_text(encoding="utf-8", errors="replace") if exists else ""
        no_kernels_profiled = "No kernels were profiled" in text
        metric_values = _ncu_metric_values(text)
        metric_coverage = {metric: metric in text for metric in NCU_REQUIRED_METRICS}
        missing_metrics = [metric for metric, present in metric_coverage.items() if not present]
        nonfinite_metrics = [
            metric for metric, values in metric_values.items() if metric_coverage[metric] and not values
        ]
        metadata_reasons: list[str] = []
        expected_profile_csv = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        if metadata.get("schema") != "lumo.track_b.ncu_archetype_profile.v1":
            metadata_reasons.append("schema_mismatch")
        if metadata.get("archetype") != archetype:
            metadata_reasons.append("archetype_mismatch")
        if metadata.get("task_id") != NCU_ARCHETYPE_TASKS[archetype]:
            metadata_reasons.append("task_id_mismatch")
        if metadata.get("required_metrics") != list(NCU_REQUIRED_METRICS):
            metadata_reasons.append("required_metrics_mismatch")
        if not isinstance(metadata.get("round"), int):
            metadata_reasons.append("round_missing")
        if not isinstance(metadata.get("runtime_config_hash"), str) or not metadata.get("runtime_config_hash"):
            metadata_reasons.append("runtime_config_hash_missing")
        elif not _runtime_config_hash_valid(metadata.get("runtime_config_hash")):
            metadata_reasons.append("runtime_config_hash_invalid")
        elif expected_runtime_config_hash and metadata.get("runtime_config_hash") != expected_runtime_config_hash:
            metadata_reasons.append("runtime_config_hash_mismatch")
        if metadata.get("profile_csv") != expected_profile_csv:
            metadata_reasons.append("profile_csv_mismatch")
        ok = (
            exists
            and size_bytes > 0
            and not no_kernels_profiled
            and not missing_metrics
            and not nonfinite_metrics
            and not metadata_reasons
        )
        if not ok:
            if not exists or size_bytes <= 0:
                reasons.append(f"{archetype}_missing_or_empty")
            elif no_kernels_profiled:
                reasons.append(f"{archetype}_no_kernels_profiled")
            elif missing_metrics:
                reasons.append(f"{archetype}_missing_required_metrics")
            elif nonfinite_metrics:
                reasons.append(f"{archetype}_nonfinite_required_metrics")
            else:
                reasons.append(f"{archetype}_metadata_invalid")
        profiles.append(
            {
                "archetype": archetype,
                "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
                "exists": exists,
                "metadata_path": str(metadata_path.relative_to(REPO_ROOT)) if metadata_path.is_relative_to(REPO_ROOT) else str(metadata_path),
                "metadata_exists": metadata_path.is_file(),
                "metadata_reasons": metadata_reasons,
                "task_id": metadata.get("task_id"),
                "expected_task_id": NCU_ARCHETYPE_TASKS[archetype],
                "round": metadata.get("round"),
                "required_metrics_metadata_match": metadata.get("required_metrics") == list(NCU_REQUIRED_METRICS),
                "runtime_config_hash": metadata.get("runtime_config_hash"),
                "size_bytes": size_bytes,
                "no_kernels_profiled": no_kernels_profiled,
                "required_metric_coverage": metric_coverage,
                "missing_metrics": missing_metrics,
                "required_metric_values": metric_values,
                "nonfinite_metrics": nonfinite_metrics,
                "ok": ok,
            }
        )
    return {
        "ok": not reasons,
        "expected_archetypes": list(NCU_ARCHETYPES),
        "required_metrics": list(NCU_REQUIRED_METRICS),
        "profile_count": sum(1 for profile in profiles if profile["ok"]),
        "expected_profile_count": len(NCU_ARCHETYPES),
        "expected_runtime_config_hash": expected_runtime_config_hash,
        "reasons": reasons,
        "profiles": profiles,
    }


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


def _ncu_metric_values(text: str) -> dict[str, list[float]]:
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


def _status(ok: bool, *, blocked: bool = False) -> str:
    if ok:
        return "complete"
    if blocked:
        return "blocked"
    return "missing"


def _step(step: str, requirement: str, evidence: dict[str, Any], ok: bool, *, blocked: bool = False) -> dict[str, Any]:
    return {
        "step": step,
        "requirement": requirement,
        "status": _status(ok, blocked=blocked),
        "evidence": evidence,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _load_json(Path(args.preflight_json)) if args.preflight_json else {}
    checks = preflight.get("checks") if isinstance(preflight.get("checks"), dict) else {}
    blockers = preflight.get("blocking_reasons") if isinstance(preflight.get("blocking_reasons"), list) else []

    trace_patch = _trace_patch_verification()
    trace_correctness_path = REPO_ROOT / "output" / "track_b_e2e" / "codex_trace_emitter_correctness.json"
    trace_correctness = _trace_correctness_verification(trace_correctness_path)
    round0_summary_path = REPO_ROOT / "output" / "track_b_e2e" / "round_0" / "round_summary.json"
    round0_summary = _round0_summary_verification(round0_summary_path)
    expected_ncu_runtime_config_hash = (
        str(round0_summary.get("runtime_config_hash"))
        if isinstance(round0_summary.get("runtime_config_hash"), str)
        else ""
    )
    ncu_profiles = _ncu_profile_verification(
        REPO_ROOT / "output" / "track_b_e2e",
        expected_runtime_config_hash=expected_ncu_runtime_config_hash,
    )
    round_proposal_prompt = _round_proposal_prompt_verification()

    steps = [
        _step(
            "A",
            "Patch Codex CLI with --trace-out and verify trace-emitter correctness.",
            {
                "trace_patch_exists": any(patch["exists"] for patch in trace_patch["patches"]),
                "trace_patch_verified": trace_patch["ok"],
                "trace_patch": trace_patch,
                "trace_correctness_artifact": str(trace_correctness_path.relative_to(REPO_ROOT)),
                "trace_correctness_exists": trace_correctness_path.is_file(),
                "trace_correctness_verified": trace_correctness["ok"],
                "trace_correctness": trace_correctness,
                "codex_trace_out_supported": checks.get("codex_trace_out_supported", {}).get("ok"),
            },
            trace_patch["ok"]
            and trace_correctness["ok"]
            and checks.get("codex_trace_out_supported", {}).get("ok") is True,
            blocked="codex_trace_out_supported" in blockers,
        ),
        _step(
            "B",
            "DCGM/NVML 100 Hz sampler emits required profile fields.",
            {
                "sampler_script": "scripts/sample_dcgm_during_task.py",
                "sampler_script_exists": _exists("scripts/sample_dcgm_during_task.py"),
                "pynvml_available": checks.get("pynvml_available", {}).get("ok"),
                "dcgmi_available": checks.get("dcgmi_available", {}).get("ok"),
                "dcgm_python_bindings_available": checks.get("dcgm_python_bindings_available", {}).get("ok"),
                "dcgm_python_bindings_modules": checks.get("dcgm_python_bindings_available", {}).get("modules"),
                "dcgm_sampler_runs": checks.get("dcgm_sampler_runs", {}).get("ok"),
                "dcgm_profile_fields_available": checks.get("dcgm_profile_fields_available", {}).get("ok"),
                "missing_profile_fields": checks.get("dcgm_profile_fields_available", {}).get(
                    "missing_profile_fields"
                ),
                "profile_fields_available_sample_count": checks.get("dcgm_profile_fields_available", {}).get(
                    "profile_fields_available_sample_count"
                ),
                "profile_fields_unavailable_reasons": checks.get("dcgm_profile_fields_available", {}).get(
                    "profile_fields_unavailable_reasons"
                ),
                "telemetry_sources": checks.get("dcgm_sampler_runs", {}).get("telemetry_sources"),
            },
            _exists("scripts/sample_dcgm_during_task.py")
            and checks.get("dcgm_sampler_runs", {}).get("ok") is True
            and checks.get("dcgm_profile_fields_available", {}).get("ok") is True,
            blocked="dcgm_profile_fields_available" in blockers,
        ),
        _step(
            "C",
            "E2E task and round runners exist.",
            {
                "runner_script": "scripts/run_track_b_e2e_task.py",
                "runner_script_exists": _exists("scripts/run_track_b_e2e_task.py"),
                "round_driver_script": "scripts/run_track_b_e2e_round.py",
                "round_driver_script_exists": _exists("scripts/run_track_b_e2e_round.py"),
            },
            _exists("scripts/run_track_b_e2e_task.py") and _exists("scripts/run_track_b_e2e_round.py"),
        ),
        _step(
            "D",
            "Per-turn vLLM metric extension captures spec_decode accepted/draft metrics keyed by request id.",
            {
                "metrics_module": "src/lumo_flywheel_serving/metrics.py",
                "compute_vllm_per_request_metrics_exists": _contains(
                    "src/lumo_flywheel_serving/metrics.py",
                    "def compute_vllm_per_request_metrics",
                ),
                "spec_decode_accepted_metric_named": _contains(
                    "src/lumo_flywheel_serving/metrics.py",
                    "spec_decode_num_accepted_tokens",
                ),
                "vllm_request_id_labels_exposed": checks.get("vllm_request_id_labels_exposed", {}).get("ok"),
                "vllm_request_metrics_side_channel_ok": checks.get("vllm_request_metrics_side_channel", {}).get("ok"),
                "vllm_request_metrics_side_channel": checks.get("vllm_request_metrics_side_channel", {}),
                "vllm_request_metrics_join_available": checks.get("vllm_request_metrics_join_available", {}).get("ok"),
            },
            _contains("src/lumo_flywheel_serving/metrics.py", "def compute_vllm_per_request_metrics")
            and checks.get("vllm_request_metrics_join_available", {}).get("ok") is True,
            blocked=(
                "vllm_request_metrics_join_available" in blockers
                or "vllm_request_id_labels_exposed" in blockers
            ),
        ),
        _step(
            "E",
            "Summary join and deterministic diagnosis rule exists.",
            {
                "summary_script": "scripts/build_track_b_e2e_summary.py",
                "exists": _exists("scripts/build_track_b_e2e_summary.py"),
                "truthful_attestation_fields": _contains(
                    "scripts/build_track_b_e2e_summary.py",
                    "truthful_measurement_attestation",
                ),
            },
            _exists("scripts/build_track_b_e2e_summary.py")
            and _contains("scripts/build_track_b_e2e_summary.py", "truthful_measurement_attestation"),
        ),
        _step(
            "F",
            "Auto research agent round proposal prompt drives the hard-gated measurement loop.",
            round_proposal_prompt,
            round_proposal_prompt["ok"],
        ),
        _step(
            "G",
            "Round 0 dry run populated and validated.",
            {
                "round0_summary": str(round0_summary_path.relative_to(REPO_ROOT)),
                "round0_summary_exists": round0_summary_path.is_file(),
                "round0_summary_verified": round0_summary["ok"],
                "round0_summary_verification": round0_summary,
                "ncu_profile_count": ncu_profiles["profile_count"],
                "expected_ncu_profile_count": ncu_profiles["expected_profile_count"],
                "ncu_profile_driver": "scripts/run_track_b_e2e_ncu_profiles.py",
                "ncu_profile_driver_exists": _exists("scripts/run_track_b_e2e_ncu_profiles.py"),
                "ncu_profiles_verified": ncu_profiles["ok"],
                "ncu_profiles": ncu_profiles,
            },
            round0_summary["ok"] and _exists("scripts/run_track_b_e2e_ncu_profiles.py") and ncu_profiles["ok"],
            blocked=bool(blockers),
        ),
    ]

    hard_gates = {
        "preflight_round0_may_run": preflight.get("round0_may_run") is True,
        "round0_summary_verified": round0_summary["ok"],
        "ncu_profiles_verified": ncu_profiles["ok"],
        "trace_correctness_verified": trace_correctness["ok"],
        "round_proposal_prompt_verified": round_proposal_prompt["ok"],
        "all_implementation_steps_complete": all(step["status"] == "complete" for step in steps),
    }
    ready = all(hard_gates.values())
    return {
        "schema": "lumo.track_b.e2e_readiness_manifest.v1",
        "generated_at": _now(),
        "git_head": _git(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git(["status", "--short", "--branch"]),
        "plan": "docs/reports/auto_research/track-b-e2e-agentic-saturation-plan-20260507.md",
        "preflight_json": args.preflight_json,
        "blocking_reasons": blockers,
        "implementation_steps": steps,
        "hard_gates": hard_gates,
        "round0_ready": ready,
        "decision": "round0_may_run" if ready else "round0_blocked",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Track B E2E readiness manifest from real artifacts.")
    parser.add_argument("--preflight-json", default="output/track_b_e2e/preflight_20260507.json")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    payload = build_manifest(args)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if payload["round0_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
