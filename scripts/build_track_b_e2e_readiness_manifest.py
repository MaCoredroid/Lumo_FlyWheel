#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    if payload.get("sample_hash_mismatch_count") not in (0, None):
        reasons.append("sample_hash_mismatch")
    if not isinstance(payload.get("median_wallclock_s"), (int, float)):
        reasons.append("median_wallclock_missing")
    if not isinstance(payload.get("aggregate_wallclock_s"), (int, float)):
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
        "reasons": reasons,
    }


def _ncu_profile_verification(output_dir: Path) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    reasons: list[str] = []
    for archetype in NCU_ARCHETYPES:
        path = output_dir / f"ncu_{archetype}.csv"
        exists = path.is_file()
        size_bytes = path.stat().st_size if exists else 0
        text = path.read_text(encoding="utf-8", errors="replace") if exists else ""
        metric_coverage = {metric: metric in text for metric in NCU_REQUIRED_METRICS}
        missing_metrics = [metric for metric, present in metric_coverage.items() if not present]
        ok = exists and size_bytes > 0 and not missing_metrics
        if not ok:
            reasons.append(
                f"{archetype}_missing_or_empty"
                if not exists or size_bytes <= 0
                else f"{archetype}_missing_required_metrics"
            )
        profiles.append(
            {
                "archetype": archetype,
                "path": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
                "exists": exists,
                "size_bytes": size_bytes,
                "required_metric_coverage": metric_coverage,
                "missing_metrics": missing_metrics,
                "ok": ok,
            }
        )
    return {
        "ok": not reasons,
        "expected_archetypes": list(NCU_ARCHETYPES),
        "required_metrics": list(NCU_REQUIRED_METRICS),
        "profile_count": sum(1 for profile in profiles if profile["ok"]),
        "expected_profile_count": len(NCU_ARCHETYPES),
        "reasons": reasons,
        "profiles": profiles,
    }


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

    trace_patch_exists = any(
        path.exists()
        for path in [
            REPO_ROOT / "vendor" / "codex-cli" / "patches" / "trace_emitter.patch",
            REPO_ROOT / "patches" / "codex" / "trace_emitter.patch",
        ]
    )
    trace_correctness_path = REPO_ROOT / "output" / "track_b_e2e" / "codex_trace_emitter_correctness.json"
    trace_correctness = _trace_correctness_verification(trace_correctness_path)
    round0_summary_path = REPO_ROOT / "output" / "track_b_e2e" / "round_0" / "round_summary.json"
    round0_summary = _round0_summary_verification(round0_summary_path)
    ncu_profiles = _ncu_profile_verification(REPO_ROOT / "output" / "track_b_e2e")

    steps = [
        _step(
            "A",
            "Patch Codex CLI with --trace-out and verify trace-emitter correctness.",
            {
                "trace_patch_exists": trace_patch_exists,
                "trace_correctness_artifact": str(trace_correctness_path.relative_to(REPO_ROOT)),
                "trace_correctness_exists": trace_correctness_path.is_file(),
                "trace_correctness_verified": trace_correctness["ok"],
                "trace_correctness": trace_correctness,
                "codex_trace_out_supported": checks.get("codex_trace_out_supported", {}).get("ok"),
            },
            trace_patch_exists
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
                "dcgm_sampler_runs": checks.get("dcgm_sampler_runs", {}).get("ok"),
                "dcgm_profile_fields_available": checks.get("dcgm_profile_fields_available", {}).get("ok"),
                "missing_profile_fields": checks.get("dcgm_profile_fields_available", {}).get(
                    "missing_profile_fields"
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
                "vllm_request_metrics_side_channel": checks.get("vllm_request_metrics_side_channel", {}).get("ok"),
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
            "Auto research agent round proposal prompt template exists.",
            {"prompt": "prompts/track_b_e2e_round_proposal.md", "exists": _exists("prompts/track_b_e2e_round_proposal.md")},
            _exists("prompts/track_b_e2e_round_proposal.md"),
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
