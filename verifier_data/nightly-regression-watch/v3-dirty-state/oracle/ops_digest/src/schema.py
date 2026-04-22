from __future__ import annotations

from typing import Any

REQUIRED_MILESTONES = ("M2_primary_fix", "M4_functional", "M5_e2e")


def _milestone_passed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if not isinstance(value, dict):
        return False
    if "pass" in value:
        return bool(value["pass"])
    if "passed" in value:
        return bool(value["passed"])
    if "passed_bool" in value:
        return bool(value["passed_bool"])
    return value.get("status") in {"passed", "ok", "success"}


def _milestone_required(value: Any) -> bool:
    if isinstance(value, dict) and "required" in value:
        return bool(value["required"])
    return True


def normalize_run(payload: dict[str, Any]) -> dict[str, Any]:
    final = payload.get("final_verdict") or {}
    final_pass = bool(final.get("pass"))

    raw_milestones = payload.get("milestones") or {}
    if isinstance(raw_milestones, dict) and "results" in raw_milestones:
        milestone_results = raw_milestones.get("results") or {}
    else:
        milestone_results = raw_milestones

    missing_required = sorted(
        milestone_id
        for milestone_id in REQUIRED_MILESTONES
        if _milestone_required(milestone_results.get(milestone_id))
        and not _milestone_passed(milestone_results.get(milestone_id))
    )

    warnings = list(payload.get("warnings") or [])
    blocking = (not final_pass) or bool(missing_required)
    reason_bits: list[str] = []
    if not final_pass:
        reason_bits.append("final verdict failed")
    if missing_required:
        reason_bits.append("missing required milestones: " + ", ".join(missing_required))
    if warnings and not blocking:
        reason_bits.append("advisory warnings: " + "; ".join(warnings))

    return {
        "run_id": payload["run_id"],
        "report_date": payload["report_date"],
        "completed_at": payload["completed_at"],
        "final_pass": final_pass,
        "missing_required": missing_required,
        "warnings": warnings,
        "is_blocking": blocking,
        "label": "Action required" if blocking else "Healthy night",
        "summary": "; ".join(reason_bits) if reason_bits else "no blocking issues",
    }
