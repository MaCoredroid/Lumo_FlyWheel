from __future__ import annotations

REQUIRED_MILESTONES = ("M2_primary_fix", "M4_functional", "M5_e2e")


def normalize_run(payload: dict) -> dict:
    # Legacy behavior: still reads the pre-rollover schema.
    final_pass = bool(payload.get("pass"))
    milestone_results = payload.get("milestones", {})
    missing_required = [
        milestone_id
        for milestone_id in REQUIRED_MILESTONES
        if milestone_results.get(milestone_id) is False
    ]
    warnings = list(payload.get("warnings") or [])
    blocking = (not final_pass) or bool(warnings)
    return {
        "run_id": payload["run_id"],
        "report_date": payload["report_date"],
        "completed_at": payload["completed_at"],
        "final_pass": final_pass,
        "missing_required": missing_required,
        "warnings": warnings,
        "is_blocking": blocking,
        "label": "Action required" if blocking else "Healthy night",
        "summary": "legacy parser path",
    }
