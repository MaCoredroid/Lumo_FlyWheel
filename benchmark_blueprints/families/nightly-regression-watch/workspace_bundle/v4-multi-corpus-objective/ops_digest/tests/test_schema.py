from __future__ import annotations

from ops_digest.src.schema import normalize_run


def test_missing_required_milestone_is_blocking() -> None:
    payload = {
        "run_id": "inline-required",
        "report_date": "2026-05-01",
        "completed_at": "2026-05-01T03:00:00Z",
        "final_verdict": {"pass": True},
        "milestones": {
            "results": {
                "M2_primary_fix": {"status": "passed", "required": True},
                "M4_functional": {"status": "missing", "required": True},
                "M5_e2e": {"status": "passed", "required": True},
            }
        },
        "warnings": [],
    }
    run = normalize_run(payload)
    assert run["is_blocking"] is True
    assert "M4_functional" in run["missing_required"]


def test_advisory_warning_does_not_page() -> None:
    payload = {
        "run_id": "inline-advisory",
        "report_date": "2026-05-02",
        "completed_at": "2026-05-02T03:00:00Z",
        "final_verdict": {"pass": True},
        "milestones": {"results": {"M2_primary_fix": True, "M4_functional": True, "M5_e2e": True}},
        "warnings": ["advisory: retry budget dipped but recovered"],
    }
    run = normalize_run(payload)
    assert run["is_blocking"] is False
    assert run["label"] == "Healthy night"
