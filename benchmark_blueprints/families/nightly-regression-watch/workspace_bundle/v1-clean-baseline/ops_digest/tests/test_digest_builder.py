from __future__ import annotations

from ops_digest.src.digest_builder import build_digest_markdown, select_latest_per_day


def test_selects_latest_run_per_day() -> None:
    runs = [
        {
            "run_id": "older",
            "report_date": "2026-05-03",
            "completed_at": "2026-05-03T01:00:00Z",
            "is_blocking": True,
            "summary": "older failure",
        },
        {
            "run_id": "newer",
            "report_date": "2026-05-03",
            "completed_at": "2026-05-03T04:00:00Z",
            "is_blocking": False,
            "summary": "newer clean rerun",
        },
    ]
    latest = select_latest_per_day(runs)
    by_day = {run["report_date"]: run["run_id"] for run in latest}
    assert by_day["2026-05-03"] == "newer"


def test_generated_digest_uses_blocking_and_healthy_sections() -> None:
    digest = build_digest_markdown(
        [
            {
                "run_id": "blocking-run",
                "report_date": "2026-05-04",
                "completed_at": "2026-05-04T03:00:00Z",
                "is_blocking": True,
                "summary": "missing required milestones: M4_functional",
            },
            {
                "run_id": "healthy-run",
                "report_date": "2026-05-05",
                "completed_at": "2026-05-05T03:00:00Z",
                "is_blocking": False,
                "summary": "advisory warnings: retry delayed",
            },
        ]
    )
    assert "## Action required" in digest
    assert "## Healthy nights" in digest
    assert "Action required: 2026-05-04" not in digest
