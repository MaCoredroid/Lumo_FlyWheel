from __future__ import annotations

from ci_app.service import run_checks
from ci_app.workflow_preview import preview_jobs, render_summary, workflow_command


def test_render_summary_mentions_every_preview_job_once() -> None:
    jobs = preview_jobs()
    summary = render_summary()

    for job in jobs:
        assert summary.count(job["job_id"]) == 1


def test_default_preview_job_order_tracks_required_checks() -> None:
    expected_suffixes = run_checks()
    actual = preview_jobs()

    assert [job["job_id"].removeprefix("ci-app-") for job in actual] == expected_suffixes


def test_workflow_command_keeps_manual_override_path() -> None:
    assert workflow_command("legacy-probe") == "python scripts/run_ci.py --package legacy-probe"
