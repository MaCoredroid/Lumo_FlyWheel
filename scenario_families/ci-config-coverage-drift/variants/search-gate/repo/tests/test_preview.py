from __future__ import annotations

from ci_app.workflow_preview import preview_jobs, render_summary


def test_preview_jobs_preserve_required_labels() -> None:
    jobs = preview_jobs()

    assert len(jobs) == 2
    assert jobs[0]["job_id"].endswith("ranking-check")
    assert jobs[1]["job_id"].endswith("fixture-check")
    assert jobs[0]["pytest_selector"].startswith("-k ")
    assert jobs[1]["pytest_selector"].startswith("-k ")


def test_preview_summary_mentions_every_required_label() -> None:
    summary = render_summary()

    assert "ranking-check" in summary
    assert "fixture-check" in summary


def test_preview_jobs_keep_requested_order_for_manual_labels() -> None:
    jobs = preview_jobs(["ranking/check", "fixture:locale_us"])

    assert jobs[0]["job_id"].endswith("ranking-check")
    assert jobs[1]["job_id"].endswith("fixture-locale-us")
