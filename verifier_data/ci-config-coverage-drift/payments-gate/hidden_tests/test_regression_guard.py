from __future__ import annotations

from ci_app.service import run_checks
from ci_app.workflow_preview import dispatch_job_ids, render_summary, workflow_command


def test_render_summary_mentions_every_required_job_once() -> None:
    ids = dispatch_job_ids()
    summary = render_summary()

    for job_id in ids:
        assert summary.count(job_id) == 1


def test_default_dispatch_job_order_tracks_required_checks() -> None:
    expected_suffixes = run_checks()
    actual = dispatch_job_ids()

    assert [job_id.rsplit("-", 2)[-2] + "-" + job_id.rsplit("-", 1)[-1] for job_id in actual] == expected_suffixes


def test_workflow_command_keeps_manual_override_path() -> None:
    assert workflow_command("legacy-probe") == "python scripts/run_ci.py --package legacy-probe"
