from __future__ import annotations

from ci_app.workflow_preview import dispatch_job_ids, render_summary


def test_preview_dispatch_ids_preserve_required_labels() -> None:
    ids = dispatch_job_ids()

    assert len(ids) == 2
    assert ids[0].endswith("queue-check")
    assert ids[1].endswith("ledger-check")


def test_preview_summary_mentions_every_required_label() -> None:
    summary = render_summary()

    assert "queue-check" in summary
    assert "ledger-check" in summary
