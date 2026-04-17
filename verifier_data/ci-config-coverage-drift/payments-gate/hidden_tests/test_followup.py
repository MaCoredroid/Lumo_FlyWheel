from __future__ import annotations

import pytest

from ci_app.workflow_preview import dispatch_job_ids, render_summary


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("receipt__audit", "ci-app-receipt-audit"),
        ("ledger///hold", "ci-app-ledger-hold"),
        ("  settlement---close  ", "ci-app-settlement-close"),
    ],
)
def test_dispatch_job_ids_collapse_separator_noise(label: str, expected: str) -> None:
    assert dispatch_job_ids([label]) == [expected]


def test_render_summary_never_emits_doubled_hyphens_for_followup_labels() -> None:
    summary = render_summary(["receipt__audit", "ledger///hold"])

    assert "--" not in summary
