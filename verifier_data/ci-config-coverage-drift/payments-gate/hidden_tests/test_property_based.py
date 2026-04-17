from __future__ import annotations

import pytest

from ci_app.workflow_preview import dispatch_job_ids, render_summary


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        (" queue-check ", "ci-app-queue-check"),
        ("ledger_check", "ci-app-ledger-check"),
        ("receipt/check", "ci-app-receipt-check"),
    ],
)
def test_dispatch_job_ids_normalize_simple_payment_labels(label: str, expected: str) -> None:
    assert dispatch_job_ids([label]) == [expected]


def test_render_summary_preserves_requested_order() -> None:
    labels = ["queue-check", "ledger_check", "receipt/check"]

    assert render_summary(labels) == (
        "ci-app-queue-check | ci-app-ledger-check | ci-app-receipt-check"
    )
