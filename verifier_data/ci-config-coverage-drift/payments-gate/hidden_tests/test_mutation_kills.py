from __future__ import annotations

import pytest

from ci_app.workflow_preview import dispatch_job_ids


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("receipt__audit", "ci-app-receipt-audit"),
        ("receipt___audit", "ci-app-receipt-audit"),
        ("receipt / / audit", "ci-app-receipt-audit"),
        ("...receipt audit...", "ci-app-receipt-audit"),
    ],
)
def test_separator_heavy_receipt_labels_collapse_to_a_single_slug(
    label: str,
    expected: str,
) -> None:
    assert dispatch_job_ids([label]) == [expected]


def test_boundary_punctuation_is_removed_from_payment_lane_labels() -> None:
    assert dispatch_job_ids(["__ledger-close__"]) == ["ci-app-ledger-close"]
