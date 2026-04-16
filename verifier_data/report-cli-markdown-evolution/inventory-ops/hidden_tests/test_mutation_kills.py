"""Layer 6 — direct tests for historically weak edges."""
from __future__ import annotations

from report_app.formatting import format_queue_count
from report_app.service import build_owner_summary


def test_format_queue_count_uses_plural_for_zero() -> None:
    assert format_queue_count(0) == "0 queued items"


def test_watchlist_rows_do_not_replace_real_totals() -> None:
    summary = build_owner_summary(
        [
            {"owner": "Noah", "label": "late-recounts", "count": 2},
            {"owner": "Noah", "label": "dock-checks", "count": 1},
        ],
        include_known_owners=True,
    )

    owner_totals = {entry["owner"]: entry["count"] for entry in summary["owner_totals"]}
    assert owner_totals["Noah"] == 3
    assert owner_totals["Ivy"] == 0


def test_owner_totals_tie_break_owner_name() -> None:
    summary = build_owner_summary(
        [
            {"owner": "Bea", "label": "returns", "count": 2},
            {"owner": "Ava", "label": "recounts", "count": 2},
        ]
    )

    assert summary["owner_totals"] == [
        {"owner": "Ava", "count": 2},
        {"owner": "Bea", "count": 2},
    ]
