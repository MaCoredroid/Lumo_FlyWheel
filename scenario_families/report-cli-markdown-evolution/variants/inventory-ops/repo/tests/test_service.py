from __future__ import annotations

from report_app.service import build_owner_summary


def test_build_owner_summary_sorts_totals_and_picks_top_owner() -> None:
    summary = build_owner_summary(
        [
            {"label": "backfills", "count": 4, "owner": "Ava"},
            {"label": "returns", "count": 2, "owner": "Zoe"},
            {"label": "cycle-counts", "count": 3, "owner": "Bea"},
            {"label": "late-recounts", "count": 1, "owner": "Bea"},
            {"label": "audit-holds", "count": 4, "owner": "Ava"},
        ]
    )

    assert summary == {
        "section_count": 5,
        "total_items": 14,
        "top_owner": {"owner": "Ava", "count": 8},
        "owner_totals": [
            {"owner": "Ava", "count": 8},
            {"owner": "Bea", "count": 4},
            {"owner": "Zoe", "count": 2},
        ],
    }
