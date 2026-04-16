from __future__ import annotations

from report_app.service import build_owner_summary


def test_build_owner_summary_sorts_totals_and_picks_top_owner() -> None:
    summary = build_owner_summary(
        [
            {"label": "stale-pages", "count": 3, "owner": "Jules"},
            {"label": "escalations", "count": 1, "owner": "Ivy"},
            {"label": "queue-audits", "count": 2, "owner": "Jules"},
            {"label": "handoff-drift", "count": 2, "owner": "Ava"},
        ]
    )

    assert summary == {
        "section_count": 4,
        "total_items": 8,
        "top_owner": {"owner": "Jules", "count": 5},
        "owner_totals": [
            {"owner": "Jules", "count": 5},
            {"owner": "Ava", "count": 2},
            {"owner": "Ivy", "count": 1},
        ],
    }
