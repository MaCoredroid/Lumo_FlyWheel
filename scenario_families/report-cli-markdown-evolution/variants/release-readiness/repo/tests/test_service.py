from __future__ import annotations

from report_app.service import build_owner_summary


def test_build_owner_summary_sorts_totals_and_picks_top_owner() -> None:
    summary = build_owner_summary(
        [
            {"label": "blocked-rollouts", "count": 2, "owner": "Sam"},
            {"label": "hotfixes", "count": 1, "owner": "Rin"},
            {"label": "preflight-checks", "count": 4, "owner": "Sam"},
            {"label": "release-notes", "count": 2, "owner": "Ava"},
        ]
    )

    assert summary == {
        "section_count": 4,
        "total_items": 9,
        "top_owner": {"owner": "Sam", "count": 6},
        "owner_totals": [
            {"owner": "Sam", "count": 6},
            {"owner": "Ava", "count": 2},
            {"owner": "Rin", "count": 1},
        ],
    }
