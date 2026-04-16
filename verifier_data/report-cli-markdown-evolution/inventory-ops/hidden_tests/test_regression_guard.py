"""Layer 4 — regression guards for non-target behavior."""
from __future__ import annotations

from report_app.service import build_owner_summary


def test_include_known_owners_flag_is_opt_in() -> None:
    summary = build_owner_summary(
        [{"owner": "Mae", "label": "blocked-picks", "count": 4}],
        include_known_owners=False,
    )

    owners = [entry["owner"] for entry in summary["owner_totals"]]
    assert owners == ["Mae"]


def test_duplicate_owner_rows_are_preserved_in_sections_contract() -> None:
    sections = [
        {"owner": "Mae", "label": "blocked-picks", "count": 4},
        {"owner": "Mae", "label": "cycle-counts", "count": 1},
    ]
    summary = build_owner_summary(sections)

    assert summary["section_count"] == 2
    assert summary["top_owner"] == {"owner": "Mae", "count": 5}
