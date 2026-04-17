from __future__ import annotations

from report_app.service import build_triage_summary


def test_build_triage_summary_sorts_owner_load_and_tracks_breaches() -> None:
    summary = build_triage_summary(
        [
            {
                "service": "billing-api",
                "severity": "sev1",
                "owner": "Jules",
                "minutes_open": 21,
                "acked": False,
            },
            {
                "service": "search-indexer",
                "severity": "sev2",
                "owner": "Ivy",
                "minutes_open": 14,
                "acked": True,
            },
            {
                "service": "checkout-web",
                "severity": "sev2",
                "owner": "Jules",
                "minutes_open": 33,
                "acked": False,
            },
            {
                "service": "fraud-worker",
                "severity": "sev3",
                "owner": "Ava",
                "minutes_open": 61,
                "acked": False,
            },
        ]
    )

    assert summary == {
        "incident_count": 4,
        "breached_count": 3,
        "highest_load_owner": {"owner": "Jules", "count": 2, "breached": 2},
        "owner_load": [
            {"owner": "Jules", "count": 2, "breached": 2},
            {"owner": "Ava", "count": 1, "breached": 1},
            {"owner": "Ivy", "count": 1, "breached": 0},
        ],
    }
