"""Layer 6 — direct assertions for historically weak edges."""
from __future__ import annotations

from report_app.formatting import format_ack_state, format_breach_count, format_incident_count
from report_app.service import build_triage_summary


def test_format_ack_state_yes_and_no() -> None:
    assert format_ack_state(True) == "yes"
    assert format_ack_state(False) == "no"


def test_format_incident_count_handles_zero_one_many() -> None:
    assert format_incident_count(0) == "0 active incidents"
    assert format_incident_count(1) == "1 active incident"
    assert format_incident_count(3) == "3 active incidents"


def test_format_breach_count_handles_zero_one_many() -> None:
    assert format_breach_count(0) == "0 ack SLA breaches"
    assert format_breach_count(1) == "1 ack SLA breach"
    assert format_breach_count(4) == "4 ack SLA breaches"


def test_owner_load_tie_breaks_breaches_before_name() -> None:
    summary = build_triage_summary(
        [
            {
                "service": "svc-a",
                "severity": "sev1",
                "owner": "Bea",
                "minutes_open": 22,
                "acked": False,
            },
            {
                "service": "svc-b",
                "severity": "sev3",
                "owner": "Bea",
                "minutes_open": 12,
                "acked": False,
            },
            {
                "service": "svc-c",
                "severity": "sev2",
                "owner": "Ava",
                "minutes_open": 18,
                "acked": True,
            },
            {
                "service": "svc-d",
                "severity": "sev3",
                "owner": "Ava",
                "minutes_open": 12,
                "acked": False,
            },
        ]
    )

    assert summary["owner_load"] == [
        {"owner": "Bea", "count": 2, "breached": 1},
        {"owner": "Ava", "count": 2, "breached": 0},
    ]


def test_summary_counts_every_runtime_incident_row() -> None:
    summary = build_triage_summary(
        [
            {
                "service": "svc-a",
                "severity": "sev2",
                "owner": "Jules",
                "minutes_open": 31,
                "acked": False,
            },
            {
                "service": "svc-a",
                "severity": "sev2",
                "owner": "Jules",
                "minutes_open": 32,
                "acked": False,
            },
        ]
    )

    assert summary["incident_count"] == 2
    assert summary["owner_load"] == [{"owner": "Jules", "count": 2, "breached": 2}]
