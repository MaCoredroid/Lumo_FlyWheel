"""Layer 5 — follow-up coverage for the ack-SLA boundary trap."""
from __future__ import annotations

from conftest import cli_json, cli_markdown
from report_app.fixtures import ACK_SLA_MINUTES
from report_app.models import Incident
from report_app.service import build_triage_summary


def test_incident_boundary_counts_as_breach_for_each_severity() -> None:
    for severity, threshold in ACK_SLA_MINUTES.items():
        incident = Incident(
            service=f"{severity}-svc",
            severity=severity,
            owner="Jules",
            minutes_open=threshold,
            acked=False,
        )
        assert incident.is_ack_sla_breached(ACK_SLA_MINUTES) is True


def test_service_summary_counts_boundary_incidents() -> None:
    summary = build_triage_summary(
        [
            {
                "service": "billing-api",
                "severity": "sev1",
                "owner": "Jules",
                "minutes_open": ACK_SLA_MINUTES["sev1"],
                "acked": False,
            },
            {
                "service": "queue-worker",
                "severity": "sev2",
                "owner": "Jules",
                "minutes_open": ACK_SLA_MINUTES["sev2"],
                "acked": False,
            },
        ]
    )

    assert summary["breached_count"] == 2
    assert summary["owner_load"] == [{"owner": "Jules", "count": 2, "breached": 2}]


def test_json_summary_counts_boundary_incident(monkeypatch) -> None:
    payload = cli_json(
        monkeypatch,
        [
            {
                "service": "billing-api",
                "severity": "sev1",
                "owner": "Jules",
                "minutes_open": ACK_SLA_MINUTES["sev1"],
                "acked": False,
            }
        ],
    )

    assert payload["summary"]["breached_count"] == 1
    assert payload["summary"]["highest_load_owner"] == {
        "owner": "Jules",
        "count": 1,
        "breached": 1,
    }


def test_markdown_owner_load_counts_boundary_incident(monkeypatch) -> None:
    output = cli_markdown(
        monkeypatch,
        [
            {
                "service": "billing-api",
                "severity": "sev1",
                "owner": "Jules",
                "minutes_open": ACK_SLA_MINUTES["sev1"],
                "acked": False,
            }
        ],
    )

    assert "1 active incident, 1 ack SLA breach." in output
    assert "Highest load owner: Jules (1 incidents, 1 breached)." in output
    assert "| Jules | 1" in output
