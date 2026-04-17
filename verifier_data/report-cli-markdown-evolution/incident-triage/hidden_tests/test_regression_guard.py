"""Layer 4 — regression guards for incident-triage."""
from __future__ import annotations

from conftest import cli_json, cli_markdown
from report_app.service import build_triage_summary


def test_json_shape_unchanged(monkeypatch) -> None:
    payload = cli_json(
        monkeypatch,
        [
            {
                "service": "billing-api",
                "severity": "sev1",
                "owner": "Jules",
                "minutes_open": 22,
                "acked": False,
            }
        ],
    )

    assert set(payload) == {"sections", "summary", "title"}
    assert set(payload["summary"]) == {"breached_count", "highest_load_owner", "incident_count", "owner_load"}


def test_acked_incident_never_counts_as_breach() -> None:
    summary = build_triage_summary(
        [
            {
                "service": "billing-api",
                "severity": "sev1",
                "owner": "Jules",
                "minutes_open": 240,
                "acked": True,
            }
        ]
    )

    assert summary["breached_count"] == 0
    assert summary["owner_load"] == [{"owner": "Jules", "count": 1, "breached": 0}]


def test_markdown_preserves_runtime_queue_order(monkeypatch) -> None:
    sections = [
        {
            "service": "zeta-api",
            "severity": "sev1",
            "owner": "Jules",
            "minutes_open": 22,
            "acked": False,
        },
        {
            "service": "alpha-worker",
            "severity": "sev2",
            "owner": "Ivy",
            "minutes_open": 18,
            "acked": True,
        },
    ]

    output = cli_markdown(monkeypatch, sections)
    assert output.index("zeta-api") < output.index("alpha-worker")


def test_owner_load_tie_breaks_owner_name_after_count_and_breaches() -> None:
    summary = build_triage_summary(
        [
            {
                "service": "svc-a",
                "severity": "sev3",
                "owner": "Bea",
                "minutes_open": 10,
                "acked": False,
            },
            {
                "service": "svc-b",
                "severity": "sev3",
                "owner": "Ava",
                "minutes_open": 10,
                "acked": False,
            },
        ]
    )

    assert summary["owner_load"] == [
        {"owner": "Ava", "count": 1, "breached": 0},
        {"owner": "Bea", "count": 1, "breached": 0},
    ]
