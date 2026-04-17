"""Layer 1 — example-based coverage for incident-triage."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import cli_json, cli_markdown
from report_app.service import TITLE


def test_cli_accepts_markdown_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    output = cli_markdown(
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

    assert output.startswith(f"# {TITLE}\n")


def test_markdown_uses_triage_headings(monkeypatch: pytest.MonkeyPatch) -> None:
    output = cli_markdown(
        monkeypatch,
        [
            {
                "service": "billing-api",
                "severity": "sev1",
                "owner": "Jules",
                "minutes_open": 22,
                "acked": False,
            },
            {
                "service": "search-indexer",
                "severity": "sev2",
                "owner": "Ivy",
                "minutes_open": 18,
                "acked": True,
            },
        ],
    )

    assert "2 active incidents, 1 ack SLA breach." in output
    assert "## Active Queue" in output
    assert "## Owner Load" in output
    assert "Highest load owner: Ivy (1 incidents, 0 breached)." not in output


def test_markdown_queue_table_preserves_every_runtime_row(monkeypatch: pytest.MonkeyPatch) -> None:
    sections = [
        {
            "service": "billing-api",
            "severity": "sev1",
            "owner": "Jules",
            "minutes_open": 22,
            "acked": False,
        },
        {
            "service": "search-indexer",
            "severity": "sev2",
            "owner": "Ivy",
            "minutes_open": 18,
            "acked": True,
        },
        {
            "service": "checkout-web",
            "severity": "sev2",
            "owner": "Jules",
            "minutes_open": 41,
            "acked": False,
        },
    ]

    output = cli_markdown(monkeypatch, sections)

    for section in sections:
        acked = "yes" if section["acked"] else "no"
        assert section["service"] in output
        assert section["severity"] in output
        assert section["owner"] in output
        assert acked in output
        assert str(section["minutes_open"]) in output


def test_markdown_owner_load_sorts_by_count_then_breaches_then_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = cli_markdown(
        monkeypatch,
        [
            {
                "service": "billing-api",
                "severity": "sev1",
                "owner": "Alex",
                "minutes_open": 22,
                "acked": False,
            },
            {
                "service": "queue-worker",
                "severity": "sev3",
                "owner": "Alex",
                "minutes_open": 12,
                "acked": False,
            },
            {
                "service": "search-indexer",
                "severity": "sev2",
                "owner": "Bea",
                "minutes_open": 41,
                "acked": False,
            },
            {
                "service": "cache-router",
                "severity": "sev2",
                "owner": "Bea",
                "minutes_open": 18,
                "acked": True,
            },
        ],
    )

    alex_index = output.index("| Alex")
    bea_index = output.index("| Bea")
    assert alex_index < bea_index


def test_usage_doc_mentions_markdown_command() -> None:
    usage = Path("docs/usage.md").read_text(encoding="utf-8")

    assert "--format markdown" in usage
    assert "ack SLA breaches" in usage
    assert "Owner Load" in usage


def test_json_output_still_matches_runtime_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = cli_json(
        monkeypatch,
        [
            {
                "service": "billing-api",
                "severity": "sev1",
                "owner": "Jules",
                "minutes_open": 22,
                "acked": False,
            },
            {
                "service": "search-indexer",
                "severity": "sev2",
                "owner": "Ivy",
                "minutes_open": 18,
                "acked": True,
            },
        ],
    )

    assert payload["title"] == TITLE
    assert payload["summary"]["incident_count"] == 2
    assert payload["summary"]["breached_count"] == 1
    assert payload["sections"][0]["service"] == "billing-api"
