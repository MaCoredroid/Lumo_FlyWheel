from __future__ import annotations

import json
import re

import pytest

import report_app.cli as cli_module
from report_app.cli import main
from report_app.service import TITLE


def test_json_output_is_still_supported() -> None:
    payload = json.loads(main([]))
    assert payload["title"] == TITLE
    assert payload["summary"]["breached_count"] == 2
    assert payload["sections"][0]["service"] == "billing-api"


def test_markdown_output_renders_runtime_queue_and_owner_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sections = [
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

    monkeypatch.setattr(cli_module, "build_sections", lambda: fake_sections)

    output = cli_module.main(["--format", "markdown"])

    assert output.startswith(f"# {TITLE}\n")
    assert "3 active incidents, 2 ack SLA breaches." in output
    assert re.search(r"\| Service\s+\| Severity\s+\| Owner\s+\| Acked\s+\| Minutes Open\s+\|", output)
    for section in fake_sections:
        acked = "yes" if section["acked"] else "no"
        assert re.search(
            rf"\| {re.escape(str(section['service']))}\s+\| {re.escape(str(section['severity']))}\s+\| {re.escape(str(section['owner']))}\s+\|",
            output,
        )
        assert re.search(rf"\| {acked}\s+\| {section['minutes_open']}\s+\|", output)
    assert "## Active Queue" in output
    assert "Highest load owner: Jules (2 incidents, 2 breached)." in output
    assert "## Owner Load" in output
    assert re.search(r"\| Owner\s+\| Active Incidents\s+\| Ack SLA Breaches\s+\|", output)
    jules_row = re.search(r"\| Jules\s+\| 2\s+\| 2\s+\|", output)
    ivy_row = re.search(r"\| Ivy\s+\| 1\s+\| 0\s+\|", output)
    assert jules_row
    assert ivy_row
    assert jules_row.start() < ivy_row.start()


def test_markdown_output_uses_breach_summary_from_runtime_incidents() -> None:
    output = cli_module.main(["--format", "markdown"])

    assert "4 active incidents, 2 ack SLA breaches." in output
    assert "Highest load owner: Jules (2 incidents, 2 breached)." in output
    assert re.search(r"\| Jules\s+\| 2\s+\| 2\s+\|", output)
    assert re.search(r"\| Ivy\s+\| 1\s+\| 0\s+\|", output)
