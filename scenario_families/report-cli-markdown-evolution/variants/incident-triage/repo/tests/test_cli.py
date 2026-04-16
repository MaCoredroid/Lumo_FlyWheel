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
    assert payload["sections"][0]["owner"]


def test_markdown_output_renders_runtime_sections_and_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sections = [
        {"label": "stale-pages", "count": 3, "owner": "Jules"},
        {"label": "escalations", "count": 1, "owner": "Ivy"},
        {"label": "queue-audits", "count": 2, "owner": "Jules"},
    ]

    monkeypatch.setattr(cli_module, "build_sections", lambda: fake_sections)

    output = cli_module.main(["--format", "markdown"])

    assert output.startswith(f"# {TITLE}\n")
    assert "3 sections covering 6 queued items" in output
    assert "| Owner | Label | Count |" in output
    for section in fake_sections:
        assert f"| {section['owner']} | {section['label']} | {section['count']} |" in output
    assert "## Owner Summary" in output
    assert "Top owner: Jules with 5 queued items" in output
    assert "## Owner Totals" in output
    assert "| Owner | Total Items |" in output
    jules_row = re.search(r"\| Jules \| 5 \|", output)
    ivy_row = re.search(r"\| Ivy \| 1 \|", output)
    assert jules_row
    assert ivy_row
    assert jules_row.start() < ivy_row.start()


def test_markdown_output_includes_owner_rollup_for_runtime_records() -> None:
    output = cli_module.main(["--format", "markdown"])

    assert "3 sections covering 6 queued items" in output
    assert "Top owner: Jules with 5 queued items" in output
    assert "| Jules | 5 |" in output
    assert "| Ivy | 1 |" in output
