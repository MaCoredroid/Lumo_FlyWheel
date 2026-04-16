"""Layer 1 — example-based hidden tests for inventory-ops."""
from __future__ import annotations

import json
import re
from pathlib import Path

from report_app.service import TITLE

from conftest import cli_json, cli_markdown


def test_cli_accepts_markdown_flag(monkeypatch) -> None:
    output = cli_markdown(
        monkeypatch,
        [{"owner": "Mae", "label": "blocked-picks", "count": 4}],
    )

    assert output.startswith(f"# {TITLE}\n")


def test_markdown_uses_handoff_headers(monkeypatch) -> None:
    output = cli_markdown(
        monkeypatch,
        [{"owner": "Mae", "label": "blocked-picks", "count": 4}],
    )

    assert "## Sections" in output
    assert "## Owner Totals" in output
    assert "| Owner " in output
    assert "| Label " in output
    assert "| Count |" in output


def test_markdown_sections_table_preserves_every_runtime_row(monkeypatch) -> None:
    sections = [
        {"owner": "Jules", "label": "stale-pages", "count": 3},
        {"owner": "Ivy", "label": "escalations", "count": 1},
        {"owner": "Jules", "label": "queue-audits", "count": 2},
    ]
    output = cli_markdown(monkeypatch, sections)

    for section in sections:
        assert f"| {section['owner']} " in output
        assert f"| {section['label']} " in output
        assert re.search(rf"\|\s+{section['count']}\s+\|", output)


def test_markdown_owner_totals_sort_desc_then_name(monkeypatch) -> None:
    sections = [
        {"owner": "Bea", "label": "returns", "count": 2},
        {"owner": "Ava", "label": "recounts", "count": 2},
        {"owner": "Zoe", "label": "holds", "count": 1},
    ]
    output = cli_markdown(monkeypatch, sections)
    totals_section = output.split("## Owner Totals", 1)[1]
    owner_rows = [line for line in totals_section.splitlines() if line.startswith("| ")][2:]

    assert owner_rows[0].startswith("| Ava")
    assert owner_rows[1].startswith("| Bea")
    assert owner_rows[2].startswith("| Zoe")
    assert "Top owner: Ava with 2 queued items." in output


def test_usage_doc_mentions_markdown_command() -> None:
    usage = Path("docs/usage.md").read_text(encoding="utf-8").lower()

    assert "--format markdown" in usage
    assert "--format json" in usage
    assert "owner totals" in usage
    assert "top owner" in usage


def test_json_output_still_matches_runtime_sections(monkeypatch) -> None:
    sections = [
        {"owner": "Mae", "label": "blocked-picks", "count": 4},
        {"owner": "Noah", "label": "late-recounts", "count": 2},
    ]
    payload = json.loads(cli_json(monkeypatch, sections))

    assert payload == {"title": TITLE, "sections": sections}
