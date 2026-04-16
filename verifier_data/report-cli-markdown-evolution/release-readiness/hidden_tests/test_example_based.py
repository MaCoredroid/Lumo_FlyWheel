"""Layer 1 — example-based hidden tests.

These are the SWE-bench-Verified layer: specific inputs, specific expected
outputs. Necessary but not sufficient (mock-heavy solutions can pass here
alone; layers 2–4 catch them).
"""
from __future__ import annotations

import json
import os
import re

import pytest

from release_readiness.cli import main, render_report_from_sections
from release_readiness.core.model import Section
from release_readiness.renderers.registry import get_registry


def _set_env(monkeypatch: pytest.MonkeyPatch, records: list, known_owners: list | None = None) -> None:
    monkeypatch.setenv("RELEASE_READINESS_SOURCE", "env")
    monkeypatch.setenv("RELEASE_READINESS_RECORDS", json.dumps(records))
    if known_owners is not None:
        monkeypatch.setenv("RELEASE_READINESS_KNOWN_OWNERS", json.dumps(known_owners))
    else:
        monkeypatch.delenv("RELEASE_READINESS_KNOWN_OWNERS", raising=False)


# --- m1_cli_accepts_markdown -----------------------------------------------

def test_cli_accepts_markdown_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, [{"owner": "Sam", "label": "x", "count": 1}])
    # Should not raise SystemExit from argparse.
    out = main(["--format", "markdown"])
    assert isinstance(out, str)
    assert out.strip()


def test_registry_exposes_markdown() -> None:
    assert "markdown" in get_registry().available_formats()


def test_markdown_renderer_comes_from_registry() -> None:
    """Agent must register markdown via entry points, not hardcode in CLI."""
    renderer = get_registry().get("markdown")
    assert renderer is not None
    assert hasattr(renderer, "render")


# --- m2_renderer_produces_correct_markdown ---------------------------------

def test_renderer_emits_title_heading(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, [{"owner": "Sam", "label": "x", "count": 1}])
    out = main(["--format", "markdown"])
    assert out.startswith("# Release Readiness Report\n")


def test_renderer_emits_owner_totals_section(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, [{"owner": "Sam", "label": "x", "count": 2}])
    out = main(["--format", "markdown"])
    assert "## Owner Totals" in out


def test_renderer_emits_sections_section(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, [{"owner": "Sam", "label": "blocked-rollouts", "count": 2}])
    out = main(["--format", "markdown"])
    assert "## Sections" in out


def test_renderer_preserves_all_owners_in_sections_table(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {"owner": "Sam", "label": "blocked-rollouts", "count": 2},
        {"owner": "Rin", "label": "hotfixes", "count": 1},
        {"owner": "Sam", "label": "preflight-checks", "count": 4},
    ]
    _set_env(monkeypatch, records)
    out = main(["--format", "markdown"])
    for r in records:
        assert f"| {r['owner']}" in out or f"| {r['owner']} " in out
        assert r["label"] in out


def test_owner_totals_sum_correctly_in_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, [
        {"owner": "Sam", "label": "a", "count": 2},
        {"owner": "Sam", "label": "b", "count": 3},
        {"owner": "Rin", "label": "c", "count": 1},
    ])
    out = main(["--format", "markdown"])
    # Sam's total = 5, Rin's total = 1. Test: both appear in totals section.
    totals_section = out.split("## Owner Totals", 1)[1]
    assert "Sam" in totals_section
    assert "Rin" in totals_section
    # 5 must appear on the Sam row, 1 on the Rin row.
    sam_row = next(line for line in totals_section.splitlines() if "Sam" in line and "|" in line)
    rin_row = next(line for line in totals_section.splitlines() if "Rin" in line and "|" in line)
    assert "5" in sam_row
    assert re.search(r"\b1\b", rin_row)


# --- m3_docs_consistent ----------------------------------------------------

def test_docs_renderers_mentions_markdown() -> None:
    """docs/renderers.md should list markdown as a registered renderer."""
    import pathlib
    for candidate in [pathlib.Path("docs/renderers.md"), pathlib.Path("/workspace/docs/renderers.md")]:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8")
            assert "markdown" in text.lower()
            return
    pytest.skip("docs/renderers.md not found at expected paths")


def test_docs_usage_mentions_markdown() -> None:
    import pathlib
    for candidate in [pathlib.Path("docs/usage.md"), pathlib.Path("/workspace/docs/usage.md")]:
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8")
            assert "markdown" in text.lower()
            return
    pytest.skip("docs/usage.md not found at expected paths")


# --- JSON regression (PASS_TO_PASS: must not break existing behavior) -----

def test_json_output_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, [{"owner": "Sam", "label": "x", "count": 2}])
    payload = json.loads(main(["--format", "json"]))
    assert payload["title"] == "Release Readiness Report"
    assert {"owner": "Sam", "label": "x", "count": 2} in payload["sections"]
