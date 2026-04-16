"""Layer 5 — follow-up (round 2).

Injected only after round 1 hidden tests pass. These tests assert the
round-2 requirement from followup/brief.md: zero-count owners must render
with correct pluralization everywhere, not just in markdown.

An agent who patched format_count locally in markdown_renderer.py will
pass round-1 tests but fail these because the JSON renderer (and any other
future consumer of format_count) still emits "0 item".
"""
from __future__ import annotations

import json

import pytest

from release_readiness.cli import main
from release_readiness.core.formatting import format_count


def _run(fmt: str, monkeypatch: pytest.MonkeyPatch, records: list, known_owners: list) -> str:
    monkeypatch.setenv("RELEASE_READINESS_SOURCE", "env")
    monkeypatch.setenv("RELEASE_READINESS_RECORDS", json.dumps(records))
    monkeypatch.setenv("RELEASE_READINESS_KNOWN_OWNERS", json.dumps(known_owners))
    return main(["--format", fmt])


# --- The direct unit-level test: format_count upstream --------------------

def test_format_count_zero_uses_plural() -> None:
    """format_count is the shared utility; it must return '0 items' not '0 item'."""
    assert format_count(0, singular="item", plural="items") == "0 items"


def test_format_count_one_uses_singular() -> None:
    assert format_count(1, singular="item", plural="items") == "1 item"


def test_format_count_many_uses_plural() -> None:
    assert format_count(5, singular="item", plural="items") == "5 items"


# --- Behavior-level tests: zero-count owners in markdown -----------------

def test_markdown_shows_zero_count_owner_with_plural(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [{"owner": "Sam", "label": "x", "count": 2}]
    known_owners = ["Sam", "Alex"]
    out = _run("markdown", monkeypatch, records, known_owners)
    totals = out.split("## Owner Totals", 1)[-1]
    alex_row = next(line for line in totals.splitlines() if "Alex" in line)
    assert "0 items" in alex_row
    assert "0 item " not in alex_row and "0 item|" not in alex_row.replace(" ", "")


def test_markdown_all_zero_owners_show_plural(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [{"owner": "Sam", "label": "x", "count": 1}]
    known_owners = ["Sam", "Alex", "Bee", "Cas"]
    out = _run("markdown", monkeypatch, records, known_owners)
    totals = out.split("## Owner Totals", 1)[-1]
    for owner in ["Alex", "Bee", "Cas"]:
        row = next(line for line in totals.splitlines() if owner in line)
        assert "0 items" in row, f"owner {owner} row missing '0 items': {row!r}"


# --- The trap-catching test: local patching of markdown_renderer fails ---

def test_format_count_is_the_fix_site_not_markdown_renderer() -> None:
    """If the agent patched format_count locally in markdown_renderer.py
    (e.g. as a private helper) rather than fixing it upstream, this test
    fails because we directly call the upstream function."""
    assert format_count(0, singular="item", plural="items") == "0 items"
    assert format_count(0, singular="owner", plural="owners") == "0 owners"
    assert format_count(0, singular="rollout", plural="rollouts") == "0 rollouts"
