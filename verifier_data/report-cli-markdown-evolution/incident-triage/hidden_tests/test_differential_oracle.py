"""Layer 3 — differential fixtures for incident-triage."""
from __future__ import annotations

import json
from pathlib import Path

from conftest import direct_markdown


def test_round1_markdown_matches_differential_fixtures() -> None:
    fixture_path = Path(__file__).with_name("_differential_fixtures.json")
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))

    for fixture in payload["fixtures"]:
        assert direct_markdown(fixture["sections"]) == fixture["markdown"], fixture["name"]
