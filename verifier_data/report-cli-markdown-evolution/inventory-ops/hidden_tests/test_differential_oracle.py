"""Layer 3 — differential fixtures against the round-1 oracle."""
from __future__ import annotations

import json
from pathlib import Path

from conftest import direct_markdown


FIXTURES = json.loads(
    (Path(__file__).with_name("_differential_fixtures.json")).read_text(encoding="utf-8")
)


def test_round1_markdown_matches_differential_fixtures() -> None:
    for fixture in FIXTURES:
        assert direct_markdown(fixture["sections"]) == fixture["expected_markdown"], fixture["name"]
