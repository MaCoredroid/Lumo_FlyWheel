from __future__ import annotations

import json
from pathlib import Path

from investigate_app.dedupe import collapse


FIXTURES = json.loads((Path(__file__).with_name("_differential_fixtures.json")).read_text(encoding="utf-8"))


def test_payments_collapse_matches_differential_fixtures() -> None:
    for fixture in FIXTURES["cases"]:
        assert collapse(fixture["events"]) == fixture["expected"], fixture["name"]
