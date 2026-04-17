from __future__ import annotations

import json
from pathlib import Path

from sync_app.service import sync_item


FIXTURES = json.loads((Path(__file__).with_name("_differential_fixtures.json")).read_text(encoding="utf-8"))


def test_visible_contract_matches_differential_fixtures() -> None:
    for fixture in FIXTURES["cases"]:
        record = fixture["record"]
        assert sync_item(record["name"], record["status"], owner=record["owner"]) == fixture["expected"]
