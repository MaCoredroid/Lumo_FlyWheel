from __future__ import annotations

import json
from pathlib import Path

from norm_app.assembler import compile_payload
from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


FIXTURES = json.loads((Path(__file__).with_name("_differential_fixtures.json")).read_text(encoding="utf-8"))


def test_round1_preview_contract_matches_differential_fixtures() -> None:
    for fixture in FIXTURES["cases"]:
        record = fixture["record"]
        expected = fixture["expected"]
        plan = build_rule_plan(record["title"], record["owner"], record["region"])

        assert compile_payload(plan) == {
            "slug": expected["slug"],
            "route_bucket": expected["route_bucket"],
            "dispatch_key": expected["dispatch_key"],
        }
        assert route_for(plan) == expected["route"]
