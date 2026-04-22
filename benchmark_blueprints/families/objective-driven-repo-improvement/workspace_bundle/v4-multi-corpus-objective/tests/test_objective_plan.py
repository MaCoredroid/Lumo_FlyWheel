
from __future__ import annotations

import json
import pathlib

import pytest

WS = pathlib.Path("/workspace") if pathlib.Path("/workspace").exists() else pathlib.Path(__file__).resolve().parents[1]
BRIEF = WS / "brief" / "manager_brief.json"
VARIANT = (WS / ".scenario_variant").read_text().strip()
INTERVENTIONS = sorted(p.stem for p in (WS / "allowed_interventions").glob("P*.md"))


def load_brief() -> dict:
    if not BRIEF.exists():
        pytest.fail("brief/manager_brief.json not found")
    return json.loads(BRIEF.read_text())


def test_brief_exists():
    assert BRIEF.exists()
    assert BRIEF.read_text().strip()


def test_schema_and_variant():
    brief = load_brief()
    assert brief["schema_version"] == "cnb55.objective_delta_brief.v1"
    assert brief["variant_id"] == VARIANT


def test_ranking_shape_matches_interventions():
    brief = load_brief()
    ranking = brief["ranking"]
    assert len(ranking) == len(INTERVENTIONS)
    assert {entry["proposal_id"] for entry in ranking} == set(INTERVENTIONS)
    assert {entry["rank"] for entry in ranking} == set(range(1, len(INTERVENTIONS) + 1))


def test_accepted_is_rank_one():
    brief = load_brief()
    accepted = brief["accepted"]
    rank1 = next(entry["proposal_id"] for entry in brief["ranking"] if entry["rank"] == 1)
    assert accepted == rank1


def test_expected_delta_and_risk_present():
    brief = load_brief()
    delta = brief["expected_delta"]
    assert delta["metric"] == "objective_delta_index"
    assert delta["direction"] == "up"
    assert isinstance(delta["low"], int) and isinstance(delta["high"], int)
    assert 0 < delta["low"] <= delta["high"]
    risk = brief["primary_risk"]
    assert isinstance(risk["statement"], str) and risk["statement"].strip()
    assert len(risk["mitigations"]) >= 2


def test_assumption_ledger_has_missing_row():
    brief = load_brief()
    assert all(item["status"] in {"observed", "to_verify", "missing"} for item in brief["assumption_ledger"])
    assert any(item["status"] == "missing" for item in brief["assumption_ledger"])
