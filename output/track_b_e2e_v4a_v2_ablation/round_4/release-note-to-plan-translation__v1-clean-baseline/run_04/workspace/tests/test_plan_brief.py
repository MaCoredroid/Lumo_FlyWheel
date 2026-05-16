from __future__ import annotations

import json
import pathlib

import pytest

WS = pathlib.Path("/workspace") if pathlib.Path("/workspace").exists() else pathlib.Path(__file__).resolve().parents[1]
BRIEF_JSON = WS / "brief" / "manager_brief.json"
BRIEF_MD = WS / "brief" / "manager_brief.md"
VARIANT_MARKER = WS / ".scenario_variant"


def _load() -> dict:
    if not BRIEF_JSON.exists():
        pytest.fail(f"brief json missing at {BRIEF_JSON}")
    try:
        return json.loads(BRIEF_JSON.read_text())
    except json.JSONDecodeError as exc:
        pytest.fail(f"brief json invalid: {exc}")


def test_brief_files_exist():
    assert BRIEF_JSON.exists()
    assert BRIEF_MD.exists()
    assert BRIEF_MD.read_text().strip()


def test_variant_matches_marker():
    data = _load()
    assert data["variant_id"] == VARIANT_MARKER.read_text().strip()


def test_first_milestone_matches_rank_1():
    data = _load()
    ordered = data["ordered_steps"]
    assert ordered[0]["step_id"] == data["first_milestone_id"]


def test_ordered_steps_are_contiguous():
    data = _load()
    ordered = data["ordered_steps"]
    assert len(ordered) >= 3
    ranks = sorted(step["rank"] for step in ordered)
    assert ranks == list(range(1, len(ordered) + 1))
    assert len({step["step_id"] for step in ordered}) == len(ordered)


def test_dependency_notes_reference_known_steps():
    data = _load()
    step_ids = {step["step_id"] for step in data["ordered_steps"]}
    assert data["dependency_notes"]
    for note in data["dependency_notes"]:
        assert note["before"] in step_ids
        assert note["after"] in step_ids
        assert note["reason"].strip()


def test_primary_risk_and_assumptions_present():
    data = _load()
    primary_risk = data["primary_risk"]
    assert primary_risk["statement"].strip()
    assert len(primary_risk["evidence"]) >= 1
    assert len(primary_risk["mitigations"]) >= 2
    assert any(row["status"] == "missing" for row in data["assumption_ledger"])
