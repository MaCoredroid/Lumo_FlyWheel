from __future__ import annotations

import json
from pathlib import Path

WS = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[1]
BRIEF = WS / "brief" / "schedule_brief.json"


def load_brief() -> dict:
    assert BRIEF.exists(), f"missing brief: {BRIEF}"
    return json.loads(BRIEF.read_text())


def backlog_ids() -> list[str]:
    return sorted(p.stem for p in (WS / "backlog").glob("B*.md"))


def test_brief_exists_and_parses():
    brief = load_brief()
    assert brief["schema_version"] == "cnb55.schedule_brief.v1"


def test_schedule_covers_all_items():
    brief = load_brief()
    scheduled = sorted(entry["item_id"] for entry in brief["schedule"])
    assert scheduled == backlog_ids()


def test_slots_are_contiguous():
    brief = load_brief()
    slots = [entry["slot"] for entry in brief["schedule"]]
    assert set(slots) == set(range(1, max(slots) + 1))


def test_risk_gate_present():
    brief = load_brief()
    assert brief["risk_gate"]["risky_item_id"]
    assert brief["risk_gate"]["must_follow"]


def test_assumption_ledger_has_missing_row():
    brief = load_brief()
    assert any(row["status"] == "missing" for row in brief["assumption_ledger"])
