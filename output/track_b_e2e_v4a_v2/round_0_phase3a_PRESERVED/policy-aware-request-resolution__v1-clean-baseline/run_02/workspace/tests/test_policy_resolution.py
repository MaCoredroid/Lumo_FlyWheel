
from __future__ import annotations

import json
import pathlib

import pytest

WS = pathlib.Path("/workspace") if pathlib.Path("/workspace").exists() else pathlib.Path(__file__).resolve().parents[1]
BRIEF = WS / "brief" / "manager_brief.json"
PROPOSALS_DIR = WS / "proposals"


def _brief() -> dict:
    if not BRIEF.exists():
        pytest.fail(f"brief not found at {BRIEF}")
    return json.loads(BRIEF.read_text(encoding="utf-8"))


def _proposal_ids() -> list[str]:
    return sorted(path.stem for path in PROPOSALS_DIR.glob("P*.md"))


def test_brief_exists_and_nonempty() -> None:
    assert BRIEF.exists()
    assert BRIEF.read_text(encoding="utf-8").strip()


def test_schema_version() -> None:
    assert _brief()["schema_version"] == "cnb55.manager_brief.v2"


def test_ranking_length_matches_proposals() -> None:
    payload = _brief()
    ranking = payload.get("ranking")
    assert isinstance(ranking, list)
    assert len(ranking) == len(_proposal_ids())


def test_accepted_is_valid_proposal() -> None:
    assert _brief()["accepted"] in _proposal_ids()


def test_each_ranking_entry_has_required_fields() -> None:
    ranking = _brief()["ranking"]
    seen_ranks = set()
    seen_ids = set()
    for entry in ranking:
        for field in ("proposal_id", "rank", "summary", "citations", "constraint_tags"):
            assert field in entry
        assert entry["proposal_id"] in _proposal_ids()
        assert entry["proposal_id"] not in seen_ids
        assert isinstance(entry["rank"], int)
        assert entry["rank"] not in seen_ranks
        assert isinstance(entry["summary"], str) and entry["summary"].strip()
        assert isinstance(entry["citations"], list)
        assert isinstance(entry["constraint_tags"], list)
        seen_ids.add(entry["proposal_id"])
        seen_ranks.add(entry["rank"])


def test_assumption_ledger_has_missing_row() -> None:
    ledger = _brief().get("assumption_ledger")
    assert isinstance(ledger, list) and ledger
    assert any(row.get("status") == "missing" for row in ledger if isinstance(row, dict))
