"""
Phase 2 visible checks for `proposal-ranking-manager-judgment`.

These tests are SAFE to read; the agent is expected to pass them. The hidden
grader runs additional tests from `verifier_data/.../hidden_tests/` that are
NOT in this repo and cannot be observed by the agent.

DO NOT MODIFY THIS FILE. Its checksum is verified by the trusted-state gate.
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import Any, Dict, List

import pytest
import yaml  # type: ignore[import]

WS = pathlib.Path("/workspace") if pathlib.Path("/workspace").exists() else pathlib.Path(__file__).resolve().parents[1]

BRIEF = WS / "brief" / "manager_brief.md"
PROPOSALS_DIR = WS / "proposals"
VARIANT_MARKER = WS / ".scenario_variant"

BLOCK_RE = re.compile(
    r"<!--\s*CNB55-MANAGER-BRIEF v1\s*-->(.*?)<!--\s*/CNB55-MANAGER-BRIEF\s*-->",
    re.DOTALL,
)


def _read_brief() -> str:
    if not BRIEF.exists():
        pytest.fail(f"brief not found at {BRIEF}; agent did not produce manager_brief.md")
    return BRIEF.read_text(encoding="utf-8")


def _parse_block(text: str) -> Dict[str, Any]:
    m = BLOCK_RE.search(text)
    if not m:
        pytest.fail("machine-readable CNB55-MANAGER-BRIEF v1 block not found")
    try:
        data = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        pytest.fail(f"machine-readable block is not valid YAML: {e}")
    if not isinstance(data, dict):
        pytest.fail("machine-readable block must be a YAML mapping")
    return data


def _expected_proposal_count() -> int:
    return sum(1 for p in PROPOSALS_DIR.glob("P*.md"))


def _proposal_ids() -> List[str]:
    return sorted(p.stem for p in PROPOSALS_DIR.glob("P*.md"))


# --- tests --------------------------------------------------------------

def test_brief_exists_and_nonempty():
    text = _read_brief()
    assert text.strip(), "brief is empty"


def test_machine_block_parses():
    text = _read_brief()
    block = _parse_block(text)
    for key in ("ranking", "accepted", "primary_risk"):
        assert key in block, f"machine block missing key: {key}"


def test_ranking_length_matches_proposals():
    text = _read_brief()
    block = _parse_block(text)
    ranking = block.get("ranking")
    assert isinstance(ranking, list), "ranking must be a YAML list"
    assert len(ranking) == _expected_proposal_count(), (
        f"ranking length {len(ranking)} does not match number of proposals "
        f"({_expected_proposal_count()}) in proposals/"
    )


def test_accepted_is_valid_proposal():
    text = _read_brief()
    block = _parse_block(text)
    accepted = block.get("accepted")
    assert accepted in _proposal_ids(), (
        f"accepted={accepted!r} is not one of the proposals {_proposal_ids()}"
    )


def test_each_ranking_entry_has_required_fields():
    text = _read_brief()
    block = _parse_block(text)
    ranks_seen = set()
    ids_seen = set()
    for entry in block["ranking"]:
        assert isinstance(entry, dict), f"ranking entry not a mapping: {entry!r}"
        for field in ("proposal_id", "rank", "reason"):
            assert field in entry, f"entry missing field {field}: {entry!r}"
        assert entry["proposal_id"] in _proposal_ids(), f"unknown proposal: {entry['proposal_id']}"
        rank = entry["rank"]
        assert isinstance(rank, int), f"rank must be int, got {type(rank).__name__}"
        assert 1 <= rank <= _expected_proposal_count(), f"rank {rank} out of range"
        assert rank not in ranks_seen, f"rank {rank} appears more than once"
        assert entry["proposal_id"] not in ids_seen, f"proposal {entry['proposal_id']} appears more than once"
        ranks_seen.add(rank)
        ids_seen.add(entry["proposal_id"])
        reason = entry.get("reason", "")
        assert isinstance(reason, str) and reason.strip(), "reason must be non-empty string"


def test_assumption_ledger_present_with_missing_row():
    text = _read_brief().lower()
    assert "assumption ledger" in text, "brief missing 'Assumption ledger' section"
    # require at least one row flagged `missing`
    assert re.search(r"\bmissing\b", text), "ledger must include at least one row flagged `missing`"


def test_no_files_written_outside_brief():
    """Smoke check: we don't scan the whole workspace here (that is a
    trusted-state check in the grader), but we at least fail loudly if the
    agent wrote a file at a common wrong location."""
    forbidden = [
        WS / "proposals" / "P_agent_wrote_here.md",
        WS / "repo_evidence" / "agent_notes.md",
        WS / "manager_brief.md",  # wrong location
    ]
    for path in forbidden:
        assert not path.exists(), f"agent wrote to forbidden path: {path}"


def test_variant_marker_present():
    assert VARIANT_MARKER.exists(), ".scenario_variant marker missing"
    assert VARIANT_MARKER.read_text().strip(), "variant marker empty"
