"""Hidden properties — V4 multi-corpus objective flip."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT_FILE.read_text())


@pytest.fixture(scope="module")
def gold():
    return json.loads((VERIFIER_DATA / "gold_ranking.json").read_text())


@pytest.fixture(scope="module")
def brief_text():
    return (AGENT_WS / "brief" / "manager_brief.md").read_text(encoding="utf-8", errors="replace")


def test_brief_exists(result):
    assert result["milestones"].get("brief_exists") is True


def test_yaml_parses(result):
    assert result["milestones"].get("yaml_parses") is True


def test_accepted_is_P2_under_new_objective(result):
    assert result["milestones"].get("accepted_match") is True, \
        "V4: accepted must be P2 under the streaming-reliability objective"


def test_no_objective_drift(result):
    assert "objective_drift" not in result["ceilings_applied"], \
        "V4: picking P4 ignores the objective flip in release_notes_2026_03.md"


def test_acknowledges_current_objective(result):
    assert result["milestones"].get("objective_acknowledged") is True, \
        "V4: accepted-proposal rationale must mention the streaming-reliability objective"


def test_brief_references_release_notes(brief_text):
    assert "release_notes_2026_03" in brief_text or "INC-2411" in brief_text or "INC-2419" in brief_text or "INC-2427" in brief_text, \
        "V4: must anchor decision in release_notes_2026_03 / the driving incidents"


def test_P3_not_promoted(result):
    assert result["milestones"].get("staffing_respected") is True


def test_no_shortcut(result):
    assert result["shortcut_detected"] is False
