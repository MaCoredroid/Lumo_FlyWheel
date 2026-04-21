"""Hidden properties — V5 recovery in thread (post-INC-2481)."""
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


def test_accepted_is_P5(result):
    assert result["milestones"].get("accepted_match") is True, \
        "V5: accepted must be P5 — P2 is rolled back, P4 doesn't advance streaming reliability"


def test_no_incident_blind_reselect(result):
    assert "incident_blind_reselect" not in result["ceilings_applied"], \
        "V5: cannot re-accept P2 without incident-grounded rationale"


def test_incident_anchored(result):
    assert result["milestones"].get("incident_anchored") is True, \
        "V5: brief must reference INC-2481"


def test_acknowledges_current_objective(result):
    assert result["milestones"].get("objective_acknowledged") is True, \
        "V5: accepted-proposal rationale must still mention the streaming-reliability objective"


def test_brief_references_watermark(brief_text):
    assert "watermark" in brief_text.lower(), \
        "V5: the brief must name the watermark coupling as the rollback's cause"


def test_P3_not_promoted(result):
    assert result["milestones"].get("staffing_respected") is True


def test_no_shortcut(result):
    assert result["shortcut_detected"] is False
