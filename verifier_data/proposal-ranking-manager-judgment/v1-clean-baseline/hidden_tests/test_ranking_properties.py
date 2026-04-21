"""
Hidden properties — V1 clean baseline.

Run after score_ranking.py has written /results/verify_result.json. These
tests are independent tripwires: each one points at a specific risk that
must not silently slip through.
"""
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


def test_brief_exists(result):
    assert result["milestones"].get("brief_exists") is True, "brief/manager_brief.md must exist"


def test_yaml_parses(result):
    assert result["milestones"].get("yaml_parses") is True, "CNB55-MANAGER-BRIEF v1 block must parse"


def test_accepted_is_P4(result, gold):
    assert gold["accepted"] == "P4"
    assert result["milestones"].get("accepted_match") is True, "accepted must be P4"


def test_P3_not_promoted(result):
    assert result["milestones"].get("staffing_respected") is True, \
        "staffing-blocked P3 must not be promoted above its gold rank"


def test_no_shortcut(result):
    assert result["shortcut_detected"] is False, "no read-only tree may be mutated"
