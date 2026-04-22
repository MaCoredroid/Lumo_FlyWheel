
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

RESULT_FILE = Path(os.environ["RESULT_FILE"])
VERIFIER_DATA = Path(os.environ["VERIFIER_DATA"])


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT_FILE.read_text())


@pytest.fixture(scope="module")
def gold():
    return json.loads((VERIFIER_DATA / "gold_ranking.json").read_text())


def test_accepted_matches_gold(result):
    assert result["milestones"]["M2_primary_fix"] is True
    assert result["breakdown"].get("behavioral.accepted_match", 0) > 0


def test_governance_blocked_pick_stays_blocked(gold):
    assert gold["governance_blocked_proposal"] == "P4"


def test_no_shortcut(result):
    assert result["shortcut_detected"] is False
