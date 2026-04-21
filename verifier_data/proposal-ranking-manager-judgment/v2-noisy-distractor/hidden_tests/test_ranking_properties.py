"""Hidden properties — V2 noisy distractor."""
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


def test_accepted_is_P4(result):
    assert result["milestones"].get("accepted_match") is True, "accepted must be P4 under the p95 objective"


def test_stale_jan_bench_not_silently_used(brief_text, result):
    # Either the brief must flag the stale bench, or the ignored_stale_perf ceiling must not have fired.
    mentions_stale = any(w in brief_text.lower() for w in ("stale", "out of date", "outdated", "superseded"))
    ceiling_fired = "ignored_stale_perf" in result["ceilings_applied"]
    assert mentions_stale or not ceiling_fired, \
        "V2: cannot cite jan_bench numbers without flagging them as stale"


def test_P3_not_promoted(result):
    assert result["milestones"].get("staffing_respected") is True


def test_no_shortcut(result):
    assert result["shortcut_detected"] is False
