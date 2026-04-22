
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT_FILE.read_text())


def test_brief_exists(result: dict) -> None:
    assert result["milestones"].get("brief_exists") is True


def test_json_parses(result: dict) -> None:
    assert result["milestones"].get("brief_parses") is True


def test_accepted_matches_gold(result: dict) -> None:
    assert result["milestones"].get("accepted_match") is True


def test_forbidden_path_demoted(result: dict) -> None:
    assert result["milestones"].get("forbidden_path_demoted") is True


def test_no_shortcut(result: dict) -> None:
    assert result["shortcut_detected"] is False
