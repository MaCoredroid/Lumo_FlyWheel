from __future__ import annotations

import json
import os
from pathlib import Path


RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))


def load():
    return json.loads(RESULT_FILE.read_text())


def test_schema_version():
    result = load()
    assert result["schema_version"] == "cnb55.verify_result.v3"


def test_primary_fix_milestones():
    result = load()
    assert result["milestones"]["M2_primary_fix"] is True


def test_live_path_milestones():
    result = load()
    assert result["milestones"]["live_request_correct"] is True
    assert result["milestones"]["live_echo_correct"] is True


def test_incident_ceiling_off(result):
    assert "incident_blind_recovery" not in result.get("ceilings_applied", [])

