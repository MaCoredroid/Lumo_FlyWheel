from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTION_ITEMS = json.loads((ROOT / "retro" / "action_items.json").read_text())
RUNBOOK = (ROOT / "repo" / "runbooks" / "queue_drain.md").read_text()


def test_runbook_contains_authoritative_sequence():
    for step in ACTION_ITEMS["verification_sequence"]:
        assert step in RUNBOOK


def test_runbook_does_not_leave_retired_command_as_an_operator_step():
    assert ACTION_ITEMS["retired_command"] not in RUNBOOK
