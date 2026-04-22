from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTION_ITEMS = ROOT / "retro" / "action_items.json"
HELPER = ROOT / "repo" / "scripts" / "queue_drain_helper.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("queue_drain_helper", HELPER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_helper_emits_authoritative_verification_command():
    import json
    expected = json.loads(ACTION_ITEMS.read_text())["verification_command"]
    helper = load_helper()
    actual = helper.build_verification_command("atlas-a")
    assert actual.startswith(expected)


def test_helper_no_longer_uses_retired_primary_command():
    import json
    payload = json.loads(ACTION_ITEMS.read_text())
    helper = load_helper()
    actual = helper.build_verification_command("atlas-a")
    assert payload["retired_command"] not in actual
