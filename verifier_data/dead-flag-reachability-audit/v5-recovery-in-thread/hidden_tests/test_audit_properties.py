from __future__ import annotations

import json
import os
from pathlib import Path

AGENT_WS = Path(os.environ.get("AGENT_WS", "/agent/workspace"))
RESULT_FILE = Path(os.environ.get("RESULT_FILE", "/results/verify_result.json"))
VERIFIER_DATA = Path(os.environ.get("VERIFIER_DATA", "/verifier_data"))
VARIANT_ID = os.environ.get("VARIANT_ID", "v1-clean-baseline")


def load_json(path: Path):
    return json.loads(path.read_text())


def test_expected_statuses():
    result = load_json(RESULT_FILE)
    gold = load_json(VERIFIER_DATA / VARIANT_ID / "gold_audit.json")
    by_flag = {row["flag"]: row["status"] for row in result.get("submitted_flags", [])}
    assert by_flag.get("ENABLE_SHADOW_PREVIEW") == gold["flags"]["ENABLE_SHADOW_PREVIEW"]["status"]
    assert by_flag.get("ENABLE_PREVIEW_V2") == gold["flags"]["ENABLE_PREVIEW_V2"]["status"]
    assert by_flag.get("PREVIEW_FORCE_LEGACY") == gold["flags"]["PREVIEW_FORCE_LEGACY"]["status"]


def test_alias_relationship():
    result = load_json(RESULT_FILE)
    submitted = {row["flag"]: row for row in result.get("submitted_flags", [])}
    entry = submitted["ENABLE_PREVIEW_V2"]
    assert entry["alias_of"] == "ENABLE_SHADOW_PREVIEW"


def test_dead_force_legacy_has_no_runtime_branch():
    result = load_json(RESULT_FILE)
    submitted = {row["flag"]: row for row in result.get("submitted_flags", [])}
    assert submitted["PREVIEW_FORCE_LEGACY"]["runtime_branch_symbol"] in (None, "")
