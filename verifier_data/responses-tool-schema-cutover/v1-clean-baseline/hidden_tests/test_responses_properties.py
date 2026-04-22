from __future__ import annotations

import json
import os
import sys
from pathlib import Path

AGENT_WS = Path(os.environ["AGENT_WS"]).resolve()
VERIFIER_DATA = Path(os.environ["VERIFIER_DATA"]).resolve()
VARIANT_ID = os.environ["VARIANT_ID"]
sys.path.insert(0, str(AGENT_WS))

from gateway.adapter import normalize_events  # noqa: E402
from gateway.reducer import render_replay  # noqa: E402

GOLD = json.loads((VERIFIER_DATA / VARIANT_ID / "gold_responses.json").read_text())


def test_hidden_replay_cases_match_gold() -> None:
    for case in GOLD["hidden_cases"]:
        fixture = AGENT_WS / case["fixture"]
        assert render_replay(fixture) == case["expected_render"]
        items = normalize_events(fixture)
        tool_calls = [item for item in items if item["kind"] == "tool_call"]
        assert [item["call_id"] for item in tool_calls] == case["expected_call_ids"]


def test_config_and_docs_match_cutover_contract() -> None:
    config_text = (AGENT_WS / "codex" / "config.toml").read_text()
    docs_text = (AGENT_WS / "docs" / "migrations" / "responses-cutover.md").read_text().lower()
    for needle in GOLD["required_config_terms"]:
        assert needle in config_text
    for needle in GOLD["forbidden_config_terms"]:
        assert needle not in config_text
    for needle in GOLD["required_doc_phrases"]:
        assert needle.lower() in docs_text
