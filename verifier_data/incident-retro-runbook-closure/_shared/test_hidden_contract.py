from __future__ import annotations

import os
from pathlib import Path

from contract_checks import inspect_surfaces, load_gold, validate_automation_schema, validate_followup_schema


WORKSPACE = Path(os.environ["AGENT_WS"])
VARIANT_ID = os.environ["VARIANT_ID"]


def test_hidden_authoritative_alignment():
    gold = load_gold(VARIANT_ID)
    surfaces = inspect_surfaces(WORKSPACE, gold)
    assert surfaces.helper_command_matches_authority
    assert surfaces.runbook_command_matches_authority
    assert surfaces.runbook_sequence_matches_authority
    assert surfaces.automation_prompt_matches_authority


def test_hidden_schema_and_note_contract():
    assert validate_automation_schema(WORKSPACE)
    assert validate_followup_schema(WORKSPACE)
