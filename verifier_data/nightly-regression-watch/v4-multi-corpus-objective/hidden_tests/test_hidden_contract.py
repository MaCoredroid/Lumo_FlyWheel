from __future__ import annotations

import os
from pathlib import Path

from contract_checks import hidden_scenarios, inspect_surfaces, load_gold, load_manifest


WORKSPACE = Path.cwd()
VARIANT_ID = os.environ["VARIANT_ID"]


def test_hidden_contract_scenarios() -> None:
    status = hidden_scenarios(WORKSPACE)
    assert status.required_milestone_blocks is True
    assert status.advisory_non_blocking is True
    assert status.latest_of_day_selected is True
    assert status.mixed_milestone_shapes_parse is True
    assert status.no_duplicate_same_day_lines is True


def test_generated_digest_and_surfaces_align() -> None:
    gold = load_gold(VARIANT_ID)
    manifest = load_manifest(VARIANT_ID)
    surfaces = inspect_surfaces(WORKSPACE, manifest, gold)
    assert surfaces.generated_digest_matches_output is True
    assert surfaces.automation_singleton_ok is True
    assert surfaces.automation_prompt_ok is True
    assert surfaces.runbook_wording_ok is True
