from __future__ import annotations

import json
import pathlib

WS = pathlib.Path("/workspace") if pathlib.Path("/workspace").exists() else pathlib.Path(__file__).resolve().parents[1]


def test_canonical_json_exists():
    target = WS / "artifacts" / "workflow_map.json"
    assert target.exists(), "artifacts/workflow_map.json missing"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cnb55.workflow_mapping.v1"
    assert payload["variant_id"] == (WS / ".scenario_variant").read_text(encoding="utf-8").strip()


def test_rendered_artifacts_exist():
    for rel in [
        "artifacts/SKILL.md",
        "artifacts/codex_triage.toml",
        "artifacts/automation_proposal.md",
        "artifacts/mapping_note.md",
    ]:
        path = WS / rel
        assert path.exists(), f"missing {rel}"
        assert path.read_text(encoding="utf-8").strip(), f"{rel} is empty"


def test_toml_parses():
    text = (WS / "artifacts" / "codex_triage.toml").read_text(encoding="utf-8")
    assert 'entrypoint = "' in text
    assert 'schedule_literal = "' in text


def test_mapping_note_has_four_decisions():
    payload = json.loads((WS / "artifacts" / "workflow_map.json").read_text(encoding="utf-8"))
    artifacts = {item["artifact"] for item in payload["mapping_note"]["decisions"]}
    assert artifacts == {"skill", "toml", "automation", "mapping_note"}


def test_rejected_candidates_present():
    payload = json.loads((WS / "artifacts" / "workflow_map.json").read_text(encoding="utf-8"))
    assert payload["rejected_candidates"], "need at least one rejected candidate"


def test_automation_proposal_splits_task_and_schedule():
    text = (WS / "artifacts" / "automation_proposal.md").read_text(encoding="utf-8")
    assert "## Task" in text
    assert "## Schedule" in text
