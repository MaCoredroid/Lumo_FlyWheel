
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from codex.config import load_config
from scripts.render_workflow_preview import render_preview


ROOT = Path(os.environ["AGENT_WS"]).resolve()
VARIANT = Path(__file__).resolve().parents[1].name
GOLD = json.loads((Path(__file__).resolve().parents[1] / "gold_repair.json").read_text())


def _temp_config(text: str) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
    tmp.write(text)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def test_actual_config_is_canonical() -> None:
    config_path = ROOT / "codex" / "config.toml"
    loaded = load_config(config_path)
    assert loaded == GOLD["expected_policy"]


def test_deprecated_input_still_parses() -> None:
    alias = GOLD["deprecated_alias"]
    config_path = _temp_config(
        f"[policy]\nsandbox = \"{alias['input']}\"\napproval_policy = \"on_request\"\n"
    )
    loaded = load_config(config_path)
    assert loaded["sandbox"] == alias["canonical"]


def test_preview_output_stays_canonical_for_alias_input() -> None:
    alias = GOLD["deprecated_alias"]
    config_path = _temp_config(
        f"[policy]\nsandbox = \"{alias['input']}\"\napproval_policy = \"on_request\"\n"
    )
    preview = render_preview(config_path)
    assert preview["sandbox"] == alias["canonical"]
    assert preview["approval_policy"] == GOLD["expected_policy"]["approval_policy"]
    assert {job["approval_policy"] for job in preview["jobs"]} == {GOLD["expected_policy"]["approval_policy"]}


def test_workflow_file_uses_one_canonical_approval_token_family() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "manual-review" not in workflow
    assert workflow.count("APPROVAL_POLICY: on_request") == 2
    assert workflow.count("SANDBOX_POLICY: workspace_write") == 2


def test_run_ci_does_not_normalize_only_the_visible_fixture() -> None:
    run_ci = (ROOT / "scripts" / "run_ci.py").read_text(encoding="utf-8")
    assert "visible_config.toml" not in run_ci
    assert 'replace("workspace_write", "workspace-write")' not in run_ci
    assert "CODEX_CONFIG_FIXTURE" not in run_ci


def test_release_note_documents_canonical_output_and_compatibility() -> None:
    note = (ROOT / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
    assert "workspace-write" in note
    assert "workspace_write" in note
    assert "on_request" in note
    assert "manual-review" not in note
    assert "fully breaking" not in note
    assert "make ci" in note


def test_archive_noise_stays_archived_when_present() -> None:
    archived = ROOT / "docs" / "archive" / "legacy-preview-contract.md"
    if archived.exists():
        active = (ROOT / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
        assert "Historical snapshot only" in archived.read_text(encoding="utf-8")
        assert "manual-review" not in active


def test_dirty_state_requires_rejecting_fixture_only_shortcut_when_present() -> None:
    dirty_state = ROOT / "repo_evidence" / "abandoned_helper_patch.md"
    if dirty_state.exists():
        note = (ROOT / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8").lower()
        assert (
            "visible fixture" in note
            or "happy-path fixture" in note
            or "fixture-only" in note
        )
        assert (
            "shortcut" in note
            or "abandoned" in note
            or "real config" in note
            or "real configs" in note
        )


def test_release_context_requires_preview_consumer_alignment_when_present() -> None:
    context = ROOT / "release_context" / "preview-consumer-contract.md"
    if context.exists():
        note = (ROOT / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
        assert "preview" in note.lower()
        assert "canonical-only" in note.lower()


def test_incident_context_requires_rollback_aware_compatibility_when_present() -> None:
    incident = ROOT / "incident_context" / "rollback_2026_04.md"
    if incident.exists():
        note = (ROOT / "docs" / "releases" / "policy-rename.md").read_text(encoding="utf-8")
        assert "rolled back" in note.lower()
        assert "workspace-write" in note
