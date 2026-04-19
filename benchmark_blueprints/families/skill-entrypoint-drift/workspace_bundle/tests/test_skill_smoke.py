from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_skill_doc_uses_live_entrypoint_and_flag() -> None:
    skill_doc = (ROOT / "skills/ops-digest/SKILL.md").read_text()
    assert "tools/digest_runner.py" in skill_doc
    assert "--summary-length short" in skill_doc
    assert "scripts/build_digest.py" not in skill_doc
    assert "--brief" not in skill_doc


def test_example_uses_skill_relative_paths() -> None:
    example = (ROOT / "skills/ops-digest/examples/weekly_digest.md").read_text()
    assert "../../tools/digest_runner.py" in example
    assert "../../fixtures/incidents/sample_events.json" in example
