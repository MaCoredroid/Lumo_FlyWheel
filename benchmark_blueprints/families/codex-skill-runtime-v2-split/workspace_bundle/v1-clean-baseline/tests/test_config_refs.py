from pathlib import Path


def test_config_references_only_canonical_skill():
    text = (Path(__file__).resolve().parents[1] / ".codex" / "config.toml").read_text()
    assert "skills/oncall_handoff/SKILL.md" in text
    assert "docs/oncall_handoff_monolith.md" not in text
