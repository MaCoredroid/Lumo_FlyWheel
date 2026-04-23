from pathlib import Path


def test_new_skill_exists_and_retires_monolith():
    root = Path(__file__).resolve().parents[1]
    skill_path = root / "skills" / "oncall_handoff" / "SKILL.md"
    assert skill_path.exists()
    assert "docs/oncall_handoff_monolith.md" not in skill_path.read_text()
