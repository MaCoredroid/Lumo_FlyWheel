from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_skill_doc_uses_live_validator() -> None:
    content = (ROOT / "skills" / "release_handoff" / "SKILL.md").read_text()
    assert "skill_verify.py" in content


def test_config_uses_live_validator() -> None:
    content = (ROOT / "codex" / "config.toml").read_text()
    assert "skill_verify.py" in content


def test_example_report_has_two_sections() -> None:
    content = (ROOT / "examples" / "expected_report.md").read_text()
    assert "# Summary" in content
    assert "# Action Items" in content


def test_usage_doc_uses_live_validator() -> None:
    content = (ROOT / "docs" / "usage.md").read_text()
    assert "skill_verify.py" in content
