from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "skill_router.toml"
DOCS_PATH = ROOT / "docs" / "skill_routing.md"
AUDIT_PATH = ROOT / "docs" / "routing_audit.md"


def test_config_examples_use_multi_field_skill_schema() -> None:
    config = tomllib.loads(CONFIG_PATH.read_text())
    skills = config["skills"]

    assert any(skill["name"] == "deploy_check" for skill in skills)
    for skill in skills:
        assert "trigger" not in skill
        assert "triggers" in skill
        assert isinstance(skill["triggers"], list)
        assert "negative_triggers" in skill
        assert "required_inputs" in skill


def test_docs_example_uses_multi_field_skill_schema() -> None:
    docs = DOCS_PATH.read_text()
    match = re.search(r"```toml\n(.*?)\n```", docs, re.DOTALL)

    assert match is not None
    parsed = tomllib.loads(match.group(1))
    skills = parsed["skills"]

    assert any(skill["name"] == "release_handoff" for skill in skills)
    for skill in skills:
        assert "trigger" not in skill
        assert "triggers" in skill
        assert "negative_triggers" in skill
        assert "required_inputs" in skill


def test_routing_audit_covers_positive_and_suppressed_cases() -> None:
    audit = AUDIT_PATH.read_text().lower()

    assert "positive match" in audit
    assert "suppressed match" in audit
