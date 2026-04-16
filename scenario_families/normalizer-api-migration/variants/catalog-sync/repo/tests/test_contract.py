from __future__ import annotations

from pathlib import Path


def test_repo_no_longer_relies_on_removed_legacy_api() -> None:
    source = Path("norm_app/assembler.py").read_text(encoding="utf-8")
    assert "build_rule_plan" in source
    assert "legacy_rules" not in source
