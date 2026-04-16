from __future__ import annotations

from pathlib import Path


def test_repo_no_longer_relies_on_removed_legacy_api() -> None:
    for path in Path("norm_app").glob("*.py"):
        assert "legacy_rules" not in path.read_text(encoding="utf-8")
