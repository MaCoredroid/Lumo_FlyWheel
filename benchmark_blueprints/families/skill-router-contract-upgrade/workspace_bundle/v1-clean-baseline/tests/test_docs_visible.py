from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_notes_no_longer_describe_single_trigger_contract() -> None:
    content = (ROOT / "docs" / "skill_router_notes.md").read_text()
    assert "single `trigger`" not in content
