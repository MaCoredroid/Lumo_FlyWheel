from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_points_to_real_skill() -> None:
    quickstart = (ROOT / "docs" / "plugin_quickstart.md").read_text()
    assert "skills/plugin_ops/SKILL.md" in quickstart
    assert (ROOT / "skills" / "plugin_ops" / "SKILL.md").exists()


def test_plugin_notes_use_stable_id() -> None:
    notes = (ROOT / "docs" / "plugin_notes.md").read_text()
    assert "plugin-ops-beta" not in notes
