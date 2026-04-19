from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_restores_deprecated_name_field() -> None:
    plugin = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text())

    assert plugin["id"] == "drive-brief"
    assert plugin["name"] == "drive-brief"
    assert (
        plugin["compatibility"]["deprecated_fields"]["name"]["value"]
        == "drive-brief"
    )
    assert (
        plugin["compatibility"]["deprecated_fields"]["name"]["remove_after"]
        == "2026.05"
    )
    assert "legacy_field_removed" not in plugin["compatibility"]
