from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_plugin_and_marketplace_ids_match() -> None:
    plugin = json.loads((ROOT / ".codex-plugin" / "plugin.json").read_text())
    marketplace = json.loads((ROOT / ".agents" / "plugins" / "marketplace.json").read_text())
    assert plugin["id"] == marketplace[0]["id"]
