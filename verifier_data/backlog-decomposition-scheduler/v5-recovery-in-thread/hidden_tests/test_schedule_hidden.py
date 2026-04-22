from __future__ import annotations

import json
from pathlib import Path

WS = Path("/workspace") if Path("/workspace").exists() else Path(__file__).resolve().parents[2]


def test_citations_reference_workspace_files():
    brief = json.loads((WS / "brief" / "schedule_brief.json").read_text())
    for entry in brief["schedule"]:
        for rel in entry["citations"]:
            assert (WS / rel).exists(), rel
