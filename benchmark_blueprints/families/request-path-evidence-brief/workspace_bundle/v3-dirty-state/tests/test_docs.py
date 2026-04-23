
from __future__ import annotations

import json
from pathlib import Path


def test_docs_reference_owner_path_surfaces() -> None:
    docs = Path("docs/cli.md").read_text(encoding="utf-8")
    data_flow = Path("docs/data_flow.md").read_text(encoding="utf-8")
    defaults = json.loads(Path("config/defaults.json").read_text(encoding="utf-8"))
    assert "--owner" in docs
    assert "owner_source" in docs
    assert "routing_key" in docs
    assert "sync_item" in data_flow
    assert defaults["owner"] in docs
