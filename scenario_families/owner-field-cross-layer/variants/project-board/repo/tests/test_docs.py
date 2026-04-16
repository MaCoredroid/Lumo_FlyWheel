from __future__ import annotations

import json
from pathlib import Path


def test_docs_and_defaults_call_out_owner_field() -> None:
    docs = Path("docs/cli.md").read_text(encoding="utf-8")
    defaults = json.loads(Path("config/defaults.json").read_text(encoding="utf-8"))
    assert "--owner" in docs
    assert "default owner" in docs.lower()
    assert defaults["owner"] in docs
