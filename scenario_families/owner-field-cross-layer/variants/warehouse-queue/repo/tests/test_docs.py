from __future__ import annotations

from pathlib import Path


def test_docs_and_defaults_call_out_owner_field() -> None:
    docs = Path("docs/cli.md").read_text(encoding="utf-8")
    defaults = Path("config/defaults.json").read_text(encoding="utf-8")
    assert "--owner" in docs
    assert '"owner"' in defaults
