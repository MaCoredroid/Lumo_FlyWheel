from __future__ import annotations

from pathlib import Path


def test_usage_doc_mentions_markdown_mode() -> None:
    usage = Path("docs/usage.md").read_text(encoding="utf-8")
    assert "--format markdown" in usage
    assert "Markdown" in usage
