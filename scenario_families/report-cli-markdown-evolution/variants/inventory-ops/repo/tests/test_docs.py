from __future__ import annotations

from pathlib import Path


def test_usage_doc_mentions_json_and_markdown_modes() -> None:
    usage = Path("docs/usage.md").read_text(encoding="utf-8")
    assert "--format markdown" in usage
    assert "--format json" in usage
    assert "owner table" in usage.lower()
    assert "owner totals" in usage.lower()
    assert "queued items" in usage.lower()
    assert "sorted by queued items" in usage.lower()
    assert "top owner" in usage.lower()
