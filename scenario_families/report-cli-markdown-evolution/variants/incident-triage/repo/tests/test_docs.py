from __future__ import annotations

from pathlib import Path


def test_usage_doc_mentions_json_and_markdown_modes() -> None:
    usage = Path("docs/usage.md").read_text(encoding="utf-8")
    assert "--format markdown" in usage
    assert "--format json" in usage
    assert "active queue" in usage.lower()
    assert "ack sla breaches" in usage.lower()
    assert "owner load" in usage.lower()
