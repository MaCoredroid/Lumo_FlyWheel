from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_router_policy_doc_no_longer_allows_first_listed_fallback() -> None:
    content = (ROOT / "docs" / "router_policy.md").read_text().lower()
    assert "first listed tool" not in content
