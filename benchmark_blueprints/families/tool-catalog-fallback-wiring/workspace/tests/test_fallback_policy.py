from __future__ import annotations

from tooling.catalog import select_tool


def test_docs_lookup_keeps_http_fetch() -> None:
    assert select_tool("docs.lookup", None) == "http_fetch"
