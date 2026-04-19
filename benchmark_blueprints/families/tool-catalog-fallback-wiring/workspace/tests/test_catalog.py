from __future__ import annotations

from tooling.catalog import select_tool


def test_preferred_browser_tool_wins_when_present() -> None:
    assert select_tool("browser.read", "chrome-devtools") == "chrome-devtools"


def test_missing_preferred_browser_uses_browser_snapshot() -> None:
    assert select_tool("browser.read", "chrome-devtool") == "browser_snapshot"
