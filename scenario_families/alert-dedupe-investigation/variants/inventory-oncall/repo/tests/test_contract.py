from __future__ import annotations

from pathlib import Path


def test_failure_log_mentions_the_window_collision() -> None:
    text = Path("logs/failure.log").read_text(encoding="utf-8")
    assert "window_start" in text
    assert "environment" in text
