from __future__ import annotations

from pathlib import Path

from investigate_app.parser import load_events


def test_failure_log_mentions_the_window_collision() -> None:
    text = Path("logs/failure.log").read_text(encoding="utf-8")
    assert "window_start" in text
    assert "environment" in text
    assert '"environment": "Stage"' in text
    assert '"environment": "staging"' in text
    assert "2026-04-15T09:00:44Z" in text


def test_load_events_normalizes_aliases_and_preserves_observed_timestamp() -> None:
    events = load_events()

    assert events[0]["environment"] == "staging"
    assert events[0]["window_start"] == "2026-04-15T09:00:00Z"
    assert events[0]["observed_at"] == "2026-04-15T09:00:07Z"
    assert events[1]["window_start"] == "2026-04-15T09:00:00Z"
    assert events[1]["observed_at"] == "2026-04-15T09:00:44Z"
