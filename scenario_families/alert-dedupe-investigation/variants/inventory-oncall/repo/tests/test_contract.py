from __future__ import annotations

from pathlib import Path

from investigate_app.parser import load_events


def test_failure_log_mentions_aliases_scope_metadata_and_second_level_window_values() -> None:
    text = Path("logs/failure.log").read_text(encoding="utf-8")
    assert '"environment": "Production"' in text
    assert '"environment": "prod"' in text
    assert '"inventory_scope": "Cycle-Counts"' in text
    assert '"dedupe_hint": "inventory.recount-drift.cycle-counts"' in text
    assert "2026-04-15T07:00:54Z" in text


def test_load_events_normalizes_aliases_window_bucket_and_preserves_scope_hint() -> None:
    events = load_events()

    assert events[0]["environment"] == "prod"
    assert events[0]["inventory_scope"] == "cycle-counts"
    assert events[0]["dedupe_hint"] == "inventory.recount-drift.cycle-counts"
    assert events[0]["window_start"] == "2026-04-15T07:00:00Z"
    assert events[0]["observed_at"] == "2026-04-15T07:00:11Z"
    assert events[1]["window_start"] == "2026-04-15T07:00:00Z"
    assert events[1]["observed_at"] == "2026-04-15T07:00:54Z"
    assert events[2]["window_start"] == "2026-04-15T08:00:00Z"
