from __future__ import annotations

from pathlib import Path

from investigate_app.parser import load_events


def test_failure_log_mentions_aliases_lane_metadata_and_second_level_window_values() -> None:
    text = Path("logs/failure.log").read_text(encoding="utf-8")
    assert '"environment": "Production"' in text
    assert '"environment": "prod"' in text
    assert '"payment_lane": "card-payins"' in text
    assert '"dedupe_hint": "payments.queue-lag.card-payins"' in text
    assert "2026-04-15T10:00:27Z" in text


def test_load_events_normalizes_aliases_window_bucket_and_preserves_lane_hint() -> None:
    events = load_events()

    assert events[0]["environment"] == "prod"
    assert events[0]["payment_lane"] == "card-payins"
    assert events[0]["dedupe_hint"] == "payments.queue-lag.card-payins"
    assert events[0]["window_start"] == "2026-04-15T10:00:00Z"
    assert events[0]["observed_at"] == "2026-04-15T10:00:27Z"
    assert events[1]["window_start"] == "2026-04-15T10:00:00Z"
    assert events[1]["observed_at"] == "2026-04-15T10:00:41Z"
    assert events[2]["window_start"] == "2026-04-15T10:05:00Z"
