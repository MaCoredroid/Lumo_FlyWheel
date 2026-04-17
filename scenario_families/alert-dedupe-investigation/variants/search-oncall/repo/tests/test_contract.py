from __future__ import annotations

from pathlib import Path

from investigate_app.parser import load_events


def test_failure_log_mentions_aliases_cluster_metadata_and_second_level_window_values() -> None:
    text = Path("logs/failure.log").read_text(encoding="utf-8")
    assert '"environment": "Stage"' in text
    assert '"environment": "staging"' in text
    assert '"search_cluster": "docs-primary"' in text
    assert '"dedupe_hint": "search.shard-saturation.docs-primary"' in text
    assert "2026-04-15T09:00:44Z" in text


def test_load_events_normalizes_aliases_window_bucket_and_preserves_cluster_hint() -> None:
    events = load_events()

    assert events[0]["environment"] == "staging"
    assert events[0]["search_cluster"] == "docs-primary"
    assert events[0]["dedupe_hint"] == "search.shard-saturation.docs-primary"
    assert events[0]["window_start"] == "2026-04-15T09:00:00Z"
    assert events[0]["observed_at"] == "2026-04-15T09:00:07Z"
    assert events[1]["window_start"] == "2026-04-15T09:00:00Z"
    assert events[1]["observed_at"] == "2026-04-15T09:00:44Z"
    assert events[2]["window_start"] == "2026-04-15T09:00:00Z"
