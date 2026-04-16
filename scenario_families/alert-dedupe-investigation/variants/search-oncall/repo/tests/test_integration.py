from __future__ import annotations

from investigate_app.dedupe import collapse
from investigate_app.parser import load_events


def test_distinct_windows_are_not_merged() -> None:
    collapsed = collapse(load_events())
    assert collapsed == [
        {
            "environment": "staging",
            "service": "search",
            "title": "Shard Saturation",
            "window_start": "2026-04-15T09:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T09:00:07Z",
            "last_seen_at": "2026-04-15T09:00:44Z",
        },
        {
            "environment": "prod",
            "service": "search",
            "title": "Shard Saturation",
            "window_start": "2026-04-15T09:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T09:00:19Z",
            "last_seen_at": "2026-04-15T09:00:19Z",
        },
    ]
