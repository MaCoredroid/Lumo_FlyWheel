from __future__ import annotations

from investigate_app.dedupe import collapse
from investigate_app.parser import load_events


def test_distinct_windows_are_not_merged() -> None:
    collapsed = collapse(load_events())
    assert collapsed == [
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "window_start": "2026-04-15T10:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T10:00:27Z",
            "last_seen_at": "2026-04-15T10:00:41Z",
        },
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "window_start": "2026-04-15T10:05:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T10:05:12Z",
            "last_seen_at": "2026-04-15T10:05:12Z",
        },
    ]
