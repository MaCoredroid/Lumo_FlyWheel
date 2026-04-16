from __future__ import annotations

from investigate_app.dedupe import collapse
from investigate_app.parser import load_events


def test_distinct_windows_are_not_merged() -> None:
    collapsed = collapse(load_events())
    assert collapsed == [
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "window_start": "2026-04-15T07:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T07:00:11Z",
            "last_seen_at": "2026-04-15T07:00:54Z",
        },
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "window_start": "2026-04-15T08:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T08:00:09Z",
            "last_seen_at": "2026-04-15T08:00:09Z",
        },
    ]
