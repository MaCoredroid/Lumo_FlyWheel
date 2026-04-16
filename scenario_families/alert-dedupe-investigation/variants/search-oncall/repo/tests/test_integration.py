from __future__ import annotations

from investigate_app.dedupe import collapse
from investigate_app.parser import load_events


def test_distinct_windows_are_not_merged() -> None:
    collapsed = collapse(load_events())
    keys = {(item["environment"], item["window_start"]) for item in collapsed}
    assert len(collapsed) == 2
    assert ("staging", "2026-04-15T09:00:00Z") in keys
    assert ("prod", "2026-04-15T09:00:00Z") in keys
