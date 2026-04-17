from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse
from investigate_app.parser import load_events


def test_load_events_canonicalize_stage_alias_and_minute_window() -> None:
    events = load_events()

    assert [event["environment"] for event in events] == ["staging", "staging", "prod"]
    assert [event["window_start"] for event in events] == [
        "2026-04-15T09:00:00Z",
        "2026-04-15T09:00:00Z",
        "2026-04-15T09:00:00Z",
    ]


def test_load_events_preserve_search_cluster_and_dedupe_hint() -> None:
    events = load_events()

    assert {event["search_cluster"] for event in events} == {"docs-primary"}
    assert {event["dedupe_hint"] for event in events} == {
        "search.shard-saturation.docs-primary",
    }


def test_collapsed_handoff_keeps_environments_windows_and_clusters_separate(
    make_event: Callable[..., dict[str, str]],
) -> None:
    events = [
        make_event(observed_at="2026-04-15T09:00:44Z"),
        make_event(search_cluster="docs-canary", observed_at="2026-04-15T09:00:38Z"),
        make_event(environment="prod", observed_at="2026-04-15T09:00:19Z"),
        make_event(window_start="2026-04-15T09:01:00Z", observed_at="2026-04-15T09:01:04Z"),
    ]

    assert collapse(events) == [
        {
            "environment": "staging",
            "service": "search",
            "title": "Shard Saturation",
            "search_cluster": "docs-canary",
            "dedupe_hint": "",
            "window_start": "2026-04-15T09:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T09:00:38Z",
            "last_seen_at": "2026-04-15T09:00:38Z",
        },
        {
            "environment": "prod",
            "service": "search",
            "title": "Shard Saturation",
            "search_cluster": "docs-primary",
            "dedupe_hint": "",
            "window_start": "2026-04-15T09:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T09:00:19Z",
            "last_seen_at": "2026-04-15T09:00:19Z",
        },
        {
            "environment": "staging",
            "service": "search",
            "title": "Shard Saturation",
            "search_cluster": "docs-primary",
            "dedupe_hint": "",
            "window_start": "2026-04-15T09:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T09:00:44Z",
            "last_seen_at": "2026-04-15T09:00:44Z",
        },
        {
            "environment": "staging",
            "service": "search",
            "title": "Shard Saturation",
            "search_cluster": "docs-primary",
            "dedupe_hint": "",
            "window_start": "2026-04-15T09:01:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T09:01:04Z",
            "last_seen_at": "2026-04-15T09:01:04Z",
        },
    ]


def test_collapsed_handoff_tracks_occurrence_bounds_for_visible_log() -> None:
    assert collapse(load_events()) == [
        {
            "environment": "staging",
            "service": "search",
            "title": "Shard Saturation",
            "search_cluster": "docs-primary",
            "dedupe_hint": "search.shard-saturation.docs-primary",
            "window_start": "2026-04-15T09:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T09:00:07Z",
            "last_seen_at": "2026-04-15T09:00:44Z",
        },
        {
            "environment": "prod",
            "service": "search",
            "title": "Shard Saturation",
            "search_cluster": "docs-primary",
            "dedupe_hint": "search.shard-saturation.docs-primary",
            "window_start": "2026-04-15T09:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T09:00:19Z",
            "last_seen_at": "2026-04-15T09:00:19Z",
        },
    ]
