from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse
from investigate_app.parser import load_events


def test_load_events_canonicalize_prod_alias_and_minute_window() -> None:
    events = load_events()

    assert [event["environment"] for event in events] == ["prod", "prod", "prod"]
    assert [event["window_start"] for event in events] == [
        "2026-04-15T07:00:00Z",
        "2026-04-15T07:00:00Z",
        "2026-04-15T08:00:00Z",
    ]


def test_load_events_preserve_inventory_scope_and_dedupe_hint() -> None:
    events = load_events()

    assert {event["inventory_scope"] for event in events} == {"cycle-counts"}
    assert {event["dedupe_hint"] for event in events} == {
        "inventory.recount-drift.cycle-counts",
    }


def test_collapsed_handoff_keeps_environments_windows_and_scopes_separate(
    make_event: Callable[..., dict[str, str]],
) -> None:
    events = [
        make_event(observed_at="2026-04-15T07:00:44Z"),
        make_event(environment="staging", observed_at="2026-04-15T07:00:19Z"),
        make_event(inventory_scope="safety-stock", observed_at="2026-04-15T07:00:38Z"),
        make_event(window_start="2026-04-15T07:01:00Z", observed_at="2026-04-15T07:01:04Z"),
    ]

    assert collapse(events) == [
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "cycle-counts",
            "dedupe_hint": "",
            "window_start": "2026-04-15T07:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T07:00:44Z",
            "last_seen_at": "2026-04-15T07:00:44Z",
        },
        {
            "environment": "staging",
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "cycle-counts",
            "dedupe_hint": "",
            "window_start": "2026-04-15T07:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T07:00:19Z",
            "last_seen_at": "2026-04-15T07:00:19Z",
        },
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "safety-stock",
            "dedupe_hint": "",
            "window_start": "2026-04-15T07:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T07:00:38Z",
            "last_seen_at": "2026-04-15T07:00:38Z",
        },
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "cycle-counts",
            "dedupe_hint": "",
            "window_start": "2026-04-15T07:01:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T07:01:04Z",
            "last_seen_at": "2026-04-15T07:01:04Z",
        },
    ]


def test_collapsed_handoff_tracks_occurrence_bounds_for_visible_log() -> None:
    assert collapse(load_events()) == [
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "cycle-counts",
            "dedupe_hint": "inventory.recount-drift.cycle-counts",
            "window_start": "2026-04-15T07:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T07:00:11Z",
            "last_seen_at": "2026-04-15T07:00:54Z",
        },
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "cycle-counts",
            "dedupe_hint": "inventory.recount-drift.cycle-counts",
            "window_start": "2026-04-15T08:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T08:00:09Z",
            "last_seen_at": "2026-04-15T08:00:09Z",
        },
    ]
