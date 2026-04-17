from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse
from investigate_app.parser import load_events


def test_load_events_canonicalize_prod_alias_and_five_minute_window() -> None:
    events = load_events()

    assert [event["environment"] for event in events] == ["prod", "prod", "prod"]
    assert [event["window_start"] for event in events] == [
        "2026-04-15T10:00:00Z",
        "2026-04-15T10:00:00Z",
        "2026-04-15T10:05:00Z",
    ]


def test_load_events_preserve_payment_lane_and_dedupe_hint() -> None:
    events = load_events()

    assert {event["payment_lane"] for event in events} == {"card-payins"}
    assert {event["dedupe_hint"] for event in events} == {
        "payments.queue-lag.card-payins",
    }


def test_collapsed_handoff_keeps_windows_and_payment_lanes_separate(
    make_event: Callable[..., dict[str, str]],
) -> None:
    events = [
        make_event(observed_at="2026-04-15T10:00:41Z"),
        make_event(payment_lane="ach-payouts", observed_at="2026-04-15T10:00:38Z"),
        make_event(window_start="2026-04-15T10:05:00Z", observed_at="2026-04-15T10:05:04Z"),
    ]

    assert collapse(events) == [
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "payment_lane": "ach-payouts",
            "dedupe_hint": "",
            "window_start": "2026-04-15T10:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T10:00:38Z",
            "last_seen_at": "2026-04-15T10:00:38Z",
        },
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "payment_lane": "card-payins",
            "dedupe_hint": "",
            "window_start": "2026-04-15T10:00:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T10:00:41Z",
            "last_seen_at": "2026-04-15T10:00:41Z",
        },
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "payment_lane": "card-payins",
            "dedupe_hint": "",
            "window_start": "2026-04-15T10:05:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T10:05:04Z",
            "last_seen_at": "2026-04-15T10:05:04Z",
        },
    ]


def test_collapsed_handoff_tracks_occurrence_bounds_for_visible_log() -> None:
    assert collapse(load_events()) == [
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "payment_lane": "card-payins",
            "dedupe_hint": "payments.queue-lag.card-payins",
            "window_start": "2026-04-15T10:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T10:00:27Z",
            "last_seen_at": "2026-04-15T10:00:41Z",
        },
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "payment_lane": "card-payins",
            "dedupe_hint": "payments.queue-lag.card-payins",
            "window_start": "2026-04-15T10:05:00Z",
            "occurrence_count": 1,
            "first_seen_at": "2026-04-15T10:05:12Z",
            "last_seen_at": "2026-04-15T10:05:12Z",
        },
    ]
