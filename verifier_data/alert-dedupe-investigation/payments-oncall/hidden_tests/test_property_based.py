from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse, fingerprint


def test_fingerprint_changes_when_environment_changes(
    make_event: Callable[..., dict[str, str]],
) -> None:
    prod = make_event()
    staging = make_event(environment="staging")

    assert fingerprint(prod) != fingerprint(staging)


def test_fingerprint_changes_when_window_changes(
    make_event: Callable[..., dict[str, str]],
) -> None:
    early = make_event(window_start="2026-04-15T10:00:00Z")
    later = make_event(window_start="2026-04-15T10:05:00Z")

    assert fingerprint(early) != fingerprint(later)


def test_fingerprint_changes_when_payment_lane_changes_without_hint(
    make_event: Callable[..., dict[str, str]],
) -> None:
    card = make_event(payment_lane="card-payins")
    ach = make_event(payment_lane="ach-payouts")

    assert fingerprint(card) != fingerprint(ach)


def test_whitespace_only_title_noise_does_not_split_a_plain_incident(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(title="Queue Lag"),
            make_event(title="Queue   Lag", observed_at="2026-04-15T10:00:41Z"),
        ]
    )

    assert collapsed == [
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "payment_lane": "card-payins",
            "dedupe_hint": "",
            "window_start": "2026-04-15T10:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T10:00:27Z",
            "last_seen_at": "2026-04-15T10:00:41Z",
        }
    ]


def test_collapse_is_order_independent_for_observed_bounds(
    make_event: Callable[..., dict[str, str]],
) -> None:
    reversed_events = [
        make_event(observed_at="2026-04-15T10:00:41Z"),
        make_event(observed_at="2026-04-15T10:00:27Z"),
    ]

    assert collapse(reversed_events) == [
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "payment_lane": "card-payins",
            "dedupe_hint": "",
            "window_start": "2026-04-15T10:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T10:00:27Z",
            "last_seen_at": "2026-04-15T10:00:41Z",
        }
    ]
