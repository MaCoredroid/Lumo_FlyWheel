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
    early = make_event(window_start="2026-04-15T07:00:00Z")
    later = make_event(window_start="2026-04-15T07:01:00Z")

    assert fingerprint(early) != fingerprint(later)


def test_fingerprint_changes_when_inventory_scope_changes_without_hint(
    make_event: Callable[..., dict[str, str]],
) -> None:
    cycle = make_event(inventory_scope="cycle-counts")
    safety = make_event(inventory_scope="safety-stock")

    assert fingerprint(cycle) != fingerprint(safety)


def test_whitespace_only_title_noise_does_not_split_a_plain_incident(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(title="Recount Drift"),
            make_event(title="Recount   Drift", observed_at="2026-04-15T07:00:44Z"),
        ]
    )

    assert collapsed == [
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "cycle-counts",
            "dedupe_hint": "",
            "window_start": "2026-04-15T07:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T07:00:11Z",
            "last_seen_at": "2026-04-15T07:00:44Z",
        }
    ]


def test_collapse_is_order_independent_for_observed_bounds(
    make_event: Callable[..., dict[str, str]],
) -> None:
    reversed_events = [
        make_event(observed_at="2026-04-15T07:00:44Z"),
        make_event(observed_at="2026-04-15T07:00:11Z"),
    ]

    assert collapse(reversed_events) == [
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "cycle-counts",
            "dedupe_hint": "",
            "window_start": "2026-04-15T07:00:00Z",
            "occurrence_count": 2,
            "first_seen_at": "2026-04-15T07:00:11Z",
            "last_seen_at": "2026-04-15T07:00:44Z",
        }
    ]
