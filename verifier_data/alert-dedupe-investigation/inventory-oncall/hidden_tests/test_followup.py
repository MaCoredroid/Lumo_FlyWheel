from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse


def test_dedupe_hint_wins_over_batch_title_noise(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(dedupe_hint="inventory.recount-drift.cycle-counts", title="Recount Drift"),
            make_event(
                dedupe_hint="inventory.recount-drift.cycle-counts",
                title="Recount Drift (batch=417)",
                observed_at="2026-04-15T07:00:44Z",
            ),
            make_event(
                dedupe_hint="inventory.recount-drift.cycle-counts",
                title="Recount Drift [batch=418]",
                observed_at="2026-04-15T07:00:52Z",
            ),
        ]
    )

    assert collapsed == [
        {
            "environment": "prod",
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "cycle-counts",
            "dedupe_hint": "inventory.recount-drift.cycle-counts",
            "window_start": "2026-04-15T07:00:00Z",
            "occurrence_count": 3,
            "first_seen_at": "2026-04-15T07:00:11Z",
            "last_seen_at": "2026-04-15T07:00:52Z",
        }
    ]


def test_different_dedupe_hints_stay_split_even_when_display_title_matches(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(dedupe_hint="inventory.recount-drift.cycle-counts"),
            make_event(
                dedupe_hint="inventory.stock-gap.cycle-counts",
                observed_at="2026-04-15T07:00:44Z",
            ),
        ]
    )

    assert len(collapsed) == 2
