from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse


def test_dedupe_hint_wins_over_processor_title_noise(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(dedupe_hint="payments.queue-lag.card-payins", title="Queue Lag"),
            make_event(
                dedupe_hint="payments.queue-lag.card-payins",
                title="Queue Lag (processor=stripe)",
                observed_at="2026-04-15T10:00:41Z",
            ),
            make_event(
                dedupe_hint="payments.queue-lag.card-payins",
                title="Queue Lag [processor=adyen]",
                observed_at="2026-04-15T10:00:52Z",
            ),
        ]
    )

    assert collapsed == [
        {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "payment_lane": "card-payins",
            "dedupe_hint": "payments.queue-lag.card-payins",
            "window_start": "2026-04-15T10:00:00Z",
            "occurrence_count": 3,
            "first_seen_at": "2026-04-15T10:00:27Z",
            "last_seen_at": "2026-04-15T10:00:52Z",
        }
    ]


def test_different_dedupe_hints_stay_split_even_when_display_title_matches(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(dedupe_hint="payments.queue-lag.card-payins"),
            make_event(
                dedupe_hint="payments.queue-depth.card-payins",
                observed_at="2026-04-15T10:00:41Z",
            ),
        ]
    )

    assert len(collapsed) == 2
