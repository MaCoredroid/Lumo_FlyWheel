from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse, fingerprint


def test_blank_dedupe_hint_falls_back_to_normalized_title(
    make_event: Callable[..., dict[str, str]],
) -> None:
    with_hint = make_event(dedupe_hint="", title="Queue Lag")
    with_spacing = make_event(dedupe_hint="", title="Queue   Lag")

    assert fingerprint(with_hint) == fingerprint(with_spacing)


def test_collapse_keeps_earliest_display_title_when_events_arrive_out_of_order(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(title="Queue Lag (processor=stripe)", observed_at="2026-04-15T10:00:41Z"),
            make_event(title="Queue Lag", observed_at="2026-04-15T10:00:27Z"),
        ]
    )

    assert collapsed[0]["title"] == "Queue Lag"
