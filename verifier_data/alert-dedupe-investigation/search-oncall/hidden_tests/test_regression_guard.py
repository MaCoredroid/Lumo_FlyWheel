from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse, fingerprint


def test_blank_dedupe_hint_falls_back_to_normalized_title(
    make_event: Callable[..., dict[str, str]],
) -> None:
    plain = make_event(dedupe_hint="", title="Shard Saturation")
    with_spacing = make_event(dedupe_hint="", title="Shard   Saturation")

    assert fingerprint(plain) == fingerprint(with_spacing)


def test_collapse_keeps_earliest_display_title_when_events_arrive_out_of_order(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(title="Shard Saturation (shard=3)", observed_at="2026-04-15T09:00:44Z"),
            make_event(title="Shard Saturation", observed_at="2026-04-15T09:00:07Z"),
        ]
    )

    assert collapsed[0]["title"] == "Shard Saturation"
