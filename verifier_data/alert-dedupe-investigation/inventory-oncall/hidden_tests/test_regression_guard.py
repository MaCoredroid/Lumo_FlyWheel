from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse, fingerprint


def test_blank_dedupe_hint_falls_back_to_normalized_title(
    make_event: Callable[..., dict[str, str]],
) -> None:
    plain = make_event(dedupe_hint="", title="Recount Drift")
    with_spacing = make_event(dedupe_hint="", title="Recount   Drift")

    assert fingerprint(plain) == fingerprint(with_spacing)


def test_collapse_keeps_earliest_display_title_when_events_arrive_out_of_order(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(title="Recount Drift (batch=417)", observed_at="2026-04-15T07:00:44Z"),
            make_event(title="Recount Drift", observed_at="2026-04-15T07:00:11Z"),
        ]
    )

    assert collapsed[0]["title"] == "Recount Drift"
