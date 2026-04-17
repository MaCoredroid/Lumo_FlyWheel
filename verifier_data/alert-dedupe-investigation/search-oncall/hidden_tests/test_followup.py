from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import collapse


def test_dedupe_hint_wins_over_shard_title_noise(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(
                dedupe_hint="search.shard-saturation.docs-primary",
                title="Shard Saturation",
            ),
            make_event(
                dedupe_hint="search.shard-saturation.docs-primary",
                title="Shard Saturation (shard=3)",
                observed_at="2026-04-15T09:00:44Z",
            ),
            make_event(
                dedupe_hint="search.shard-saturation.docs-primary",
                title="Shard Saturation [shard=7]",
                observed_at="2026-04-15T09:00:52Z",
            ),
        ]
    )

    assert collapsed == [
        {
            "environment": "staging",
            "service": "search",
            "title": "Shard Saturation",
            "search_cluster": "docs-primary",
            "dedupe_hint": "search.shard-saturation.docs-primary",
            "window_start": "2026-04-15T09:00:00Z",
            "occurrence_count": 3,
            "first_seen_at": "2026-04-15T09:00:07Z",
            "last_seen_at": "2026-04-15T09:00:52Z",
        }
    ]


def test_different_dedupe_hints_stay_split_even_when_display_title_matches(
    make_event: Callable[..., dict[str, str]],
) -> None:
    collapsed = collapse(
        [
            make_event(dedupe_hint="search.shard-saturation.docs-primary"),
            make_event(
                dedupe_hint="search.shard-pressure.docs-primary",
                observed_at="2026-04-15T09:00:44Z",
            ),
        ]
    )

    assert len(collapsed) == 2
