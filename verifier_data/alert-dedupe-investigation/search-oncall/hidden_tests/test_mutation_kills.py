from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import fingerprint


def test_fingerprint_prefers_dedupe_hint_over_title_noise(
    make_event: Callable[..., dict[str, str]],
) -> None:
    stable = make_event(dedupe_hint="search.shard-saturation.docs-primary", title="Shard Saturation")
    noisy = make_event(
        dedupe_hint="search.shard-saturation.docs-primary",
        title="Shard Saturation (shard=3)",
    )

    assert fingerprint(stable) == fingerprint(noisy)


def test_blank_dedupe_hint_falls_back_to_title_identity(
    make_event: Callable[..., dict[str, str]],
) -> None:
    saturation = make_event(dedupe_hint="", title="Shard Saturation")
    pressure = make_event(dedupe_hint="", title="Shard Pressure")

    assert fingerprint(saturation) != fingerprint(pressure)


def test_dedupe_hint_is_casefolded_before_fingerprint(
    make_event: Callable[..., dict[str, str]],
) -> None:
    lower = make_event(dedupe_hint="search.shard-saturation.docs-primary")
    upper = make_event(dedupe_hint="SEARCH.SHARD-SATURATION.DOCS-PRIMARY")

    assert fingerprint(lower) == fingerprint(upper)
