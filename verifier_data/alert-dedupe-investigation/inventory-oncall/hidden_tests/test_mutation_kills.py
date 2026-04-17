from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import fingerprint


def test_fingerprint_prefers_dedupe_hint_over_title_noise(
    make_event: Callable[..., dict[str, str]],
) -> None:
    stable = make_event(dedupe_hint="inventory.recount-drift.cycle-counts", title="Recount Drift")
    noisy = make_event(
        dedupe_hint="inventory.recount-drift.cycle-counts",
        title="Recount Drift (batch=417)",
    )

    assert fingerprint(stable) == fingerprint(noisy)


def test_blank_dedupe_hint_falls_back_to_title_identity(
    make_event: Callable[..., dict[str, str]],
) -> None:
    recount = make_event(dedupe_hint="", title="Recount Drift")
    gap = make_event(dedupe_hint="", title="Stock Gap")

    assert fingerprint(recount) != fingerprint(gap)


def test_dedupe_hint_is_casefolded_before_fingerprint(
    make_event: Callable[..., dict[str, str]],
) -> None:
    lower = make_event(dedupe_hint="inventory.recount-drift.cycle-counts")
    upper = make_event(dedupe_hint="INVENTORY.RECOUNT-DRIFT.CYCLE-COUNTS")

    assert fingerprint(lower) == fingerprint(upper)
