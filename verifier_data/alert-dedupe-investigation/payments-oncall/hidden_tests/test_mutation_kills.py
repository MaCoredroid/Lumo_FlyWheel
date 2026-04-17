from __future__ import annotations

from collections.abc import Callable

from investigate_app.dedupe import fingerprint


def test_fingerprint_prefers_dedupe_hint_over_title_noise(
    make_event: Callable[..., dict[str, str]],
) -> None:
    stable = make_event(dedupe_hint="payments.queue-lag.card-payins", title="Queue Lag")
    noisy = make_event(
        dedupe_hint="payments.queue-lag.card-payins",
        title="Queue Lag (processor=stripe)",
    )

    assert fingerprint(stable) == fingerprint(noisy)


def test_blank_dedupe_hint_falls_back_to_title_identity(
    make_event: Callable[..., dict[str, str]],
) -> None:
    queue_lag = make_event(dedupe_hint="", title="Queue Lag")
    depth_drift = make_event(dedupe_hint="", title="Queue Depth Drift")

    assert fingerprint(queue_lag) != fingerprint(depth_drift)


def test_dedupe_hint_is_casefolded_before_fingerprint(
    make_event: Callable[..., dict[str, str]],
) -> None:
    lower = make_event(dedupe_hint="payments.queue-lag.card-payins")
    upper = make_event(dedupe_hint="PAYMENTS.QUEUE-LAG.CARD-PAYINS")

    assert fingerprint(lower) == fingerprint(upper)
