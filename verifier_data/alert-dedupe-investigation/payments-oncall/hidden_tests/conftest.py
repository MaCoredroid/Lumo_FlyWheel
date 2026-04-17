from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path.cwd()


@pytest.fixture()
def make_event() -> Callable[..., dict[str, str]]:
    def _make_event(**overrides: str) -> dict[str, str]:
        event = {
            "environment": "prod",
            "service": "payments",
            "title": "Queue Lag",
            "payment_lane": "card-payins",
            "dedupe_hint": "",
            "window_start": "2026-04-15T10:00:00Z",
            "observed_at": "2026-04-15T10:00:27Z",
        }
        event.update(overrides)
        return event

    return _make_event
