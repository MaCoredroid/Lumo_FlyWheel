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
            "service": "inventory",
            "title": "Recount Drift",
            "inventory_scope": "cycle-counts",
            "dedupe_hint": "",
            "window_start": "2026-04-15T07:00:00Z",
            "observed_at": "2026-04-15T07:00:11Z",
        }
        event.update(overrides)
        return event

    return _make_event
