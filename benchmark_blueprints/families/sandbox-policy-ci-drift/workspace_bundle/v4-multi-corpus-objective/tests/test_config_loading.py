
from __future__ import annotations

from codex.config import load_config
from tests.helpers import fixture_path


def test_config_loading_returns_canonical_tokens() -> None:
    loaded = load_config(fixture_path())
    assert loaded == {
        "sandbox": "workspace_write",
        "approval_policy": "on_request",
    }
