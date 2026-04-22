
from __future__ import annotations

from codex.config import load_config
from tests.helpers import fixture_path


def test_parser_accepts_visible_fixture() -> None:
    loaded = load_config(fixture_path())
    assert loaded["sandbox"] == "workspace_write"
    assert loaded["approval_policy"] == "on_request"
