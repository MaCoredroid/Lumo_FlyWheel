from __future__ import annotations

from pathlib import Path

import pytest

from release_readiness.config import Settings


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(Settings.model_fields):
        monkeypatch.delenv(f"RELEASE_READINESS_{key.upper()}", raising=False)
    s = Settings()
    assert s.title == "Release Readiness Report"
    assert s.source == "env"
    assert s.fs_path == Path("records.json")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELEASE_READINESS_TITLE", "Custom Title")
    monkeypatch.setenv("RELEASE_READINESS_SOURCE", "fs")
    monkeypatch.setenv("RELEASE_READINESS_FS_PATH", "/tmp/custom.json")
    s = Settings()
    assert s.title == "Custom Title"
    assert s.source == "fs"
    assert s.fs_path == Path("/tmp/custom.json")
