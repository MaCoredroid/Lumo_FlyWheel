from __future__ import annotations

import json
from pathlib import Path

import pytest

import sync_app


@pytest.fixture(scope="session")
def package_dir() -> Path:
    return Path(sync_app.__file__).resolve().parent


@pytest.fixture(scope="session")
def default_owner() -> str:
    return json.loads(Path("config/defaults.json").read_text(encoding="utf-8"))["owner"]
