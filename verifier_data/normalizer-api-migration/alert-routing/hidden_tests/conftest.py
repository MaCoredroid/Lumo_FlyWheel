from __future__ import annotations

from pathlib import Path

import pytest

import norm_app
import norm_app.cli as cli_module


@pytest.fixture
def package_dir() -> Path:
    return Path(norm_app.__file__).resolve().parent


@pytest.fixture
def sample_record() -> dict[str, str]:
    return dict(cli_module.SAMPLE)
