from __future__ import annotations

import copy

import pytest

import norm_app.cli as cli_module


_DEFAULT_SAMPLE = copy.deepcopy(cli_module.SAMPLE)


@pytest.fixture(autouse=True)
def _restore_sample() -> None:
    cli_module.SAMPLE = copy.deepcopy(_DEFAULT_SAMPLE)
    yield
    cli_module.SAMPLE = copy.deepcopy(_DEFAULT_SAMPLE)
