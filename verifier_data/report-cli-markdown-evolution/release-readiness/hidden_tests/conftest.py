"""Conftest for hidden tests.

Hidden tests run in the TRUSTED grading container against the agent's
workspace (mounted read-only per LLD-13 §6). They use the installed
release_readiness package from the agent's environment, so they must not
import anything from the agent's own tests/ directory.
"""
from __future__ import annotations

import pytest

from release_readiness.renderers.registry import reset_registry_for_testing


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()
