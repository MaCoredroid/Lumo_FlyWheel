import pytest

from release_readiness.renderers.registry import reset_registry_for_testing


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_registry_for_testing()
    yield
    reset_registry_for_testing()
