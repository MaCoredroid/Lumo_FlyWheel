from __future__ import annotations

import pytest

from release_readiness.renderers.json_renderer import JsonRenderer
from release_readiness.renderers.registry import RendererRegistry, get_registry


def test_json_renderer_is_discovered() -> None:
    registry = get_registry()
    assert "json" in registry.available_formats()
    renderer = registry.get("json")
    assert isinstance(renderer, JsonRenderer)


def test_unknown_format_raises_value_error() -> None:
    registry = get_registry()
    with pytest.raises(ValueError, match="unknown format"):
        registry.get("nope")


def test_manual_registration() -> None:
    class _FakeRenderer:
        def render(self, report) -> str:  # type: ignore[no-untyped-def]
            return "fake"

    registry = RendererRegistry()
    registry.register("fake", _FakeRenderer)
    assert "fake" in registry.available_formats()
