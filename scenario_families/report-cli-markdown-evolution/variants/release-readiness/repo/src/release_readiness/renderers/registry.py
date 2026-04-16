"""Renderer registry.

Discovers renderers via Python entry points declared in `pyproject.toml` under
the `release_readiness.renderers` group. To add a new renderer:

  1. Create a module under `release_readiness.renderers` that defines a class
     implementing the `Renderer` protocol.
  2. Add an entry under `[project.entry-points."release_readiness.renderers"]`
     in `pyproject.toml` mapping a format name to the class.
  3. Reinstall the package (`pip install -e .`) so setuptools re-reads entry
     points.

The CLI reads `available_formats()` to populate `--format` choices, so a
format not registered here will not be exposed to users.
"""
from __future__ import annotations

from importlib.metadata import entry_points
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from release_readiness.renderers.base import Renderer

_ENTRY_POINT_GROUP = "release_readiness.renderers"


class RendererRegistry:
    def __init__(self) -> None:
        self._classes: dict[str, type[Renderer]] = {}
        self._load_from_entry_points()

    def _load_from_entry_points(self) -> None:
        for ep in entry_points(group=_ENTRY_POINT_GROUP):
            cls = ep.load()
            self._classes[ep.name] = cls

    def available_formats(self) -> list[str]:
        return sorted(self._classes.keys())

    def get(self, fmt: str) -> Renderer:
        if fmt not in self._classes:
            available = ", ".join(self.available_formats()) or "<none>"
            raise ValueError(f"unknown format {fmt!r}; available: {available}")
        return self._classes[fmt]()

    def register(self, fmt: str, cls: type[Renderer]) -> None:
        """Manually register a renderer. Intended for tests."""
        self._classes[fmt] = cls


# Module-level singleton: built once per process.
_registry: RendererRegistry | None = None


def get_registry() -> RendererRegistry:
    global _registry
    if _registry is None:
        _registry = RendererRegistry()
    return _registry


def reset_registry_for_testing() -> None:
    """Force re-discovery of entry points. Test-only."""
    global _registry
    _registry = None
