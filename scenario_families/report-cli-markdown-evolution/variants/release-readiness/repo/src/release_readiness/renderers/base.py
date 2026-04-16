"""Renderer protocol.

All renderers implement this minimal interface. Renderers are instantiated
by the registry once per CLI invocation and called via `render(report)`.
"""
from __future__ import annotations

from typing import Protocol

from release_readiness.core.model import Report


class Renderer(Protocol):
    """A stateless renderer that turns a Report into a string."""

    def render(self, report: Report) -> str: ...
