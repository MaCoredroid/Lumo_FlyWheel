from __future__ import annotations

from release_readiness.renderers.registry import get_renderer


def export_report(report: dict, *, output: str = "json"):
    renderer = get_renderer(output)
    return renderer(report)
