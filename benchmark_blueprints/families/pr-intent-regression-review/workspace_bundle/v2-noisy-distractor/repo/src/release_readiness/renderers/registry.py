from __future__ import annotations

from release_readiness.renderers.json_renderer import render_json
from release_readiness.renderers.markdown_renderer import render_markdown

_RENDERERS = {
    "json": render_markdown,
    "markdown": render_markdown,
}


def get_renderer(name: str):
    if name not in _RENDERERS:
        raise KeyError(f"unknown renderer: {name}")
    return _RENDERERS[name]
