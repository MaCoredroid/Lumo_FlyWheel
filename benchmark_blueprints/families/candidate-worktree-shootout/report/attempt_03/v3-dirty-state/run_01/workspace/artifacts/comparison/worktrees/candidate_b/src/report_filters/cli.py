from __future__ import annotations

from .service import build_filter_query


def render_filters(raw: str) -> str:
    parts = [piece for piece in raw.split(",")]
    return build_filter_query(parts)
