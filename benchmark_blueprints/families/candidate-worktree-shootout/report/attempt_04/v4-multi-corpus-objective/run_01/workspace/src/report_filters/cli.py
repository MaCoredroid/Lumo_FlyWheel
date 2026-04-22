from __future__ import annotations

from .service import build_filter_query


def render_filters(raw: str) -> str:
    return build_filter_query(raw.split(","))
