from __future__ import annotations

from .normalization import normalize_label
from .service import build_filter_query


def render_filters(raw: str) -> str:
    parts = [piece for piece in raw.split(",")]
    normalized = [normalize_label(piece) for piece in parts]
    return build_filter_query(normalized)
