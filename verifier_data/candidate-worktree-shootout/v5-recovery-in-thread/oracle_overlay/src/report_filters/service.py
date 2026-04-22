from __future__ import annotations

from .normalization import normalize_label


def compile_filters(labels: list[str]) -> list[str]:
    compiled: list[str] = []
    for label in labels:
        cleaned = normalize_label(label)
        if cleaned:
            compiled.append(cleaned)
    return compiled


def build_filter_query(labels: list[str]) -> str:
    return ",".join(compile_filters(labels))
