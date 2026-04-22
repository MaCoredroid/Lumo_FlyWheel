from __future__ import annotations

import re

_SEPARATORS = re.compile(r"[-_/]+")
_WHITESPACE = re.compile(r"\s+")


def normalize_label(label: str) -> str:
    normalized = _SEPARATORS.sub(" ", label.strip().lower())
    normalized = _WHITESPACE.sub(" ", normalized)
    return normalized.strip()
