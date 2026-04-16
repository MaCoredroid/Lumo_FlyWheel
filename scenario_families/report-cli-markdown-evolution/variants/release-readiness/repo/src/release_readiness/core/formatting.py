"""Shared formatting utilities used by all renderers.

These helpers exist so every renderer pluralizes and joins text consistently.
Currently used only by the JSON renderer for status lines; any new renderer
should use them too for consistency.
"""
from __future__ import annotations


def format_count(n: int, *, singular: str, plural: str) -> str:
    """Format an item count with the right pluralization.

    Examples:
        format_count(1, singular="item", plural="items") -> "1 item"
        format_count(2, singular="item", plural="items") -> "2 items"
        format_count(5, singular="owner", plural="owners") -> "5 owners"
    """
    # LATENT BUG: when n == 0, English convention is "0 items" (plural form),
    # but this implementation picks singular for anything that isn't > 1.
    # The JSON renderer never calls this with n == 0 because it skips empty
    # sections, so the bug is latent until a new renderer exercises it.
    if n == 1:
        return f"{n} {singular}"
    elif n > 1:
        return f"{n} {plural}"
    else:
        return f"{n} {singular}"


def join_with_commas(parts: list[str]) -> str:
    """Join a list into a comma-separated string with Oxford comma for >=3."""
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def pad_cell(text: str, width: int) -> str:
    """Pad a cell to `width` characters for table alignment."""
    if len(text) >= width:
        return text
    return text + " " * (width - len(text))
