from __future__ import annotations

from collections.abc import Iterable, Sequence


def format_queue_count(total_items: int) -> str:
    if total_items <= 1:
        return f"{total_items} queued item"
    return f"{total_items} queued items"


def render_markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
) -> str:
    normalized_headers = [str(header) for header in headers]
    normalized_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in normalized_headers]

    for row in normalized_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def _render_line(values: Sequence[str]) -> str:
        padded = [value.ljust(widths[index]) for index, value in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    lines = [_render_line(normalized_headers), separator]
    lines.extend(_render_line(row) for row in normalized_rows)
    return "\n".join(lines)
