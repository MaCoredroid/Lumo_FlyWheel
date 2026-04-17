from __future__ import annotations

from collections.abc import Iterable


def format_incident_count(count: int) -> str:
    noun = "active incident" if count == 1 else "active incidents"
    return f"{count} {noun}"


def format_breach_count(count: int) -> str:
    noun = "ack SLA breach" if count == 1 else "ack SLA breaches"
    return f"{count} {noun}"


def format_ack_state(acked: bool) -> str:
    return "yes" if acked else "no"


def render_markdown_table(headers: tuple[str, ...], rows: Iterable[Iterable[object]]) -> str:
    rendered_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render_row(cells: Iterable[str]) -> str:
        padded = [cell.ljust(widths[index]) for index, cell in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    header_row = render_row(headers)
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body_rows = [render_row(row) for row in rendered_rows]
    return "\n".join([header_row, divider, *body_rows])
