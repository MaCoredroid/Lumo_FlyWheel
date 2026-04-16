from __future__ import annotations

from report_app.formatting import format_queue_count, render_markdown_table


def render_inventory_markdown(
    title: str,
    sections: list[dict[str, object]],
    owner_summary: dict[str, object],
) -> str:
    section_rows = [
        [section["owner"], section["label"], section["count"]]
        for section in sections
    ]
    owner_totals = owner_summary["owner_totals"]
    owner_rows = [
        [entry["owner"], entry["count"]]
        for entry in owner_totals
    ]

    lines = [f"# {title}", ""]
    lines.append(
        f"{owner_summary['section_count']} sections covering "
        f"{format_queue_count(int(owner_summary['total_items']))}."
    )

    top_owner = owner_summary.get("top_owner")
    if isinstance(top_owner, dict):
        lines.append(
            f"Top owner: {top_owner['owner']} with "
            f"{format_queue_count(int(top_owner['count']))}."
        )

    lines.extend(
        [
            "",
            "## Sections",
            render_markdown_table(
                ("Owner", "Label", "Count"),
                section_rows,
            ),
            "",
            "## Owner Totals",
            render_markdown_table(
                ("Owner", "Total Items"),
                owner_rows,
            ),
        ]
    )
    return "\n".join(lines)
