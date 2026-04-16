from __future__ import annotations

from report_app.formatting import format_queue_count, render_markdown_table


def test_format_queue_count_handles_one_and_many() -> None:
    assert format_queue_count(1) == "1 queued item"
    assert format_queue_count(4) == "4 queued items"


def test_render_markdown_table_aligns_headers_and_rows() -> None:
    table = render_markdown_table(
        ("Owner", "Total Items"),
        (("Mae", 5), ("Noah", 2)),
    )

    assert table.splitlines()[0].startswith("| Owner")
    assert "| Mae" in table
    assert "| Noah" in table
