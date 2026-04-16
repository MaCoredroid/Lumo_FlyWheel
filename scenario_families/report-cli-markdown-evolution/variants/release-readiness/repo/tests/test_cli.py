from __future__ import annotations

import json

from report_app.cli import main
from report_app.service import TITLE


def test_json_output_is_still_supported() -> None:
    payload = json.loads(main([]))
    assert payload["title"] == TITLE
    assert payload["sections"][0]["owner"]


def test_markdown_output_renders_heading_and_owner_table() -> None:
    output = main(["--format", "markdown"])
    assert output.startswith(f"# {TITLE}")
    assert "| Owner |" in output
    assert "Sam" in output
