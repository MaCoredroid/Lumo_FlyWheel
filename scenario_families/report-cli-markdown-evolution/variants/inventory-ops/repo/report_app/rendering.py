from __future__ import annotations

import json

from report_app.markdown import render_inventory_markdown


def render_json(title: str, sections: list[dict[str, object]]) -> str:
    return json.dumps({"title": title, "sections": sections}, sort_keys=True)


def render_markdown(
    title: str,
    sections: list[dict[str, object]],
    owner_summary: dict[str, object],
) -> str:
    return render_inventory_markdown(title, sections, owner_summary)
