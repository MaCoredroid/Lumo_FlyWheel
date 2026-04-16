from __future__ import annotations

import json


def render_json(title: str, sections: list[dict[str, object]]) -> str:
    return json.dumps({"title": title, "sections": sections}, sort_keys=True)
