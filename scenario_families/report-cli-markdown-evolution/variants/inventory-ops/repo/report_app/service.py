from __future__ import annotations

TITLE = "Inventory Ops Report"
REPORT_SLUG = "inventory-ops"
RECORDS = [{'label': 'blocked-picks', 'count': 4, 'owner': 'Mae'}, {'label': 'late-recounts', 'count': 2, 'owner': 'Noah'}]


def build_sections() -> list[dict[str, object]]:
    return [dict(item) for item in RECORDS]
