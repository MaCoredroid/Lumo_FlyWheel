from __future__ import annotations

TITLE = "Incident Triage Report"
REPORT_SLUG = "incident-triage"
RECORDS = [{'label': 'stale-pages', 'count': 3, 'owner': 'Jules'}, {'label': 'escalations', 'count': 1, 'owner': 'Ivy'}]


def build_sections() -> list[dict[str, object]]:
    return [dict(item) for item in RECORDS]
