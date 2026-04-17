from __future__ import annotations

from report_app.fixtures import ACK_SLA_MINUTES, RECORDS, REPORT_SLUG, TITLE
from report_app.models import Incident
from report_app.summaries import summarize_incidents


def build_sections() -> list[dict[str, object]]:
    return [Incident.from_mapping(item).as_dict() for item in RECORDS]


def build_triage_summary(sections: list[dict[str, object]]) -> dict[str, object]:
    summary = summarize_incidents(
        (Incident.from_mapping(item) for item in sections),
        ack_sla_minutes=ACK_SLA_MINUTES,
    )
    return summary.as_dict()
