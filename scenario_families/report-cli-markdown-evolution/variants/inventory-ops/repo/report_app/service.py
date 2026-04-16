from __future__ import annotations

from report_app.fixtures import KNOWN_OWNERS, RECORDS, REPORT_SLUG, TITLE
from report_app.models import Section
from report_app.summaries import summarize_sections


def build_sections() -> list[dict[str, object]]:
    return [Section.from_mapping(item).as_dict() for item in RECORDS]


def build_owner_summary(
    sections: list[dict[str, object]],
    *,
    include_known_owners: bool = False,
) -> dict[str, object]:
    del include_known_owners
    summary = summarize_sections(Section.from_mapping(item) for item in sections)
    return summary.as_dict()
