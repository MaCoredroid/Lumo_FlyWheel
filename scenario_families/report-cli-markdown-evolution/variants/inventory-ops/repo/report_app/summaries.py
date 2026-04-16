from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from report_app.models import OwnerTotal, QueueSummary, Section


def summarize_sections(
    sections: Iterable[Section],
    *,
    known_owners: Sequence[str] | None = None,
    include_known_owners: bool = False,
) -> QueueSummary:
    copied_sections = [Section(item.owner, item.label, item.count) for item in sections]
    owner_counts: dict[str, int] = defaultdict(int)

    for section in copied_sections:
        owner_counts[section.owner] += section.count

    if include_known_owners:
        for owner in known_owners or ():
            owner_counts.setdefault(owner, 0)

    owner_totals = tuple(
        OwnerTotal(owner=owner, count=count)
        for owner, count in sorted(
            owner_counts.items(),
            key=lambda item: (-item[1], item[0].lower()),
        )
    )
    top_owner = owner_totals[0] if owner_totals else None

    return QueueSummary(
        section_count=len(copied_sections),
        total_items=sum(section.count for section in copied_sections),
        top_owner=top_owner,
        owner_totals=owner_totals,
    )
