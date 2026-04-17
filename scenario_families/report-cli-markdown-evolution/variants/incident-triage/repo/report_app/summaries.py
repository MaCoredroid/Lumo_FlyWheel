from __future__ import annotations

from collections.abc import Iterable, Mapping

from report_app.models import Incident, OwnerLoad, TriageSummary


def summarize_incidents(
    incidents: Iterable[Incident],
    *,
    ack_sla_minutes: Mapping[str, int],
) -> TriageSummary:
    incident_list = list(incidents)
    owner_totals: dict[str, dict[str, int | str]] = {}
    breached_count = 0

    for incident in incident_list:
        breached = incident.is_ack_sla_breached(ack_sla_minutes)
        breached_count += int(breached)
        row = owner_totals.setdefault(
            incident.owner,
            {
                "owner": incident.owner,
                "count": 0,
                "breached": 0,
            },
        )
        row["count"] = int(row["count"]) + 1
        row["breached"] = int(row["breached"]) + int(breached)

    owner_load = tuple(
        OwnerLoad(
            owner=str(row["owner"]),
            count=int(row["count"]),
            breached=int(row["breached"]),
        )
        for row in sorted(
            owner_totals.values(),
            key=lambda item: (-int(item["count"]), -int(item["breached"]), str(item["owner"]).lower()),
        )
    )
    highest_load_owner = owner_load[0].as_dict() if owner_load else None
    return TriageSummary(
        incident_count=len(incident_list),
        breached_count=breached_count,
        highest_load_owner=highest_load_owner,
        owner_load=owner_load,
    )
