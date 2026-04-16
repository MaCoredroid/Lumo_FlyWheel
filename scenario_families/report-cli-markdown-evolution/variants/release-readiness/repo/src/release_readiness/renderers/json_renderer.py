"""JSON renderer for Report objects."""
from __future__ import annotations

import json

from release_readiness.core.model import Report


class JsonRenderer:
    """Renders a Report as deterministic, sorted JSON."""

    def render(self, report: Report) -> str:
        payload = {
            "title": report.title,
            "sections": [
                {"owner": s.owner, "label": s.label, "count": s.count}
                for s in report.sections
            ],
            "owner_totals": [
                {"owner": ot.owner, "total": ot.total}
                for ot in report.owner_totals
            ],
            "known_owners": list(report.known_owners),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
