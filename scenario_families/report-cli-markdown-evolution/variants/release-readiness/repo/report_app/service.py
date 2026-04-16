from __future__ import annotations

TITLE = "Release Readiness Report"
REPORT_SLUG = "release-readiness"
RECORDS = [
    {"label": "blocked-rollouts", "count": 2, "owner": "Sam"},
    {"label": "hotfixes", "count": 1, "owner": "Rin"},
    {"label": "preflight-checks", "count": 4, "owner": "Sam"},
]


def build_sections() -> list[dict[str, object]]:
    return [dict(item) for item in RECORDS]
