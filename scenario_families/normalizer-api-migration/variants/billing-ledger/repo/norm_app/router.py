from __future__ import annotations

from norm_app.legacy_rules import build_plan


def route_for(record: dict[str, str]) -> str:
    plan = build_plan(record["title"], record["owner"], record["region"])
    return f"{plan.route_bucket}:{plan.slug}"
