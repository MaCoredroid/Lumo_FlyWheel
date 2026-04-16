from __future__ import annotations

from norm_app.legacy_rules import build_plan


def compile_payload(record: dict[str, str]) -> dict[str, str]:
    plan = build_plan(record["title"], record["owner"], record["region"])
    return {"slug": plan.slug, "route_bucket": plan.route_bucket}
