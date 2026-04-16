from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RulePlan:
    slug: str
    route_bucket: str
    owner: str


def build_rule_plan(title: str, owner: str, region: str) -> RulePlan:
    slug = title.strip().lower().replace(" ", "-")
    return RulePlan(slug=slug, route_bucket=f"{region}:{owner}", owner=owner)
