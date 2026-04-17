from __future__ import annotations

import pytest

from norm_app.assembler import compile_payload
from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


@pytest.mark.parametrize("title", [" API Gateway Latency ", "DB\tPrimary", "Cache\nNode"])
def test_slug_never_contains_whitespace(title: str) -> None:
    plan = build_rule_plan(title, "ops", "us-east")

    assert not any(char.isspace() for char in plan.slug)


def test_route_bucket_stays_region_owner_scoped() -> None:
    plan = build_rule_plan("API Gateway Latency", "ops", "us-east")

    assert plan.route_bucket == "us-east:ops"


def test_route_and_payload_share_dispatch_identity() -> None:
    plan = build_rule_plan("API Gateway Latency", "ops", "us-east")
    payload = compile_payload(plan)

    assert route_for(plan) == f"{plan.route_bucket}:{payload['slug']}?dispatch={payload['dispatch_key']}"
