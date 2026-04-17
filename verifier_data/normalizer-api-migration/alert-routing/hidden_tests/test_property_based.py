from __future__ import annotations

import pytest

from norm_app.assembler import compile_payload
from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


@pytest.mark.parametrize(
    ("title", "expected_slug"),
    [
        ("API Gateway Latency", "api-gateway-latency"),
        (" API   Gateway   Latency ", "api-gateway-latency"),
        ("API\tGateway\nLatency", "api-gateway-latency"),
        ("DB\tPrimary\nDown", "db-primary-down"),
    ],
)
def test_dispatch_key_title_component_matches_slug_for_whitespace_variations(
    title: str,
    expected_slug: str,
) -> None:
    plan = build_rule_plan(title, "ops", "us-east")

    assert plan.slug == expected_slug
    assert plan.dispatch_key.rsplit(":", 1)[1] == expected_slug


@pytest.mark.parametrize(
    "title",
    [
        "API Gateway Latency",
        " DB   Primary Down ",
        "Cache\tNode\nCPU High",
    ],
)
def test_route_suffix_matches_payload_dispatch_key_for_whitespace_variations(title: str) -> None:
    plan = build_rule_plan(title, "ops", "us-east")
    payload = compile_payload(plan)
    route = route_for(plan)

    assert route.endswith(f"?dispatch={payload['dispatch_key']}")
    assert f":{payload['slug']}?dispatch=" in route


def test_ruleplan_instances_are_hashable_for_preview_caching() -> None:
    plan_a = build_rule_plan("API Gateway Latency", "ops", "us-east")
    plan_b = build_rule_plan("API Gateway Latency", "ops", "us-east")

    assert {plan_a, plan_b} == {plan_a}
