from __future__ import annotations

import pytest

from norm_app.assembler import compile_payload
from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


@pytest.mark.parametrize(
    ("title", "expected_slug"),
    [
        ("Missing SKU", "missing-sku"),
        (" Missing   SKU ", "missing-sku"),
        ("Missing\tSKU", "missing-sku"),
        ("Variant\nImage Drift", "variant-image-drift"),
    ],
)
def test_dispatch_key_title_component_matches_slug_for_whitespace_only_variations(
    title: str,
    expected_slug: str,
) -> None:
    plan = build_rule_plan(title, "catalog", "ap-south")

    assert plan.slug == expected_slug
    assert plan.dispatch_key.rsplit(":", 1)[1] == expected_slug


@pytest.mark.parametrize(
    "title",
    [
        "Missing SKU",
        " Variant   Image Drift ",
        "Catalog\tImage\nMismatch",
    ],
)
def test_route_suffix_matches_payload_dispatch_key_for_whitespace_only_variations(title: str) -> None:
    plan = build_rule_plan(title, "catalog", "ap-south")
    payload = compile_payload(plan)
    route = route_for(plan)

    assert route.endswith(f"?dispatch={payload['dispatch_key']}")
    assert f":{payload['slug']}?dispatch=" in route


def test_ruleplan_instances_are_hashable_for_preview_caching() -> None:
    plan_a = build_rule_plan("Missing SKU", "catalog", "ap-south")
    plan_b = build_rule_plan("Missing SKU", "catalog", "ap-south")

    assert {plan_a, plan_b} == {plan_a}
