from __future__ import annotations

import pytest

from norm_app.assembler import compile_payload
from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


@pytest.mark.parametrize(
    ("title", "expected_slug"),
    [
        ("[catalog sync][ap-south] Missing SKU", "missing-sku"),
        ("catalog feed :: ap south :: Missing SKU", "missing-sku"),
        ("ap-south catalog sync - Missing SKU", "missing-sku"),
    ],
)
def test_build_rule_plan_strips_boundary_source_wrappers(title: str, expected_slug: str) -> None:
    plan = build_rule_plan(title, "catalog", "ap-south")

    assert plan.slug == expected_slug


@pytest.mark.parametrize(
    ("title", "expected_dispatch_key"),
    [
        ("Missing SKU :: catalog sync ap-south", "ap-south:catalog:missing-sku"),
        ("Missing SKU [catalog][ap-south][feed]", "ap-south:catalog:missing-sku"),
        ("SKU 42 Drift catalog feed ap-south", "ap-south:catalog:sku-42-drift"),
    ],
)
def test_dispatch_key_stays_stable_when_title_echoes_source_labels(
    title: str,
    expected_dispatch_key: str,
) -> None:
    plan = build_rule_plan(title, "catalog", "ap-south")

    assert plan.dispatch_key == expected_dispatch_key
    assert compile_payload(plan)["dispatch_key"] == expected_dispatch_key
    assert route_for(plan).endswith(f"?dispatch={expected_dispatch_key}")
