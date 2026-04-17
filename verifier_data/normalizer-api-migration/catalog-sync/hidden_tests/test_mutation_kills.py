from __future__ import annotations

from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


def test_slug_strips_repeated_source_tokens_from_both_boundaries() -> None:
    plan = build_rule_plan(
        "catalog sync ap south Missing SKU catalog feed",
        "catalog",
        "ap-south",
    )

    assert plan.slug == "missing-sku"


def test_canonical_slug_preserves_digits_after_source_token_stripping() -> None:
    plan = build_rule_plan(
        "catalog feed / ap-south / SKU 42 Drift / catalog sync",
        "catalog",
        "ap-south",
    )

    assert plan.slug == "sku-42-drift"
    assert route_for(plan) == (
        "ap-south:catalog:sku-42-drift"
        "?dispatch=ap-south:catalog:sku-42-drift"
    )
