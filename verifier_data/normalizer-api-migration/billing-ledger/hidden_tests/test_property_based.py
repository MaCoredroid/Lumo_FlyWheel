from __future__ import annotations

import pytest

from norm_app.assembler import compile_payload
from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


@pytest.mark.parametrize(
    ("title", "expected_slug"),
    [
        ("Invoice Retry", "invoice-retry"),
        (" Invoice   Retry ", "invoice-retry"),
        ("Invoice\tRetry", "invoice-retry"),
        ("Invoice\nRetry", "invoice-retry"),
        ("Ledger\tClose\nRequest", "ledger-close-request"),
    ],
)
def test_dispatch_key_title_component_matches_slug_for_whitespace_only_variations(
    title: str,
    expected_slug: str,
) -> None:
    plan = build_rule_plan(title, "finance", "eu-west")

    assert plan.slug == expected_slug
    assert plan.dispatch_key.rsplit(":", 1)[1] == expected_slug


@pytest.mark.parametrize(
    "title",
    [
        "Invoice Retry",
        " Refund   Hold ",
        "Ledger\tClose\nRequest",
    ],
)
def test_route_suffix_matches_payload_dispatch_key_for_whitespace_only_variations(title: str) -> None:
    plan = build_rule_plan(title, "finance", "eu-west")
    payload = compile_payload(plan)
    route = route_for(plan)

    assert route.endswith(f"?dispatch={payload['dispatch_key']}")
    assert f":{payload['slug']}?dispatch=" in route


def test_ruleplan_instances_are_hashable_for_preview_caching() -> None:
    plan_a = build_rule_plan("Invoice Retry", "finance", "eu-west")
    plan_b = build_rule_plan("Invoice Retry", "finance", "eu-west")

    assert {plan_a, plan_b} == {plan_a}


@pytest.mark.parametrize("title", [" Invoice Retry ", "Invoice\tRetry", "Ledger\nClose"])
def test_slug_never_contains_whitespace(title: str) -> None:
    plan = build_rule_plan(title, "finance", "eu-west")

    assert not any(char.isspace() for char in plan.slug)
