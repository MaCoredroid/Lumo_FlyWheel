from __future__ import annotations

import pytest

from norm_app.assembler import compile_payload
from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


@pytest.mark.parametrize(
    ("title", "expected_slug"),
    [
        (" Refund / Retry _ Queue ", "refund-retry-queue"),
        ("Chargeback---Retry!!!", "chargeback-retry"),
        ("Invoice 2024 / Retry 7", "invoice-2024-retry-7"),
    ],
)
def test_build_rule_plan_normalizes_separator_noise(title: str, expected_slug: str) -> None:
    plan = build_rule_plan(title, "finance", "eu-west")

    assert plan.slug == expected_slug


@pytest.mark.parametrize(
    ("title", "expected_dispatch_key"),
    [
        (" Refund / Retry _ Queue ", "eu-west:finance:refund-retry-queue"),
        ("Chargeback---Retry!!!", "eu-west:finance:chargeback-retry"),
    ],
)
def test_dispatch_key_uses_canonical_slug_for_separator_noise(
    title: str,
    expected_dispatch_key: str,
) -> None:
    plan = build_rule_plan(title, "finance", "eu-west")

    assert plan.dispatch_key == expected_dispatch_key
    assert compile_payload(plan)["dispatch_key"] == expected_dispatch_key


@pytest.mark.parametrize(
    "title",
    [
        " Refund / Retry _ Queue ",
        "Chargeback---Retry!!!",
        "Invoice___Retry///Hold",
    ],
)
def test_route_never_emits_doubled_hyphens_for_separator_noise(title: str) -> None:
    plan = build_rule_plan(title, "finance", "eu-west")
    route = route_for(plan)

    assert "--" not in plan.slug
    assert "--" not in route
