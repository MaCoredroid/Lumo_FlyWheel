from __future__ import annotations

import pytest

from norm_app.assembler import compile_payload
from norm_app.rules_v2 import build_rule_plan


@pytest.mark.parametrize(
    ("title", "expected_slug"),
    [
        ("[SEV2] FIRING: API Gateway Latency (page)", "api-gateway-latency"),
        ("(acked) API Gateway Latency [resolved]", "api-gateway-latency"),
        ("RESOLVED :: DB Primary Down [SEV1]", "db-primary-down"),
    ],
)
def test_build_rule_plan_ignores_boundary_lifecycle_wrappers(title: str, expected_slug: str) -> None:
    plan = build_rule_plan(title, "ops", "us-east")

    assert plan.slug == expected_slug


@pytest.mark.parametrize(
    "title",
    [
        "[SEV2] FIRING: API Gateway Latency (page)",
        "(acked) API Gateway Latency [resolved]",
        "resolved - API Gateway Latency - acked",
    ],
)
def test_dispatch_key_stays_stable_across_lifecycle_variants(title: str) -> None:
    plan = build_rule_plan(title, "ops", "us-east")

    assert compile_payload(plan)["dispatch_key"] == "us-east:ops:api-gateway-latency"
