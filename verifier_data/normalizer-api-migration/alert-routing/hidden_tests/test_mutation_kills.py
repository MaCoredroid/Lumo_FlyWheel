from __future__ import annotations

from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


def test_slug_normalization_drops_leading_severity_and_trailing_status_tokens() -> None:
    plan = build_rule_plan("[SEV1] Cache Node 7 CPU High [ACKED]", "ops", "us-east")

    assert plan.slug == "cache-node-7-cpu-high"


def test_slug_normalization_drops_boundary_page_token_but_preserves_digits() -> None:
    plan = build_rule_plan("page - Cache Node 7 CPU High - resolved", "ops", "us-east")

    assert plan.slug == "cache-node-7-cpu-high"


def test_route_uses_the_same_canonical_dispatch_suffix_after_wrapper_stripping() -> None:
    plan = build_rule_plan("[SEV1] Cache Node 7 CPU High [ACKED]", "ops", "us-east")

    assert route_for(plan) == (
        "us-east:ops:cache-node-7-cpu-high"
        "?dispatch=us-east:ops:cache-node-7-cpu-high"
    )
