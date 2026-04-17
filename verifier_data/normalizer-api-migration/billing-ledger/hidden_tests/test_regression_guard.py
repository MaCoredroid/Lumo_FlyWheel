from __future__ import annotations

from norm_app.assembler import compile_payload
from norm_app.router import route_for
from norm_app.rules_v2 import build_rule_plan


def test_route_suffix_stays_in_sync_with_dispatch_key() -> None:
    plan = build_rule_plan("Refund Hold", "finance", "eu-west")
    route = route_for(plan)

    assert route.split("?dispatch=", 1)[1] == plan.dispatch_key


def test_route_bucket_format_stays_stable() -> None:
    plan = build_rule_plan("Refund Hold", "finance", "eu-west")

    assert plan.route_bucket == "eu-west:finance"
    assert route_for(plan).startswith(f"{plan.route_bucket}:{plan.slug}")


def test_compile_payload_keeps_dispatch_key_field() -> None:
    plan = build_rule_plan("Refund Hold", "finance", "eu-west")
    payload = compile_payload(plan)

    assert payload["dispatch_key"] == plan.dispatch_key
    assert set(payload) == {"slug", "route_bucket", "dispatch_key"}
