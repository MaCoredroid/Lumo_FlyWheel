from __future__ import annotations

import pytest

import norm_app.cli as cli_module
from norm_app.assembler import compile_payload
from norm_app.router import route_for
from norm_app.rules_v2 import RulePlan, build_rule_plan


def test_build_rule_plan_exposes_canonical_dispatch_key() -> None:
    plan = build_rule_plan("  Disk   Pressure  ", cli_module.SAMPLE["owner"], cli_module.SAMPLE["region"])

    assert plan.dispatch_key == "us-east:ops:disk-pressure"


def test_compile_payload_and_router_accept_ruleplan_instances() -> None:
    plan = build_rule_plan(
        cli_module.SAMPLE["title"],
        cli_module.SAMPLE["owner"],
        cli_module.SAMPLE["region"],
    )

    assert compile_payload(plan) == {
        "slug": plan.slug,
        "route_bucket": plan.route_bucket,
        "dispatch_key": plan.dispatch_key,
    }
    assert route_for(plan) == f"{plan.route_bucket}:{plan.slug}?dispatch={plan.dispatch_key}"


def test_preview_builds_ruleplan_once_and_threads_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_plan = build_rule_plan(
        cli_module.SAMPLE["title"],
        cli_module.SAMPLE["owner"],
        cli_module.SAMPLE["region"],
    )
    calls: list[tuple[str, str, str]] = []

    def fake_build_rule_plan(title: str, owner: str, region: str) -> RulePlan:
        calls.append((title, owner, region))
        return expected_plan

    monkeypatch.setattr(cli_module, "build_rule_plan", fake_build_rule_plan)

    payload = cli_module.preview()

    assert calls == [
        (
            cli_module.SAMPLE["title"],
            cli_module.SAMPLE["owner"],
            cli_module.SAMPLE["region"],
        )
    ]
    assert payload == {
        "slug": expected_plan.slug,
        "route_bucket": expected_plan.route_bucket,
        "dispatch_key": expected_plan.dispatch_key,
        "route": f"{expected_plan.route_bucket}:{expected_plan.slug}?dispatch={expected_plan.dispatch_key}",
    }
