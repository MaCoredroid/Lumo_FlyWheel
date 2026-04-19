from __future__ import annotations

from router.skill_router import route


def test_release_handoff_uses_trigger_list() -> None:
    assert route("please draft a release handoff", {}) == "release_handoff"


def test_negative_trigger_suppresses_deploy_match() -> None:
    assert route("update the deploy policy doc", {"env": "prod"}) == "general_helper"


def test_missing_required_input_blocks_deploy_check() -> None:
    assert route("check the deploy rollout", {}) == "general_helper"
