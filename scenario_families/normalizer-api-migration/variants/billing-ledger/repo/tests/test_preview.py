from __future__ import annotations

from norm_app.cli import preview


def test_preview_uses_ruleplan_v2_shape() -> None:
    payload = preview()
    assert payload["slug"] == "invoice-retry"
    assert payload["route_bucket"] == "eu-west:finance"
    assert payload["route"].startswith("eu-west:finance")
