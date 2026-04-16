from __future__ import annotations

from norm_app.cli import preview


def test_preview_uses_ruleplan_v2_shape() -> None:
    payload = preview()
    assert payload["slug"] == "disk-pressure"
    assert payload["route_bucket"] == "us-east:ops"
    assert payload["route"].startswith("us-east:ops")
