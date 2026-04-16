from __future__ import annotations

from norm_app.cli import preview


def test_preview_uses_ruleplan_v2_shape() -> None:
    payload = preview()
    assert payload["slug"] == "missing-sku"
    assert payload["route_bucket"] == "ap-south:catalog"
    assert payload["route"].startswith("ap-south:catalog")
