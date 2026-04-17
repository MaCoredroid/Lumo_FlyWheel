from __future__ import annotations

import json

from sync_app.cli import main
from sync_app.service import sync_item


def test_raw_owner_field_is_not_replaced_by_slug() -> None:
    payload = sync_item("Launch Checklist", "pending", owner="PM Platform / Core")

    assert payload["owner"] == "PM Platform / Core"
    assert payload["routing_key"].split(":", 1)[0] == "pm-platform-core"


def test_cli_uses_canonical_owner_prefix_for_separator_heavy_label() -> None:
    payload = json.loads(
        main(
            [
                "--name",
                "Launch Checklist",
                "--status",
                "pending",
                "--owner",
                "PM Platform / Core",
            ]
        )
    )

    assert payload["routing_key"] == "pm-platform-core:launch-checklist"


def test_routing_key_owner_prefix_never_leaks_separator_noise() -> None:
    payload = sync_item("Launch Checklist", "pending", owner="PM.Platform/Core")
    prefix = payload["routing_key"].split(":", 1)[0]

    assert prefix == "pm-platform-core"
    assert "." not in prefix
    assert "/" not in prefix
    assert " " not in prefix
