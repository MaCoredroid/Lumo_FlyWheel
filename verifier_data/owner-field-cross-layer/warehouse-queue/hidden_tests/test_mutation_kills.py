from __future__ import annotations

import json

from sync_app.cli import main
from sync_app.service import sync_item


def test_raw_name_field_is_not_replaced_by_slug() -> None:
    payload = sync_item("Picker / Backlog", "pending", owner="ops-lead")

    assert payload["name"] == "Picker / Backlog"
    assert payload["routing_key"].split(":", 1)[1] == "picker-backlog"


def test_cli_uses_canonical_queue_suffix_for_separator_heavy_label() -> None:
    payload = json.loads(
        main(
            [
                "--name",
                "Picker::Backlog",
                "--status",
                "pending",
                "--owner",
                "ops-lead",
            ]
        )
    )

    assert payload["routing_key"] == "ops-lead:picker-backlog"


def test_routing_key_queue_suffix_never_leaks_separator_noise() -> None:
    payload = sync_item("Picker / Backlog", "pending", owner="ops-lead")
    suffix = payload["routing_key"].split(":", 1)[1]

    assert suffix == "picker-backlog"
    assert "/" not in suffix
    assert "_" not in suffix
    assert " " not in suffix
