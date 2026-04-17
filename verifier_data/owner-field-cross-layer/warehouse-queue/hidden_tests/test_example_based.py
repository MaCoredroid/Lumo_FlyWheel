from __future__ import annotations

import json

from sync_app.cli import main
from sync_app.service import sync_item


def test_service_threads_explicit_owner_fields() -> None:
    assert sync_item("Picker Backlog", "pending", owner="ops-lead") == {
        "name": "Picker Backlog",
        "status": "pending",
        "owner": "ops-lead",
        "owner_source": "explicit",
        "routing_key": "ops-lead:picker-backlog",
    }


def test_service_uses_default_owner_when_owner_missing(default_owner: str) -> None:
    assert sync_item("Picker Backlog", "pending") == {
        "name": "Picker Backlog",
        "status": "pending",
        "owner": default_owner,
        "owner_source": "default",
        "routing_key": f"{default_owner}:picker-backlog",
    }


def test_cli_accepts_owner_flag_and_json_contract() -> None:
    payload = json.loads(
        main(
            [
                "--name",
                "Picker Backlog",
                "--status",
                "pending",
                "--owner",
                "ops-lead",
            ]
        )
    )

    assert payload == {
        "name": "Picker Backlog",
        "status": "pending",
        "owner": "ops-lead",
        "owner_source": "explicit",
        "routing_key": "ops-lead:picker-backlog",
    }


def test_routing_key_normalizes_whitespace_only_names() -> None:
    payload_a = sync_item("Picker   Backlog", "blocked", owner="ops-lead")
    payload_b = sync_item("Picker\tBacklog", "blocked", owner="ops-lead")

    assert payload_a["routing_key"] == payload_b["routing_key"] == "ops-lead:picker-backlog"
    assert payload_a["owner_source"] == payload_b["owner_source"] == "explicit"
