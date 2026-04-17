from __future__ import annotations

import json

from sync_app.cli import main
from sync_app.service import sync_item


def test_service_threads_explicit_owner_fields() -> None:
    assert sync_item("Patch Train", "pending", owner="release-captain") == {
        "name": "Patch Train",
        "status": "pending",
        "owner": "release-captain",
        "owner_source": "explicit",
        "routing_key": "release-captain:patch-train",
    }


def test_service_uses_default_owner_when_owner_missing(default_owner: str) -> None:
    assert sync_item("Patch Train", "pending") == {
        "name": "Patch Train",
        "status": "pending",
        "owner": default_owner,
        "owner_source": "default",
        "routing_key": f"{default_owner}:patch-train",
    }


def test_cli_accepts_owner_flag_and_json_contract() -> None:
    payload = json.loads(
        main(
            [
                "--name",
                "Patch Train",
                "--status",
                "pending",
                "--owner",
                "release-captain",
            ]
        )
    )

    assert payload == {
        "name": "Patch Train",
        "status": "pending",
        "owner": "release-captain",
        "owner_source": "explicit",
        "routing_key": "release-captain:patch-train",
    }


def test_routing_key_normalizes_whitespace_only_names() -> None:
    payload_a = sync_item("Patch   Train", "blocked", owner="release-captain")
    payload_b = sync_item("Patch\tTrain", "blocked", owner="release-captain")

    assert payload_a["routing_key"] == payload_b["routing_key"] == "release-captain:patch-train"
    assert payload_a["owner_source"] == payload_b["owner_source"] == "explicit"
