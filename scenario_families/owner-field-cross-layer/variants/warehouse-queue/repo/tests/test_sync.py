from __future__ import annotations

import json
from pathlib import Path

from sync_app.cli import main
from sync_app.service import sync_item


def _default_owner() -> str:
    return json.loads(Path("config/defaults.json").read_text(encoding="utf-8"))["owner"]


def test_owner_is_persisted_by_the_service() -> None:
    payload = sync_item("Picker  Backlog", "pending", owner="ops-lead")
    assert payload == {
        "name": "Picker  Backlog",
        "status": "pending",
        "owner": "ops-lead",
        "owner_source": "explicit",
        "routing_key": "ops-lead:picker-backlog",
    }


def test_service_uses_default_owner_when_not_provided() -> None:
    payload = sync_item("Picker Backlog", "pending")
    assert payload == {
        "name": "Picker Backlog",
        "status": "pending",
        "owner": _default_owner(),
        "owner_source": "default",
        "routing_key": f"{_default_owner()}:picker-backlog",
    }


def test_cli_accepts_owner_flag_and_preserves_existing_fields() -> None:
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


def test_cli_uses_default_owner_when_flag_is_missing() -> None:
    payload = json.loads(main(["--name", "Picker Backlog", "--status", "pending"]))
    assert payload == {
        "name": "Picker Backlog",
        "status": "pending",
        "owner": _default_owner(),
        "owner_source": "default",
        "routing_key": f"{_default_owner()}:picker-backlog",
    }
