from __future__ import annotations

import json
from pathlib import Path

from sync_app.cli import main
from sync_app.service import sync_item


def _default_owner() -> str:
    return json.loads(Path("config/defaults.json").read_text(encoding="utf-8"))["owner"]


def test_owner_is_persisted_by_the_service() -> None:
    payload = sync_item("launch-checklist", "pending", owner="pm-oncall")
    assert payload == {
        "name": "launch-checklist",
        "status": "pending",
        "owner": "pm-oncall",
    }


def test_service_uses_default_owner_when_not_provided() -> None:
    payload = sync_item("launch-checklist", "pending")
    assert payload == {
        "name": "launch-checklist",
        "status": "pending",
        "owner": _default_owner(),
    }


def test_cli_accepts_owner_flag_and_preserves_existing_fields() -> None:
    payload = json.loads(
        main(
            [
                "--name",
                "launch-checklist",
                "--status",
                "pending",
                "--owner",
                "pm-oncall",
            ]
        )
    )
    assert payload == {
        "name": "launch-checklist",
        "status": "pending",
        "owner": "pm-oncall",
    }


def test_cli_uses_default_owner_when_flag_is_missing() -> None:
    payload = json.loads(main(["--name", "launch-checklist", "--status", "pending"]))
    assert payload["owner"] == _default_owner()
