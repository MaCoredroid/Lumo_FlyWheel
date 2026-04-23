
from __future__ import annotations

import json
from pathlib import Path

from sync_app.cli import main
from sync_app.service import sync_item


def _default_owner() -> str:
    return json.loads(Path("config/defaults.json").read_text(encoding="utf-8"))["owner"]


def test_service_resolves_explicit_owner_before_serialization() -> None:
    payload = sync_item("Launch Checklist", "pending", owner="pm-oncall")
    assert payload == {
        "name": "Launch Checklist",
        "status": "pending",
        "owner": "pm-oncall",
        "owner_source": "explicit",
        "routing_key": "pm-oncall:launch-checklist",
    }


def test_service_uses_default_owner_when_flag_is_missing() -> None:
    payload = sync_item("Launch Checklist", "pending")
    assert payload == {
        "name": "Launch Checklist",
        "status": "pending",
        "owner": _default_owner(),
        "owner_source": "default",
        "routing_key": f"{_default_owner()}:launch-checklist",
    }


def test_cli_accepts_owner_flag_and_preserves_existing_fields() -> None:
    payload = json.loads(
        main(
            [
                "--name",
                "Launch Checklist",
                "--status",
                "pending",
                "--owner",
                "pm-oncall",
            ]
        )
    )
    assert payload == {
        "name": "Launch Checklist",
        "status": "pending",
        "owner": "pm-oncall",
        "owner_source": "explicit",
        "routing_key": "pm-oncall:launch-checklist",
    }
