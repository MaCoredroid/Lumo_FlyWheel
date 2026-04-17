from __future__ import annotations

import json

from sync_app.cli import main
from sync_app.service import sync_item


def test_service_threads_explicit_owner_fields() -> None:
    assert sync_item("Launch Checklist", "pending", owner="pm-oncall") == {
        "name": "Launch Checklist",
        "status": "pending",
        "owner": "pm-oncall",
        "owner_source": "explicit",
        "routing_key": "pm-oncall:launch-checklist",
    }


def test_service_uses_default_owner_when_owner_missing(default_owner: str) -> None:
    assert sync_item("Launch Checklist", "pending") == {
        "name": "Launch Checklist",
        "status": "pending",
        "owner": default_owner,
        "owner_source": "default",
        "routing_key": f"{default_owner}:launch-checklist",
    }


def test_cli_accepts_owner_flag_and_json_contract() -> None:
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


def test_routing_key_normalizes_whitespace_only_names() -> None:
    payload_a = sync_item("Quarterly   Roadmap", "blocked", owner="product-ops")
    payload_b = sync_item("Quarterly\tRoadmap", "blocked", owner="product-ops")

    assert payload_a["routing_key"] == payload_b["routing_key"] == "product-ops:quarterly-roadmap"
    assert payload_a["owner_source"] == payload_b["owner_source"] == "explicit"
