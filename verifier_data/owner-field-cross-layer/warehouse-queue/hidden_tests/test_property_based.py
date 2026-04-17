from __future__ import annotations

import json

import pytest

from sync_app.cli import main
from sync_app.service import sync_item


@pytest.mark.parametrize(
    ("name", "expected_slug"),
    [
        ("Picker Backlog", "picker-backlog"),
        (" Picker   Backlog ", "picker-backlog"),
        ("Picker\tBacklog", "picker-backlog"),
        ("Receiving\nHold", "receiving-hold"),
    ],
)
def test_whitespace_only_name_variations_keep_canonical_suffix(name: str, expected_slug: str) -> None:
    payload = sync_item(name, "pending", owner="ops-lead")

    assert payload["routing_key"] == f"ops-lead:{expected_slug}"


@pytest.mark.parametrize("owner", ["ops-lead", "dock-coordinator", "returns-captain"])
def test_sluglike_owner_values_flow_through_routing_prefix(owner: str) -> None:
    payload = sync_item("Picker Backlog", "pending", owner=owner)

    assert payload["routing_key"].split(":", 1)[0] == owner


def test_cli_and_service_agree_for_sluglike_owner_inputs() -> None:
    cli_payload = json.loads(
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

    assert cli_payload == sync_item("Picker Backlog", "pending", owner="ops-lead")
