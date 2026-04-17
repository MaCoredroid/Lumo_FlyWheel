from __future__ import annotations

import json

import pytest

from sync_app.cli import main
from sync_app.service import sync_item


@pytest.mark.parametrize(
    ("name", "expected_slug"),
    [
        ("Patch Train", "patch-train"),
        (" Patch   Train ", "patch-train"),
        ("Patch\tTrain", "patch-train"),
        ("Hotfix\nQueue", "hotfix-queue"),
    ],
)
def test_whitespace_only_name_variations_keep_canonical_suffix(name: str, expected_slug: str) -> None:
    payload = sync_item(name, "pending", owner="release-captain")

    assert payload["routing_key"] == f"release-captain:{expected_slug}"


@pytest.mark.parametrize("owner", ["release-captain", "ops-approver", "qa-gatekeeper"])
def test_sluglike_owner_values_flow_through_routing_prefix(owner: str) -> None:
    payload = sync_item("Patch Train", "pending", owner=owner)

    assert payload["routing_key"].split(":", 1)[0] == owner


def test_cli_and_service_agree_for_sluglike_owner_inputs() -> None:
    cli_payload = json.loads(
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

    assert cli_payload == sync_item("Patch Train", "pending", owner="release-captain")
