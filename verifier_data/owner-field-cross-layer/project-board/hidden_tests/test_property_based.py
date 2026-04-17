from __future__ import annotations

import json

import pytest

from sync_app.cli import main
from sync_app.service import sync_item


@pytest.mark.parametrize(
    ("name", "expected_slug"),
    [
        ("Launch Checklist", "launch-checklist"),
        (" Launch   Checklist ", "launch-checklist"),
        ("Launch\tChecklist", "launch-checklist"),
        ("Quarterly\nRoadmap", "quarterly-roadmap"),
    ],
)
def test_whitespace_only_name_variations_keep_canonical_suffix(name: str, expected_slug: str) -> None:
    payload = sync_item(name, "pending", owner="project-ops")

    assert payload["routing_key"] == f"project-ops:{expected_slug}"


@pytest.mark.parametrize("owner", ["pm-oncall", "design-systems", "release-captain"])
def test_sluglike_owner_values_flow_through_routing_prefix(owner: str) -> None:
    payload = sync_item("Launch Checklist", "pending", owner=owner)

    assert payload["routing_key"].split(":", 1)[0] == owner


def test_cli_and_service_agree_for_sluglike_owner_inputs() -> None:
    cli_payload = json.loads(
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

    assert cli_payload == sync_item("Launch Checklist", "pending", owner="pm-oncall")
