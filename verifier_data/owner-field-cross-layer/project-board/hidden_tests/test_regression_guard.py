from __future__ import annotations

import pytest

from sync_app.service import sync_item


def test_owner_source_tracks_explicit_and_default_paths(default_owner: str) -> None:
    default_payload = sync_item("Launch Checklist", "pending")
    explicit_payload = sync_item("Launch Checklist", "pending", owner="product-ops")

    assert default_payload["owner_source"] == "default"
    assert default_payload["owner"] == default_owner
    assert explicit_payload["owner_source"] == "explicit"
    assert explicit_payload["owner"] == "product-ops"


@pytest.mark.parametrize("name", [" Launch Checklist ", "Launch\tChecklist", "Launch\nChecklist"])
def test_routing_key_never_contains_whitespace_for_sluglike_inputs(name: str) -> None:
    payload = sync_item(name, "pending", owner="pm-oncall")

    assert not any(char.isspace() for char in payload["routing_key"])


def test_payload_keys_remain_stable() -> None:
    payload = sync_item("Launch Checklist", "pending", owner="pm-oncall")

    assert set(payload) == {"name", "status", "owner", "owner_source", "routing_key"}
