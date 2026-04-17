from __future__ import annotations

import pytest

from sync_app.service import sync_item


def test_owner_source_tracks_explicit_and_default_paths(default_owner: str) -> None:
    default_payload = sync_item("Patch Train", "pending")
    explicit_payload = sync_item("Patch Train", "pending", owner="ops-approver")

    assert default_payload["owner_source"] == "default"
    assert default_payload["owner"] == default_owner
    assert explicit_payload["owner_source"] == "explicit"
    assert explicit_payload["owner"] == "ops-approver"


@pytest.mark.parametrize("name", [" Patch Train ", "Patch\tTrain", "Patch\nTrain"])
def test_routing_key_never_contains_whitespace_for_sluglike_inputs(name: str) -> None:
    payload = sync_item(name, "pending", owner="release-captain")

    assert not any(char.isspace() for char in payload["routing_key"])


def test_payload_keys_remain_stable() -> None:
    payload = sync_item("Patch Train", "pending", owner="release-captain")

    assert set(payload) == {"name", "status", "owner", "owner_source", "routing_key"}
