from __future__ import annotations

import pytest

from sync_app.service import sync_item


@pytest.mark.parametrize(
    ("name", "expected_suffix"),
    [
        ("Picker / Backlog", "picker-backlog"),
        ("Picker::Backlog", "picker-backlog"),
        ("Picker__Backlog", "picker-backlog"),
    ],
)
def test_routing_key_normalizes_separator_heavy_queue_labels(
    name: str,
    expected_suffix: str,
) -> None:
    payload = sync_item(name, "pending", owner="ops-lead")

    assert payload["name"] == name
    assert payload["routing_key"] == f"ops-lead:{expected_suffix}"


def test_equivalent_queue_spellings_share_canonical_routing_key() -> None:
    first = sync_item("Picker / Backlog", "pending", owner="ops-lead")
    second = sync_item("Picker-Backlog", "pending", owner="ops-lead")

    assert first["routing_key"] == second["routing_key"] == "ops-lead:picker-backlog"
