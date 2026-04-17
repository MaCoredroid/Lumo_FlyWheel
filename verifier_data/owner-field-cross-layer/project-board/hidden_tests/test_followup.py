from __future__ import annotations

import pytest

from sync_app.service import sync_item


@pytest.mark.parametrize(
    ("owner", "expected_prefix"),
    [
        ("PM Platform / Core", "pm-platform-core"),
        ("PM.Platform/Core", "pm-platform-core"),
        ("PM   Platform :: Core", "pm-platform-core"),
    ],
)
def test_routing_key_normalizes_separator_heavy_owner_labels(
    owner: str,
    expected_prefix: str,
) -> None:
    payload = sync_item("Launch Checklist", "pending", owner=owner)

    assert payload["owner"] == owner
    assert payload["routing_key"] == f"{expected_prefix}:launch-checklist"


def test_equivalent_owner_spellings_share_canonical_routing_key() -> None:
    first = sync_item("Launch Checklist", "pending", owner="PM Platform / Core")
    second = sync_item("Launch Checklist", "pending", owner="pm-platform-core")

    assert first["routing_key"] == second["routing_key"] == "pm-platform-core:launch-checklist"
