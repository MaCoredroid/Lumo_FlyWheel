from __future__ import annotations

import pytest

from sync_app.service import sync_item


@pytest.mark.parametrize(
    ("name", "expected_suffix"),
    [
        ("Patch/Train [2026.04]", "patch-train-2026-04"),
        ("Patch::Train__2026.04", "patch-train-2026-04"),
        ("Patch Train / 2026_04", "patch-train-2026-04"),
    ],
)
def test_routing_key_normalizes_separator_heavy_release_labels(
    name: str,
    expected_suffix: str,
) -> None:
    payload = sync_item(name, "pending", owner="release-captain")

    assert payload["name"] == name
    assert payload["routing_key"] == f"release-captain:{expected_suffix}"


def test_equivalent_release_spellings_share_canonical_routing_key() -> None:
    first = sync_item("Patch/Train [2026.04]", "pending", owner="release-captain")
    second = sync_item("Patch-Train-2026-04", "pending", owner="release-captain")

    assert first["routing_key"] == second["routing_key"] == "release-captain:patch-train-2026-04"
