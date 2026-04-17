from __future__ import annotations

import json

from sync_app.cli import main
from sync_app.service import sync_item


def test_raw_name_field_is_not_replaced_by_slug() -> None:
    payload = sync_item("Patch/Train [2026.04]", "pending", owner="release-captain")

    assert payload["name"] == "Patch/Train [2026.04]"
    assert payload["routing_key"].split(":", 1)[1] == "patch-train-2026-04"


def test_cli_uses_canonical_release_suffix_for_separator_heavy_label() -> None:
    payload = json.loads(
        main(
            [
                "--name",
                "Patch::Train__2026.04",
                "--status",
                "pending",
                "--owner",
                "release-captain",
            ]
        )
    )

    assert payload["routing_key"] == "release-captain:patch-train-2026-04"


def test_routing_key_release_suffix_never_leaks_separator_noise() -> None:
    payload = sync_item("Patch/Train [2026.04]", "pending", owner="release-captain")
    suffix = payload["routing_key"].split(":", 1)[1]

    assert suffix == "patch-train-2026-04"
    assert "/" not in suffix
    assert "[" not in suffix
    assert "." not in suffix
    assert "_" not in suffix
    assert " " not in suffix
