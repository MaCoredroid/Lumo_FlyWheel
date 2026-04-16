from __future__ import annotations

import json

from sync_app.cli import main
from sync_app.service import sync_item


def test_owner_is_persisted_by_the_service() -> None:
    payload = sync_item("patch-train", "pending", owner="release-captain")
    assert payload["owner"] == "release-captain"


def test_cli_accepts_owner_flag() -> None:
    payload = json.loads(
        main(
            [
                "--name",
                "patch-train",
                "--status",
                "pending",
                "--owner",
                "release-captain",
            ]
        )
    )
    assert payload["owner"] == "release-captain"
