from __future__ import annotations

from sync_app.store import make_record


def sync_item(name: str, status: str, owner: str | None = None) -> dict[str, str]:
    del owner
    return make_record(name, status)
