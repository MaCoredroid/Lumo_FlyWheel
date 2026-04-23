
from __future__ import annotations

import json
from pathlib import Path

from sync_app.serializer import build_routing_key, serialize_payload
from sync_app.store import make_record


def _load_defaults() -> dict[str, str]:
    defaults_path = Path(__file__).resolve().parents[1] / "config" / "defaults.json"
    return json.loads(defaults_path.read_text(encoding="utf-8"))


def _resolve_owner(owner: str | None) -> tuple[str, str]:
    if owner and owner.strip():
        return owner.strip(), "explicit"
    return _load_defaults()["owner"], "default"


def sync_item(name: str, status: str, owner: str | None = None) -> dict[str, str]:
    effective_owner, owner_source = _resolve_owner(owner)
    record = make_record(name=name, status=status, owner=effective_owner)
    routing_key = build_routing_key(effective_owner, name)
    return serialize_payload(record, owner_source=owner_source, routing_key=routing_key)
