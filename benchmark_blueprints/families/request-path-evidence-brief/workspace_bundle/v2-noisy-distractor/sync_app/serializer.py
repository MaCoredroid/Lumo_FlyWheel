
from __future__ import annotations


def _slugify(text: str) -> str:
    return "-".join(text.lower().split())


def build_routing_key(owner: str, name: str) -> str:
    return f"{_slugify(owner)}:{_slugify(name)}"


def serialize_payload(
    record: dict[str, str],
    owner_source: str,
    routing_key: str,
) -> dict[str, str]:
    payload = dict(record)
    payload["owner_source"] = owner_source
    payload["routing_key"] = routing_key
    return payload


def draft_owner_source_from_record(record: dict[str, str]) -> str:
    return record.get("owner_source", "unknown")
