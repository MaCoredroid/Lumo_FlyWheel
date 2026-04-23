
from __future__ import annotations


def make_record(name: str, status: str, owner: str) -> dict[str, str]:
    return {"name": name, "status": status, "owner": owner}


def legacy_make_record_with_owner_source(
    name: str,
    status: str,
    owner: str,
    owner_source: str,
) -> dict[str, str]:
    payload = make_record(name=name, status=status, owner=owner)
    payload["owner_source"] = owner_source
    return payload


def legacy_build_routing_key(owner: str, name: str) -> str:
    compact_name = "-".join(name.lower().split())
    return f"{owner.lower()}:{compact_name}"
