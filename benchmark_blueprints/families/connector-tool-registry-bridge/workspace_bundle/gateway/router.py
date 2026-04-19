from __future__ import annotations

from typing import Any

from .tool_registry import (
    build_available_index,
    canonical_handler_name,
    canonicalize_tool_id,
    load_registry,
    load_routing_config,
    ordered_owners,
)


def resolve_tool(tool_id: str, available_tools: Any = None) -> dict[str, Any]:
    canonical_tool_id = canonicalize_tool_id(tool_id)
    registry = load_registry()
    config = load_routing_config()
    available_index = build_available_index(available_tools)

    if available_tools is None:
        owners = {registry[canonical_tool_id].owner} if canonical_tool_id in registry else set()
    else:
        owners = available_index.get(canonical_tool_id, set())

    ordered = ordered_owners(canonical_tool_id, owners, registry=registry, config=config)
    spec = registry.get(canonical_tool_id)

    return {
        "tool_id": canonical_tool_id,
        "handler": spec.handler if spec is not None else canonical_handler_name(canonical_tool_id),
        "owner": ordered[0] if ordered else "unknown",
        "owners": ordered,
    }


def choose_owner(tool_id: str, available_tools: Any = None) -> str:
    return resolve_tool(tool_id, available_tools=available_tools)["owner"]
