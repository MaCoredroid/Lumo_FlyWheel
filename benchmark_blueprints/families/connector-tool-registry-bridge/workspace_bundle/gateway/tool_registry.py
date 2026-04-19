from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = WORKSPACE_ROOT / "registry" / "tool_catalog.yaml"
CONFIG_PATH = WORKSPACE_ROOT / ".codex" / "config.toml"

CONNECTOR_KEYS = ("connector", "owner", "provider", "source")
TOOL_ID_KEYS = ("id", "tool_id", "name")
HANDLER_KEYS = ("handler", "handler_name")
CONTAINER_KEYS = ("tools", "items", "discovered_tools", "result", "data", "payload", "event")


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    owner: str
    handler: str


@dataclass(frozen=True)
class RoutingConfig:
    preferred_owners: dict[str, str]
    fallback_order: dict[str, list[str]]
    default_precedence: list[str]


def canonicalize_tool_id(tool_id: str | None) -> str | None:
    return tool_id


def canonical_handler_name(tool_id: str) -> str:
    return f"{tool_id.replace('.', '_')}_handler"


def canonicalize_handler_name(handler_name: str | None, tool_id: str | None = None) -> str | None:
    if tool_id is not None:
        return canonical_handler_name(tool_id)
    return handler_name


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    return value


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, ToolSpec]:
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped == "tools:":
            continue
        if stripped.startswith("- "):
            current = {}
            entries.append(current)
            stripped = stripped[2:]
        if current is None or ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        current[key.strip()] = _parse_scalar(raw_value)

    registry: dict[str, ToolSpec] = {}
    for entry in entries:
        tool_id = canonicalize_tool_id(str(entry["id"]))
        owner = str(entry["owner"])
        handler = canonicalize_handler_name(entry.get("handler"), tool_id)
        registry[tool_id] = ToolSpec(tool_id=tool_id, owner=owner, handler=handler)
    return registry


def load_routing_config(path: Path = CONFIG_PATH) -> RoutingConfig:
    preferred_owners: dict[str, str] = {}
    fallback_order: dict[str, list[str]] = {}
    default_precedence: list[str] = []
    current_section: str | None = None

    for raw_line in path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped[1:-1]
            continue
        if "=" not in stripped or current_section is None:
            continue

        raw_key, raw_value = stripped.split("=", 1)
        key = raw_key.strip().strip("'\"")
        value = _parse_scalar(raw_value)

        if current_section == "routing.preferred_owners" and isinstance(value, str):
            preferred_owners[canonicalize_tool_id(key)] = value
        elif current_section == "routing.fallback_order" and isinstance(value, list):
            fallback_order[canonicalize_tool_id(key)] = [str(item) for item in value]
        elif current_section == "routing.connector_precedence" and key == "default" and isinstance(value, list):
            default_precedence = [str(item) for item in value]
        elif current_section == "routing" and isinstance(value, str) and "." in key:
            preferred_owners[canonicalize_tool_id(key)] = value

    return RoutingConfig(
        preferred_owners=preferred_owners,
        fallback_order=fallback_order,
        default_precedence=default_precedence,
    )


def _first_string(payload: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def iter_discovered_tools(payload: Any, connector: str | None = None) -> Iterable[dict[str, str]]:
    if isinstance(payload, list):
        for item in payload:
            yield from iter_discovered_tools(item, connector)
        return

    if not isinstance(payload, dict):
        return

    next_connector = _first_string(payload, CONNECTOR_KEYS) or connector
    tool_id = _first_string(payload, TOOL_ID_KEYS)

    if tool_id is not None:
        yield {
            "id": tool_id,
            "owner": next_connector or "unknown",
            "handler": _first_string(payload, HANDLER_KEYS) or "",
        }

    for key in CONTAINER_KEYS:
        if key in payload:
            yield from iter_discovered_tools(payload[key], next_connector)


def _collect_from_tool_sequence(tools: Iterable[Any], index: dict[str, set[str]]) -> None:
    for item in tools:
        if isinstance(item, str):
            tool_id = canonicalize_tool_id(item)
            if tool_id is not None:
                index.setdefault(tool_id, set())
            continue
        if not isinstance(item, dict):
            continue
        tool_id = canonicalize_tool_id(_first_string(item, TOOL_ID_KEYS))
        owner = _first_string(item, CONNECTOR_KEYS)
        if tool_id is None:
            continue
        index.setdefault(tool_id, set())
        if owner:
            index[tool_id].add(owner)


def build_available_index(available_tools: Any) -> dict[str, set[str]]:
    if available_tools is None:
        return {}

    if isinstance(available_tools, (str, Path)):
        return available_index_from_session(Path(available_tools))

    index: dict[str, set[str]] = {}
    if isinstance(available_tools, dict):
        if all(isinstance(value, dict) for value in available_tools.values()):
            for tool_id, metadata in available_tools.items():
                canonical_tool = canonicalize_tool_id(tool_id)
                owners = metadata.get("owners", [])
                owner = metadata.get("owner")
                index.setdefault(canonical_tool, set())
                if isinstance(owners, list):
                    index[canonical_tool].update(str(item) for item in owners)
                if isinstance(owner, str):
                    index[canonical_tool].add(owner)
            return index

        if all(isinstance(value, (list, tuple, set)) for value in available_tools.values()):
            for connector, tools in available_tools.items():
                for tool in tools:
                    if isinstance(tool, str):
                        tool_id = canonicalize_tool_id(tool)
                    elif isinstance(tool, dict):
                        tool_id = canonicalize_tool_id(_first_string(tool, TOOL_ID_KEYS))
                    else:
                        continue
                    if tool_id is None:
                        continue
                    index.setdefault(tool_id, set()).add(str(connector))
            return index

    if isinstance(available_tools, Iterable) and not isinstance(available_tools, (str, bytes)):
        _collect_from_tool_sequence(available_tools, index)
    return index


def ordered_owners(
    tool_id: str,
    available_owners: Iterable[str],
    registry: dict[str, ToolSpec] | None = None,
    config: RoutingConfig | None = None,
) -> list[str]:
    registry = registry or load_registry()
    config = config or load_routing_config()

    owners = {owner for owner in available_owners if owner}
    if not owners:
        return []

    preferred = config.preferred_owners.get(tool_id)
    fallback = config.fallback_order.get(tool_id, [])
    registry_owner = registry.get(tool_id).owner if tool_id in registry else None

    ordered: list[str] = []
    for owner in [preferred, *fallback, registry_owner, *config.default_precedence]:
        if owner and owner in owners and owner not in ordered:
            ordered.append(owner)

    for owner in sorted(owners):
        if owner not in ordered:
            ordered.append(owner)
    return ordered


def available_index_from_session(session_path: Path) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for raw_line in session_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        for record in iter_discovered_tools(payload):
            tool_id = canonicalize_tool_id(record["id"])
            owner = record["owner"]
            if tool_id is None:
                continue
            index.setdefault(tool_id, set())
            if owner:
                index[tool_id].add(owner)
    return index


def replay_discovery_session(session_path: Path) -> dict[str, dict[str, Any]]:
    registry = load_registry()
    config = load_routing_config()
    available = available_index_from_session(session_path)

    tool_map: dict[str, dict[str, Any]] = {}
    live_tool_ids = set(available)

    for tool_id in sorted(live_tool_ids):
        spec = registry.get(tool_id)
        owners = available.get(tool_id, set())
        ordered = ordered_owners(tool_id, owners, registry=registry, config=config)
        handler = spec.handler if spec is not None else canonical_handler_name(tool_id)
        tool_map[tool_id] = {
            "handler": canonicalize_handler_name(handler, tool_id),
            "owner": ordered[0] if ordered else "unknown",
            "owners": ordered,
        }

    return tool_map
