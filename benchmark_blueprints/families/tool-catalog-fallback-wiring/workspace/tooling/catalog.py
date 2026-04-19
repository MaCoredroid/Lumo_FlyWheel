from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "tooling" / "tool_manifest.yaml"
ROUTER_CONFIG_PATH = ROOT / "config" / "tool_router.toml"

DEFAULT_TOOL_CAPABILITIES = {
    "chrome-devtools": {"browser.read"},
    "http_fetch": {"docs.lookup"},
    "browser_snapshot": {"browser.read"},
}

DEFAULT_TOOL_POLICIES = {
    "chrome-devtools": "interactive",
    "http_fetch": "safe",
    "browser_snapshot": "safe",
}

DEFAULT_PREFERRED_TOOLS = {
    "browser.read": "chrome-devtools",
}

DEFAULT_CATALOG = {
    "browser.read": ["http_fetch", "browser_snapshot"],
    "docs.lookup": ["http_fetch"],
}

DEFAULT_ELIGIBLE_FALLBACK_POLICIES = ["safe", "interactive"]


def _strip_scalar(value: str) -> str:
    value = value.strip()
    if value[:1] == value[-1:] and value[:1] in {'"', "'"}:
        return value[1:-1]
    return value


def _load_manifest() -> tuple[dict[str, str], dict[str, list[str]], dict[str, set[str]], dict[str, str]]:
    preferred_tools = dict(DEFAULT_PREFERRED_TOOLS)
    fallbacks = {capability: list(tool_ids) for capability, tool_ids in DEFAULT_CATALOG.items()}
    tool_capabilities = {
        tool_id: set(capabilities)
        for tool_id, capabilities in DEFAULT_TOOL_CAPABILITIES.items()
    }
    tool_policies = dict(DEFAULT_TOOL_POLICIES)

    current_section: str | None = None
    current_tool: str | None = None
    current_capability: str | None = None
    tool_list_name: str | None = None

    for raw_line in MANIFEST_PATH.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0:
            current_tool = None
            current_capability = None
            tool_list_name = None

            if stripped.endswith(":"):
                current_section = stripped[:-1]
                continue

            key, _, value = stripped.partition(":")
            if key == "preferred_browser_tool":
                preferred_tools["browser.read"] = _strip_scalar(value)
            continue

        if current_section == "preferred_tools" and indent == 2:
            capability, _, tool_id = stripped.partition(":")
            preferred_tools[capability] = _strip_scalar(tool_id)
            continue

        if current_section == "fallbacks":
            if indent == 2 and stripped.endswith(":"):
                current_capability = stripped[:-1]
                fallbacks[current_capability] = []
                continue
            if indent == 4 and stripped.startswith("- ") and current_capability:
                fallbacks[current_capability].append(_strip_scalar(stripped[2:]))
            continue

        if current_section == "tools":
            if indent == 2 and stripped.endswith(":"):
                current_tool = stripped[:-1]
                tool_capabilities.setdefault(current_tool, set())
                tool_policies.setdefault(current_tool, "")
                tool_list_name = None
                continue
            if indent == 4 and stripped.startswith("capabilities:"):
                tool_list_name = "capabilities"
                continue
            if indent == 4 and stripped.startswith("policy:") and current_tool:
                _, _, policy = stripped.partition(":")
                tool_policies[current_tool] = _strip_scalar(policy)
                tool_list_name = None
                continue
            if (
                indent == 6
                and stripped.startswith("- ")
                and current_tool
                and tool_list_name == "capabilities"
            ):
                tool_capabilities[current_tool].add(_strip_scalar(stripped[2:]))

    return preferred_tools, fallbacks, tool_capabilities, tool_policies


def _load_router_config() -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    preferred_tools = dict(DEFAULT_PREFERRED_TOOLS)
    fallbacks = {capability: list(tool_ids) for capability, tool_ids in DEFAULT_CATALOG.items()}
    eligible_policies = list(DEFAULT_ELIGIBLE_FALLBACK_POLICIES)

    current_section: str | None = None
    for raw_line in ROUTER_CONFIG_PATH.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()

        if current_section == "router" and key == "eligible_fallback_policies":
            eligible_policies = [
                _strip_scalar(item.strip())
                for item in value.strip("[]").split(",")
                if item.strip()
            ]
            continue

        if current_section == "router.preferred_tools":
            preferred_tools[key] = _strip_scalar(value)
            continue

        if current_section == "router.fallbacks":
            fallbacks[key] = [
                _strip_scalar(item.strip())
                for item in value.strip("[]").split(",")
                if item.strip()
            ]

    return preferred_tools, fallbacks, eligible_policies


def _load_catalog() -> tuple[dict[str, str], dict[str, list[str]], dict[str, set[str]], dict[str, str], list[str]]:
    manifest_preferred, manifest_fallbacks, tool_capabilities, tool_policies = _load_manifest()
    router_preferred, router_fallbacks, eligible_policies = _load_router_config()

    preferred_tools = dict(manifest_preferred)
    preferred_tools.update(router_preferred)

    fallbacks = {capability: list(tool_ids) for capability, tool_ids in manifest_fallbacks.items()}
    fallbacks.update(router_fallbacks)

    return preferred_tools, fallbacks, tool_capabilities, tool_policies, eligible_policies


def _supports_capability(tool_id: str, capability: str, tool_capabilities: dict[str, set[str]]) -> bool:
    return capability in tool_capabilities.get(tool_id, set())


def _policy_is_eligible(tool_id: str, tool_policies: dict[str, str], eligible_policies: list[str]) -> bool:
    return tool_policies.get(tool_id) in eligible_policies


def _candidate_is_eligible(
    tool_id: str | None,
    capability: str,
    tool_capabilities: dict[str, set[str]],
    tool_policies: dict[str, str],
    eligible_policies: list[str],
) -> bool:
    if not tool_id:
        return False
    return _supports_capability(tool_id, capability, tool_capabilities) and _policy_is_eligible(
        tool_id,
        tool_policies,
        eligible_policies,
    )


def _fallback_policy_rank(tool_id: str, tool_policies: dict[str, str], eligible_policies: list[str]) -> int:
    policy = tool_policies.get(tool_id)
    try:
        return eligible_policies.index(policy)
    except ValueError:
        return len(eligible_policies)


def select_tool(capability: str, preferred_tool: str | None) -> str:
    configured_preferred, catalog, tool_capabilities, tool_policies, eligible_policies = _load_catalog()
    preferred_candidate = preferred_tool or configured_preferred.get(capability)

    if _candidate_is_eligible(
        preferred_candidate,
        capability,
        tool_capabilities,
        tool_policies,
        eligible_policies,
    ):
        return preferred_candidate

    fallbacks = [
        tool_id
        for tool_id in catalog.get(capability, [])
        if _candidate_is_eligible(
            tool_id,
            capability,
            tool_capabilities,
            tool_policies,
            eligible_policies,
        )
    ]
    if not fallbacks:
        raise KeyError(f"No eligible tool found for capability {capability!r}")

    return min(
        enumerate(fallbacks),
        key=lambda item: (
            _fallback_policy_rank(item[1], tool_policies, eligible_policies),
            item[0],
        ),
    )[1]
