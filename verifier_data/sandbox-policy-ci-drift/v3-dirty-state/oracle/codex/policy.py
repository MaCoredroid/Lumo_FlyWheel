
from __future__ import annotations

SANDBOX_NAMES = {
    "read_only": "read_only",
    "workspace_write": "workspace_write",
    "danger_full_access": "danger_full_access",
}
DEPRECATED_SANDBOX_ALIASES = {
    "workspace-write": "workspace_write",
}
APPROVAL_NAMES = {"never", "on_request", "on_failure", "untrusted"}


def parse_sandbox(value: str) -> str:
    raw = str(value).strip()
    if raw in SANDBOX_NAMES:
        return SANDBOX_NAMES[raw]
    if raw in DEPRECATED_SANDBOX_ALIASES:
        return DEPRECATED_SANDBOX_ALIASES[raw]
    raise ValueError(f"unsupported sandbox policy: {value!r}")


def parse_approval_policy(value: str) -> str:
    raw = str(value).strip()
    if raw in APPROVAL_NAMES:
        return raw
    raise ValueError(f"unsupported approval policy: {value!r}")


def normalize_policy(policy: dict[str, str]) -> dict[str, str]:
    return {
        "sandbox": parse_sandbox(policy["sandbox"]),
        "approval_policy": parse_approval_policy(policy["approval_policy"]),
    }


def preview_contract(policy: dict[str, str]) -> dict[str, str]:
    return normalize_policy(policy)
