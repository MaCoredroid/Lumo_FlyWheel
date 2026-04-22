
from __future__ import annotations

OLD_SANDBOX_NAMES = {
    "read-only": "read_only",
    "workspace-write": "workspace_write",
    "danger-full-access": "danger_full_access",
}
APPROVAL_NAMES = {"never", "on_request", "on_failure", "untrusted"}


def parse_sandbox(value: str) -> str:
    raw = str(value).strip()
    if raw in OLD_SANDBOX_NAMES:
        return OLD_SANDBOX_NAMES[raw]
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
    parsed = normalize_policy(policy)
    approval = parsed["approval_policy"]
    if approval == "on_request":
        approval = "manual-review"
    return {
        "sandbox": parsed["sandbox"],
        "approval_policy": approval,
    }
