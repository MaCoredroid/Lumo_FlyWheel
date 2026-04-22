from __future__ import annotations

from backend.workspaces.models import DEFAULT_RISK_LEVEL


def normalize_workspace_row(raw_row: dict, default_policy: dict | None = None) -> dict:
    normalized = {
        "workspace_id": raw_row["workspace_id"],
        "workspace_name": raw_row["workspace_name"],
        "risk_level": raw_row.get("risk_level", DEFAULT_RISK_LEVEL),
    }
    if raw_row.get("approval_state"):
        normalized["approval_state"] = raw_row["approval_state"]
    if raw_row.get("approval_mode"):
        normalized["approval_mode"] = raw_row["approval_mode"]
    return normalized
