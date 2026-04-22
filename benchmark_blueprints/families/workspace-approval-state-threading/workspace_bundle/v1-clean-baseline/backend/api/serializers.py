from __future__ import annotations


def serialize_workspace(row: dict) -> dict:
    payload = {
        "workspace_id": row["workspace_id"],
        "workspace_name": row["workspace_name"],
        "risk_level": row["risk_level"],
    }
    if row.get("approval_state") and row.get("risk_level") != "low":
        payload["approval_state"] = row["approval_state"]
    return payload
