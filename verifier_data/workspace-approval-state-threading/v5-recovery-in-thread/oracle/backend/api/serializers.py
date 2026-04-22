from __future__ import annotations


def serialize_workspace(row: dict) -> dict:
    return {
        'workspace_id': row['workspace_id'],
        'workspace_name': row['workspace_name'],
        'risk_level': row['risk_level'],
        'approval_state': row['approval_state'],
        'approval_state_source': row['approval_state_source'],
    }
