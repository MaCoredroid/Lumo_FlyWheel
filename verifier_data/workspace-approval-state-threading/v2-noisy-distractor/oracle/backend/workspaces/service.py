from __future__ import annotations

from backend.workspaces.models import (
    DEFAULT_APPROVAL_STATE,
    DEFAULT_RISK_LEVEL,
    EXPLICIT_APPROVAL_SOURCE,
    LEGACY_APPROVAL_SOURCE,
)


def normalize_workspace_row(raw_row: dict, default_policy: dict | None = None) -> dict:
    default_state = (default_policy or {}).get('approval_state', DEFAULT_APPROVAL_STATE)
    approval_state = raw_row.get('approval_state') or default_state
    approval_state_source = EXPLICIT_APPROVAL_SOURCE if raw_row.get('approval_state') else LEGACY_APPROVAL_SOURCE
    return {
        'workspace_id': raw_row['workspace_id'],
        'workspace_name': raw_row['workspace_name'],
        'risk_level': raw_row.get('risk_level', DEFAULT_RISK_LEVEL),
        'approval_state': approval_state,
        'approval_state_source': approval_state_source,
    }
