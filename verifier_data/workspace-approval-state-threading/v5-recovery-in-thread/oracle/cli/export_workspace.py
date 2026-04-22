from __future__ import annotations

import json
from pathlib import Path

from backend.api.serializers import serialize_workspace
from backend.workspaces.service import normalize_workspace_row

SEED_PATH = Path(__file__).resolve().parents[1] / 'seed_data' / 'mixed_workspaces.json'
DEFAULT_POLICY = {'approval_state': 'manual_review'}


def export_workspace_snapshot(rows: list[dict], default_policy: dict | None = None) -> list[dict]:
    policy = default_policy or DEFAULT_POLICY
    return [serialize_workspace(normalize_workspace_row(row, policy)) for row in rows]


def main() -> None:
    rows = json.loads(SEED_PATH.read_text())
    print(json.dumps(export_workspace_snapshot(rows), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
