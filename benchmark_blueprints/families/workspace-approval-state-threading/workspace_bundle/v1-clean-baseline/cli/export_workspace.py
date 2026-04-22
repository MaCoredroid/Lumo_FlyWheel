from __future__ import annotations

import json
from pathlib import Path

from backend.workspaces.service import normalize_workspace_row

SEED_PATH = Path(__file__).resolve().parents[1] / 'seed_data' / 'mixed_workspaces.json'


def export_workspace_snapshot(rows: list[dict], default_policy: dict | None = None) -> list[dict]:
    export_rows = []
    for row in rows:
        normalized = normalize_workspace_row(row, default_policy=default_policy)
        export_rows.append({
            'workspace_id': normalized['workspace_id'],
            'workspace_name': normalized['workspace_name'],
            'risk_level': normalized['risk_level'],
        })
    return export_rows


def main() -> None:
    rows = json.loads(SEED_PATH.read_text())
    print(json.dumps(export_workspace_snapshot(rows), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
