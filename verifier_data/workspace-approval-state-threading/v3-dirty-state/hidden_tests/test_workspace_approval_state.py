from __future__ import annotations

import json
import os
import sys
from pathlib import Path
import importlib.util


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    agent_ws = Path(os.environ['AGENT_WS'])
    if str(agent_ws) not in sys.path:
        sys.path.insert(0, str(agent_ws))
    gold = json.loads(Path(os.environ['GOLD_FILE']).read_text())
    rows = json.loads((agent_ws / 'seed_data' / 'mixed_workspaces.json').read_text())
    service = _load_module('service_mod', agent_ws / 'backend' / 'workspaces' / 'service.py')
    serializers = _load_module('serializers_mod', agent_ws / 'backend' / 'api' / 'serializers.py')
    cli = _load_module('cli_mod', agent_ws / 'cli' / 'export_workspace.py')
    normalized = [service.normalize_workspace_row(row, {'approval_state': 'manual_review'}) for row in rows]
    serialized = [serializers.serialize_workspace(row) for row in normalized]
    assert cli.export_workspace_snapshot(rows, {'approval_state': 'manual_review'}) == serialized
    legacy = serialized[2]
    assert legacy['approval_state'] == 'manual_review'
    assert legacy['approval_state_source'] == 'legacy_fallback'
    preview = json.loads((agent_ws / 'artifacts' / 'preview' / 'workspace_admin_capture.json').read_text())
    assert preview['filtered_row']['workspace_id'] == gold['preview_filtered_row']['workspace_id']
    assert preview['filtered_row']['approval_state'] == gold['preview_filtered_row']['approval_state']


if __name__ == '__main__':
    main()
