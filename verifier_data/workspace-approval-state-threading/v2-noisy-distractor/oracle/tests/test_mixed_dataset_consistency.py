import json
import unittest
from pathlib import Path

from backend.api.serializers import serialize_workspace
from backend.workspaces.service import normalize_workspace_row
from cli.export_workspace import export_workspace_snapshot


class MixedDatasetConsistencyTests(unittest.TestCase):
    def test_service_api_cli_agree(self) -> None:
        rows = json.loads(Path('seed_data/mixed_workspaces.json').read_text())
        normalized = [normalize_workspace_row(row, {'approval_state': 'manual_review'}) for row in rows]
        serialized = [serialize_workspace(row) for row in normalized]
        cli_rows = export_workspace_snapshot(rows, {'approval_state': 'manual_review'})
        self.assertEqual(serialized, cli_rows)
        legacy = serialized[2]
        self.assertEqual(legacy['approval_state'], 'manual_review')
        self.assertEqual(legacy['approval_state_source'], 'legacy_fallback')


if __name__ == '__main__':
    unittest.main()
