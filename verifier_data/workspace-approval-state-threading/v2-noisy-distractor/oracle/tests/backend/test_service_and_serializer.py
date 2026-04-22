import json
import unittest
from pathlib import Path

from backend.api.serializers import serialize_workspace
from backend.workspaces.service import normalize_workspace_row


class WorkspaceContractTests(unittest.TestCase):
    def test_legacy_row_fallback_and_serializer(self) -> None:
        rows = json.loads(Path('seed_data/mixed_workspaces.json').read_text())
        legacy = normalize_workspace_row(rows[2], {'approval_state': 'manual_review'})
        self.assertEqual(legacy['approval_state'], 'manual_review')
        self.assertEqual(legacy['approval_state_source'], 'legacy_fallback')
        payload = serialize_workspace(legacy)
        self.assertEqual(payload['approval_state'], 'manual_review')
        self.assertEqual(payload['approval_state_source'], 'legacy_fallback')


if __name__ == '__main__':
    unittest.main()
