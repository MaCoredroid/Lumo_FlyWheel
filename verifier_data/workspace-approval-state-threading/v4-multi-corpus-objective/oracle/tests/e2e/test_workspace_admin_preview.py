import json
import unittest
from pathlib import Path


class WorkspacePreviewTests(unittest.TestCase):
    def test_preview_matches_current_contract(self) -> None:
        data = json.loads(Path('artifacts/preview/workspace_admin_capture.json').read_text())
        self.assertEqual(data['screenshot_name'], 'workspace-admin-approval-state.png')
        self.assertIn('approval_state', data['columns'])
        self.assertEqual(data['filtered_row']['workspace_id'], 'ws-blocked-02')
        self.assertEqual(data['filtered_row']['approval_state'], 'blocked')


if __name__ == '__main__':
    unittest.main()
