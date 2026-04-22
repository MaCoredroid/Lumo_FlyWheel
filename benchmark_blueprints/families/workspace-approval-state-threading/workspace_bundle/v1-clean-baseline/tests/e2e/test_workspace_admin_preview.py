import json
import unittest
from pathlib import Path


class WorkspacePreviewTests(unittest.TestCase):
    def test_preview_uses_old_name(self) -> None:
        data = json.loads(Path('artifacts/preview/workspace_admin_capture.json').read_text())
        self.assertEqual(data['screenshot_name'], 'workspace-admin-risk-level.png')


if __name__ == '__main__':
    unittest.main()
