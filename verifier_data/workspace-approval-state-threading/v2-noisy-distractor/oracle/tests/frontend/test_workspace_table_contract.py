import unittest
from pathlib import Path


class WorkspaceTableContractTests(unittest.TestCase):
    def test_table_mentions_approval_state_and_fallback(self) -> None:
        text = Path('frontend/src/components/WorkspaceTable.tsx').read_text()
        self.assertIn('approval_state', text)
        self.assertIn('Approval state', text)
        self.assertIn('manual_review', text)


if __name__ == '__main__':
    unittest.main()
