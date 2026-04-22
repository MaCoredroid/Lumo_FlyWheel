import unittest
from pathlib import Path


class WorkspaceTableContractTests(unittest.TestCase):
    def test_table_mentions_risk_level(self) -> None:
        text = Path('frontend/src/components/WorkspaceTable.tsx').read_text()
        self.assertIn('risk_level', text)


if __name__ == '__main__':
    unittest.main()
