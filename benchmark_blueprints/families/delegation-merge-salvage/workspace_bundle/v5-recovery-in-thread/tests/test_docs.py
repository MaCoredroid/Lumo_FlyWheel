from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DocsTests(unittest.TestCase):
    def test_usage_has_json_example(self) -> None:
        text = (ROOT / "docs" / "usage.md").read_text()
        self.assertIn("--format json", text)


if __name__ == "__main__":
    unittest.main()
