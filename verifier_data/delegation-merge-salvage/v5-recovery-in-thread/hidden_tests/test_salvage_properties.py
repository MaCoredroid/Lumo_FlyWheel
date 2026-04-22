from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watchlist_report.renderers.json_renderer import render_json
from watchlist_report.renderers.markdown_renderer import render_markdown
from watchlist_report.service import build_report


class HiddenSalvageTests(unittest.TestCase):
    def test_json_contract_fixture(self) -> None:
        expected = (ROOT / "tests" / "fixtures" / "json" / "baseline_report.json").read_text()
        actual = render_json(build_report(include_watchlist=False))
        self.assertEqual(actual, expected)

    def test_markdown_watchlist(self) -> None:
        out = render_markdown(build_report(include_watchlist=True))
        self.assertIn("## Watchlist Follow-Up", out)
        self.assertIn("AAPL", out)


if __name__ == "__main__":
    unittest.main()
