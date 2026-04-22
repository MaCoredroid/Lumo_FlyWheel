from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watchlist_report.renderers.json_renderer import render_json
from watchlist_report.service import build_report


class ServiceTests(unittest.TestCase):
    def test_json_renderer_returns_valid_json(self) -> None:
        payload = render_json(build_report(include_watchlist=False))
        parsed = json.loads(payload)
        self.assertIn("alerts", parsed)
        self.assertEqual(parsed["summary"]["total_symbols"], 4)


if __name__ == "__main__":
    unittest.main()
