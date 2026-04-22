from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from watchlist_report.cli import main


class CLITests(unittest.TestCase):
    def test_markdown_includes_watchlist_follow_up(self) -> None:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            rc = main(["--format", "markdown", "--include-watchlist"])
        self.assertEqual(rc, 0)
        out = stream.getvalue()
        self.assertIn("## Watchlist Follow-Up", out)
        self.assertIn("AAPL", out)


if __name__ == "__main__":
    unittest.main()
