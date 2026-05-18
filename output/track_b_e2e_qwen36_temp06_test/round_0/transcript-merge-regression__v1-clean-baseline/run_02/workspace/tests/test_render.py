from __future__ import annotations

import unittest
from pathlib import Path

from replay.render import render_paths


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "artifacts" / "sessions"


class RenderTests(unittest.TestCase):
    def test_post_completion_debug_only_fragment_is_not_rendered(self) -> None:
        rendered = render_paths([SESSIONS / "debug_after_completion.jsonl"])
        self.assertIn("TOOL fetch_logs: live tail output", rendered)
        self.assertNotIn("debug replay duplicate", rendered)


if __name__ == "__main__":
    unittest.main()
