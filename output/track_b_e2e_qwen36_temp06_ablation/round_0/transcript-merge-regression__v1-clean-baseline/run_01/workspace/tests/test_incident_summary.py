from __future__ import annotations

import unittest
from pathlib import Path

from replay.incident_summary import summarize_paths


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "artifacts" / "sessions"


class SummaryTests(unittest.TestCase):
    def test_incident_summary_counts_merged_tool_blocks(self) -> None:
        summary = summarize_paths(
            [
                SESSIONS / "visible_collision_part1.jsonl",
                SESSIONS / "visible_collision_part2.jsonl",
            ]
        )
        self.assertEqual(summary["count_source"], "merged_events")
        self.assertEqual(summary["tool_output_blocks"], 2)


if __name__ == "__main__":
    unittest.main()
