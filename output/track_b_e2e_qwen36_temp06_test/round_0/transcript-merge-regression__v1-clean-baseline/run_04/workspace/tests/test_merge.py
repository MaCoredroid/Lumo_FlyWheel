from __future__ import annotations

import unittest
from pathlib import Path

from replay.merge import merge_paths


ROOT = Path(__file__).resolve().parents[1]
SESSIONS = ROOT / "artifacts" / "sessions"


class MergeTests(unittest.TestCase):
    def test_same_name_tool_outputs_remain_distinct(self) -> None:
        merged = merge_paths(
            [
                SESSIONS / "visible_collision_part1.jsonl",
                SESSIONS / "visible_collision_part2.jsonl",
            ]
        )
        tool_ids = [event["event_id"] for event in merged if event.get("kind") == "tool_output"]
        self.assertEqual(tool_ids, ["tool-weather-01", "tool-weather-02"])


if __name__ == "__main__":
    unittest.main()
