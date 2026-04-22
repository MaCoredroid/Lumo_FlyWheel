from __future__ import annotations

import os
import unittest
from pathlib import Path

from replay.incident_summary import summarize_paths
from replay.merge import merge_paths
from replay.render import render_paths


ROOT = Path(os.environ["VERIFIER_DATA_VARIANT_DIR"])
HIDDEN = ROOT.parent / "_shared" / "hidden_inputs"


class HiddenTranscriptChecks(unittest.TestCase):
    def test_same_name_tool_outputs_survive(self) -> None:
        merged = merge_paths(
            [
                HIDDEN / "same_name_collision_part1.jsonl",
                HIDDEN / "same_name_collision_part2.jsonl",
            ]
        )
        tool_ids = [event["event_id"] for event in merged if event.get("kind") == "tool_output"]
        self.assertEqual(tool_ids, ["tool-hidden-01", "tool-hidden-02"])

    def test_interleaved_fragments_keep_distinct_identity(self) -> None:
        merged = merge_paths(
            [
                HIDDEN / "interleaved_fragments_part1.jsonl",
                HIDDEN / "interleaved_fragments_part2.jsonl",
            ]
        )
        mapping = {event["event_id"]: event.get("content", "") for event in merged if event.get("kind") == "tool_output"}
        self.assertEqual(mapping, {"tool-query-01": "alpha tail", "tool-query-02": "bravo tail"})

    def test_deferred_output_survives_but_debug_noise_does_not(self) -> None:
        merged = merge_paths([HIDDEN / "deferred_output.jsonl"])
        tool_ids = [event["event_id"] for event in merged if event.get("kind") == "tool_output"]
        self.assertEqual(tool_ids, ["tool-log-01", "tool-log-02"])
        rendered = render_paths([HIDDEN / "deferred_output.jsonl"])
        self.assertIn("TOOL fetch_logs: live tail output", rendered)
        self.assertNotIn("debug replay duplicate", rendered)

    def test_summary_counts_merged_events(self) -> None:
        summary = summarize_paths(
            [
                HIDDEN / "same_name_collision_part1.jsonl",
                HIDDEN / "same_name_collision_part2.jsonl",
            ]
        )
        self.assertEqual(summary["count_source"], "merged_events")
        self.assertEqual(summary["tool_output_blocks"], 2)


if __name__ == "__main__":
    unittest.main()
