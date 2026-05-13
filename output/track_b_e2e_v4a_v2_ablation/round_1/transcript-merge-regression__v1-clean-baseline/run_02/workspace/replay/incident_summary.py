from __future__ import annotations

from pathlib import Path

from replay.merge import merge_paths


def summarize_events(events: list[dict]) -> dict:
    return {
        "count_source": "merged_events",
        "tool_output_blocks": sum(1 for event in events if event.get("kind") == "tool_output"),
        "assistant_blocks": sum(1 for event in events if event.get("kind") == "assistant"),
    }


def summarize_paths(paths: list[str | Path]) -> dict:
    return summarize_events(merge_paths(paths))
