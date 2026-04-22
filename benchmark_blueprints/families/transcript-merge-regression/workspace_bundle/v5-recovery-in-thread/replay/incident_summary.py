from __future__ import annotations

from pathlib import Path

from replay.merge import merge_paths
from replay.render import render_events


def summarize_events(events: list[dict]) -> dict:
    rendered = render_events(events)
    return {
        "count_source": "rendered_lines",
        "tool_output_blocks": sum(1 for line in rendered.splitlines() if line.startswith("TOOL ")),
        "assistant_blocks": sum(1 for line in rendered.splitlines() if line.startswith("ASSISTANT:")),
    }


def summarize_paths(paths: list[str | Path]) -> dict:
    return summarize_events(merge_paths(paths))
