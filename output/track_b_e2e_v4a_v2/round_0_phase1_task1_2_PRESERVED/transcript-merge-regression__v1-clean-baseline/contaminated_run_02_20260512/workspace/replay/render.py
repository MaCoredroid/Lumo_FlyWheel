from __future__ import annotations

from pathlib import Path

from replay.merge import merge_paths


def render_events(events: list[dict]) -> str:
    lines: list[str] = []
    for event in sorted(events, key=lambda row: row.get("sequence", 0)):
        if event.get("kind") == "assistant":
            lines.append(f"ASSISTANT: {event.get('content', '').strip()}")
        elif event.get("kind") == "tool_output":
            lines.append(f"TOOL {event.get('tool_name')}: {event.get('content', '').strip()}")
    return "\n".join(line for line in lines if line)


def render_paths(paths: list[str | Path]) -> str:
    return render_events(merge_paths(paths))
