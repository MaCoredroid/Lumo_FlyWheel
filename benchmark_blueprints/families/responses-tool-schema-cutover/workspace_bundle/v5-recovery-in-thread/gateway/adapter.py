from __future__ import annotations

import json
from pathlib import Path


def load_events(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def normalize_events(path: str | Path) -> list[dict]:
    events = load_events(path)
    assistant_chunks: list[str] = []
    calls_by_tool: dict[str, dict] = {}
    results_by_tool: dict[str, str] = {}

    for sequence, event in enumerate(events, start=1):
        if event.get("type") != "response.output_item.added":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    assistant_chunks.append(part.get("text", ""))
        elif item_type == "tool_call":
            tool_name = item.get("tool_name")
            calls_by_tool[tool_name] = {
                "kind": "tool_call",
                "sequence": sequence,
                "tool_name": tool_name,
                "arguments": item.get("arguments", ""),
            }
        elif item_type == "tool_result":
            results_by_tool[item.get("tool_name")] = item.get("output", "")

    normalized: list[dict] = []
    if assistant_chunks:
        normalized.append(
            {
                "kind": "assistant_text",
                "sequence": 0,
                "text": " ".join(chunk for chunk in assistant_chunks if chunk).strip(),
            }
        )

    for tool_name, call in calls_by_tool.items():
        normalized.append(call)
        if tool_name in results_by_tool:
            normalized.append(
                {
                    "kind": "tool_result",
                    "sequence": call["sequence"] + 100,
                    "tool_name": tool_name,
                    "output": results_by_tool[tool_name],
                }
            )
    return normalized
