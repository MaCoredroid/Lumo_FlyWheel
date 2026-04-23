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
    tool_calls: list[dict] = []
    tool_results: list[dict] = []

    for sequence, event in enumerate(events, start=1):
        if event.get("type") != "response.output_item.added":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        item_sequence = event.get("sequence", sequence)
        if item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    assistant_chunks.append(part.get("text", ""))
        elif item_type == "tool_call":
            tool_calls.append(
                {
                    "kind": "tool_call",
                    "sequence": item_sequence,
                    "tool_name": item.get("tool_name"),
                    "arguments": item.get("arguments", ""),
                    "call_id": item.get("call_id"),
                }
            )
        elif item_type == "tool_result":
            tool_results.append(
                {
                    "kind": "tool_result",
                    "sequence": item_sequence,
                    "tool_name": item.get("tool_name"),
                    "output": item.get("output", ""),
                    "call_id": item.get("call_id"),
                }
            )

    normalized: list[dict] = []
    if assistant_chunks:
        normalized.append(
            {
                "kind": "assistant_text",
                "sequence": 0,
                "text": " ".join(chunk for chunk in assistant_chunks if chunk).strip(),
            }
        )

    normalized.extend(tool_calls)
    normalized.extend(tool_results)
    return normalized
