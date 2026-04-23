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
    normalized_items: list[dict] = []

    for event in events:
        if event.get("type") != "response.output_item.added":
            continue
        sequence = event.get("sequence", 0)
        item = event.get("item", {})
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    assistant_chunks.append(part.get("text", ""))
        elif item_type == "tool_call":
            normalized_items.append(
                {
                    "kind": "tool_call",
                    "sequence": sequence,
                    "call_id": item.get("call_id"),
                    "tool_name": item.get("tool_name"),
                    "arguments": item.get("arguments", ""),
                }
            )
        elif item_type == "tool_result":
            normalized_items.append(
                {
                    "kind": "tool_result",
                    "sequence": sequence,
                    "call_id": item.get("call_id"),
                    "tool_name": item.get("tool_name"),
                    "output": item.get("output", ""),
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
    normalized.extend(normalized_items)
    return normalized
