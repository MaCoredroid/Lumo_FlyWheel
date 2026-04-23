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
    events = sorted(load_events(path), key=lambda event: event.get("sequence", 0))
    assistant_chunks: list[str] = []
    tool_items: list[dict] = []
    assistant_sequence: int | None = None

    for event in events:
        if event.get("type") != "response.output_item.added":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        sequence = event.get("sequence", 0)
        if item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text = part.get("text", "")
                    if text:
                        assistant_chunks.append(text)
                        if assistant_sequence is None:
                            assistant_sequence = sequence
        elif item_type == "tool_call":
            tool_items.append(
                {
                    "kind": "tool_call",
                    "sequence": sequence,
                    "tool_name": item.get("tool_name"),
                    "arguments": item.get("arguments", ""),
                    "call_id": item.get("call_id"),
                }
            )
        elif item_type == "tool_result":
            tool_items.append(
                {
                    "kind": "tool_result",
                    "sequence": sequence,
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
                "sequence": assistant_sequence or 0,
                "text": " ".join(chunk for chunk in assistant_chunks if chunk).strip(),
            }
        )

    normalized.extend(tool_items)
    return normalized
