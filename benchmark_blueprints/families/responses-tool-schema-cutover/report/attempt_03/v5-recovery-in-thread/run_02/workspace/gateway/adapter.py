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
    assistant_sequence: int | None = None
    normalized: list[dict] = []

    for fallback_sequence, event in enumerate(events, start=1):
        sequence = event.get("sequence", fallback_sequence)
        if event.get("type") != "response.output_item.added":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        if item_type == "message":
            if assistant_sequence is None:
                assistant_sequence = sequence
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    assistant_chunks.append(part.get("text", ""))
        elif item_type == "tool_call":
            normalized.append(
                {
                    "kind": "tool_call",
                    "sequence": sequence,
                    "tool_name": item.get("tool_name"),
                    "arguments": item.get("arguments", ""),
                    "call_id": item.get("call_id"),
                }
            )
        elif item_type == "tool_result":
            normalized.append(
                {
                    "kind": "tool_result",
                    "sequence": sequence,
                    "tool_name": item.get("tool_name"),
                    "output": item.get("output", ""),
                    "call_id": item.get("call_id"),
                }
            )
    if assistant_chunks:
        normalized.insert(
            0,
            {
                "kind": "assistant_text",
                "sequence": assistant_sequence or 0,
                "text": " ".join(chunk for chunk in assistant_chunks if chunk).strip(),
            },
        )
    return normalized
