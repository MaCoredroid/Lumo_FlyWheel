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
    normalized: list[dict] = []

    for fallback_sequence, event in enumerate(events, start=1):
        if event.get("type") != "response.output_item.added":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        sequence = event.get("sequence", fallback_sequence)
        if item_type == "message":
            text_parts: list[str] = []
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text = part.get("text", "")
                    if text:
                        text_parts.append(text)
            if text_parts:
                normalized.append(
                    {
                        "kind": "assistant_text",
                        "sequence": sequence,
                        "text": " ".join(text_parts).strip(),
                    }
                )
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
    normalized.sort(key=lambda item: item.get("sequence", 0))
    return normalized
