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

    for event in events:
        sequence = event.get("sequence", 0)
        if event.get("type") != "response.output_item.added":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        if item_type == "message":
            text_parts: list[str] = []
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text_parts.append(part.get("text", ""))
            text = " ".join(chunk for chunk in text_parts if chunk).strip()
            if text:
                normalized.append(
                    {
                        "kind": "assistant_text",
                        "sequence": sequence,
                        "text": text,
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
    return normalized
