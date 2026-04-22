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
    normalized: list[dict] = []
    for fallback_sequence, event in enumerate(load_events(path), start=1):
        event_type = event.get("type")
        sequence = int(event.get("sequence", fallback_sequence))
        if event_type != "response.output_item.added":
            continue
        item = event.get("item", {})
        item_type = item.get("type")
        if item_type == "message":
            parts = [
                part.get("text", "")
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            ]
            text = " ".join(part for part in parts if part).strip()
            if text:
                normalized.append(
                    {
                        "kind": "assistant_text",
                        "sequence": sequence,
                        "text": text,
                    }
                )
        elif item_type in {"tool_call", "function_call"}:
            normalized.append(
                {
                    "kind": "tool_call",
                    "sequence": sequence,
                    "call_id": item.get("call_id"),
                    "tool_name": item.get("tool_name") or item.get("name"),
                    "arguments": item.get("arguments", ""),
                }
            )
        elif item_type in {"tool_result", "function_call_output"}:
            normalized.append(
                {
                    "kind": "tool_result",
                    "sequence": sequence,
                    "call_id": item.get("call_id"),
                    "tool_name": item.get("tool_name") or item.get("name"),
                    "output": item.get("output", ""),
                }
            )
    normalized.sort(key=lambda item: (item["sequence"], item["kind"]))
    return normalized
