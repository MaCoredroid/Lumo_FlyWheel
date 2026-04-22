from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def stable_event_identity(event: dict) -> str:
    if event.get("event_id"):
        return str(event["event_id"])
    return f"{event.get('kind')}:{event.get('sequence', 0)}:{event.get('tool_name', '')}"


def merge_records(records: list[dict]) -> list[dict]:
    merged: list[dict] = []
    by_key: dict[str, dict] = {}
    seen_completion = False
    for event in sorted(
        records,
        key=lambda row: (
            row.get("sequence", 0),
            row.get("chunk_index", 0),
            row.get("event_id", ""),
        ),
    ):
        if event.get("kind") == "response.completed":
            seen_completion = True
            continue
        if seen_completion and event.get("debug_only"):
            continue
        key = stable_event_identity(event)
        existing = by_key.get(key)
        if existing is None:
            current = dict(event)
            current["_parts"] = [(event.get("chunk_index", 0), event.get("content", ""))]
            by_key[key] = current
            merged.append(current)
            continue
        chunk = (event.get("chunk_index", 0), event.get("content", ""))
        if chunk not in existing["_parts"]:
            existing["_parts"].append(chunk)
        existing["sequence"] = max(existing.get("sequence", 0), event.get("sequence", 0))
        existing["debug_only"] = existing.get("debug_only", False) and event.get("debug_only", False)
    for event in merged:
        event["_parts"].sort(key=lambda item: item[0])
        event["content"] = "".join(piece for _, piece in event["_parts"])
        del event["_parts"]
    return merged


def merge_paths(paths: list[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return merge_records(records)
