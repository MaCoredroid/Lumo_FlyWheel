from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def _event_key(event: dict) -> object:
    # Use event_id for stable identity - applies to all event kinds including tool_output
    return event.get("event_id") or f"{event.get('kind')}:{event.get('sequence', 0)}"


def merge_records(records: list[dict]) -> list[dict]:
    merged: list[dict] = []
    by_key: dict[object, dict] = {}
    for event in sorted(
        records,
        key=lambda row: (
            row.get("sequence", 0),
            row.get("chunk_index", 0),
            row.get("event_id", ""),
        ),
    ):
        # Skip completion events - they are not rendered
        if event.get("kind") == "response.completed":
            continue
        # Skip debug_only events - they should not appear in output
        if event.get("debug_only"):
            continue
        key = _event_key(event)
        existing = by_key.get(key)
        if existing is None:
            current = dict(event)
            current["content_parts"] = [event.get("content", "")]
            by_key[key] = current
            merged.append(current)
        else:
            existing["content_parts"].append(event.get("content", ""))
            existing["sequence"] = max(existing.get("sequence", 0), event.get("sequence", 0))
    for event in merged:
        event["content"] = "".join(event.pop("content_parts", []))
    return merged


def merge_paths(paths: list[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return merge_records(records)
