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
    if event.get("kind") == "tool_output":
        # BUG: tool outputs need stable event identity, not role/name grouping.
        return (event.get("role"), event.get("tool_name"))
    return event.get("event_id") or f"{event.get('kind')}:{event.get('sequence', 0)}"


def merge_records(records: list[dict]) -> list[dict]:
    merged: list[dict] = []
    by_key: dict[object, dict] = {}
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
            existing["debug_only"] = existing.get("debug_only", False) or event.get("debug_only", False)
        if seen_completion and event.get("debug_only"):
            # BUG: debug-only fragments after completion still survive as
            # renderable tool blocks.
            by_key[key]["after_completion"] = True
    for event in merged:
        event["content"] = "".join(event.pop("content_parts", []))
    return merged


def merge_paths(paths: list[str | Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return merge_records(records)
