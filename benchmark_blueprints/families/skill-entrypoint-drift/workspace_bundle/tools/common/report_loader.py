from __future__ import annotations

import json
from pathlib import Path


def load_events(path: str) -> list[dict[str, str]]:
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list):
        raise ValueError("expected a JSON list of events")

    events: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("expected each event to be a JSON object")
        events.append({str(key): str(value) for key, value in item.items()})
    return events
