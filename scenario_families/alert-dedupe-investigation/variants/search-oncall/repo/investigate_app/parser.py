from __future__ import annotations

import json
from pathlib import Path


def load_events(path: str = "logs/failure.log") -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        raw = json.loads(line)
        observed_at = raw.get("observed_at") or raw["window_start"]
        events.append(
            {
                "environment": raw["environment"],
                "service": raw["service"].strip(),
                "title": raw["title"].strip(),
                "search_cluster": raw["search_cluster"].strip(),
                "dedupe_hint": raw.get("dedupe_hint", "").strip(),
                "window_start": raw.get("window_start", observed_at),
                "observed_at": observed_at,
            }
        )
    return events
