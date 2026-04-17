from __future__ import annotations

from collections.abc import Iterable

def fingerprint(event: dict[str, str]) -> str:
    return f"{event['service']}::{event['title']}"


def collapse(events: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, str | int]] = {}
    for event in events:
        key = fingerprint(event)
        current = grouped.get(key)
        if current is None:
            grouped[key] = {
                "environment": event["environment"],
                "service": event["service"],
                "title": event["title"],
                "payment_lane": event["payment_lane"],
                "dedupe_hint": event["dedupe_hint"],
                "window_start": event["window_start"],
                "occurrence_count": 1,
                "first_seen_at": event["observed_at"],
                "last_seen_at": event["observed_at"],
            }
            continue
        current["occurrence_count"] = int(current["occurrence_count"]) + 1
        current["first_seen_at"] = min(str(current["first_seen_at"]), event["observed_at"])
        current["last_seen_at"] = max(str(current["last_seen_at"]), event["observed_at"])
    return [_coerce_group(record) for record in grouped.values()]


def _coerce_group(record: dict[str, str | int]) -> dict[str, str | int]:
    return {
        "environment": str(record["environment"]),
        "service": str(record["service"]),
        "title": str(record["title"]),
        "payment_lane": str(record["payment_lane"]),
        "dedupe_hint": str(record["dedupe_hint"]),
        "window_start": str(record["window_start"]),
        "occurrence_count": int(record["occurrence_count"]),
        "first_seen_at": str(record["first_seen_at"]),
        "last_seen_at": str(record["last_seen_at"]),
    }
