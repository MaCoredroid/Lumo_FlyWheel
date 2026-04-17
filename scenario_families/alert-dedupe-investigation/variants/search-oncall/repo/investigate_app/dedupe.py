from __future__ import annotations


def fingerprint(event: dict[str, str]) -> str:
    return f"{event['service']}::{event['title']}"


def collapse(events: list[dict[str, str]]) -> list[dict[str, str | int]]:
    grouped: dict[str, dict[str, str]] = {}
    for event in events:
        grouped[fingerprint(event)] = event
    return list(grouped.values())
