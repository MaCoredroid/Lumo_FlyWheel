from __future__ import annotations

from datetime import UTC, datetime


ENVIRONMENT_ALIASES = {
    "prod": "prod",
    "production": "prod",
    "staging": "staging",
}


def canonical_environment(value: str) -> str:
    token = " ".join(value.strip().split()).casefold()
    return ENVIRONMENT_ALIASES.get(token, token)


def canonical_window_start(observed_at: str) -> str:
    timestamp = datetime.fromisoformat(observed_at.replace("Z", "+00:00")).astimezone(UTC)
    window_minute = timestamp.minute - (timestamp.minute % 5)
    return timestamp.replace(minute=window_minute, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_title(value: str) -> str:
    return " ".join(value.strip().split())


def incident_identity(event: dict[str, str]) -> str:
    hint = event.get("dedupe_hint", "").strip()
    if hint:
        return hint.casefold()
    return canonical_title(event["title"]).casefold()
