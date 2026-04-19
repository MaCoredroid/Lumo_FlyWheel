from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


DEFAULT_PROMPT = (
    "Open the review digest inbox item and continue the review-digest workflow in "
    "this thread."
)
DEFAULT_RRULE = "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=9;BYMINUTE=0"


def _coerce_str(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _read_input(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    raw = source.read_text()
    if source.suffix == ".json":
        payload = json.loads(raw)
    elif source.suffix == ".toml":
        payload = tomllib.loads(raw)
    else:
        raise ValueError(f"unsupported input format: {source.suffix or '<none>'}")

    if not isinstance(payload, dict):
        raise TypeError("automation payload must deserialize to an object")
    return payload


def _extract_destination(payload: dict[str, Any]) -> str:
    destination = _first_present(payload, "destination", "target", "resume_destination")
    if isinstance(destination, dict):
        destination = (
            destination.get("type")
            or destination.get("kind")
            or destination.get("destination")
            or destination.get("target")
        )
    if isinstance(destination, str) and destination.strip().lower() == "thread":
        return "thread"

    if any(
        key in payload
        for key in ("thread_id", "threadId", "raw_thread_id", "resume_thread_id")
    ):
        return "thread"

    return "thread"


def _extract_status(payload: dict[str, Any]) -> str:
    status = _first_present(payload, "status", "state")
    if status is not None:
        text = _coerce_str(status, default="ACTIVE").strip().upper()
        return text or "ACTIVE"

    paused = _first_present(payload, "paused", "is_paused")
    if isinstance(paused, bool):
        return "PAUSED" if paused else "ACTIVE"

    return "ACTIVE"


def _extract_prompt(payload: dict[str, Any]) -> str:
    prompt = _first_present(payload, "prompt", "task_prompt", "prompt_text", "body")
    text = _coerce_str(prompt, default=DEFAULT_PROMPT)
    return text or DEFAULT_PROMPT


def _extract_rrule(payload: dict[str, Any]) -> str:
    schedule = _first_present(payload, "rrule", "schedule", "cadence")
    text = _coerce_str(schedule, default=DEFAULT_RRULE)
    return text or DEFAULT_RRULE


def normalize_heartbeat(payload: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {
        "name": _coerce_str(_first_present(payload, "name", "title"), default="Review Digest"),
        "kind": "heartbeat",
        "destination": _extract_destination(payload),
        "status": _extract_status(payload),
        "rrule": _extract_rrule(payload),
        "prompt": _extract_prompt(payload),
    }

    return normalized


def _toml_line(key: str, value: str) -> str:
    return f"{key} = {json.dumps(value)}"


def serialize_heartbeat(payload: dict[str, Any]) -> str:
    normalized = normalize_heartbeat(payload)
    lines = [
        _toml_line("name", normalized["name"]),
        _toml_line("kind", normalized["kind"]),
        _toml_line("destination", normalized["destination"]),
        _toml_line("status", normalized["status"]),
        _toml_line("rrule", normalized["rrule"]),
        _toml_line("prompt", normalized["prompt"]),
    ]
    return "\n".join(lines) + "\n"


def dump_review_digest() -> str:
    return serialize_heartbeat({})


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        sys.stdout.write(dump_review_digest())
        return 0

    payload = _read_input(args[0])
    sys.stdout.write(serialize_heartbeat(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
