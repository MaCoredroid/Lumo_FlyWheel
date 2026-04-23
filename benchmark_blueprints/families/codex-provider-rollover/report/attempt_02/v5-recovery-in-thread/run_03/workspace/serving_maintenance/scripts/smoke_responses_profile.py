#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_PROVIDER = "responses_proxy"
EXPECTED_BASE_URL = "http://127.0.0.1:11434/v1/responses"


class SmokeError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_scalar(raw: str):
    text = raw.strip()
    if text in {"true", "false"}:
        return text == "true"
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        return text


def load_config(path: Path) -> dict:
    data = {}
    current = data
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = data
            for part in line[1:-1].split("."):
                current = current.setdefault(part, {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        current[key.strip()] = parse_scalar(value)
    return data


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def validate_profile(config: dict) -> tuple[str, dict]:
    selected = config.get("provider")
    providers = config.get("model_providers", {})
    provider = providers.get(selected)
    require(
        selected == EXPECTED_PROVIDER,
        f"selected provider must be {EXPECTED_PROVIDER!r}, got {selected!r}",
    )
    require(provider is not None, f"missing provider config for {selected!r}")
    require(
        provider.get("base_url") == EXPECTED_BASE_URL,
        f"provider {selected!r} must use {EXPECTED_BASE_URL!r}",
    )
    require(
        provider.get("wire_api") == "responses",
        f"provider {selected!r} must use wire_api='responses'",
    )
    require(
        provider.get("store") is True,
        f"provider {selected!r} must set store = true for follow-up continuity",
    )
    return selected, provider


def validate_turn_request(request: dict, *, label: str, selected: str) -> None:
    require(request.get("provider") == selected, f"{label} request provider mismatch")
    require(
        request.get("base_url") == EXPECTED_BASE_URL,
        f"{label} request base_url mismatch",
    )
    require(request.get("store") is True, f"{label} request store must be true")


def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected, provider = validate_profile(config)
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)

    turn1_request = turn1.get("request", {})
    turn1_response = turn1.get("response", {})
    turn2_request = turn2.get("request", {})
    turn2_response = turn2.get("response", {})

    validate_turn_request(turn1_request, label="turn1", selected=selected)
    validate_turn_request(turn2_request, label="turn2", selected=selected)

    turn1_id = turn1_response.get("id")
    request_previous_response_id = turn2_request.get("previous_response_id")
    response_previous_response_id = turn2_response.get("previous_response_id")

    require(turn1_id, "turn1 response must include an id")
    require(
        request_previous_response_id == turn1_id,
        "turn2 request previous_response_id must exactly match turn1 response id",
    )
    require(
        response_previous_response_id == turn1_id,
        "turn2 response previous_response_id must exactly match turn1 response id",
    )

    return {
        "selected_provider": selected,
        "base_url": provider.get("base_url"),
        "turn1_id": turn1_id,
        "observed_previous_response_id": request_previous_response_id,
        "response_previous_response_id": response_previous_response_id,
        "continuity_ok": True,
        "store_ok": True,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--turn1", type=Path, required=True)
    parser.add_argument("--turn2", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = run_smoke(args.config, args.turn1, args.turn2)
    except SmokeError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
