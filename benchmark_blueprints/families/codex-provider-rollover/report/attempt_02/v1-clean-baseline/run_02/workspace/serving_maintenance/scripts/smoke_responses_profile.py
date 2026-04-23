#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_PROVIDER = "responses_proxy"
EXPECTED_BASE_URL = "http://127.0.0.1:11434/v1/responses"

def load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def parse_scalar(raw: str):
    text = raw.strip()
    if text in {'true', 'false'}:
        return text == 'true'
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
        if not line or line.startswith('#'):
            continue
        if line.startswith('[') and line.endswith(']'):
            current = data
            for part in line[1:-1].split('.'):
                current = current.setdefault(part, {})
            continue
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        current[key.strip()] = parse_scalar(value)
    return data

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)

def require_request_shape(label: str, payload: dict, expected_previous_response_id: str | None) -> None:
    request = payload.get('request', {})
    require(request.get('provider') == EXPECTED_PROVIDER, f"{label} request provider must be {EXPECTED_PROVIDER}")
    require(request.get('base_url') == EXPECTED_BASE_URL, f"{label} request base_url must be {EXPECTED_BASE_URL}")
    require(request.get('store') is True, f"{label} request must set store=true")
    observed_previous = request.get('previous_response_id')
    require(
        observed_previous == expected_previous_response_id,
        f"{label} request previous_response_id must exactly match {expected_previous_response_id!r}, got {observed_previous!r}",
    )

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    selected_provider = providers.get(selected, {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    require(selected == EXPECTED_PROVIDER, f"selected provider must be {EXPECTED_PROVIDER}, got {selected!r}")
    require(selected_provider.get('base_url') == EXPECTED_BASE_URL, f"provider base_url must be {EXPECTED_BASE_URL}")
    require(selected_provider.get('wire_api') == 'responses', "provider wire_api must be responses")
    require(selected_provider.get('store') is True, "provider must set store=true for maintained follow-up chaining")

    require_request_shape("turn1", turn1, None)
    turn1_response = turn1.get('response', {})
    turn1_id = turn1_response.get('id')
    require(turn1_id, "turn1 response id is required")
    require(turn1_response.get('status') == 'completed', "turn1 response status must be completed")

    require_request_shape("turn2", turn2, turn1_id)
    turn2_response = turn2.get('response', {})
    require(turn2_response.get('status') == 'completed', "turn2 response status must be completed")
    require(
        turn2_response.get('previous_response_id') == turn1_id,
        "turn2 response previous_response_id must echo the exact turn1 response id",
    )

    observed_previous = turn2.get('request', {}).get('previous_response_id')
    continuity_ok = observed_previous == turn1_id
    store_ok = (
        selected_provider.get('store') is True
        and turn1.get('request', {}).get('store') is True
        and turn2.get('request', {}).get('store') is True
    )
    return {
        'selected_provider': selected,
        'base_url': selected_provider.get('base_url'),
        'turn1_id': turn1_id,
        'turn2_id': turn2_response.get('id'),
        'expected_previous_response_id': turn1_id,
        'observed_previous_response_id': observed_previous,
        'response_previous_response_id': turn2_response.get('previous_response_id'),
        'continuity_ok': continuity_ok,
        'store_ok': store_ok,
        'status': 'ok',
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--turn1', type=Path, required=True)
    parser.add_argument('--turn2', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config, args.turn1, args.turn2), indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
