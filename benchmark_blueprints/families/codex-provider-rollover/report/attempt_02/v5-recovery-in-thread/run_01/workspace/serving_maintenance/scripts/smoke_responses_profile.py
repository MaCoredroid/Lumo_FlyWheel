#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
        raise ValueError(message)

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    provider = providers.get(selected, {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    turn1_request = turn1.get('request', {})
    turn1_response = turn1.get('response', {})
    turn2_request = turn2.get('request', {})
    turn2_response = turn2.get('response', {})
    turn1_id = turn1_response.get('id')
    observed_previous_response_id = turn2_request.get('previous_response_id')
    response_previous_response_id = turn2_response.get('previous_response_id')

    require(selected == EXPECTED_PROVIDER, f"selected provider must be {EXPECTED_PROVIDER}, got {selected!r}")
    require(provider.get('base_url') == EXPECTED_BASE_URL, f"provider base_url must be {EXPECTED_BASE_URL}, got {provider.get('base_url')!r}")
    require(provider.get('wire_api') == 'responses', f"provider wire_api must be 'responses', got {provider.get('wire_api')!r}")
    require(provider.get('store') is True, f"provider store must be true, got {provider.get('store')!r}")

    require(turn1_request.get('provider') == EXPECTED_PROVIDER, f"turn1 provider must be {EXPECTED_PROVIDER}, got {turn1_request.get('provider')!r}")
    require(turn1_request.get('base_url') == EXPECTED_BASE_URL, f"turn1 base_url must be {EXPECTED_BASE_URL}, got {turn1_request.get('base_url')!r}")
    require(turn1_request.get('store') is True, f"turn1 store must be true, got {turn1_request.get('store')!r}")
    require(turn1_id, 'turn1 response is missing id')

    require(turn2_request.get('provider') == EXPECTED_PROVIDER, f"turn2 provider must be {EXPECTED_PROVIDER}, got {turn2_request.get('provider')!r}")
    require(turn2_request.get('base_url') == EXPECTED_BASE_URL, f"turn2 base_url must be {EXPECTED_BASE_URL}, got {turn2_request.get('base_url')!r}")
    require(turn2_request.get('store') is True, f"turn2 store must be true, got {turn2_request.get('store')!r}")
    require(observed_previous_response_id == turn1_id, f"turn2 request previous_response_id must equal turn1 id {turn1_id!r}, got {observed_previous_response_id!r}")
    require(response_previous_response_id == turn1_id, f"turn2 response previous_response_id must equal turn1 id {turn1_id!r}, got {response_previous_response_id!r}")

    return {
        'selected_provider': selected,
        'base_url': provider.get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': observed_previous_response_id,
        'response_previous_response_id': response_previous_response_id,
        'continuity_ok': observed_previous_response_id == turn1_id and response_previous_response_id == turn1_id,
        'store_ok': provider.get('store') is True and turn1_request.get('store') is True and turn2_request.get('store') is True,
        'status': 'ok',
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--turn1', type=Path, required=True)
    parser.add_argument('--turn2', type=Path, required=True)
    args = parser.parse_args()
    try:
        payload: dict[str, Any] = run_smoke(args.config, args.turn1, args.turn2)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
