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

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    provider = providers.get(selected, {})
    turn1_request = turn1.get('request', {})
    turn2_request = turn2.get('request', {})
    turn2_response = turn2.get('response', {})
    turn1_id = turn1.get('response', {}).get('id')
    request_previous_response_id = turn2_request.get('previous_response_id')
    response_previous_response_id = turn2_response.get('previous_response_id')
    continuity_ok = (
        bool(turn1_id)
        and request_previous_response_id == turn1_id
        and response_previous_response_id == turn1_id
    )
    store_ok = (
        provider.get('store') is True
        and turn1_request.get('store') is True
        and turn2_request.get('store') is True
    )
    return {
        'selected_provider': selected,
        'base_url': provider.get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': request_previous_response_id,
        'response_previous_response_id': response_previous_response_id,
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
    payload = run_smoke(args.config, args.turn1, args.turn2)
    print(json.dumps(payload, indent=2, sort_keys=True))
    require(payload['selected_provider'] == EXPECTED_PROVIDER, 'selected provider must be responses_proxy')
    require(payload['base_url'] == EXPECTED_BASE_URL, 'base_url must target the canonical responses proxy')
    require(payload['store_ok'], 'store must remain enabled for both turns and the selected provider')
    require(
        payload['continuity_ok'],
        'previous_response_id must exactly match the prior response id in both the request and follow-up response',
    )
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
