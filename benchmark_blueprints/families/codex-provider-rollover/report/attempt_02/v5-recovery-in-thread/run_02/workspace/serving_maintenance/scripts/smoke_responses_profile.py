#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_PROVIDER = "responses_proxy"
EXPECTED_BASE_URL = "http://127.0.0.1:11434/v1/responses"
EXPECTED_PROFILE = "maintenance-responses"

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

def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    selected_provider = providers.get(selected, {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    turn1_id = turn1.get('response', {}).get('id')
    request_previous_response_id = turn2.get('request', {}).get('previous_response_id')
    response_previous_response_id = turn2.get('response', {}).get('previous_response_id')
    return {
        'profile': config.get('profile'),
        'selected_provider': selected,
        'base_url': selected_provider.get('base_url'),
        'turn1_id': turn1_id,
        'request_previous_response_id': request_previous_response_id,
        'response_previous_response_id': response_previous_response_id,
        'continuity_ok': (
            bool(turn1_id)
            and request_previous_response_id == turn1_id
            and response_previous_response_id == turn1_id
        ),
        'store_ok': selected_provider.get('store') is True and turn2.get('request', {}).get('store') is True,
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

    if payload['profile'] != EXPECTED_PROFILE:
        return fail(f"expected profile {EXPECTED_PROFILE}, got {payload['profile']!r}")
    if payload['selected_provider'] != EXPECTED_PROVIDER:
        return fail(f"expected provider {EXPECTED_PROVIDER}, got {payload['selected_provider']!r}")
    if payload['base_url'] != EXPECTED_BASE_URL:
        return fail(f"expected base_url {EXPECTED_BASE_URL}, got {payload['base_url']!r}")
    if not payload['store_ok']:
        return fail('store must remain true for the selected provider and the follow-up request')
    if not payload['continuity_ok']:
        return fail(
            'previous_response_id continuity failed: turn2 request/response must exactly match turn1 response id'
        )
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
