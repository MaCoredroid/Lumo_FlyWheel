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

def fail(message: str, **payload) -> int:
    details = {**payload, "status": "error", "error": message}
    print(json.dumps(details, indent=2, sort_keys=True))
    return 1

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    turn1_id = turn1.get('response', {}).get('id')
    provider = providers.get(selected, {})
    observed_previous_response_id = turn2.get('request', {}).get('previous_response_id')
    response_previous_response_id = turn2.get('response', {}).get('previous_response_id')
    store_ok = bool(provider.get('store')) and bool(turn1.get('request', {}).get('store')) and bool(turn2.get('request', {}).get('store'))
    continuity_ok = (
        bool(turn1_id)
        and observed_previous_response_id == turn1_id
        and response_previous_response_id == turn1_id
    )
    return {
        'selected_provider': selected,
        'base_url': provider.get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': observed_previous_response_id,
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
    if payload['selected_provider'] != EXPECTED_PROVIDER:
        return fail(
            'selected provider is not the canonical maintenance profile',
            **payload,
            expected_provider=EXPECTED_PROVIDER,
        )
    if payload['base_url'] != EXPECTED_BASE_URL:
        return fail(
            'selected provider base_url is not the canonical responses endpoint',
            **payload,
            expected_base_url=EXPECTED_BASE_URL,
        )
    if not payload['store_ok']:
        return fail(
            'store must stay enabled across both turns for previous_response_id follow-up continuity',
            **payload,
        )
    if not payload['continuity_ok']:
        return fail(
            'turn two previous_response_id must exactly match turn one response id in both request and response',
            **payload,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
