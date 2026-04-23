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

def collect_errors(config: dict, turn1: dict, turn2: dict) -> list[str]:
    errors: list[str] = []
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    provider_cfg = providers.get(selected, {})
    turn1_request = turn1.get('request', {})
    turn1_response = turn1.get('response', {})
    turn2_request = turn2.get('request', {})
    turn2_response = turn2.get('response', {})
    turn1_id = turn1_response.get('id')
    turn2_request_prev = turn2_request.get('previous_response_id')
    turn2_response_prev = turn2_response.get('previous_response_id')

    if config.get('profile') != 'maintenance-responses':
        errors.append('config profile must be maintenance-responses')
    if selected != EXPECTED_PROVIDER:
        errors.append(f'config provider must be {EXPECTED_PROVIDER}')
    if provider_cfg.get('base_url') != EXPECTED_BASE_URL:
        errors.append(f'config base_url must be {EXPECTED_BASE_URL}')
    if provider_cfg.get('wire_api') != 'responses':
        errors.append('config wire_api must stay responses')
    if provider_cfg.get('store') is not True:
        errors.append('config store must be true for the maintenance profile')

    if turn1_request.get('provider') != EXPECTED_PROVIDER:
        errors.append('turn1 provider must use responses_proxy')
    if turn1_request.get('base_url') != EXPECTED_BASE_URL:
        errors.append('turn1 base_url must use the proxy responses route')
    if turn1_request.get('store') is not True:
        errors.append('turn1 store must be true')
    if not turn1_id:
        errors.append('turn1 response id is missing')

    if turn2_request.get('provider') != EXPECTED_PROVIDER:
        errors.append('turn2 provider must use responses_proxy')
    if turn2_request.get('base_url') != EXPECTED_BASE_URL:
        errors.append('turn2 base_url must use the proxy responses route')
    if turn2_request.get('store') is not True:
        errors.append('turn2 store must be true')
    if turn2_request_prev != turn1_id:
        errors.append('turn2 request previous_response_id must exactly match turn1 response id')
    if turn2_response_prev != turn1_id:
        errors.append('turn2 response previous_response_id must exactly match turn1 response id')

    return errors

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    turn1_id = turn1.get('response', {}).get('id')
    turn2_request_prev = turn2.get('request', {}).get('previous_response_id')
    turn2_response_prev = turn2.get('response', {}).get('previous_response_id')
    errors = collect_errors(config, turn1, turn2)
    return {
        'profile': config.get('profile'),
        'selected_provider': selected,
        'base_url': providers.get(selected, {}).get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': turn2_request_prev,
        'response_previous_response_id': turn2_response_prev,
        'continuity_ok': (
            bool(turn1_id)
            and turn2_request_prev == turn1_id
            and turn2_response_prev == turn1_id
        ),
        'store_ok': (
            config.get('model_providers', {}).get(EXPECTED_PROVIDER, {}).get('store') is True
            and turn1.get('request', {}).get('store') is True
            and turn2.get('request', {}).get('store') is True
        ),
        'errors': errors,
        'status': 'ok' if not errors else 'failed',
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--turn1', type=Path, required=True)
    parser.add_argument('--turn2', type=Path, required=True)
    args = parser.parse_args()
    result = run_smoke(args.config, args.turn1, args.turn2)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['status'] == 'ok' else 1

if __name__ == '__main__':
    raise SystemExit(main())
