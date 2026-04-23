#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_PROVIDER = "responses_proxy"
EXPECTED_BASE_URL = "http://127.0.0.1:11434/v1/responses"
EXPECTED_WIRE_API = "responses"

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

def validate_smoke(config: dict, turn1: dict, turn2: dict) -> list[str]:
    errors: list[str] = []
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    provider_config = providers.get(selected, {})
    turn1_request = turn1.get('request', {})
    turn1_response = turn1.get('response', {})
    turn2_request = turn2.get('request', {})
    turn2_response = turn2.get('response', {})
    turn1_id = turn1_response.get('id')
    request_previous_id = turn2_request.get('previous_response_id')
    response_previous_id = turn2_response.get('previous_response_id')

    if selected != EXPECTED_PROVIDER:
        errors.append(
            f"selected provider must be {EXPECTED_PROVIDER}, got {selected!r}"
        )
    if provider_config.get('base_url') != EXPECTED_BASE_URL:
        errors.append(
            f"provider base_url must be {EXPECTED_BASE_URL}, got "
            f"{provider_config.get('base_url')!r}"
        )
    if provider_config.get('wire_api') != EXPECTED_WIRE_API:
        errors.append(
            f"provider wire_api must be {EXPECTED_WIRE_API!r}, got "
            f"{provider_config.get('wire_api')!r}"
        )
    if provider_config.get('store') is not True:
        errors.append("provider config must keep store=true for follow-up continuity")
    if turn1_request.get('provider') != EXPECTED_PROVIDER:
        errors.append(
            f"turn1 provider must be {EXPECTED_PROVIDER}, got "
            f"{turn1_request.get('provider')!r}"
        )
    if turn1_request.get('base_url') != EXPECTED_BASE_URL:
        errors.append(
            f"turn1 base_url must be {EXPECTED_BASE_URL}, got "
            f"{turn1_request.get('base_url')!r}"
        )
    if turn1_request.get('store') is not True:
        errors.append("turn1 must set store=true")
    if turn1_response.get('status') != 'completed':
        errors.append(
            f"turn1 response must be completed, got {turn1_response.get('status')!r}"
        )
    if not turn1_id:
        errors.append("turn1 response is missing id")
    if turn2_request.get('provider') != EXPECTED_PROVIDER:
        errors.append(
            f"turn2 provider must be {EXPECTED_PROVIDER}, got "
            f"{turn2_request.get('provider')!r}"
        )
    if turn2_request.get('base_url') != EXPECTED_BASE_URL:
        errors.append(
            f"turn2 base_url must be {EXPECTED_BASE_URL}, got "
            f"{turn2_request.get('base_url')!r}"
        )
    if turn2_request.get('store') is not True:
        errors.append("turn2 must set store=true")
    if not request_previous_id:
        errors.append("turn2 request is missing previous_response_id")
    elif request_previous_id != turn1_id:
        errors.append(
            "turn2 request previous_response_id must exactly match turn1 response id"
        )
    if response_previous_id != turn1_id:
        errors.append(
            "turn2 response previous_response_id must exactly match turn1 response id"
        )
    return errors

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    errors = validate_smoke(config, turn1, turn2)
    turn1_id = turn1.get('response', {}).get('id')
    observed_previous_response_id = turn2.get('request', {}).get('previous_response_id')
    echoed_previous_response_id = turn2.get('response', {}).get('previous_response_id')
    return {
        'selected_provider': selected,
        'base_url': providers.get(selected, {}).get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': observed_previous_response_id,
        'echoed_previous_response_id': echoed_previous_response_id,
        'continuity_ok': (
            observed_previous_response_id == turn1_id
            and echoed_previous_response_id == turn1_id
        ),
        'store_ok': (
            providers.get(selected, {}).get('store') is True
            and turn1.get('request', {}).get('store') is True
            and turn2.get('request', {}).get('store') is True
        ),
        'errors': errors,
        'status': 'ok' if not errors else 'error',
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--turn1', type=Path, required=True)
    parser.add_argument('--turn2', type=Path, required=True)
    args = parser.parse_args()
    payload = run_smoke(args.config, args.turn1, args.turn2)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload['status'] == 'ok' else 1

if __name__ == '__main__':
    raise SystemExit(main())
