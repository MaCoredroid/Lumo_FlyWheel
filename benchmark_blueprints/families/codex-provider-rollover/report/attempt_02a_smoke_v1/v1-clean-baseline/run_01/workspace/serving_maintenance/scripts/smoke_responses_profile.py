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

def check_equal(errors: list[str], label: str, observed, expected) -> bool:
    if observed != expected:
        errors.append(f"{label}: expected {expected!r}, observed {observed!r}")
        return False
    return True

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
    turn2_request_previous = turn2_request.get('previous_response_id')
    turn2_response_previous = turn2_response.get('previous_response_id')
    errors: list[str] = []

    provider_ok = all(
        [
            check_equal(errors, 'selected provider', selected, EXPECTED_PROVIDER),
            check_equal(errors, 'provider base_url', provider.get('base_url'), EXPECTED_BASE_URL),
            check_equal(errors, 'provider wire_api', provider.get('wire_api'), EXPECTED_WIRE_API),
            check_equal(errors, 'provider store', provider.get('store'), True),
        ]
    )
    store_ok = all(
        [
            check_equal(errors, 'turn1 request store', turn1_request.get('store'), True),
            check_equal(errors, 'turn2 request store', turn2_request.get('store'), True),
        ]
    )
    continuity_ok = all(
        [
            check_equal(errors, 'turn1 response id', bool(turn1_id), True),
            check_equal(errors, 'turn2 request previous_response_id', turn2_request_previous, turn1_id),
            check_equal(errors, 'turn2 response previous_response_id', turn2_response_previous, turn1_id),
        ]
    )

    return {
        'selected_provider': selected,
        'base_url': provider.get('base_url'),
        'provider_ok': provider_ok,
        'turn1_id': turn1_id,
        'observed_request_previous_response_id': turn2_request_previous,
        'observed_response_previous_response_id': turn2_response_previous,
        'continuity_ok': continuity_ok,
        'store_ok': store_ok,
        'status': 'ok' if provider_ok and store_ok and continuity_ok else 'failed',
        'errors': errors,
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
