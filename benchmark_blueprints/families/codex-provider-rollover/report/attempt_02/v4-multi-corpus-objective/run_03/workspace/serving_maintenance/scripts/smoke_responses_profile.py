#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_PROFILE = "maintenance-responses"
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

def check_equal(errors: list[str], label: str, observed, expected) -> bool:
    if observed != expected:
        errors.append(f"{label}: expected {expected!r}, observed {observed!r}")
        return False
    return True

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    profile = config.get('profile')
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    selected_provider = providers.get(selected, {})
    expected_provider = providers.get(EXPECTED_PROVIDER, {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    turn1_request = turn1.get('request', {})
    turn1_response = turn1.get('response', {})
    turn2_request = turn2.get('request', {})
    turn2_response = turn2.get('response', {})
    turn1_id = turn1_response.get('id')
    observed_request_prev = turn2_request.get('previous_response_id')
    observed_response_prev = turn2_response.get('previous_response_id')
    errors: list[str] = []

    check_equal(errors, 'profile', profile, EXPECTED_PROFILE)
    check_equal(errors, 'selected provider', selected, EXPECTED_PROVIDER)
    check_equal(errors, 'provider base_url', expected_provider.get('base_url'), EXPECTED_BASE_URL)
    check_equal(errors, 'provider wire_api', expected_provider.get('wire_api'), 'responses')
    if expected_provider.get('store') is not True:
        errors.append(f"provider store: expected True, observed {expected_provider.get('store')!r}")

    check_equal(errors, 'turn1 provider', turn1_request.get('provider'), EXPECTED_PROVIDER)
    check_equal(errors, 'turn2 provider', turn2_request.get('provider'), EXPECTED_PROVIDER)
    check_equal(errors, 'turn1 base_url', turn1_request.get('base_url'), EXPECTED_BASE_URL)
    check_equal(errors, 'turn2 base_url', turn2_request.get('base_url'), EXPECTED_BASE_URL)
    if turn1_request.get('store') is not True:
        errors.append(f"turn1 store: expected True, observed {turn1_request.get('store')!r}")
    if turn2_request.get('store') is not True:
        errors.append(f"turn2 store: expected True, observed {turn2_request.get('store')!r}")
    if not turn1_id:
        errors.append('turn1 response id missing')
    if observed_request_prev != turn1_id:
        errors.append(
            "request previous_response_id continuity: "
            f"expected {turn1_id!r}, observed {observed_request_prev!r}"
        )
    if observed_response_prev != turn1_id:
        errors.append(
            "response previous_response_id continuity: "
            f"expected {turn1_id!r}, observed {observed_response_prev!r}"
        )

    continuity_ok = (
        bool(turn1_id)
        and observed_request_prev == turn1_id
        and observed_response_prev == turn1_id
    )
    store_ok = (
        expected_provider.get('store') is True
        and turn1_request.get('store') is True
        and turn2_request.get('store') is True
    )
    return {
        'profile': profile,
        'selected_provider': selected,
        'base_url': selected_provider.get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': observed_request_prev,
        'observed_response_previous_response_id': observed_response_prev,
        'continuity_ok': continuity_ok,
        'store_ok': store_ok,
        'status': 'ok' if not errors else 'failed',
        'errors': errors,
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--turn1', type=Path, required=True)
    parser.add_argument('--turn2', type=Path, required=True)
    args = parser.parse_args()
    result = run_smoke(args.config, args.turn1, args.turn2)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result['status'] != 'ok':
        print('\n'.join(result['errors']), file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
