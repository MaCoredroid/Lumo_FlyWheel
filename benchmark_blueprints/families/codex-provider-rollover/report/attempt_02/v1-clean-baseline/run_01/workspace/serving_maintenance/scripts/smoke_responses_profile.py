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

def expect_equal(errors: list[str], label: str, observed: Any, expected: Any) -> bool:
    ok = observed == expected
    if not ok:
        errors.append(f"{label}: expected {expected!r}, observed {observed!r}")
    return ok

def expect_true(errors: list[str], label: str, observed: Any) -> bool:
    ok = bool(observed)
    if not ok:
        errors.append(f"{label}: expected truthy value, observed {observed!r}")
    return ok

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    selected_provider = providers.get(selected, {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    turn1_request = turn1.get('request', {})
    turn1_response = turn1.get('response', {})
    turn2_request = turn2.get('request', {})
    turn2_response = turn2.get('response', {})
    turn1_id = turn1_response.get('id')
    turn2_previous_request = turn2_request.get('previous_response_id')
    turn2_previous_response = turn2_response.get('previous_response_id')

    errors: list[str] = []
    expect_equal(errors, 'selected_provider', selected, EXPECTED_PROVIDER)
    expect_equal(errors, 'provider.base_url', selected_provider.get('base_url'), EXPECTED_BASE_URL)
    expect_equal(errors, 'provider.wire_api', selected_provider.get('wire_api'), 'responses')
    expect_equal(errors, 'provider.store', selected_provider.get('store'), True)

    expect_equal(errors, 'turn1.request.provider', turn1_request.get('provider'), EXPECTED_PROVIDER)
    expect_equal(errors, 'turn1.request.base_url', turn1_request.get('base_url'), EXPECTED_BASE_URL)
    expect_equal(errors, 'turn1.request.store', turn1_request.get('store'), True)
    expect_true(errors, 'turn1.response.id', turn1_id)

    expect_equal(errors, 'turn2.request.provider', turn2_request.get('provider'), EXPECTED_PROVIDER)
    expect_equal(errors, 'turn2.request.base_url', turn2_request.get('base_url'), EXPECTED_BASE_URL)
    expect_equal(errors, 'turn2.request.store', turn2_request.get('store'), True)
    expect_equal(errors, 'turn2.request.previous_response_id', turn2_previous_request, turn1_id)
    expect_equal(errors, 'turn2.response.previous_response_id', turn2_previous_response, turn1_id)

    continuity_ok = (
        bool(turn1_id)
        and turn2_previous_request == turn1_id
        and turn2_previous_response == turn1_id
    )
    store_ok = selected_provider.get('store') is True and turn1_request.get('store') is True and turn2_request.get('store') is True

    return {
        'selected_provider': selected,
        'base_url': selected_provider.get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': turn2_previous_request,
        'response_previous_response_id': turn2_previous_response,
        'continuity_ok': continuity_ok,
        'store_ok': store_ok,
        'status': 'ok' if not errors else 'error',
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
