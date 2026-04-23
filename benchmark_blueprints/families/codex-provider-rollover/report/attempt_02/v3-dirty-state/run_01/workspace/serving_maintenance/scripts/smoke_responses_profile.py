#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
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

def ensure(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    active_provider = providers.get(selected, {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    turn1_request = turn1.get('request', {})
    turn1_response = turn1.get('response', {})
    turn2_request = turn2.get('request', {})
    turn2_response = turn2.get('response', {})
    turn1_id = turn1_response.get('id')
    observed_previous_response_id = turn2_request.get('previous_response_id')
    echoed_previous_response_id = turn2_response.get('previous_response_id')
    errors: list[str] = []

    ensure(selected == EXPECTED_PROVIDER, errors, f"selected provider must be {EXPECTED_PROVIDER}")
    ensure(
        active_provider.get('base_url') == EXPECTED_BASE_URL,
        errors,
        f"selected provider base_url must be {EXPECTED_BASE_URL}",
    )
    ensure(turn1_request.get('provider') == EXPECTED_PROVIDER, errors, "turn1 provider drifted")
    ensure(turn1_request.get('base_url') == EXPECTED_BASE_URL, errors, "turn1 base_url drifted")
    ensure(turn1_request.get('store') is True, errors, "turn1 store must be true")
    ensure(turn1_id is not None, errors, "turn1 response.id is required")
    ensure(turn2_request.get('provider') == EXPECTED_PROVIDER, errors, "turn2 provider drifted")
    ensure(turn2_request.get('base_url') == EXPECTED_BASE_URL, errors, "turn2 base_url drifted")
    ensure(turn2_request.get('store') is True, errors, "turn2 store must be true")
    ensure(
        observed_previous_response_id == turn1_id,
        errors,
        "turn2 request previous_response_id must exactly match turn1 response.id",
    )
    ensure(
        echoed_previous_response_id == turn1_id,
        errors,
        "turn2 response previous_response_id must exactly match turn1 response.id",
    )

    continuity_ok = (
        turn1_id is not None
        and observed_previous_response_id == turn1_id
        and echoed_previous_response_id == turn1_id
    )
    store_ok = turn1_request.get('store') is True and turn2_request.get('store') is True
    return {
        'selected_provider': selected,
        'base_url': active_provider.get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': observed_previous_response_id,
        'echoed_previous_response_id': echoed_previous_response_id,
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
    if payload['errors']:
        print("Smoke failed:", file=sys.stderr)
        for error in payload['errors']:
            print(f"- {error}", file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
