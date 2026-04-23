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

def expect(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    provider_cfg = providers.get(selected, {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    turn1_request = turn1.get('request', {})
    turn1_response = turn1.get('response', {})
    turn2_request = turn2.get('request', {})
    turn2_response = turn2.get('response', {})
    turn1_id = turn1_response.get('id')
    request_previous_response_id = turn2_request.get('previous_response_id')
    response_previous_response_id = turn2_response.get('previous_response_id')

    failures: list[str] = []
    expect(selected == EXPECTED_PROVIDER, f"provider must be {EXPECTED_PROVIDER}", failures)
    expect(provider_cfg.get('base_url') == EXPECTED_BASE_URL, "profile base_url must target the responses endpoint", failures)
    expect(provider_cfg.get('store') is True, "profile store must stay enabled for follow-up continuity", failures)
    expect(turn1_request.get('provider') == EXPECTED_PROVIDER, "turn1 provider did not use responses_proxy", failures)
    expect(turn1_request.get('base_url') == EXPECTED_BASE_URL, "turn1 base_url did not use the responses endpoint", failures)
    expect(turn1_request.get('store') is True, "turn1 store must be true", failures)
    expect(turn1_response.get('status') == 'completed', "turn1 must complete before running the follow-up smoke", failures)
    expect(bool(turn1_id), "turn1 response.id is required", failures)
    expect(turn2_request.get('provider') == EXPECTED_PROVIDER, "turn2 provider did not use responses_proxy", failures)
    expect(turn2_request.get('base_url') == EXPECTED_BASE_URL, "turn2 base_url did not use the responses endpoint", failures)
    expect(turn2_request.get('store') is True, "turn2 store must be true to preserve follow-up state", failures)
    expect(turn2_response.get('status') == 'completed', "turn2 must complete successfully", failures)
    expect(
        request_previous_response_id == turn1_id,
        "turn2 request.previous_response_id must exactly match turn1 response.id",
        failures,
    )
    expect(
        response_previous_response_id == turn1_id,
        "turn2 response.previous_response_id must exactly match turn1 response.id",
        failures,
    )
    continuity_ok = (
        bool(turn1_id)
        and request_previous_response_id == turn1_id
        and response_previous_response_id == turn1_id
    )
    store_ok = provider_cfg.get('store') is True and turn1_request.get('store') is True and turn2_request.get('store') is True
    status = 'ok' if not failures else 'error'
    return {
        'selected_provider': selected,
        'base_url': provider_cfg.get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': request_previous_response_id,
        'response_previous_response_id': response_previous_response_id,
        'continuity_ok': continuity_ok,
        'store_ok': store_ok,
        'status': status,
        'failures': failures,
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
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
