#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_PROVIDER = "responses_proxy"
EXPECTED_BASE_URL = "http://127.0.0.1:11434/v1/responses"
EXPECTED_ENV_KEY = "OPENAI_API_KEY"

class SmokeFailure(RuntimeError):
    pass

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

def parse_config(path: Path) -> dict:
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

def load_config(path: Path) -> tuple[str, dict]:
    data = parse_config(path)
    selected = data.get('provider')
    providers = data.get('model_providers', {})
    if selected != EXPECTED_PROVIDER:
        raise SmokeFailure(f'selected provider must be {EXPECTED_PROVIDER}, got {selected}')
    stanza = providers.get(selected, {})
    if stanza.get('base_url') != EXPECTED_BASE_URL:
        raise SmokeFailure('selected provider must use the proxy-backed Responses path')
    if stanza.get('wire_api') != 'responses':
        raise SmokeFailure('selected provider must use wire_api=responses')
    if stanza.get('env_key') != EXPECTED_ENV_KEY:
        raise SmokeFailure('selected provider must use OPENAI_API_KEY')
    if stanza.get('store') is not True:
        raise SmokeFailure('selected provider must keep store = true')
    return selected, stanza

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    selected, stanza = load_config(config_path)
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    turn1_id = turn1.get('response', {}).get('id')
    observed_previous = turn2.get('request', {}).get('previous_response_id')
    if observed_previous != turn1_id:
        raise SmokeFailure('turn two must use exact previous_response_id continuity')
    if turn2.get('request', {}).get('provider') != selected:
        raise SmokeFailure('turn two must use the selected provider id')
    if turn2.get('request', {}).get('base_url') != stanza.get('base_url'):
        raise SmokeFailure('turn two must use the selected provider base_url')
    if turn2.get('request', {}).get('store') is not True:
        raise SmokeFailure('turn two must keep store = true')
    response_previous = turn2.get('response', {}).get('previous_response_id')
    if response_previous not in (None, turn1_id):
        raise SmokeFailure('turn two response must preserve previous_response_id continuity')
    return {
        'selected_provider': selected,
        'base_url': stanza.get('base_url'),
        'expected_previous_response_id': turn1_id,
        'observed_previous_response_id': observed_previous,
        'continuity_ok': True,
        'store_ok': True,
        'status': 'ok',
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--turn1', type=Path, required=True)
    parser.add_argument('--turn2', type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = run_smoke(args.config, args.turn1, args.turn2)
    except SmokeFailure as exc:
        print(str(exc), flush=True)
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
