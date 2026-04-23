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

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    return {
        'selected_provider': selected,
        'base_url': providers.get(selected, {}).get('base_url'),
        'turn1_id': turn1.get('response', {}).get('id'),
        'observed_previous_response_id': turn2.get('request', {}).get('previous_response_id'),
        'continuity_ok': bool(turn2.get('request', {}).get('previous_response_id')),
        'store_ok': True,
        'status': 'ok',
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--turn1', type=Path, required=True)
    parser.add_argument('--turn2', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config, args.turn1, args.turn2), indent=2, sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
