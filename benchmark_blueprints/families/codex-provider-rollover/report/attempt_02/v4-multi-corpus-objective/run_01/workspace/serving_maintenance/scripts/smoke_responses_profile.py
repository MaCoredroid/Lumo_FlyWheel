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

def fail(summary: dict, *errors: str) -> tuple[int, dict]:
    payload = dict(summary)
    payload['errors'] = list(errors)
    payload['status'] = 'failed'
    return 1, payload

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> tuple[int, dict]:
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
    observed_previous_response_id = turn2_request.get('previous_response_id')
    response_previous_response_id = turn2_response.get('previous_response_id')

    summary = {
        'selected_provider': selected,
        'base_url': provider.get('base_url'),
        'turn1_id': turn1_id,
        'observed_previous_response_id': observed_previous_response_id,
        'response_previous_response_id': response_previous_response_id,
        'continuity_ok': False,
        'store_ok': False,
        'status': 'ok',
    }

    if selected != EXPECTED_PROVIDER:
        return fail(summary, f"selected provider must be {EXPECTED_PROVIDER}")
    if provider.get('base_url') != EXPECTED_BASE_URL:
        return fail(summary, f"provider base_url must be {EXPECTED_BASE_URL}")
    if provider.get('wire_api') != 'responses':
        return fail(summary, "provider wire_api must be responses")
    if provider.get('store') is not True:
        return fail(summary, "provider store must be true for follow-up retrieval")
    if turn1_request.get('provider') != EXPECTED_PROVIDER or turn2_request.get('provider') != EXPECTED_PROVIDER:
        return fail(summary, "fixtures must exercise the responses_proxy provider on both turns")
    if turn1_request.get('base_url') != EXPECTED_BASE_URL or turn2_request.get('base_url') != EXPECTED_BASE_URL:
        return fail(summary, "fixtures must target the proxy-backed Responses route")
    if turn1_request.get('store') is not True:
        return fail(summary, "turn one must enable store so the follow-up response can be retrieved")
    if turn2_request.get('store') is not True:
        return fail(summary, "turn two must keep store enabled for the maintained follow-up chain")
    if not turn1_id:
        return fail(summary, "turn one response id is missing")

    continuity_ok = (
        observed_previous_response_id == turn1_id
        and response_previous_response_id == turn1_id
    )
    store_ok = turn1_request.get('store') is True and turn2_request.get('store') is True
    summary['continuity_ok'] = continuity_ok
    summary['store_ok'] = store_ok

    if not continuity_ok:
        return fail(
            summary,
            "turn two must carry the exact previous_response_id from turn one in both request and response",
        )

    return 0, summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--turn1', type=Path, required=True)
    parser.add_argument('--turn2', type=Path, required=True)
    args = parser.parse_args()
    status, payload = run_smoke(args.config, args.turn1, args.turn2)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return status

if __name__ == '__main__':
    raise SystemExit(main())
