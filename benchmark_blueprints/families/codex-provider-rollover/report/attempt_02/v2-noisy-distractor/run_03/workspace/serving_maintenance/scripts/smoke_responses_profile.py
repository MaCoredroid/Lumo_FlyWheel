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

def build_summary(config: dict, turn1: dict, turn2: dict) -> dict:
    selected = config.get('provider')
    providers = config.get('model_providers', {})
    provider = providers.get(selected, {})

    turn1_request = turn1.get('request', {})
    turn1_response = turn1.get('response', {})
    turn2_request = turn2.get('request', {})
    turn2_response = turn2.get('response', {})

    turn1_id = turn1_response.get('id')
    observed_request_previous = turn2_request.get('previous_response_id')
    observed_response_previous = turn2_response.get('previous_response_id')

    selected_provider_ok = selected == EXPECTED_PROVIDER
    config_endpoint_ok = provider.get('base_url') == EXPECTED_BASE_URL
    wire_api_ok = provider.get('wire_api') == EXPECTED_WIRE_API
    config_store_ok = provider.get('store') is True
    request_provider_ok = (
        turn1_request.get('provider') == EXPECTED_PROVIDER
        and turn2_request.get('provider') == EXPECTED_PROVIDER
    )
    request_base_url_ok = (
        turn1_request.get('base_url') == EXPECTED_BASE_URL
        and turn2_request.get('base_url') == EXPECTED_BASE_URL
    )
    request_store_ok = (
        turn1_request.get('store') is True
        and turn2_request.get('store') is True
    )
    continuity_ok = (
        bool(turn1_id)
        and observed_request_previous == turn1_id
        and observed_response_previous == turn1_id
    )
    response_status_ok = turn2_response.get('status') == 'completed'

    errors = []
    if not selected_provider_ok:
        errors.append(
            f"selected provider must be {EXPECTED_PROVIDER}, got {selected!r}"
        )
    if not config_endpoint_ok:
        errors.append(
            f"selected provider base_url must be {EXPECTED_BASE_URL}, got {provider.get('base_url')!r}"
        )
    if not wire_api_ok:
        errors.append(
            f"selected provider wire_api must be {EXPECTED_WIRE_API}, got {provider.get('wire_api')!r}"
        )
    if not config_store_ok:
        errors.append("selected provider must set store = true")
    if not turn1_id:
        errors.append("turn one response id is required before checking continuity")
    if not request_provider_ok:
        errors.append("turn fixtures must target responses_proxy on both turns")
    if not request_base_url_ok:
        errors.append(
            f"turn fixtures must use {EXPECTED_BASE_URL} on both turns"
        )
    if not request_store_ok:
        errors.append(
            "turn requests must keep store=true so previous_response_id follow-up state persists"
        )
    if turn1_id and observed_request_previous != turn1_id:
        errors.append(
            "turn two request previous_response_id must exactly match turn one response id"
        )
    if turn1_id and observed_response_previous != turn1_id:
        errors.append(
            "turn two response previous_response_id must echo the exact turn one response id"
        )
    if not response_status_ok:
        errors.append("turn two response status must be completed")

    return {
        "selected_provider": selected,
        "base_url": provider.get("base_url"),
        "turn1_id": turn1_id,
        "observed_previous_response_id": observed_request_previous,
        "response_previous_response_id": observed_response_previous,
        "continuity_ok": continuity_ok,
        "store_ok": config_store_ok and request_store_ok,
        "status": "ok" if not errors else "failed",
        "errors": errors,
    }

def run_smoke(config_path: Path, turn1_path: Path, turn2_path: Path) -> dict:
    config = load_config(config_path)
    turn1 = load_json(turn1_path)
    turn2 = load_json(turn2_path)
    return build_summary(config, turn1, turn2)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--turn1', type=Path, required=True)
    parser.add_argument('--turn2', type=Path, required=True)
    args = parser.parse_args()
    result = run_smoke(args.config, args.turn1, args.turn2)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1

if __name__ == '__main__':
    raise SystemExit(main())
