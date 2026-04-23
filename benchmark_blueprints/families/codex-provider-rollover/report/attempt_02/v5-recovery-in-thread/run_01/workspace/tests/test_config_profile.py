from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'serving_maintenance' / '.codex' / 'config.toml'
EXPECTED_PROVIDER = 'responses_proxy'
EXPECTED_BASE_URL = 'http://127.0.0.1:11434/v1/responses'

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

def load_config() -> dict:
    data = {}
    current = data
    for raw in CONFIG.read_text().splitlines():
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

class ConfigProfileTests(unittest.TestCase):
    def test_selected_provider_is_proxy(self) -> None:
        data = load_config()
        self.assertEqual(data['provider'], EXPECTED_PROVIDER)
        provider = data['model_providers'][EXPECTED_PROVIDER]
        self.assertEqual(provider['base_url'], EXPECTED_BASE_URL)
        self.assertEqual(provider['wire_api'], 'responses')
        self.assertIs(provider['store'], True)

if __name__ == '__main__':
    unittest.main()
