from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'serving_maintenance' / '.codex' / 'config.toml'
SCRIPT = ROOT / 'serving_maintenance' / 'scripts' / 'smoke_responses_profile.py'
FIXTURES = ROOT / 'serving_maintenance' / 'fixtures' / 'http'

def run(turn2_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            '--config', str(CONFIG),
            '--turn1', str(FIXTURES / 'turn1_ok.json'),
            '--turn2', str(FIXTURES / turn2_name),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

class SmokeProfileTests(unittest.TestCase):
    def test_good_followup_passes(self) -> None:
        proc = run('turn2_ok.json')
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        payload = json.loads(proc.stdout)
        self.assertTrue(payload['continuity_ok'])
        self.assertTrue(payload['store_ok'])

    def test_missing_store_fails(self) -> None:
        proc = run('turn2_missing_store.json')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('store', proc.stdout + proc.stderr)

if __name__ == '__main__':
    unittest.main()
