from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVIDER_DOC = ROOT / 'serving_maintenance' / 'docs' / 'provider_rollover.md'
SMOKE_DOC = ROOT / 'serving_maintenance' / 'docs' / 'smoke.md'

class DocsSyncTests(unittest.TestCase):
    def test_docs_reference_current_provider_and_continuity(self) -> None:
        provider_text = PROVIDER_DOC.read_text()
        smoke_text = SMOKE_DOC.read_text()
        self.assertIn('responses_proxy', provider_text)
        self.assertIn('http://127.0.0.1:11434/v1/responses', provider_text + smoke_text)
        self.assertIn('previous_response_id', smoke_text)

if __name__ == '__main__':
    unittest.main()
