
from __future__ import annotations

import json
import os
from pathlib import Path

RESULT_FILE = Path(os.environ["RESULT_FILE"])


def test_functional_milestone_present() -> None:
    result = json.loads(RESULT_FILE.read_text())
    assert "M4_functional" in result["milestones"]


def test_no_shortcut_for_oracle() -> None:
    result = json.loads(RESULT_FILE.read_text())
    assert isinstance(result["shortcut_detected"], bool)
