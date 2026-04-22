import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incident_handoff.adapter import normalize_response_items


def test_adapter_extracts_message_blocks():
    payload = json.loads((ROOT / "transcripts" / "interleaved_tool_turn.json").read_text())
    events = normalize_response_items(payload)
    assert events[0] == {"kind": "assistant_text", "text": "Checking"}
