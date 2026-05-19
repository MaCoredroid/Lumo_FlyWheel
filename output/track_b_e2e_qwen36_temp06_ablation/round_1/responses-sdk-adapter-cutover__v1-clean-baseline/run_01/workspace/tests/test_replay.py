import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incident_handoff.replay import replay_from_serialized, serialize_events


def test_replay_preserves_tool_call_ids():
    events = [
        {"kind": "assistant_text", "text": "Checking"},
        {"kind": "tool_call", "call_id": "call-1", "name": "lookup_owner", "arguments": "{\"id\": 1}"},
        {"kind": "tool_result", "call_id": "call-1", "output": "oncall-primary"},
    ]
    assert replay_from_serialized(serialize_events(events)) == events
