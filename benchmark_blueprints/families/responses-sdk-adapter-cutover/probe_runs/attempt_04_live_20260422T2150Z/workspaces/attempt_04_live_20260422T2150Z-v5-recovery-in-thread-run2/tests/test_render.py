import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from incident_handoff.render import render_transcript


def test_render_includes_tool_result_call_id():
    rendered = render_transcript(
        [
            {"kind": "assistant_text", "text": "Checking"},
            {"kind": "tool_result", "call_id": "call-1", "output": "oncall-primary"},
        ]
    )
    assert "tool_result(call-1): oncall-primary" in rendered
