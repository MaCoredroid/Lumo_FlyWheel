from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_prompt_requests_thread_reply() -> None:
    content = (ROOT / "automations" / "release_handoff" / "prompt.md").read_text().lower()
    assert "reply in-thread" in content
    assert "handoff.md" not in content
