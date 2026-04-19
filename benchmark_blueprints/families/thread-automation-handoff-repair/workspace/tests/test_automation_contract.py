from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kind_is_heartbeat() -> None:
    content = (ROOT / "automations" / "release_handoff" / "automation.toml").read_text()
    assert 'kind = "heartbeat"' in content


def test_destination_is_thread() -> None:
    content = (ROOT / "automations" / "release_handoff" / "automation.toml").read_text()
    assert 'destination = "thread"' in content


def test_status_is_active() -> None:
    content = (ROOT / "automations" / "release_handoff" / "automation.toml").read_text()
    assert 'status = "ACTIVE"' in content
