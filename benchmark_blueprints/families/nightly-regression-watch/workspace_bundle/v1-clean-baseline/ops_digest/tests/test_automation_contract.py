from __future__ import annotations

from pathlib import Path


def test_single_active_watch_definition() -> None:
    automation_dir = Path(__file__).resolve().parents[1] / "automation"
    tomls = sorted(automation_dir.glob("*.toml"))
    assert [path.name for path in tomls] == ["nightly_regression_watch.toml"]
    text = tomls[0].read_text()
    assert 'status = "ACTIVE"' in text
    assert "Action required" in text
    assert "flag anything marked fail" not in text
