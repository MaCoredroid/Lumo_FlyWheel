from pathlib import Path


def test_primary_automation_is_canonical_and_runnable():
    root = Path(__file__).resolve().parents[1]
    primary = (root / "automations" / "handoff-primary.toml").read_text()
    duplicate = (root / "automations" / "handoff-copy.toml").read_text()
    assert "canonical = true" in primary
    assert "scripts/run_handoff.py" in primary
    assert "canonical = true" not in duplicate
