from __future__ import annotations

from pathlib import Path

from drive_brief.settings import TOGGLE_LABEL, UNSET_SUMMARY, render_settings_panel


ROOT = Path(__file__).resolve().parents[1]


def test_toggle_is_visible_when_optional_value_is_unset() -> None:
    rendered = render_settings_panel()

    assert TOGGLE_LABEL in rendered
    assert UNSET_SUMMARY in rendered


def test_toggle_is_visible_when_optional_value_is_configured() -> None:
    rendered = render_settings_panel("shared-drive")

    assert TOGGLE_LABEL in rendered
    assert "Optional connector value: shared-drive" in rendered


def test_ui_source_mentions_toggle_without_early_unset_return() -> None:
    source = (ROOT / "drive_brief" / "ui" / "settings_panel.tsx").read_text()

    assert "Connector fallback" in source
    assert 'return "<section><h1>Settings</h1></section>";' not in source
