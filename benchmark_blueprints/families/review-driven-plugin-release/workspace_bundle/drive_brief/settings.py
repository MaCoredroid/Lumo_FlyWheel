from __future__ import annotations

TOGGLE_LABEL = "Connector fallback"
UNSET_SUMMARY = "Optional connector value is unset. Connector fallback remains available."


def render_settings_panel(optional_value: str | None = None) -> str:
    summary = (
        f"Optional connector value: {optional_value}"
        if optional_value
        else UNSET_SUMMARY
    )
    return (
        "<section><h1>Settings</h1>"
        f"<p>{summary}</p>"
        f'<label><input type="checkbox" name="connector_fallback" />{TOGGLE_LABEL}</label>'
        "</section>"
    )
