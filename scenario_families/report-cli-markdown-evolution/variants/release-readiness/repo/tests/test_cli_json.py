from __future__ import annotations

import json

import pytest

from release_readiness.cli import main


def _set_records(records: list[dict[str, object]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELEASE_READINESS_RECORDS", json.dumps(records))
    monkeypatch.setenv("RELEASE_READINESS_SOURCE", "env")


def test_json_output_is_still_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_records(
        [
            {"owner": "Sam", "label": "blocked-rollouts", "count": 2},
            {"owner": "Rin", "label": "hotfixes", "count": 1},
        ],
        monkeypatch,
    )
    payload = json.loads(main(["--format", "json"]))
    assert payload["title"] == "Release Readiness Report"
    owners_in_sections = {s["owner"] for s in payload["sections"]}
    assert owners_in_sections == {"Sam", "Rin"}


def test_json_output_has_owner_totals_field(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_records(
        [{"owner": "Sam", "label": "a", "count": 2}, {"owner": "Sam", "label": "b", "count": 3}],
        monkeypatch,
    )
    payload = json.loads(main(["--format", "json"]))
    assert {"owner": "Sam", "total": 5} in payload["owner_totals"]


def test_cli_rejects_unknown_format(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_records([], monkeypatch)
    with pytest.raises(SystemExit):
        main(["--format", "xml"])


def test_cli_exposes_markdown_format(monkeypatch: pytest.MonkeyPatch) -> None:
    """The registry must expose 'markdown' as an available --format choice.

    This test will fail until a markdown renderer is registered.
    """
    from release_readiness.renderers.registry import get_registry
    assert "markdown" in get_registry().available_formats()


def test_markdown_output_has_title_heading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markdown output must begin with a level-1 heading matching the title."""
    _set_records(
        [{"owner": "Sam", "label": "blocked-rollouts", "count": 2}],
        monkeypatch,
    )
    out = main(["--format", "markdown"])
    assert out.startswith("# Release Readiness Report\n")


def test_markdown_output_has_owner_totals_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markdown output must include an owner-totals section."""
    _set_records(
        [{"owner": "Sam", "label": "a", "count": 2}, {"owner": "Rin", "label": "b", "count": 1}],
        monkeypatch,
    )
    out = main(["--format", "markdown"])
    assert "## Owner Totals" in out
