from __future__ import annotations

import json
from collections.abc import Iterable

import pytest

import report_app.cli as cli_module
from report_app.rendering import render_markdown
from report_app.service import TITLE, build_triage_summary


def _copy_sections(sections: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(section) for section in sections]


def cli_markdown(
    monkeypatch: pytest.MonkeyPatch,
    sections: Iterable[dict[str, object]],
) -> str:
    monkeypatch.setattr(cli_module, "build_sections", lambda: _copy_sections(sections))
    return cli_module.main(["--format", "markdown"])


def cli_json(
    monkeypatch: pytest.MonkeyPatch,
    sections: Iterable[dict[str, object]],
) -> dict[str, object]:
    monkeypatch.setattr(cli_module, "build_sections", lambda: _copy_sections(sections))
    return json.loads(cli_module.main(["--format", "json"]))


def direct_markdown(sections: Iterable[dict[str, object]]) -> str:
    copied = _copy_sections(sections)
    triage_summary = build_triage_summary(copied)
    return render_markdown(TITLE, copied, triage_summary)
