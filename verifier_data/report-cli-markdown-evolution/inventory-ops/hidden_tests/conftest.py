from __future__ import annotations

from collections.abc import Iterable

import pytest

import report_app.cli as cli_module
from report_app.rendering import render_markdown
from report_app.service import TITLE, build_owner_summary


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
) -> str:
    monkeypatch.setattr(cli_module, "build_sections", lambda: _copy_sections(sections))
    return cli_module.main(["--format", "json"])


def direct_markdown(
    sections: Iterable[dict[str, object]],
    *,
    include_known_owners: bool = False,
) -> str:
    copied = _copy_sections(sections)
    owner_summary = build_owner_summary(
        copied,
        include_known_owners=include_known_owners,
    )
    return render_markdown(TITLE, copied, owner_summary)
