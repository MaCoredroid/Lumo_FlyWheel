from __future__ import annotations

import json
from pathlib import Path

import pytest

from release_readiness.adapters.fs_source import FsSource


def test_load_returns_empty_tuple_when_file_missing(tmp_path: Path) -> None:
    source = FsSource(tmp_path / "nope.json")
    assert source.load() == ()


def test_load_parses_valid_records(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text(
        json.dumps([{"owner": "Sam", "label": "hotfixes", "count": 1}]),
        encoding="utf-8",
    )
    sections = FsSource(path).load()
    assert len(sections) == 1
    assert sections[0].owner == "Sam"


def test_load_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"owner": "Sam"}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected list"):
        FsSource(path).load()
