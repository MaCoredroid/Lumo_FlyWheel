"""Filesystem source adapter.

Reads a JSON file containing a list of section records from a configured
path. The file format is an array of `{owner, label, count}` objects.
"""
from __future__ import annotations

import json
from pathlib import Path

from release_readiness.core.model import Section, sections_from_iterable


class FsSource:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> tuple[Section, ...]:
        if not self._path.exists():
            return ()
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"expected list in {self._path}, got {type(data).__name__}")
        return sections_from_iterable(data)
