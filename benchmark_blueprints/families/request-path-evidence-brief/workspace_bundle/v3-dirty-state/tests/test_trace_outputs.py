
from __future__ import annotations

import json
from pathlib import Path


def test_path_outputs_exist_and_parse() -> None:
    base = Path("artifacts")
    path_map = base / "path_map.json"
    brief = base / "request_path_brief.md"
    correction = base / "docs_correction.md"
    assert path_map.exists()
    assert brief.exists()
    assert correction.exists()

    data = json.loads(path_map.read_text(encoding="utf-8"))
    assert data["schema_version"] == "cnb55.request_path_map.v1"
    assert data["variant_id"] == Path(".scenario_variant").read_text(encoding="utf-8").strip()
    assert isinstance(data["live_path"], list) and data["live_path"]
    assert isinstance(data["field_derivations"], dict)
    assert isinstance(data["test_observations"], list) and data["test_observations"]
    assert isinstance(data["rejected_decoys"], list) and data["rejected_decoys"]
