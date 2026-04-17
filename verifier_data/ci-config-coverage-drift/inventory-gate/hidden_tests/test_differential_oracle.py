from __future__ import annotations

import json
from pathlib import Path

from ci_app.workflow_preview import preview_jobs


FIXTURES = json.loads((Path(__file__).with_name("_differential_fixtures.json")).read_text(encoding="utf-8"))


def test_round1_preview_jobs_match_differential_fixtures() -> None:
    for fixture in FIXTURES["cases"]:
        assert preview_jobs(fixture["labels"]) == fixture["expected"]
