from __future__ import annotations

import json
from pathlib import Path

from ci_app.workflow_preview import dispatch_job_ids


FIXTURES = json.loads((Path(__file__).with_name("_differential_fixtures.json")).read_text(encoding="utf-8"))


def test_round1_dispatch_ids_match_differential_fixtures() -> None:
    for fixture in FIXTURES["cases"]:
        assert dispatch_job_ids(fixture["labels"]) == fixture["expected"]
