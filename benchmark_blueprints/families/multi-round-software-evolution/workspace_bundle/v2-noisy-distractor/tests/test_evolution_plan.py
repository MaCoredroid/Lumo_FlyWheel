from __future__ import annotations

import json
from pathlib import Path


def test_round_plan_exists() -> None:
    assert Path("brief/round_plan.json").exists()


def test_round_plan_shape() -> None:
    data = json.loads(Path("brief/round_plan.json").read_text())
    assert data["schema_version"] == "cnb55.evolution_plan.v1"
    assert data["variant_id"] == Path(".scenario_variant").read_text().strip()

    focus = data["selected_focus"]
    assert focus["focus_id"]
    assert focus["title"]
    assert focus["why_now"]
    assert isinstance(focus["citations"], list) and len(focus["citations"]) >= 2

    do_not_touch = data["do_not_touch"]
    assert isinstance(do_not_touch, list) and do_not_touch
    for row in do_not_touch:
        assert row["area"]
        assert row["reason"]
        assert isinstance(row["citations"], list) and row["citations"]

    metric = data["partial_progress_metric"]
    assert metric["metric_id"]
    assert metric["baseline"]
    assert metric["target"]
    assert metric["guardrail"]
    assert isinstance(metric["measurement_plan"], list) and len(metric["measurement_plan"]) >= 2

    ledger = data["assumption_ledger"]
    assert isinstance(ledger, list) and ledger
    assert any(row["status"] == "missing" for row in ledger)
