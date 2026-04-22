from __future__ import annotations

import json
from pathlib import Path


def test_oracle_and_gold_align():
    root = Path(__file__).resolve().parents[1]
    gold = json.loads((root / "gold_ranking.json").read_text())
    oracle = json.loads((root / "oracle" / "manager_brief.json").read_text())
    assert oracle["first_milestone_id"] == gold["first_milestone_id"]
    assert [step["step_id"] for step in oracle["ordered_steps"]] == gold["gold_order"]
