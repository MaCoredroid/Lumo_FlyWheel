import json
import pathlib


def test_hidden_fixture_contract():
    variant_root = pathlib.Path(__file__).resolve().parents[1]
    variant = variant_root.name
    gold = json.loads((variant_root / "gold_reference.json").read_text())
    assert gold["variant_id"] == variant
