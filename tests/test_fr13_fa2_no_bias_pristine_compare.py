from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def _module():
    path = Path("scripts/fr13_fa2_no_bias_pristine_compare.py")
    spec = importlib.util.spec_from_file_location("fr13_fa2_compare", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_bias_compare_input_includes_decode_and_prefill_cases(tmp_path) -> None:
    path = tmp_path / "input.pt"

    assert _module().make_input(path) == 0

    payload = torch.load(path, map_location="cpu", weights_only=False)
    cases = {case["name"]: case for case in payload["cases"]}

    assert set(cases) == {
        "float16_decode",
        "float16_prefill",
        "bfloat16_decode",
        "bfloat16_prefill",
    }
    for name, case in cases.items():
        cu_q = case["cu_q"].tolist()
        cu_k = case["cu_k"].tolist()
        if name.endswith("_decode"):
            assert cu_q == [0, 9, 12]
            assert cu_k == [0, 71, 88]
        else:
            assert cu_q == [0, 71, 88]
            assert cu_k == [0, 71, 88]
