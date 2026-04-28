"""Unit tests for P5b fp8_e5m2 KV purity attestation script.

The vLLM-dependent path (server bring-up + probe capture) is exercised by the
operator script itself; these tests cover the deterministic bits — bundle
kv_cache_dtype override and the overshoot computation — that surface bugs at
unit-test time before the 10-minute integration cost.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "p5b_fp8_kv_purity_attestation.py"


def _load_p5b_module():
    spec = importlib.util.spec_from_file_location("p5b_attest", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["p5b_attest"] = module
    spec.loader.exec_module(module)
    return module


def test_override_kv_cache_dtype_writes_sibling_bundle(tmp_path: Path) -> None:
    p5b = _load_p5b_module()
    base = tmp_path / "base.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "tuned_config_bundle": {
                    "model_id": "qwen3.5-27b",
                    "vllm_config": {"max_num_seqs": 4, "kv_cache_dtype": "fp8_e5m2"},
                    "kernel_selection": {
                        "attention_backend": "vllm-default",
                        "kv_cache_dtype": "fp8_e5m2",
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "bf16.yaml"
    p5b._override_kv_cache_dtype(base, "bf16", out)
    payload = yaml.safe_load(out.read_text(encoding="utf-8"))
    bundle = payload["tuned_config_bundle"]
    assert bundle["vllm_config"]["kv_cache_dtype"] == "bf16"
    assert bundle["kernel_selection"]["kv_cache_dtype"] == "bf16"
    # Other fields are preserved.
    assert bundle["vllm_config"]["max_num_seqs"] == 4
    assert bundle["kernel_selection"]["attention_backend"] == "vllm-default"
    # The base file is unchanged.
    base_payload = yaml.safe_load(base.read_text(encoding="utf-8"))
    assert base_payload["tuned_config_bundle"]["vllm_config"]["kv_cache_dtype"] == "fp8_e5m2"


def test_compute_overshoot_zero_when_within_tolerance() -> None:
    p5b = _load_p5b_module()
    candidate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reference = np.array([1.0005, 2.0008, 3.0009], dtype=np.float32)
    # rtol=1e-3 + atol=1e-3 -> allowed ~ 0.001 + 0.001*|ref|, all diffs are within.
    overshoot = p5b._compute_overshoot(candidate, reference, rtol=1e-3, atol=1e-3)
    assert overshoot == 0.0


def test_compute_overshoot_positive_when_diff_exceeds_tolerance() -> None:
    p5b = _load_p5b_module()
    candidate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    # 0.05 absolute diff vs allowed ~ 0.003 -> 0.047 overshoot at element 1.
    reference = np.array([1.0, 2.05, 3.0], dtype=np.float32)
    overshoot = p5b._compute_overshoot(candidate, reference, rtol=1e-3, atol=1e-3)
    assert overshoot > 0.04
    assert overshoot < 0.05


def test_compute_overshoot_inf_on_shape_mismatch() -> None:
    p5b = _load_p5b_module()
    candidate = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    reference = np.array([1.0, 2.0], dtype=np.float32)
    overshoot = p5b._compute_overshoot(candidate, reference, rtol=1e-3, atol=1e-3)
    assert overshoot == float("inf")
