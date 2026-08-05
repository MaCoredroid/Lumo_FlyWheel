from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/fr13_fixed32_gdn_gqa_group3_bv16_source_bound_sm121a_20260805"
)


def test_bv16_exact_scope_and_resource_tradeoff() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    assert summary["schema"] == (
        "fr13.fixed32.gdn_gqa_group3_bv16_source_bound.sm121a.codegen.v1"
    )
    contract = summary["compile_contract"]
    assert contract["batches"] == [1, 4]
    assert contract["physical_rows_per_request"] == 32
    assert contract["draft_vocab_k"] == 65_536
    assert contract["draft_vocab_root"] == 1
    assert contract["selector"] == "gqa_group3_bv16"
    assert contract["candidate_default_off"] is True
    assert contract["programs_per_48_layer_event"]["removed"] == {
        "b1": 6_144,
        "b4": 24_576,
    }
    assert contract["repeated_qk_bytes_removed_per_48_layer_event"] == {
        "b1": 96 * 1024 * 1024,
        "b4": 384 * 1024 * 1024,
    }

    expected = {
        "baseline_bv8_base": (108, 1_972, 74, 54),
        "candidate_bv16_base": (128, 2_602, 89, 54),
        "baseline_bv8_committer_stack": (118, 2_078, 74, 82),
        "candidate_bv16_committer_stack": (112, 2_790, 89, 82),
    }
    for variant, metrics in expected.items():
        for batch in ("b1", "b4"):
            row = summary["variants"][variant]["builds"][batch]
            assert (
                row["registers_per_thread"],
                row["static_sass_instructions"],
                row["ldg"],
                row["stg"],
            ) == metrics
            assert row["stack_bytes_per_thread"] == 0
            assert row["local_bytes_per_thread"] == 0
            assert row["ldl"] == row["stl"] == row["calls"] == 0


def test_bv16_artifact_verifier_passes_without_gpu() -> None:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    result = subprocess.run(
        ["python3", str(ARTIFACT / "verify_codegen_outputs.py")],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert "PASS: exact BV16 source-bound SM121a evidence" in result.stdout
