from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/fr13_fixed32_gdn_gqa_group3_source_20260803"


def test_work_model_is_source_only_and_exactly_bound() -> None:
    model = json.loads((ARTIFACT / "work_model.json").read_text())

    assert model["status"] == "source_only_default_off"
    assert model["source_commit"] == (
        "fafd7a80f4e6080e83b5c024f975bd5399e569b7"
    )
    guard = model["guard_scope"]
    assert guard["validation_phase"] == "before_triton_dispatch"
    assert guard["served_runtime_wired"] is False
    assert guard["tensor_metadata"] == [
        "device",
        "dtype",
        "shape",
        "contiguity",
        "canonical_stride",
    ]
    assert guard["pointer_domains"] == [
        "h0_index_addresses_and_bank_ids",
        "raw_and_prescaled_fixed32_descriptors",
        "active_output_ring_counter_and_flag_storage",
    ]
    assert model["qualification"] == {
        "draft_vocab_k": 65_536,
        "draft_vocab_root": 1,
        "physical_rows": 32,
        "logical_tree_limit": 32,
        "modes": ["tail6_fixed32", "hydra27_fixed32"],
        "batches": [1, 4],
    }
    assert model["physical_work"]["b1"]["ctas_removed_per_event"] == 24_576
    assert model["physical_work"]["b1"]["qk_bytes_removed_per_event"] == (
        402_653_184
    )
    assert model["physical_work"]["b4"]["ctas_removed_per_event"] == 98_304
    assert model["physical_work"]["b4"]["qk_bytes_removed_per_event"] == (
        1_610_612_736
    )
    assert model["claim_scope"] == (
        "static_source_work_only_no_runtime_speed_claim"
    )


def test_published_source_hashes_match() -> None:
    entries = (ARTIFACT / "source_hashes.sha256").read_text().splitlines()
    assert len(entries) == 2
    for entry in entries:
        expected, relative = entry.split("  ", 1)
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected


def test_artifact_states_every_unrun_runtime_gate() -> None:
    readme = (ARTIFACT / "README.md").read_text()
    results = (ARTIFACT / "test_results.txt").read_text()

    for gate in (
        "SM121a",
        "real SWE-Verified byte A/B",
        "real four-task B1 and B4 full-wall",
        "16-task confirmation",
    ):
        assert gate in readme
    assert "does not claim a speedup" in readme
    assert "gpu_compile=NOT_RUN_no_gpu" in results
    assert "real_swe_verified_byte_ab=NOT_RUN" in results
    assert "real_four_task_timing=NOT_RUN" in results
