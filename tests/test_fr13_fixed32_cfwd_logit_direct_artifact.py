from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/fr13_fixed32_cfwd_logit_direct_source_20260803"


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
        "source_rows_parent_slots_and_uniform_levels",
        "child_counts_nodes_and_draft_token_ids",
        "finite_half_open_uniform_range",
        "exact_disjoint_workspace_storage",
    ]
    assert model["qualification"] == {
        "draft_vocab_k": 65_536,
        "draft_vocab_root": 1,
        "physical_rows": 32,
        "logical_tree_limit": 32,
        "modes": ["tail6_fixed32", "hydra27_fixed32"],
        "batches": [1, 4],
        "vocab_size": 151_936,
        "fanout": 3,
        "walk_cap": 12,
    }
    pipeline = model["pipeline"]
    assert pipeline["incumbent_full_vocab_fp32_rows_per_request"] == 192
    assert pipeline["producer_dispatch_sites_removed_static"] == 2
    assert pipeline["physical_kernel_launches_removed"] == "pending_gpu_trace"
    assert pipeline["integer_commit_launches_before"] == 1
    assert pipeline["integer_commit_launches_after"] == 1
    assert model["physical_work"]["b1"] == {
        "incumbent_full_vocab_materialized_bytes": 116_686_848,
        "candidate_active_block_stat_bytes": 142_560,
        "candidate_block_stat_workspace_bytes": 245_760,
        "full_vocab_materialized_bytes_removed": 116_544_288,
    }
    assert model["physical_work"]["b4"] == {
        "incumbent_full_vocab_materialized_bytes": 466_747_392,
        "candidate_active_block_stat_bytes": 570_240,
        "candidate_block_stat_workspace_bytes": 983_040,
        "full_vocab_materialized_bytes_removed": 466_177_152,
    }
    assert model["claim_scope"] == (
        "static_source_work_only_no_runtime_speed_claim"
    )


def test_published_source_hashes_match() -> None:
    entries = (ARTIFACT / "source_hashes.sha256").read_text().splitlines()
    assert len(entries) == 3
    for entry in entries:
        expected, relative = entry.split("  ", 1)
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected


def test_artifact_states_every_unrun_runtime_gate() -> None:
    readme = (ARTIFACT / "README.md").read_text()
    results = (ARTIFACT / "test_results.txt").read_text()

    for gate in (
        "SM121a",
        "real SWE-Verified",
        "Nsight",
        "real four-task B1 and B4 full-wall",
        "16-task confirmation",
    ):
        assert gate in readme
    assert "does not claim a speedup" in readme
    assert "gpu_compile=NOT_RUN_no_gpu" in results
    assert "real_swe_verified_product_ab=NOT_RUN" in results
    assert "nsight_launch_and_dram_trace=NOT_RUN" in results
    assert "real_four_task_b1_b4_timing=NOT_RUN" in results
