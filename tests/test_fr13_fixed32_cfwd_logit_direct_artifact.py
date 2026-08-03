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
        "a4406489ee681a65cb564b09ce1109ecd49867f3"
    )
    assert model["qualification"] == {
        "draft_vocab_k": 65_536,
        "draft_vocab_root": 1,
        "physical_rows": 32,
        "logical_tree_limit": 32,
        "modes": ["tail6_fixed32", "hydra27_fixed32"],
        "batches": [1, 4],
        "vocab_size": 248_320,
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
        "incumbent_full_vocab_materialized_bytes": 190_709_760,
        "candidate_active_block_stat_bytes": 14_640,
        "candidate_block_stat_workspace_bytes": 15_360,
        "full_vocab_materialized_bytes_removed": 190_695_120,
    }
    assert model["physical_work"]["b4"] == {
        "incumbent_full_vocab_materialized_bytes": 762_839_040,
        "candidate_active_block_stat_bytes": 58_560,
        "candidate_block_stat_workspace_bytes": 61_440,
        "full_vocab_materialized_bytes_removed": 762_780_480,
    }
    assert model["claim_scope"] == (
        "static_source_work_only_no_runtime_speed_claim"
    )
    assert model["offline_codegen"] == {
        "status": "pass",
        "architecture": "sm_121a",
        "block_v": 4096,
        "active_vocab_blocks": 61,
        "summary": "codegen_summary.json",
    }


def test_published_source_hashes_match() -> None:
    entries = (ARTIFACT / "source_hashes.sha256").read_text().splitlines()
    assert len(entries) == 3
    for entry in entries:
        expected, relative = entry.split("  ", 1)
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected


def test_codegen_summary_is_bound_and_spill_free() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())

    assert summary["status"] == "pass"
    assert summary["architecture"] == "sm_121a"
    assert summary["vocab_size"] == 248_320
    assert summary["block_v"] == 4096
    assert summary["max_blocks"] == 64
    assert summary["active_vocab_blocks"] == 61
    assert summary["source_sha256"] == (
        "2003bb878f61ba09e10d08335ce98387a95aabfb0b0a2d03f03100217afcec05"
    )
    assert [(item["kernel"], item["ctas_b1"], item["ctas_b4"]) for item in summary["kernels"]] == [
        ("block_stats", 1_830, 7_320),
        ("direct_decision", 30, 120),
    ]
    assert [item["registers_per_thread"] for item in summary["kernels"]] == [
        76,
        80,
    ]
    for kernel in summary["kernels"]:
        assert kernel["stack_bytes"] == 0
        assert kernel["local_bytes"] == 0
        assert kernel["spill_loads"] == 0
        assert kernel["spill_stores"] == 0
        assert kernel["calls"] == 0


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
    assert "offline_sm121a_codegen=PASS" in results
    assert "gpu_execution=NOT_RUN" in results
    assert "real_swe_verified_product_ab=NOT_RUN" in results
    assert "nsight_launch_and_dram_trace=NOT_RUN" in results
    assert "real_four_task_b1_b4_timing=NOT_RUN" in results
