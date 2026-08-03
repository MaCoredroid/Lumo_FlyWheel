from __future__ import annotations

import json
from pathlib import Path


ARTIFACT = Path(
    "results/fr13_fixed32_gdn_static_descriptors_sm121a_rejection_20260803"
)
KERNEL = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")


def test_static_descriptor_experiment_is_rejected_and_reverted() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())

    assert summary["status"] == "rejected_resource_regression"
    assert summary["offline_only"] is True
    assert summary["timing_claim"] is False
    assert summary["candidate_default_off"] is True
    assert summary["compile_context"] == {
        "block_v": 8,
        "cuda_compiler": "13.0.88",
        "dim_k": 128,
        "dim_v": 128,
        "draft_vocab_k": 65536,
        "fixed_physical_rows": 32,
        "num_k_heads_tp4": 4,
        "num_v_heads_tp4": 12,
        "num_warps": 8,
        "root_reduction": 1,
        "scan_align": True,
        "target": "sm_121a",
        "triton": "3.6.0",
    }
    assert summary["comparison"] == {
        "cubin_bytes_delta": -6520,
        "ldg_sites_delta": -15,
        "registers_per_thread_delta": 13,
        "sass_instructions_delta": -256,
        "selector_off_sass_identity": True,
    }
    assert summary["incumbent"]["stack_bytes"] == 0
    assert summary["candidate_codegen"]["stack_bytes"] == 0
    assert summary["candidate_codegen"]["registers_per_thread"] > summary[
        "incumbent"
    ]["registers_per_thread"]

    source = KERNEL.read_text()
    assert "STATIC_DESCRIPTORS" not in source
    assert "_fr13_fixed32_gdn_static_branch_node" not in source
