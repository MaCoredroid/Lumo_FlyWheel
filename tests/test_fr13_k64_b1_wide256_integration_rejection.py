from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT / "results" / "fr13_fixed32_k64_b1_wide256_integration_rejection_20260801"
)
DECISION = ARTIFACT / "rejection.json"
FORCED_BUILD = (
    ROOT / "results" / "fr13_fixed32_streamk_wide256_stock_symbol_exact_build_20260801"
)


def _decision() -> dict[str, object]:
    return json.loads(DECISION.read_text(encoding="ascii"))


def test_decision_is_fail_closed_on_the_exact_k64_target() -> None:
    decision = _decision()
    assert decision["schema"] == (
        "fr13.fixed32.k64_b1_wide256_integration_rejection.v1"
    )
    assert decision["status"] == "REJECTED_NO_PRODUCTION_CANDIDATE"
    assert decision["base_commit"] == ("c3ee2fece6daa17927ec216ff0135c5cf3ebb1e0")

    target = decision["integration_target"]
    assert target["draft_vocab_root"] == 1
    assert target["draft_vocab_k"] == 65536
    assert target["physical_drafts"] == 31
    assert target["physical_rows_root_inclusive"] == 32
    assert target["block_map_path"] == (
        "/workspace/scripts/fr13_dvk_subset_blocks.json"
    )
    assert target["block_map_sha256"] == (
        "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
    )

    topologies = {row["logical_topology"]: row for row in target["topologies"]}
    assert topologies == {
        "Tail23": {
            "logical_topology": "Tail23",
            "mode": "tail6_fixed32",
            "logical_drafts": 23,
            "valid_mask": "0x7a9ce7ff",
        },
        "Hydra27": {
            "logical_topology": "Hydra27",
            "mode": "hydra27_fixed32",
            "logical_drafts": 27,
            "valid_mask": "0x7abdffff",
        },
    }
    assert target["required_projection_shapes_mnk"] == [
        [32, 5120, 6144],
        [32, 5120, 17408],
        [32, 16384, 5120],
        [32, 34816, 5120],
        [32, 14336, 5120],
    ]


def test_no_candidate_or_downstream_route_was_issued() -> None:
    decision = _decision()
    route = decision["decision"]
    assert route["newest_audited_candidate"] == "wide256_dataparallel"
    assert route["candidate_selected"] is None
    for field in (
        "production_pass_issued",
        "k64_gate_prepared",
        "timing_pair_prepared",
        "b4_configured",
        "gdn_combined",
        "overlap_combined",
        "performance_measurement",
        "acceptance_valid",
    ):
        assert route[field] is False
    assert decision["analytical_context"] == {
        "optimistic_max_recovery_ms_per_event": 10.923627,
        "classification": "model_only_not_measurement",
        "realizable_payoff_claimed": False,
    }
    assert decision["next_valid_action"]["timing_before_gate_forbidden"] is True


def test_forced_streamk_is_static_and_byte_rejected() -> None:
    forced = _decision()["candidates"][0]
    assert forced["candidate"] == "streamk_force_wide256"
    assert forced["binary"]["sha256"] == (
        "f7d5c01ca79829fbfff4c93949d057bd740905165b0b6793b3c0007629add962"
    )
    static = forced["static_audit"]
    assert static["exact_stock_symbol_and_resource_matches"] == 6
    assert static["missing_or_changed_stock_records"] == 0
    assert static["wide256_stack_bytes"] == 8
    assert static["wide256_local_bytes"] == 0
    assert static["passes_zero_stack_requirement"] is False

    live = forced["real_b1_byte_diagnostic"]
    assert live["comparisons"] == live["mismatching_comparisons"] == 240
    assert live["total_compared_bytes"] == 234881024
    assert live["differing_bytes"] == 11511
    assert live["required_shape_coverage_complete"] is False
    assert live["formal_live_gate_reducer_ran"] is False
    assert live["served_result"] == "stock"
    assert forced["production_eligible"] is False


def test_newest_data_parallel_candidate_is_static_pass_byte_fail() -> None:
    candidate = _decision()["candidates"][1]
    assert candidate["candidate"] == "wide256_dataparallel"
    assert candidate["binary"] == {
        "path": "/home/mark/fr13_streamk_build/bin/_C_stable_libtorch.wide256_dataparallel_b1_gate_ready.abi3.so",
        "sha256": "5b921ab7b428f2c5cfeefc0daed0314ff903d73bb0d4f8a790b17234c9d60890",
        "bytes": 112787936,
        "mode": "0555",
        "runpath": "/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64",
    }
    static = candidate["static_audit"]
    assert static["tile_mnk"] == [256, 32, 128]
    assert static["tile_scheduler"] == "stock_data_parallel"
    assert static["k_split"] is False
    assert static["candidate_kernel_records"] == 2
    assert static["exact_stock_symbol_and_resource_matches"] == 6
    assert static["missing_or_changed_stock_records"] == 0
    assert static["stack_bytes"] == static["local_bytes"] == 0
    assert static["passes_zero_stack_requirement"] is True

    build = candidate["build_provenance"]
    assert build["sha256sums_present"] is True
    assert build["declared_file_missing_from_git_tree"].endswith("/build.log")
    assert build["complete"] is False

    live = candidate["real_b1_byte_diagnostic"]
    shapes = live["shapes"]
    assert sum(shape["comparisons"] for shape in shapes) == 256
    assert all(
        shape["comparisons"] == shape["mismatching_comparisons"] for shape in shapes
    )
    assert sum(shape["compared_bytes"] for shape in shapes) == 249888768
    assert sum(shape["differing_bytes"] for shape in shapes) == 10504
    assert live["comparisons"] == live["mismatching_comparisons"] == 256
    assert live["missing_required_shapes_mnk"] == [[32, 14336, 5120]]
    assert live["required_shape_coverage_complete"] is False
    assert live["formal_live_gate_reducer_ran"] is False
    assert live["served_result"] == "stock"
    assert live["production_enabled"] is live["timing_eligible"] is False
    assert candidate["production_eligible"] is False


def test_republished_artifact_is_reduced_and_content_free() -> None:
    allowed = {"README.md", "SHA256SUMS", "rejection.json", "test_results.txt"}
    published = {path.name for path in ARTIFACT.iterdir()}
    assert published == allowed


def test_existing_forced_build_attestation_remains_pinned() -> None:
    candidate_path = FORCED_BUILD / "candidate.json"
    candidate = json.loads(candidate_path.read_text(encoding="ascii"))
    assert hashlib.sha256(candidate_path.read_bytes()).hexdigest() == (
        "da26f699862d017fe2821893c99eaf01a7a746b423fe3df1a4b81ef6ed159926"
    )
    assert candidate["binary"]["sha256"] == (
        "f7d5c01ca79829fbfff4c93949d057bd740905165b0b6793b3c0007629add962"
    )
    assert candidate["stock_equivalence"]["exact_symbol_and_resource_matches"] == 6
    assert candidate["stock_equivalence"]["missing_or_changed_records"] == 0
    assert candidate["candidate_kernels"]["wide256_stack_bytes"] == 8
