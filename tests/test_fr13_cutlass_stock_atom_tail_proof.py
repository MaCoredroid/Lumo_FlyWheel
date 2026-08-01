from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/fr13_cutlass_stock_atom_tail_proof.py")
    spec = importlib.util.spec_from_file_location("fr13_stock_atom_tail_proof", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_with_anchors(anchors: tuple[str, ...]) -> bytes:
    return ("\n".join(anchors) + "\n").encode("utf-8")


def test_contract_arithmetic_and_minimum_tile_are_pinned() -> None:
    module = _module()

    assert module.STOCK_TILE_MNK == (128, 32, 128)
    assert module.STOCK_SCALE_GRANULARITY_MNK == (128, 1, 128)
    assert (
        module.STOCK_COMPLETE_TILE_BYTES
        - module.THEORETICAL_PARTITIONED_BYTES
        == module.THEORETICAL_OPPORTUNITY_BYTES
        == 2_982_150_144
    )
    assert module.MANDATORY_EXPLICIT_BYTES == 23_823_646_720
    assert module.THEORETICAL_OPPORTUNITY_MS == 10.923627
    assert module.SEPARATE_SAME_TILE_RESIDUAL_MS == 12.688211
    assert [tile for tile in range(1, 128) if tile % 128 == 0] == []


def test_audit_is_bound_to_exact_pinned_source_identities() -> None:
    module = _module()

    assert module.SOURCE_BASE_COMMIT == (
        "bcde8591f8ee9b0a563c9ddc4924450643cf4114"
    )
    assert module.VLLM_COMMIT == "fe9c3d6c5f66c873d196800384ed6880687b9e52"
    assert module.CUTLASS_COMMIT == "da5e086dab31d63815acafdac9a9c5893b1c69e2"
    assert module.BLOCK_MAP_SHA256 == (
        "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
    )
    assert "using TileShape = Shape<_128, _32, _128>;" in module.VLLM_REQUIRED
    assert any("% ScaleGranularityM" in item for item in module.BUILDER_REQUIRED)
    assert any("Tile<PermTileM" in item for item in module.BUILDER_REQUIRED)


def test_source_identity_or_anchor_drift_fails_closed() -> None:
    module = _module()
    source = _source_with_anchors(module.BUILDER_REQUIRED)
    source_hash = module._sha256(source)

    with pytest.raises(module.ProofError, match="source identity mismatch"):
        module._require_source(
            data=source,
            path="builder",
            expected_sha256="0" * 64,
            anchors=module.BUILDER_REQUIRED,
        )

    with pytest.raises(module.ProofError, match="source anchors missing"):
        module._require_source(
            data=source,
            path="builder",
            expected_sha256=source_hash,
            anchors=module.BUILDER_REQUIRED + ("missing geometry anchor",),
        )


def test_checked_in_proof_stops_before_build_and_gpu() -> None:
    module = _module()
    proof_path = Path(
        "results/fr13_fixed32_sm121_stock_atom_tail_proof_20260801/proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="ascii"))

    assert proof["schema"] == module.SCHEMA
    assert proof["status"] == "BLOCKED_NEEDS_USER_HELP"
    assert proof["source_proof"]["minimum_builder_legal_output_tile_m"] == 128
    assert proof["source_proof"]["legal_positive_output_tile_m_below_128"] == []
    assert proof["source_proof"]["candidate_viable_under_hard_contract"] is False
    assert proof["wide256_rejection"]["differing_bytes"] == 10_504
    assert proof["implementation"] == {
        "build_attempted": False,
        "candidate_binary": None,
        "candidate_selector": None,
        "dispatcher_patch": False,
        "gpu_launched": False,
        "kernel_patch": False,
        "same_process_comparator": False,
        "stop_reason": (
            "The pinned builder has no legal sub-128 output tile for the "
            "128-wide scale block, and a sub-128 tile changes TiledMma "
            "permutation/fragment association before code generation."
        ),
    }


def test_runtime_patch_does_not_advertise_blocked_candidate() -> None:
    patch = Path("scripts/fr13_patch_cutlass_fixed32_wave.py").read_text(
        encoding="utf-8"
    )

    assert "stock_atom_tail" not in patch
    assert "output_tail_partitioner" not in patch
