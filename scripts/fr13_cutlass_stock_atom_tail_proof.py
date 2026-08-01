#!/usr/bin/env python3
"""Audit whether an SM121 B1 output tail can keep the stock CUTLASS atom.

This audit is intentionally source-only.  It reads pinned Git blobs rather
than a possibly patched checkout and fails closed on every source anchor.  A
successful audit run can still conclude that the candidate is not viable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "fr13.fixed32.cutlass_stock_atom_tail_source_proof.v1"
SOURCE_BASE_COMMIT = "bcde8591f8ee9b0a563c9ddc4924450643cf4114"
VLLM_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
CUTLASS_COMMIT = "da5e086dab31d63815acafdac9a9c5893b1c69e2"

VLLM_DISPATCH_PATH = (
    "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/"
    "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"
)
CUTLASS_BUILDER_PATH = (
    "include/cutlass/gemm/collective/builders/"
    "sm120_blockwise_mma_builder.inl"
)
CUTLASS_MMA_PATH = "include/cute/arch/mma_sm120.hpp"
CUTLASS_SCHEDULER_PATH = "include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp"

EXPECTED_SHA256 = {
    VLLM_DISPATCH_PATH: (
        "6e1df3f4701f58f233b3831b848c7bbf7936e6cb34b3bc28ded208fd66c48a7f"
    ),
    CUTLASS_BUILDER_PATH: (
        "40409c39fbbc5f023e8030472efab2a7b94baf41109eaa59fb009e52ce0d6509"
    ),
    CUTLASS_MMA_PATH: (
        "4848512808aa9c1c461445ff4b35f5a02f25ce734f81a5994176e0a46e57c37a"
    ),
    CUTLASS_SCHEDULER_PATH: (
        "54ebcaee08d4fc0663169f97c7fa665cec90d29ce5d01336c2c714f2a911b010"
    ),
}

BLOCK_MAP_PATH = Path("scripts/fr13_dvk_subset_blocks.json")
BLOCK_MAP_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)

MANDATORY_EXPLICIT_BYTES = 23_823_646_720
MANDATORY_EXPLICIT_FLOOR_MS = 88.439932
STOCK_COMPLETE_TILE_BYTES = 26_877_100_032
THEORETICAL_PARTITIONED_BYTES = 23_894_949_888
THEORETICAL_OPPORTUNITY_BYTES = 2_982_150_144
THEORETICAL_OPPORTUNITY_MS = 10.923627
SEPARATE_SAME_TILE_RESIDUAL_MS = 12.688211

STOCK_TILE_MNK = (128, 32, 128)
STOCK_SCALE_GRANULARITY_MNK = (128, 1, 128)
STOCK_CLUSTER_MNK = (1, 1, 1)

VLLM_REQUIRED = (
    "using ElementAB = cutlass::float_e4m3_t;",
    "using ElementAccumulator = float;",
    "using ElementCompute = float;",
    "using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<",
    "cutlass::gemm::collective::StageCountAutoCarveout<",
    "struct sm120_blockwise_fp8_config_swapab",
    "using TileShape = Shape<_128, _32, _128>;",
    "using ClusterShape = Shape<_1, _1, _1>;",
    "OutType, 128, 1, 128, TileShape, ClusterShape,",
    "EpilogueSchedule, KernelSchedule, true>;",
)

BUILDER_REQUIRED = (
    "using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));",
    "using PermTileN = decltype(cute::min(size<1>(TileShape_MNK{}),  _32{}));",
    "Layout<Shape<_4,_2,_1>>, Layout<Shape<_2,_2,_1>>>;",
    "cute::rr_op_selector_sm120<ElementA, ElementB, ElementAccumulator>()",
    "Tile<PermTileM, PermTileN, _32>{}",
    "static_assert(size<0>(TileShape_MNK{}) % ScaleGranularityM == 0",
    "static_assert(size<2>(TileShape_MNK{}) == ScaleGranularityK",
)

MMA_REQUIRED = (
    "return SM120_16x8x32_TN<ElementA, ElementB, ElementC>{};",
)

SCHEDULER_REQUIRED = (
    "auto grid_shape    = shape(ceil_div(problem_shape_mnkl, blk_shape));",
    "product_each(ceil_div(select<0,1,3>(problem_shape_mnkl), take<0,2>(tile_shape_mnk)))",
)


class ProofError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob(repo: Path, commit: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ProofError(f"cannot read {commit}:{relative_path}: {detail}")
    return result.stdout


def _require_source(
    *, data: bytes, path: str, expected_sha256: str, anchors: tuple[str, ...]
) -> dict[str, Any]:
    actual_sha256 = _sha256(data)
    if actual_sha256 != expected_sha256:
        raise ProofError(
            f"source identity mismatch for {path}: "
            f"{actual_sha256} != {expected_sha256}"
        )
    text = data.decode("utf-8")
    missing = [anchor for anchor in anchors if anchor not in text]
    if missing:
        raise ProofError(f"source anchors missing from {path}: {missing!r}")
    return {
        "path": path,
        "sha256": actual_sha256,
        "required_anchor_count": len(anchors),
        "all_required_anchors_present": True,
    }


def prove_sources(
    *,
    vllm_source: bytes,
    builder_source: bytes,
    mma_source: bytes,
    scheduler_source: bytes,
    block_map: bytes,
) -> dict[str, Any]:
    sources = [
        _require_source(
            data=vllm_source,
            path=VLLM_DISPATCH_PATH,
            expected_sha256=EXPECTED_SHA256[VLLM_DISPATCH_PATH],
            anchors=VLLM_REQUIRED,
        ),
        _require_source(
            data=builder_source,
            path=CUTLASS_BUILDER_PATH,
            expected_sha256=EXPECTED_SHA256[CUTLASS_BUILDER_PATH],
            anchors=BUILDER_REQUIRED,
        ),
        _require_source(
            data=mma_source,
            path=CUTLASS_MMA_PATH,
            expected_sha256=EXPECTED_SHA256[CUTLASS_MMA_PATH],
            anchors=MMA_REQUIRED,
        ),
        _require_source(
            data=scheduler_source,
            path=CUTLASS_SCHEDULER_PATH,
            expected_sha256=EXPECTED_SHA256[CUTLASS_SCHEDULER_PATH],
            anchors=SCHEDULER_REQUIRED,
        ),
    ]
    block_map_sha256 = _sha256(block_map)
    if block_map_sha256 != BLOCK_MAP_SHA256:
        raise ProofError(
            f"K64 block map identity mismatch: "
            f"{block_map_sha256} != {BLOCK_MAP_SHA256}"
        )

    legal_smaller_tile_m = [
        tile_m
        for tile_m in range(1, STOCK_TILE_MNK[0])
        if tile_m % STOCK_SCALE_GRANULARITY_MNK[0] == 0
    ]
    if legal_smaller_tile_m:
        raise ProofError(
            f"unexpected legal sub-128 output tiles: {legal_smaller_tile_m!r}"
        )

    opportunity_bytes = STOCK_COMPLETE_TILE_BYTES - THEORETICAL_PARTITIONED_BYTES
    if opportunity_bytes != THEORETICAL_OPPORTUNITY_BYTES:
        raise ProofError("tail opportunity byte arithmetic drifted")

    return {
        "schema": SCHEMA,
        "status": "BLOCKED_NEEDS_USER_HELP",
        "source_base_commit": SOURCE_BASE_COMMIT,
        "source_identity": {
            "vllm_commit": VLLM_COMMIT,
            "cutlass_commit": CUTLASS_COMMIT,
            "files": sources,
        },
        "workload_contract": {
            "batch_size": 1,
            "physical_rows": 32,
            "draft_vocab_k": 65_536,
            "draft_vocab_root": 1,
            "draft_vocab_block_map": str(BLOCK_MAP_PATH),
            "draft_vocab_block_map_sha256": block_map_sha256,
        },
        "stock_instantiation": {
            "layout": "swap_ab",
            "tile_mnk": list(STOCK_TILE_MNK),
            "scale_granularity_mnk": list(STOCK_SCALE_GRANULARITY_MNK),
            "cluster_mnk": list(STOCK_CLUSTER_MNK),
            "instruction_mnk": [16, 8, 32],
            "cooperative_atom_layout_mnk": [4, 2, 1],
            "permutation_tile_mnk": [128, 32, 32],
            "k_tile_sequence": 128,
            "accumulator": "fp32",
            "epilogue": "stock_linear_combination_bf16_or_fp16",
            "tile_scheduler": "stock_persistent_sm100_for_arch_sm120_family",
        },
        "source_proof": {
            "minimum_builder_legal_output_tile_m": 128,
            "legal_positive_output_tile_m_below_128": legal_smaller_tile_m,
            "sub_128_changes_permutation_type": True,
            "sub_128_fails_scale_granularity_constraint": True,
            "problem_tail_below_128_still_has_one_stock_work_tile": True,
            "stock_fragment_association_retained_by_smaller_partition": False,
            "candidate_viable_under_hard_contract": False,
        },
        "traffic_contract": {
            "mandatory_explicit_bytes": MANDATORY_EXPLICIT_BYTES,
            "mandatory_explicit_floor_ms": MANDATORY_EXPLICIT_FLOOR_MS,
            "stock_complete_tile_bytes": STOCK_COMPLETE_TILE_BYTES,
            "theoretical_partitioned_bytes": THEORETICAL_PARTITIONED_BYTES,
            "theoretical_opportunity_bytes": opportunity_bytes,
            "theoretical_opportunity_ms": THEORETICAL_OPPORTUNITY_MS,
            "separate_same_tile_residual_ms": SEPARATE_SAME_TILE_RESIDUAL_MS,
            "opportunity_realized": False,
        },
        "wide256_rejection": {
            "status": "binding_negative_evidence",
            "real_comparisons_all_drifted": True,
            "differing_bytes": 10_504,
            "revived": False,
        },
        "implementation": {
            "dispatcher_patch": False,
            "kernel_patch": False,
            "candidate_selector": None,
            "candidate_binary": None,
            "same_process_comparator": False,
            "gpu_launched": False,
            "build_attempted": False,
            "stop_reason": (
                "The pinned builder has no legal sub-128 output tile for the "
                "128-wide scale block, and a sub-128 tile changes TiledMma "
                "permutation/fragment association before code generation."
            ),
        },
        "remaining_live_gate_risk": (
            "No exact-safe candidate exists under the stated stock-atom "
            "contract, so binary, real-task comparator, and timing gates are "
            "intentionally inapplicable."
        ),
    }


def audit(*, repo_root: Path, vllm_root: Path, cutlass_root: Path) -> dict[str, Any]:
    block_map_path = repo_root / BLOCK_MAP_PATH
    if not block_map_path.is_file() or block_map_path.is_symlink():
        raise ProofError(f"block map is missing, non-regular, or symlinked: {block_map_path}")
    return prove_sources(
        vllm_source=_git_blob(vllm_root, VLLM_COMMIT, VLLM_DISPATCH_PATH),
        builder_source=_git_blob(cutlass_root, CUTLASS_COMMIT, CUTLASS_BUILDER_PATH),
        mma_source=_git_blob(cutlass_root, CUTLASS_COMMIT, CUTLASS_MMA_PATH),
        scheduler_source=_git_blob(
            cutlass_root, CUTLASS_COMMIT, CUTLASS_SCHEDULER_PATH
        ),
        block_map=block_map_path.read_bytes(),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--vllm-root", type=Path, required=True)
    parser.add_argument("--cutlass-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = audit(
        repo_root=args.repo_root.resolve(),
        vllm_root=args.vllm_root.resolve(),
        cutlass_root=args.cutlass_root.resolve(),
    )
    if args.output is not None:
        _write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
