#!/usr/bin/env python3
"""Static lossless-fusion audit for fixed32 Hydra27 drafter attention.

This module is deliberately host-only.  It models the exact production
geometry, recurrence, buffer ownership, and byte/work counts without importing
Torch, Triton, or a vLLM runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from scripts import fr13_fixed32_topology as topology
except ModuleNotFoundError:  # Direct execution from scripts/.
    import fr13_fixed32_topology as topology


SCHEMA = "fr13.fixed32.dfwd_attention_batch_audit.v1"
CREATED_AT_UTC = "2026-08-03T09:34:45Z"
BASE_COMMIT = "c49c8eb5370e4d4035aceffaa8476aea31f921f5"
VLLM_BASE_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
PRISTINE_UNIFIED_ATTENTION_SHA256 = (
    "0d74baf7862c0feaef7a5da2f2cf35b761ff7e35c1cc23affbb41679eb2c3903"
)
BASE_PATCHER_SHA256 = (
    "494b6d4c1475204a7bc5148ac386d8455ecca575a2d0b643a74967c9e68a4046"
)
BASE_TOPOLOGY_SOURCE_SHA256 = (
    "146bdeace494f79cf15ef4fe54ee6c71d38cf086b5f2acfbf5149b46d102e311"
)
CURATED_NSYS_ATTRIBUTION_SHA256 = (
    "685c410a0ba09d00c8b244bfa06809530337cea383b88fa92a5da1013eadf2d0"
)

MODE = "hydra27_fixed32"
VALID_MASK = 0x7ABDFFFF
LOGICAL_DRAFTS = 27
PHYSICAL_ROWS = 32
DRAFT_VOCAB_ROOT = 1
DRAFT_VOCAB_K = 65_536
BATCH_SIZE = 1
POST_ROOT_SITES = 4
QUERY_ROWS_PER_SITE = 1
QUERY_HEADS = 24
KV_HEADS = 4
QUERY_HEADS_PER_KV = 6
HEAD_SIZE = 256
PAGE_ROWS = 1024
TILE_ROWS = 32
BLOCK_M = 8
BLOCK_Q = 1
DTYPE_BYTES = 2
INDEX_BYTES = 4

Q_SHAPE = (1, QUERY_HEADS, HEAD_SIZE)
Q_STRIDES = (QUERY_HEADS * HEAD_SIZE, HEAD_SIZE, 1)
KV_INNER_SHAPE = (PAGE_ROWS, KV_HEADS, HEAD_SIZE)
KV_INNER_STRIDES = (KV_HEADS * HEAD_SIZE, HEAD_SIZE, 1)
Q_BYTES_PER_SITE = QUERY_ROWS_PER_SITE * QUERY_HEADS * HEAD_SIZE * DTYPE_BYTES
OUTPUT_BYTES_PER_SITE = Q_BYTES_PER_SITE
KV_BYTES_PER_SEQUENCE_ROW = KV_HEADS * HEAD_SIZE * DTYPE_BYTES * 2
CTAS_PER_SITE = KV_HEADS
PHYSICAL_FLOPS_PER_TILE = (
    2 * BLOCK_M * HEAD_SIZE * TILE_ROWS * 2 * KV_HEADS
)
USEFUL_FLOPS_PER_TILE = (
    2 * QUERY_HEADS_PER_KV * HEAD_SIZE * TILE_ROWS * 2 * KV_HEADS
)
VECTOR_PAGE_INDEX_BYTES_PER_TILE = TILE_ROWS * INDEX_BYTES * KV_HEADS
SCALAR_PAGE_INDEX_BYTES_PER_TILE = INDEX_BYTES * KV_HEADS
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AttentionSite:
    level: int
    sequence_updates_before_site: int
    q_shape: tuple[int, int, int]
    q_strides: tuple[int, int, int]
    out_shape: tuple[int, int, int]
    out_strides: tuple[int, int, int]
    kv_inner_shape: tuple[int, int, int]
    kv_inner_strides: tuple[int, int, int]
    q_dtype: str
    kv_dtype: str
    out_dtype: str
    metadata_dtype: str
    cu_seqlens_q_shape: tuple[int, ...]
    cu_seqlens_q_values: tuple[int, ...]
    seqused_k_shape: tuple[int, ...]
    block_table_rank: int
    block_table_batch_rows: int
    block_table_last_stride: int
    q_storage: str
    out_storage: str
    key_cache_storage: str
    value_cache_storage: str
    sequence_lengths_storage: str
    block_table_storage: str


@dataclass(frozen=True)
class AuditContract:
    mode: str
    valid_mask: int
    logical_drafts: int
    physical_rows: int
    draft_vocab_root: int
    draft_vocab_k: int
    batch_size: int
    post_root_sites: int
    kernel: str
    block_m: int
    block_q: int
    tile_rows: int
    page_rows: int
    causal: bool
    full_window: bool
    max_query_length: int
    sites: tuple[AttentionSite, ...]


def _site(level: int) -> AttentionSite:
    return AttentionSite(
        level=level,
        sequence_updates_before_site=level,
        q_shape=Q_SHAPE,
        q_strides=Q_STRIDES,
        out_shape=Q_SHAPE,
        out_strides=Q_STRIDES,
        kv_inner_shape=KV_INNER_SHAPE,
        kv_inner_strides=KV_INNER_STRIDES,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        out_dtype="bfloat16",
        metadata_dtype="int32",
        cu_seqlens_q_shape=(2,),
        cu_seqlens_q_values=(0, 1),
        seqused_k_shape=(1,),
        block_table_rank=2,
        block_table_batch_rows=1,
        block_table_last_stride=1,
        # The graph records four model calls against the same workspaces.
        q_storage="mtp_attention_q_workspace",
        out_storage="mtp_attention_out_workspace",
        key_cache_storage="mtp_paged_key_cache",
        value_cache_storage="mtp_paged_value_cache",
        sequence_lengths_storage="mtp_sequence_lengths",
        block_table_storage="mtp_block_table",
    )


def fixed32_hydra27_contract() -> AuditContract:
    return AuditContract(
        mode=MODE,
        valid_mask=VALID_MASK,
        logical_drafts=LOGICAL_DRAFTS,
        physical_rows=PHYSICAL_ROWS,
        draft_vocab_root=DRAFT_VOCAB_ROOT,
        draft_vocab_k=DRAFT_VOCAB_K,
        batch_size=BATCH_SIZE,
        post_root_sites=POST_ROOT_SITES,
        kernel="kernel_unified_attention_2d",
        block_m=BLOCK_M,
        block_q=BLOCK_Q,
        tile_rows=TILE_ROWS,
        page_rows=PAGE_ROWS,
        causal=True,
        full_window=True,
        max_query_length=1,
        sites=tuple(_site(level) for level in range(POST_ROOT_SITES)),
    )


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, dict[str, str]]:
    expected = {
        "scripts/fr10_phase4_patch_vllm_tree_gdn.py": BASE_PATCHER_SHA256,
        "scripts/fr13_fixed32_topology.py": BASE_TOPOLOGY_SOURCE_SHA256,
    }
    manifest: dict[str, dict[str, str]] = {}
    for relative, digest in expected.items():
        actual = _sha256(REPO_ROOT / relative)
        _require(actual == digest, "audit source drifted: " + relative)
        manifest[relative] = {"sha256": actual}
    manifest["scripts/fr13_dfwd_attention_batch_audit.py"] = {
        "sha256": _sha256(Path(__file__).resolve())
    }
    return manifest


def validate_contract(contract: AuditContract) -> AuditContract:
    """Fail closed unless every algebra, layout, and alias invariant is exact."""
    _require(contract.mode == MODE, "mode must be hydra27_fixed32")
    _require(contract.valid_mask == VALID_MASK, "Hydra27 valid mask drifted")
    _require(
        contract.logical_drafts == LOGICAL_DRAFTS,
        "Hydra27 logical draft count drifted",
    )
    _require(contract.physical_rows == PHYSICAL_ROWS, "physical rows must be 32")
    _require(contract.draft_vocab_root == 1, "draft vocabulary root must be 1")
    _require(contract.draft_vocab_k == 65_536, "draft vocabulary K must be 65536")
    _require(contract.batch_size == 1, "DFWD attention audit requires B1")
    _require(contract.post_root_sites == 4, "post-root attention site count must be 4")
    _require(
        contract.kernel == "kernel_unified_attention_2d",
        "attention kernel identity drifted",
    )
    _require((contract.block_m, contract.block_q) == (8, 1), "BM8 geometry drifted")
    _require((contract.tile_rows, contract.page_rows) == (32, 1024), "paged tile geometry drifted")
    _require(contract.page_rows % contract.tile_rows == 0, "attention tile crosses a KV page")
    _require(contract.causal is True, "drafter attention must remain causal")
    _require(contract.full_window is True, "drafter attention must use a full window")
    _require(contract.max_query_length == 1, "max query length must remain 1")
    _require(len(contract.sites) == POST_ROOT_SITES, "site tuple length drifted")
    _require(topology.MTP_FORWARD_CALLS == POST_ROOT_SITES, "topology MTP call count drifted")
    _require(topology.PHYSICAL_ROWS == PHYSICAL_ROWS, "topology physical rows drifted")
    _require(
        topology.HYDRA27_ACTIVE_DRAFTS == LOGICAL_DRAFTS,
        "topology active draft count drifted",
    )
    _require(topology.HYDRA27_VALID_MASK == VALID_MASK, "topology Hydra27 mask drifted")

    storage_fields = (
        "q_storage",
        "out_storage",
        "key_cache_storage",
        "value_cache_storage",
        "sequence_lengths_storage",
        "block_table_storage",
    )
    for expected_level, site in enumerate(contract.sites):
        _require(site.level == expected_level, "attention levels are not ordered 0..3")
        _require(
            site.sequence_updates_before_site == expected_level,
            "sequence metadata update count drifted",
        )
        _require(site.q_shape == Q_SHAPE, "query shape drifted")
        _require(site.q_strides == Q_STRIDES, "query layout is not contiguous [1,24,256]")
        _require(site.out_shape == Q_SHAPE, "output shape drifted")
        _require(site.out_strides == Q_STRIDES, "output layout is not contiguous [1,24,256]")
        _require(site.kv_inner_shape == KV_INNER_SHAPE, "paged KV inner shape drifted")
        _require(site.kv_inner_strides == KV_INNER_STRIDES, "paged KV inner layout drifted")
        _require(
            (site.q_dtype, site.kv_dtype, site.out_dtype)
            == ("bfloat16", "bfloat16", "bfloat16"),
            "attention data dtype drifted",
        )
        _require(site.metadata_dtype == "int32", "attention metadata dtype drifted")
        _require(
            site.cu_seqlens_q_shape == (2,)
            and site.cu_seqlens_q_values == (0, 1),
            "query sequence metadata is not exact B1 Q1",
        )
        _require(site.seqused_k_shape == (1,), "sequence length layout drifted")
        _require(
            site.block_table_rank == 2
            and site.block_table_batch_rows == 1
            and site.block_table_last_stride == 1,
            "block-table layout drifted",
        )
        storages = tuple(getattr(site, field) for field in storage_fields)
        _require(len(set(storages)) == len(storages), "within-site buffers alias")

    # The captured graph reuses these addresses. Deferring attention would read
    # the final query/length unless every site added a snapshot copy.
    for field in storage_fields:
        _require(
            len({getattr(site, field) for site in contract.sites}) == 1,
            "cross-level workspace ownership drifted for " + field,
        )
    return contract


def recurrence_edges() -> tuple[dict[str, Any], ...]:
    edges: list[dict[str, Any]] = []
    for level in range(POST_ROOT_SITES - 1):
        edges.append(
            {
                "from_attention_site": level,
                "to_attention_site": level + 1,
                "blocking_path": [
                    f"attention_{level}.output",
                    f"model_{level}.post_attention_hidden",
                    f"lm_head_{level}.logits",
                    f"top3_{level}.spine_rank0",
                    f"model_{level + 1}.input_ids",
                    f"attention_{level + 1}.query",
                ],
                "second_blocking_path": [
                    f"attention_{level}.output",
                    f"model_{level}.returned_hidden_state",
                    f"model_{level + 1}.hidden_state_input",
                    f"attention_{level + 1}.query",
                ],
                "state_transition": {
                    "sequence_length_update": "in_place_increment_or_rollover",
                    "kv_suffix_write_before_next_attention": True,
                },
            }
        )
    return tuple(edges)


def fusion_verdict(contract: AuditContract | None = None) -> dict[str, Any]:
    validated = validate_contract(contract or fixed32_hydra27_contract())
    sites = validated.sites
    return {
        "eligible": False,
        "scope": "attention-only launch batching across four post-root MTP forwards",
        "classification": "recurrent_nonconsecutive_attention_sites",
        "current_attention_launches_per_event": POST_ROOT_SITES,
        "minimum_lossless_attention_launches_per_event": POST_ROOT_SITES,
        "launches_removed": 0,
        "blocked_edges": list(recurrence_edges()),
        "wide_branch_note": (
            "Hydra27 runner-up branches are packing-only, but rank-0 spine token "
            "and returned hidden state remain recurrent."
        ),
        "alias_guard": {
            "query_workspace_reused_across_sites": len({s.q_storage for s in sites}) == 1,
            "output_workspace_reused_across_sites": len({s.out_storage for s in sites}) == 1,
            "sequence_lengths_mutated_in_place": len(
                {s.sequence_lengths_storage for s in sites}
            )
            == 1,
            "paged_kv_suffix_mutated_between_sites": True,
            "deferred_batching_requires_query_and_length_snapshots": True,
        },
        "reason": (
            "Attention i must finish before post-attention model work, K64 top3, "
            "and metadata/KV transition can produce the query for attention i+1. "
            "The intervening graph nodes cannot execute behind one same-stream "
            "attention kernel."
        ),
    }


def static_ledger(first_sequence_length: int | None = None) -> dict[str, Any]:
    """Return exact symbolic counts and, optionally, one integer evaluation."""
    ledger: dict[str, Any] = {
        "accounting_scope": {
            "payload_bytes": "mandatory Q, output, K, and V tensor traffic",
            "page_index_bytes": "source-level logical block-table loads",
            "work": "QK and PV dot-product FLOPs",
            "physical_hbm_or_compiler_metadata_transactions_included": False,
        },
        "sequence_length_precondition": "S>0 and S+3<=max_model_len",
        "sequence_lengths": ["S", "S+1", "S+2", "S+3"],
        "sum_sequence_rows": "4*S+6",
        "launches": {
            "kernel_unified_attention_2d": 4,
            "ctas_per_launch": CTAS_PER_SITE,
            "ctas_per_event": POST_ROOT_SITES * CTAS_PER_SITE,
            "lossless_attention_only_fused_launches": None,
        },
        "bytes": {
            "q_reads_per_event": POST_ROOT_SITES * Q_BYTES_PER_SITE,
            "output_writes_per_event": POST_ROOT_SITES * OUTPUT_BYTES_PER_SITE,
            "kv_payload_per_sequence_row": KV_BYTES_PER_SEQUENCE_ROW,
            "kv_payload_reads_per_event": "4096*(4*S+6)",
            "vector_page_index_loads_per_full_tile": VECTOR_PAGE_INDEX_BYTES_PER_TILE,
            "scalar_page_index_loads_per_full_tile": SCALAR_PAGE_INDEX_BYTES_PER_TILE,
            "scalar_page_index_logical_saving_per_full_tile": (
                VECTOR_PAGE_INDEX_BYTES_PER_TILE - SCALAR_PAGE_INDEX_BYTES_PER_TILE
            ),
            "scalar_page_index_saving_vs_kv_payload_fraction": (
                (VECTOR_PAGE_INDEX_BYTES_PER_TILE - SCALAR_PAGE_INDEX_BYTES_PER_TILE)
                / (KV_BYTES_PER_SEQUENCE_ROW * TILE_ROWS)
            ),
        },
        "work": {
            "tile_rows": TILE_ROWS,
            "bm8_physical_qk_pv_flops_per_full_tile_across_four_kv_heads": (
                PHYSICAL_FLOPS_PER_TILE
            ),
            "live_head_qk_pv_flops_per_full_tile_across_four_kv_heads": (
                USEFUL_FLOPS_PER_TILE
            ),
            "physical_to_live_head_work_ratio": (
                PHYSICAL_FLOPS_PER_TILE / USEFUL_FLOPS_PER_TILE
            ),
            "tile_count_per_event": (
                "ceil(S/32)+ceil((S+1)/32)+ceil((S+2)/32)+ceil((S+3)/32)"
            ),
        },
        "same_width_common_prefix_stage": {
            "original_common_prefix_reads": "4*4096*S",
            "staged_common_prefix_traffic": "6*4096*S",
            "traffic_delta_bytes": "+8192*S",
            "standalone_preparation_launch_delta": 1,
            "verdict": "strictly_more_traffic",
        },
        "index_preparation": {
            "page_rows": PAGE_ROWS,
            "tile_rows": TILE_ROWS,
            "tiles_per_page": PAGE_ROWS // TILE_ROWS,
            "block_table_stable_across_sites": True,
            "once_per_event_expanded_tile_map_can_be_exact": True,
            "source_page_table_unique_bytes": "4*ceil((S+3)/1024)",
            "expanded_tile_map_write_bytes": "4*ceil((S+3)/32)",
            "expanded_map_entries_per_source_page": PAGE_ROWS // TILE_ROWS,
            "expanded_map_consumer_scalar_bytes": "16*T",
            "direct_scalar_lookup_needs_expanded_map": False,
            "physical_byte_saving_proven_host_only": False,
        },
    }
    if first_sequence_length is not None:
        sequence_length = int(first_sequence_length)
        if sequence_length <= 0:
            raise ValueError("first sequence length must be positive")
        lengths = [sequence_length + level for level in range(POST_ROOT_SITES)]
        tile_counts = [
            (length + TILE_ROWS - 1) // TILE_ROWS for length in lengths
        ]
        tile_count = sum(tile_counts)
        common_prefix_bytes = sequence_length * KV_BYTES_PER_SEQUENCE_ROW
        ledger["evaluation"] = {
            "first_sequence_length": sequence_length,
            "sequence_lengths": lengths,
            "sum_sequence_rows": sum(lengths),
            "tile_counts": tile_counts,
            "tile_count": tile_count,
            "kv_payload_read_bytes": sum(lengths) * KV_BYTES_PER_SEQUENCE_ROW,
            "bm8_physical_qk_pv_flops": tile_count * PHYSICAL_FLOPS_PER_TILE,
            "live_head_qk_pv_flops": tile_count * USEFUL_FLOPS_PER_TILE,
            "vector_page_index_logical_bytes": (
                tile_count * VECTOR_PAGE_INDEX_BYTES_PER_TILE
            ),
            "scalar_page_index_logical_bytes": (
                tile_count * SCALAR_PAGE_INDEX_BYTES_PER_TILE
            ),
            "same_width_common_prefix_stage_extra_bytes": 2 * common_prefix_bytes,
        }
    return ledger


def build_audit() -> dict[str, Any]:
    contract = validate_contract(fixed32_hydra27_contract())
    sources = source_manifest()
    return {
        "schema": SCHEMA,
        "created_at_utc": CREATED_AT_UTC,
        "status": "STOP_RECURRENCE_BLOCKS_ATTENTION_ONLY_FUSION",
        "scope": {
            "base_commit": BASE_COMMIT,
            "topology": MODE,
            "valid_mask": f"0x{VALID_MASK:08x}",
            "logical_drafts": LOGICAL_DRAFTS,
            "physical_rows": PHYSICAL_ROWS,
            "draft_vocab_root": DRAFT_VOCAB_ROOT,
            "draft_vocab_k": DRAFT_VOCAB_K,
            "batch_size": BATCH_SIZE,
            "gpu_used": False,
            "docker_used": False,
            "runtime_used": False,
            "synthetic_timing_used": False,
            "main_worktree_touched": False,
        },
        "source_evidence": {
            "vllm_base_commit": VLLM_BASE_COMMIT,
            "pristine_unified_attention_sha256": PRISTINE_UNIFIED_ATTENTION_SHA256,
            "base_patcher_sha256": BASE_PATCHER_SHA256,
            "topology_parent_sha256": topology.PHYSICAL_PARENT_SHA256,
            "topology_ancestry_sha256": topology.TREE_ANCESTRY_SHA256,
            "curated_nsys_attribution_sha256": CURATED_NSYS_ATTRIBUTION_SHA256,
            "old_attribution_only_unified_attention_ms_per_event": 6.967564,
            "old_attribution_is_current_speed_claim": False,
            "repository_sources": sources,
        },
        "exact_contract": json.loads(json.dumps(asdict(contract))),
        "fusion_verdict": fusion_verdict(contract),
        "static_ledger": static_ledger(),
        "next_safe_reduction_review": {
            "candidate": "scalar paged-block lookup for each aligned 32-row tile",
            "algebraically_safe_under_exact_contract": True,
            "logical_index_byte_saving_fraction_of_full_tile_kv_payload": (
                (VECTOR_PAGE_INDEX_BYTES_PER_TILE - SCALAR_PAGE_INDEX_BYTES_PER_TILE)
                / (KV_BYTES_PER_SEQUENCE_ROW * TILE_ROWS)
            ),
            "physical_hbm_saving_proven": False,
            "reason_not_integrated": (
                "All 32 current lane addresses select the same cached block-table "
                "entry. Host inspection cannot prove a physical transaction or "
                "runtime win, and the optimistic logical saving is below 0.4%."
            ),
        },
        "implementation": {
            "runtime_kernel_changed": False,
            "launcher_changed": False,
            "production_credentials_changed": False,
            "default_behavior_changed": False,
            "stop_reason": (
                "No lossless attention-only fusion or defensible shared-KV/index "
                "traffic reduction remains after the recurrent barriers."
            ),
        },
        "sanitization": {
            "absolute_host_paths": False,
            "request_payloads": False,
            "environment_dump": False,
            "runtime_logs": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--first-sequence-length", type=int)
    args = parser.parse_args()
    payload = build_audit()
    if args.first_sequence_length is not None:
        payload["static_ledger"] = static_ledger(args.first_sequence_length)
    rendered = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.write_text(rendered, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
