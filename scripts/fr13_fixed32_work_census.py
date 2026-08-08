#!/usr/bin/env python3
"""Fail-closed validator for the fixed-32 hot-path work census.

Version 7 separates semantic activity from physical work. Every event carries
an arm-local ``event_index`` and the global SFWD ``forward_step_index`` plus
exact counters for every fixed-capacity publish, pack, remap, conv, and
committer route. Ordered per-request SHA-256 values permit an exact,
privacy-safe join to the authenticated ingress ledger without publishing
EngineCore request IDs. Conv pregather is attested as one in-graph launch
before all 48 GDN consumers, rather than a host-side stage after the prior
event. A final terminal record binds the complete ordered event body with
SHA-256, so a truncated or partially flushed JSONL cannot pass.

Examples:

    python3 scripts/fr13_fixed32_work_census.py \
        --tail tail-census.jsonl --hydra hydra-census.jsonl

    python3 scripts/fr13_fixed32_work_census.py \
        --tail tail-b1.jsonl --hydra hydra-b1.jsonl --batch-size 1

    python3 scripts/fr13_fixed32_work_census.py --self-test

Validation requires at least one fully occupied B1 and B4 event by default.
Event counts and global forward-index origins may differ between arms; every
event must still reduce to the same exact per-request work signature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fr13_fixed32_topology import (
    ACCEPTED_PATH_CAPACITY,
    ARCTIC_LOOKUP_CALLS_PER_REQUEST,
    ARCTIC_LOOKUP_CHAINS,
    ARCTIC_LOOKUP_TOKENS_PER_REQUEST,
    ARCTIC_MAIN_TAIL_LENGTH,
    COMMIT_PATH_CAP,
    COMMITTER_NEUTRALIZE_OPS,
    COMMITTER_RING_GATHERS,
    CONV_COMMIT_LAYERS,
    CONV_PREGATHER_BLOCK,
    CONV_PREGATHER_LAYERS,
    CONV_PREGATHER_ROW_ELEMS,
    GDN_CONV_CHANNELS,
    GDN_CONV_KERNEL_SIZE,
    GDN_CONV_STATE_LENGTH,
    GDN_LAUNCHES,
    GDN_LAYERS,
    GDN_LEVEL_MAX_LENGTHS,
    GDN_LEVEL_PATH_COUNTS,
    GDN_PADDED_SLOTS,
    GDN_PATH_PROGRAMS,
    HYDRA27_ACTIVE_DRAFTS,
    HYDRA27_VALID_MASK,
    KV_REMAP_CACHE_TENSORS,
    KV_REMAP_DRAFTER_APPLY_CACHE_CALLS,
    KV_REMAP_DRAFTER_CACHE_TENSORS,
    KV_REMAP_DRAFTER_PAIR_SLOTS,
    KV_REMAP_DRAFTER_PREPARE_CALLS,
    KV_REMAP_GROUPS,
    KV_REMAP_PAIR_SLOTS,
    KV_REMAP_PATH_CAPACITY,
    KV_REMAP_PLANES,
    KV_REMAP_TARGET_APPLY_CACHE_CALLS,
    KV_REMAP_TARGET_CACHE_TENSORS,
    KV_REMAP_TARGET_PAIR_SLOTS,
    KV_REMAP_TARGET_PREPARE_CALLS,
    MODEL_LAYERS,
    MTP_FORWARD_CALLS,
    OUTPUT_PUBLISH_CAPACITY,
    PHYSICAL_DRAFTS,
    PHYSICAL_PARENT_SHA256,
    PHYSICAL_ROWS,
    RESCUE_CARRY_SLOTS_PER_REQUEST,
    REQUEST_KEY_PATH_CAPACITY,
    SAMPLER_MAX_FANOUT,
    TAIL6_ACTIVE_DRAFTS,
    TAIL6_VALID_MASK,
    TAW_CHILD_LANES,
    TAW_PATH_SCATTER_SLOTS,
    TAW_ROW_SCATTER_SLOTS,
    TAW_UNIFORM_SLOTS,
    TREE_ATTENTION_LAYERS,
    TREE_ANCESTRY_SHA256,
    WALK_CAP,
)

SCHEMA = "fr13-fixed32-work-census-v12"
TERMINAL_SCHEMA = "fr13-fixed32-work-census-terminal-v12"
REPORT_SCHEMA = "fr13-fixed32-work-census-report-v12"
ARM_REPORT_SCHEMA = "fr13-fixed32-work-census-arm-report-v12"
SELF_TEST_SCHEMA = "fr13-fixed32-work-census-self-test-v12"

TAIL_MODE = "tail6_fixed32"
HYDRA_MODE = "hydra27_fixed32"
MODE_SEMANTICS = {
    TAIL_MODE: {
        "active_nodes": TAIL6_ACTIVE_DRAFTS,
        "valid_mask": TAIL6_VALID_MASK,
    },
    HYDRA_MODE: {
        "active_nodes": HYDRA27_ACTIVE_DRAFTS,
        "valid_mask": HYDRA27_VALID_MASK,
    },
}
SUPPORTED_BATCH_SIZES = (1, 2, 3, 4)
SUPPORTED_CAMPAIGN_CAPACITIES = (1, 4)

VERIFY_ROWS_PER_REQUEST = PHYSICAL_ROWS
TREE_ROWS_PER_REQUEST = PHYSICAL_ROWS
TREE_BIAS_SHAPE = (PHYSICAL_ROWS, PHYSICAL_ROWS)
TREE_CALLS_PER_EVENT = TREE_ATTENTION_LAYERS
GDN_SCAN_CALLS_PER_REQUEST = GDN_LAYERS

GDN_LAUNCHES_PER_SCAN = GDN_LAUNCHES
GDN_PATH_PROGRAMS_PER_SCAN = GDN_PATH_PROGRAMS
GDN_PADDED_SLOTS_PER_SCAN = GDN_PADDED_SLOTS
GDN_NODES_PER_SCAN = PHYSICAL_ROWS
GDN_CRITICAL_PATH = WALK_CAP
GDN_GRID_Z = GDN_LEVEL_PATH_COUNTS
GDN_MAX_PATH_LENGTHS = GDN_LEVEL_MAX_LENGTHS
GDN_EXPORT_OR_MASK = 0x4213

TAW_LOOP_ITERATIONS = WALK_CAP
TAW_CHILD_LANES_PER_REQUEST = TAW_CHILD_LANES
TAW_ROWS_PER_REQUEST = WALK_CAP
TAW_BUFFER_CAPACITY = OUTPUT_PUBLISH_CAPACITY
TAW_ROUTE = "fixed32_pytorch_exact_float_triton_integer_commit"
TAW_NATIVE_PRECOMPUTE_ROUTE = (
    "fixed32_native_precompute_byte_ab_reference_return"
)
TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE = (
    "fixed32_native_precompute_production_candidate_return"
)
TAW_CFWD_LOGIT_DIRECT_PRODUCTION_ROUTE = (
    "fixed32_cfwd_logit_direct_production_candidate_return"
)
TAW_EXACT_COMMIT_LAUNCHES = WALK_CAP
TAW_EXACT_COMMIT_PROGRAMS_PER_REQUEST = WALK_CAP
TAW_ALL_PARENT_SELF_ROWS_PER_REQUEST = 13
TAW_ALL_PARENT_TARGET_ROWS_PER_REQUEST = 17
TAW_SOURCE_CONTRACT_SCHEMA = "fr13-fixed32-taw-all-parent-v7"
TAW_SOURCE_CONTRACT_SHA256 = (
    "2b1cc55c6ec3d45c2d6ad0a21be4dc76685df4c974ae7fcfa421d5824a5c1ffb"
)
TAW_CFWD_LOGIT_DIRECT_SOURCE_SCHEMA = (
    "fr13.fixed32.cfwd_logit_direct.integration_source.v2"
)
TAW_CFWD_LOGIT_DIRECT_SOURCE_SHA256 = (
    "421465c6c04de8c26e3ea724a7d2f0d3f00fe50b4fdc9f57c35e71e71212297b"
)
TAW_TENSOR_CALL_CENSUS = {
    "walk_levels": 12,
    "full_vocab_row_gathers": 24,
    "full_vocab_fp32_casts": 24,
    "full_vocab_softmax_calls": 24,
    "full_vocab_normalizations": 36,
    "full_vocab_cdf_calls": 24,
    "source_cdf_calls": 12,
    "qmix_zero_fills": 12,
    "qmix_scatter_add_calls": 12,
    "residual_subtract_calls": 12,
    "residual_clamp_calls": 12,
    "residual_where_calls": 24,
    "output_scatter_calls": 0,
    "path_scatter_calls": 0,
    "exact_commit_launches": 12,
    "exact_commit_programs_per_request": 12,
    "floating_sampling_reimplementation": False,
}
TAW_NATIVE_PRECOMPUTE_TENSOR_CALL_CENSUS = {
    **TAW_TENSOR_CALL_CENSUS,
    "walk_levels": 13,
    "full_vocab_row_gathers": 54,
    "full_vocab_fp32_casts": 26,
    "full_vocab_softmax_calls": 26,
    "full_vocab_normalizations": 83,
    "full_vocab_cdf_calls": 54,
    "source_cdf_calls": 29,
    "qmix_zero_fills": 29,
    "qmix_scatter_add_calls": 29,
    "residual_subtract_calls": 29,
    "residual_clamp_calls": 29,
    "residual_where_calls": 58,
    "exact_commit_launches": 13,
    "exact_commit_programs_per_request": 13,
}
TAW_NATIVE_PRECOMPUTE_PRODUCTION_TENSOR_CALL_CENSUS = {
    **TAW_TENSOR_CALL_CENSUS,
    "walk_levels": 1,
    "full_vocab_row_gathers": 30,
    "full_vocab_fp32_casts": 2,
    "full_vocab_softmax_calls": 2,
    "full_vocab_normalizations": 47,
    "full_vocab_cdf_calls": 30,
    "source_cdf_calls": 17,
    "qmix_zero_fills": 17,
    "qmix_scatter_add_calls": 17,
    "residual_subtract_calls": 17,
    "residual_clamp_calls": 17,
    "residual_where_calls": 34,
    "exact_commit_launches": 1,
    "exact_commit_programs_per_request": 1,
}
TAW_CFWD_LOGIT_DIRECT_PRODUCTION_TENSOR_CALL_CENSUS = {
    **TAW_TENSOR_CALL_CENSUS,
    "walk_levels": 1,
    "full_vocab_row_gathers": 0,
    "full_vocab_fp32_casts": 0,
    "full_vocab_softmax_calls": 0,
    "full_vocab_normalizations": 0,
    "full_vocab_cdf_calls": 0,
    "source_cdf_calls": 0,
    "qmix_zero_fills": 0,
    "qmix_scatter_add_calls": 0,
    "residual_subtract_calls": 0,
    "residual_clamp_calls": 0,
    "residual_where_calls": 0,
    "exact_commit_launches": 1,
    "exact_commit_programs_per_request": 1,
}
TAW_COUNT_ROUTE = "preseeded_cuda_fixed31"
TAW_RNG_ROUTE = "bulk_device_generator"
TAW_VOCAB_SIZE = 248_320

COMMITTER_PATH_CAPACITY = COMMIT_PATH_CAP
COMMITTER_RING_GATHER_OPS = COMMITTER_RING_GATHERS
COMMITTER_RING_LAYER_PATH_ROWS_PER_REQUEST = (
    COMMITTER_RING_GATHERS * GDN_LAYERS * COMMIT_PATH_CAP
)

OUTPUT_PUBLISH_ROUTE = "device_fixed32"
ACCEPTED_PATH_PACK_ROUTE = "device_fixed16"
REQUEST_KEY_PACK_ROUTE = "device_rowmap"
KV_REMAP_ROUTE = "syncfree_target16_postsample_drafter1_postforward"
CONV_COMMIT_ROUTE = "fixed32_direct_source_col0"
CONV_COMMIT_SOURCE_ROWS = PHYSICAL_ROWS + GDN_CONV_KERNEL_SIZE
CONV_ROW_GUARD_ROUTE = (
    "fixed32_triton_alias3_ownerpath_warp32_physical32_v4"
)
CONV_ROW_GUARD_PROGRAMS_PER_REQUEST = CONV_COMMIT_LAYERS
CONV_ROW_GUARD_ALIAS_WIDTH = 3
CONV_ROW_GUARD_COMPARE_CAPACITY = 16
CONV_ROW_GUARD_PATH_PROGRAMS_PER_REQUEST = 1
CONV_ROW_GUARD_PATH_VECTOR_LOADS_PER_REQUEST = 1
CONV_ROW_GUARD_ALIAS_PROGRAMS_PER_EVENT = 1
CONV_ROW_GUARD_ALIAS_VECTOR_LOADS_PER_EVENT = 2
CONV_ROW_GUARD_SELECTED_ROW_LOADS_PER_PROGRAM = 0
CONV_ROW_GUARD_PEER_TOPOLOGY_PROOF = "preseed_lease_audit"
CONV_PREGATHER_ROUTE = "in_graph_preconsume"
COMMITTER_ROUTE = "fixed16_device_fill_graph"
if TREE_ATTENTION_LAYERS + GDN_LAYERS != MODEL_LAYERS:
    raise RuntimeError(
        "fixed32 topology model partition is inconsistent: "
        f"{TREE_ATTENTION_LAYERS}+{GDN_LAYERS}!={MODEL_LAYERS}"
    )

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "event_id",
        "event_index",
        "forward_step_index",
        "producer_pid",
        "event_complete",
        "mode",
        "batch_size",
        "physical_drafts",
        "verify_rows",
        "active_nodes",
        "valid_mask",
        "batch_purity",
        "drafter",
        "drafter_runtime",
        "tree_attn",
        "gdn",
        "taw",
        "output_publish",
        "accepted_path_pack",
        "request_key_pack",
        "kv_remap",
        "conv_commit",
        "conv_pregather",
        "committer",
        "failures",
    }
)
GDN_COMPARATOR_SCHEMA = "fr13.fixed32.gdn_single_launch.comparator_event.v1"
GDN_COMPARATOR_CANDIDATES = frozenset(
    (
        "fixed32_gdn_single_launch_tree_v2",
        "fixed32_gdn_single_launch_gqa_group3_v1",
    )
)
GDN_COMPARATOR_KEYS = frozenset(
    {
        "schema",
        "mode",
        "batch_size",
        "runtime_capture_manifest_sha256",
        "structural_graph_signature",
        "reference",
        "candidate",
        "reference_physical_launches_per_request_layer",
        "candidate_physical_launches_per_request_layer",
        "records",
        "compared_byte_surfaces",
        "raw_byte_equal",
        "state_restored",
        "reference_served",
        "candidate_served",
        "comparison_order",
        "census_event_id",
        "census_event_index",
        "census_forward_step_index",
        "request_id_sha256s",
        "observed_task_marker",
    }
)
GDN_COMPARATOR_SURFACES = (
    "output",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "flags",
    "counter",
)
BATCH_PURITY_KEYS = frozenset(
    {
        "batch_rows",
        "spec_rows",
        "physical_draft_counts",
        "mixed_pseudo_rows",
        "all_physical_31",
    }
)
DRAFTER_KEYS = frozenset(
    {
        "mtp_forward_calls",
        "mtp_forward_rows",
        "arctic_lookup_calls",
        "arctic_requested_tokens",
        "main_tail_length",
        "rescue_chains",
        "carry_fill_slots",
        "pack_columns",
        "packed_rows",
    }
)
DRAFTER_RUNTIME_KEYS = frozenset(
    {
        "association",
        "forward_step_index",
        "batch_size",
        "request_ids_sha256",
        "request_id_sha256s",
        "proposal_begins",
        "proposal_ends",
        "graph_id",
        "graph_signature",
        "graph_captures",
        "graph_replays",
        "mtp_observation",
        "mtp_forward_calls",
        "mtp_forward_rows",
        "arctic_ledger",
        "arctic_lookup_calls",
        "arctic_requested_tokens",
        "merge_fill_calls",
        "merge_fill_columns",
        "merge_fill_rows",
        "rescue_carry_slots",
        "publish_shape",
        "physical_parent_sha256",
        "outer_handoff_calls",
    }
)
DRAFTER_LEDGER_KEYS = frozenset({"kind", "calls", "tokens"})
TREE_KEYS = frozenset(
    {
        "calls",
        "q_rows",
        "bias_shape",
        "physical_parent_digest",
        "bias_digest",
    }
)
GDN_KEYS = frozenset(
    {
        "scan_calls",
        "launches",
        "path_programs",
        "padded_slots",
        "nodes",
        "critical_path",
        "grid_z",
        "max_path_lengths",
        "export_or_mask",
    }
)
TAW_KEYS = frozenset(
    {
        "route",
        "preseeded_batches",
        "topology_cache_hit",
        "cache_misses",
        "table_shape",
        "buffer_capacity",
        "loop_iterations",
        "uniform_slots",
        "child_lanes",
        "target_rows",
        "self_rows",
        "self_cdf_rows",
        "source_cdf_rows",
        "residual_cdf_rows",
        "qmix_rows",
        "residual_rows",
        "row_scatter_slots",
        "path_scatter_slots",
        "exact_commit_launches",
        "exact_commit_programs",
        "floating_sampling_reimplementation",
        "source_contract_schema",
        "source_contract_sha256",
        "tensor_call_census",
        "count_route",
        "rng_route",
        "vocab_size",
        "count_shape",
        "count_dtype",
        "count_stride",
        "count_contiguous",
        "draft_shape",
        "draft_dtype",
        "draft_stride",
        "draft_contiguous",
        "parent_shape",
        "parent_dtype",
        "parent_stride",
        "parent_contiguous",
        "bonus_shape",
        "bonus_dtype",
        "bonus_stride",
        "bonus_contiguous",
        "target_shape",
        "target_dtype",
        "target_stride",
        "target_contiguous",
        "self_shape",
        "self_dtype",
        "self_stride",
        "self_contiguous",
        "uniform_shape",
        "uniform_dtype",
        "uniform_stride",
        "uniform_contiguous",
        "child_table_shape",
        "child_counts_shape",
        "output_shape",
        "output_lens_shape",
        "accepted_path_shape",
        "accepted_lens_shape",
        "last_row_shape",
        "exact_current_shape",
        "exact_alive_shape",
    }
)
TAW_TENSOR_CALL_CENSUS_KEYS = frozenset(TAW_TENSOR_CALL_CENSUS)
OUTPUT_PUBLISH_KEYS = frozenset(
    {
        "route",
        "capacity",
        "requests",
        "launches",
        "slots_written",
        "accepted_rows_written",
        "host_materializations",
        "host_scalar_writes",
        "dtoh",
        "h2d",
        "fallback",
    }
)
ACCEPTED_PATH_PACK_KEYS = frozenset(
    {
        "route",
        "capacity",
        "requests",
        "pack_launches",
        "slots_written",
        "source_walk_slots",
        "lens_written",
        "host_path_items",
        "overflow",
        "fallback",
    }
)
REQUEST_KEY_PACK_KEYS = frozenset(
    {
        "route",
        "sampler_rows",
        "spec_rows",
        "map_passes",
        "path_slots_gathered",
        "lens_gathered",
        "zero_launches",
        "gather_launches",
        "host_dict_inserts",
        "host_hash_lookups",
        "missing",
        "fallback",
    }
)
KV_REMAP_KEYS = frozenset(
    {
        "route",
        "path_capacity",
        "pair_slots",
        "target_pair_slots",
        "drafter_pair_slots",
        "kv_groups",
        "target_cache_tensors",
        "drafter_cache_tensors",
        "kv_cache_tensors",
        "kv_planes",
        "target_prepare_calls",
        "drafter_prepare_calls",
        "prepare_calls",
        "target_apply_cache_calls",
        "drafter_apply_cache_calls",
        "apply_cache_calls",
        "src_pair_rows",
        "dst_pair_rows",
        "identity_safe_writes",
        "host_syncs",
        "skips",
        "fallback",
    }
)
CONV_COMMIT_KEYS = frozenset(
    {
        "route",
        "layers",
        "requests",
        "row_elems",
        "channels",
        "state_length",
        "source_rows_per_batch",
        "block",
        "direct_launches",
        "gather_launches",
        "scatter_launches",
        "direct_programs",
        "committed_rows",
        "source_staging_reused",
        "source_pointer_entries",
        "row_guard_route",
        "row_guard_kernel_launches",
        "row_guard_programs",
        "row_guard_physical_rows",
        "row_guard_path_capacity",
        "row_guard_alias_width",
        "row_guard_compare_capacity",
        "row_guard_path_validation_programs",
        "row_guard_path_vector_loads",
        "row_guard_alias_validation_programs",
        "row_guard_alias_vector_loads",
        "row_guard_selected_row_loads",
        "row_guard_peer_topology_proof",
        "row_guard_torch_index_transforms",
        "row_guard_async_scalar_reductions",
        "row_guard_async_assertions",
        "full_node_writebacks",
        "conv_remaps",
        "host_syncs",
        "skips",
        "fallback",
    }
)
CONV_PREGATHER_KEYS = frozenset(
    {
        "route",
        "layout_sha256",
        "stage_calls",
        "stage_before_all_consumes",
        "layers",
        "requests",
        "row_elems",
        "programs",
        "staged_rows",
        "consume_calls",
        "consume_hits",
        "consume_fallbacks",
        "freshness_matches",
    }
)
COMMITTER_KEYS = frozenset(
    {
        "route",
        "layers",
        "requests",
        "path_capacity",
        "layout_slots",
        "ring_gather_ops",
        "ring_layer_path_rows",
        "neutralize_ops",
        "fused_layer_calls",
        "graph_replays",
        "graph_captures",
        "host_lens_readbacks",
        "host_flag_readbacks",
        "pointer_table_rebuilds",
        "overflow",
        "fallback",
        "graph_dead",
    }
)
FAILURE_KEYS = frozenset(
    {
        "fallback",
        "overflow",
        "graph_dead",
        "mixed_pseudo",
        "taw_cache_miss",
    }
)
TERMINAL_KEYS = frozenset(
    {
        "schema",
        "mode",
        "producer_pid",
        "final",
        "event_count",
        "first_event_index",
        "last_event_index",
        "first_forward_step_index",
        "last_forward_step_index",
        "events_sha256",
        "batch_histogram",
        "drafter_graph_registry",
        "forward_graph_registry",
        "conv_pregather_auxiliary",
        "nonpure_dispatch",
        "nonpure_committer_replays_by_batch",
        "scope",
    }
)
NONPURE_DISPATCH_KEYS = frozenset(
    {
        "guarded_steps",
        "piecewise_steps",
        "none_steps",
        "forbidden_full_steps",
    }
)
DRAFTER_GRAPH_REGISTRY_KEYS = frozenset(
    {
        "batch_size",
        "graph_signature",
        "captures",
        "capture_origin",
        "measured_replays",
        "unmeasured_replays",
    }
)
FORWARD_GRAPH_REGISTRY_KEYS = frozenset(
    {
        "batch_size",
        "graph_signature",
        "conv_layout_sha256",
        "captures",
        "capture_origin",
        "stage_calls",
        "stage_before_all_consumes",
        "layers",
        "requests",
        "row_elems",
        "programs",
        "ssi_pointer_entries",
        "ssi_groups",
        "source_validations",
        "staged_rows",
        "consume_calls",
        "consume_hits",
        "consume_fallbacks",
        "freshness_matches",
        "measured_replays",
    }
)
CONV_PREGATHER_AUXILIARY_KEYS = frozenset(
    {
        "profile_capture_stages",
        "aux_capture_stages",
        "host_actual_stages",
        "host_actual_stages_by_batch",
    }
)
SCOPE_KEYS = frozenset(
    {
        "direct_event_observations",
        "contract_derived",
        "data_dependent_unproven",
    }
)
FIXED_WORK_SCOPE = {
    "direct_event_observations": [
        "forward_graph_replay",
        "drafter_graph_replay",
        "drafter_arctic_fill_call_boundaries",
        "drafter_bx31_publish",
        "preforward_request_key_tensor_ops",
        "post_taw_output_path_request_key_tensor_ops",
        "committer_replay_delta",
        "conv_commit_direct_launch_delta",
        "conv_pregather_capture_manifest_bound_replay",
        "kv_fixed16_geometry",
        "taw_live_layout_and_route",
    ],
    "contract_derived": [
        "tree_attn_inner_calls",
        "gdn_inner_launches",
        "gdn_export_or_mask",
        "taw.ast_pinned_fixed_12_iteration_tensor_call_census",
        "committer_graph_inner_ops",
        "kv_inner_apply_calls",
        "conv_commit_inner_launch_programs",
        "conv_pregather_inner_launch_programs",
    ],
    "data_dependent_unproven": [
        "rejection_sampler.parse_output.accepted_length_host_materialization",
        "scheduler.slot_lifecycle",
        "taw.indexed_addresses_cache_atomic_contention_and_cycles",
        "tree_attn.kv_sequence_lengths_block_addresses_cache_and_cycles",
        "committer.accepted_path_gather_addresses_cache_and_cycles",
        "committer.direct_ring_rows_root_plus_accepted_depth",
        "kv_remap.accepted_path_src_dst_addresses_cache_and_cycles",
    ],
}


class CensusError(RuntimeError):
    """A census record or campaign failed a fixed-work requirement."""


class DuplicateJsonKey(ValueError):
    """A JSON object contained a duplicate key."""


@dataclass(frozen=True)
class ValidatedEvent:
    """Validated semantic identity plus its normalized work signature."""

    source: str
    event_id: str
    event_index: int
    forward_step_index: int
    producer_pid: int
    mode: str
    batch_size: int
    conv_layout_sha256: str
    drafter_graph_signature: str
    drafter_graph_captures: int
    normalized_work: dict[str, Any]


LocatedRecord = tuple[object, str]


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CensusError(f"{label}: expected an object")
    if not all(isinstance(key, str) for key in value):
        raise CensusError(f"{label}: object keys must be strings")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise CensusError(
            f"{label}: schema keys differ; missing={missing}, unknown={unknown}"
        )


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CensusError(f"{label}: expected an integer")
    if value < minimum:
        raise CensusError(f"{label}: expected >= {minimum}, got {value}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CensusError(f"{label}: expected a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    digest = _string(value, label)
    if SHA256_RE.fullmatch(digest) is None:
        raise CensusError(f"{label}: expected lowercase SHA-256 hex")
    return digest


def _string_tuple(value: object, label: str, *, length: int) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise CensusError(f"{label}: expected a string array of length {length}")
    return tuple(
        _string(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _integer_tuple(value: object, label: str, *, length: int) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise CensusError(f"{label}: expected an integer array of length {length}")
    return tuple(value)


def _integer_pairs(
    value: object, label: str, *, length: int
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list) or len(value) != length:
        raise CensusError(f"{label}: expected {length} integer pairs")
    pairs = []
    for index, raw_pair in enumerate(value):
        pair = _integer_tuple(raw_pair, f"{label}[{index}]", length=2)
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)


def _tensor_layout(
    section: Mapping[str, Any],
    *,
    prefix: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    expected_stride: tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    shape = _integer_tuple(
        section[f"{prefix}_shape"],
        f"{label}.{prefix}_shape",
        length=len(expected_shape),
    )
    dtype = _string(section[f"{prefix}_dtype"], f"{label}.{prefix}_dtype")
    stride = _integer_tuple(
        section[f"{prefix}_stride"],
        f"{label}.{prefix}_stride",
        length=len(expected_stride),
    )
    contiguous = section[f"{prefix}_contiguous"]
    if contiguous is not True:
        raise CensusError(f"{label}.{prefix}_contiguous: expected literal true")
    _expect(shape, expected_shape, f"{label}.{prefix}_shape")
    _expect(dtype, expected_dtype, f"{label}.{prefix}_dtype")
    _expect(stride, expected_stride, f"{label}.{prefix}_stride")
    return {
        "shape": list(shape),
        "dtype": dtype,
        "stride": list(stride),
        "contiguous": True,
    }


def _expect(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise CensusError(f"{label}: expected {expected!r}, got {actual!r}")


def _fixed_section(
    raw: object,
    *,
    keys: frozenset[str],
    expected: Mapping[str, bool | int | str],
    label: str,
) -> dict[str, bool | int | str]:
    section = _mapping(raw, label)
    _exact_keys(section, keys, label)
    values: dict[str, bool | int | str] = {}
    for name, expected_value in expected.items():
        field_label = f"{label}.{name}"
        if isinstance(expected_value, bool):
            actual: bool | int | str = section[name]
            if actual is not expected_value:
                raise CensusError(
                    f"{field_label}: expected literal {str(expected_value).lower()}"
                )
        elif isinstance(expected_value, str):
            actual = _string(section[name], field_label)
        else:
            actual = _integer(section[name], field_label)
        _expect(actual, expected_value, field_label)
        values[name] = actual
    return values


def _drafter_graph_signature(batch_size: int) -> str:
    manifest = {
        "schema": "fr13-fixed32-drafter-graph-manifest-v2",
        "batch_size": batch_size,
        "mtp_forward_calls": MTP_FORWARD_CALLS,
        "mtp_forward_rows": MTP_FORWARD_CALLS * batch_size,
        "tree_attn_calls": MTP_FORWARD_CALLS,
        "tree_attn_rows": MTP_FORWARD_CALLS * batch_size,
        "tree_attn_layer": "mtp.layers.0.self_attn.attn",
        "tree_attn_bias_shape": [1, 1],
    }
    canonical = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _fixture_conv_layout_signature(batch_size: int) -> str:
    payload = {
        "schema": "fr13-fixed32-fixture-conv-layout-v1",
        "batch_size": batch_size,
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


UNFUSED_KERNEL_SHAPE = "conv_pregather"
FUSED_KERNEL_SHAPE = "sfwd_conv_postprep_fused"
KERNEL_SHAPES = (UNFUSED_KERNEL_SHAPE, FUSED_KERNEL_SHAPE)

STRUCTURAL_MANIFEST_SCHEMA = "fr13-fixed32-forward-graph-structural-manifest-v1"
STRUCTURAL_MANIFEST_FUSED_SCHEMA = (
    "fr13-fixed32-forward-graph-structural-manifest-sfwd-fused-v1"
)

SFWD_CONV_POSTPREP_ROUTE = "fused_conv_postprep_single_kernel"

# Two canonical structural references, each pinned. An arm matches exactly
# one. These are written-down constants, never derived from what a runtime
# was observed to do: a runtime that disagrees with its arm's reference is a
# failure, not a new reference.
FORWARD_GRAPH_STRUCTURAL_SIGNATURES = {
    UNFUSED_KERNEL_SHAPE: {
        1: "2373bfbd2ac6ab7a6fd67af5570385f2aea2a16a1e80b804bdf12e092f319423",
        2: "508a856a418e5954083e8aaf93efa1e6f89b65562f3c20414418b9dd640e5362",
        3: "f451f42fc2803a8a3a7d7359e39487ba944fc27618a043d5026d766f2e94cba7",
        4: "025bc236c194ee88a512ccb633b0247cfa3e4a15e17975061083b62d7be921cb",
    },
    FUSED_KERNEL_SHAPE: {
        1: "f15d7b06d0e3c72ae6d53aa43c2f27501ef0c4238cbcd01e65844eaf1b63b875",
        2: "f85e059f6937e2d0cd63d9173bf088c545b18bd6fccceb76b1d20cfcb05dc7f1",
        3: "8d0c4bc4e81ea41831a45c4c069ee0e2ab67a585379f1347d8918a6b399d011d",
        4: "f8a2ca1fc246a3058430a653b1fd27796622e21bdf80db5b93a2d19e831ed547",
    },
}


def _validated_kernel_shape(kernel_shape: str) -> str:
    if kernel_shape not in KERNEL_SHAPES:
        raise ValueError(
            f"forward graph kernel_shape must be in {KERNEL_SHAPES}, "
            f"got {kernel_shape!r}"
        )
    return kernel_shape


def forward_graph_structural_manifest(
    batch_size: int, *, kernel_shape: str = UNFUSED_KERNEL_SHAPE
) -> dict[str, Any]:
    """Return the mode-independent final-FULL graph contract for one B.

    ``kernel_shape`` selects which canonical reference applies. The workload
    identity -- descriptor geometry, tree attention, GDN -- is shared and
    strict across both: that is what makes a stock/candidate pair comparable.
    Only the conv work structure differs, because the SFWD fusion replaces the
    pregather stage kernel and the 48 per-layer consumes with one fused kernel
    per layer.
    """
    _validated_kernel_shape(kernel_shape)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size not in SUPPORTED_BATCH_SIZES
    ):
        raise ValueError(
            f"forward graph batch_size must be in {SUPPORTED_BATCH_SIZES}"
        )
    if kernel_shape == FUSED_KERNEL_SHAPE:
        return {
            "schema": STRUCTURAL_MANIFEST_FUSED_SCHEMA,
            "batch_size": batch_size,
            "descriptor_geometry": {
                "physical_drafts": PHYSICAL_DRAFTS,
                "verify_rows_per_request": VERIFY_ROWS_PER_REQUEST,
                "verify_rows": VERIFY_ROWS_PER_REQUEST * batch_size,
                "model_layers": MODEL_LAYERS,
            },
            "tree_attention": {
                "layers": TREE_ATTENTION_LAYERS,
                "calls_per_event": TREE_CALLS_PER_EVENT,
                "q_rows_per_call": TREE_ROWS_PER_REQUEST * batch_size,
                "bias_shape": list(TREE_BIAS_SHAPE),
                "physical_parent_sha256": PHYSICAL_PARENT_SHA256,
            },
            "gdn": {
                "layers": GDN_LAYERS,
                "scan_calls": GDN_SCAN_CALLS_PER_REQUEST * batch_size,
                "launches_per_scan": GDN_LAUNCHES_PER_SCAN,
                "path_programs_per_scan": GDN_PATH_PROGRAMS_PER_SCAN,
                "padded_slots_per_scan": GDN_PADDED_SLOTS_PER_SCAN,
                "nodes_per_scan": GDN_NODES_PER_SCAN,
                "critical_path": GDN_CRITICAL_PATH,
                "grid_z": list(GDN_GRID_Z),
                "max_path_lengths": list(GDN_MAX_PATH_LENGTHS),
                "export_or_mask": GDN_EXPORT_OR_MASK,
            },
            "sfwd_conv_postprep": {
                "route": SFWD_CONV_POSTPREP_ROUTE,
                "layers": CONV_PREGATHER_LAYERS,
                "requests": batch_size,
                "calls": CONV_PREGATHER_LAYERS,
                "calls_per_layer": 1,
                "capture_guard": True,
                # The fused kernel subsumes both of these; they are pinned at
                # zero so a run that also took the unfused route fails.
                "stage_calls": 0,
                "consume_calls": 0,
                "source_validations": 0,
                "freshness_matches": 0,
                "staged_rows": 0,
            },
        }
    conv_pregather_programs = (
        CONV_PREGATHER_LAYERS
        * batch_size
        * (
            (
                CONV_PREGATHER_ROW_ELEMS
                + CONV_PREGATHER_BLOCK
                - 1
            )
            // CONV_PREGATHER_BLOCK
        )
    )
    return {
        "schema": "fr13-fixed32-forward-graph-structural-manifest-v1",
        "batch_size": batch_size,
        "descriptor_geometry": {
            "physical_drafts": PHYSICAL_DRAFTS,
            "verify_rows_per_request": VERIFY_ROWS_PER_REQUEST,
            "verify_rows": VERIFY_ROWS_PER_REQUEST * batch_size,
            "model_layers": MODEL_LAYERS,
        },
        "tree_attention": {
            "layers": TREE_ATTENTION_LAYERS,
            "calls_per_event": TREE_CALLS_PER_EVENT,
            "q_rows_per_call": TREE_ROWS_PER_REQUEST * batch_size,
            "bias_shape": list(TREE_BIAS_SHAPE),
            "physical_parent_sha256": PHYSICAL_PARENT_SHA256,
        },
        "gdn": {
            "layers": GDN_LAYERS,
            "scan_calls": GDN_SCAN_CALLS_PER_REQUEST * batch_size,
            "launches_per_scan": GDN_LAUNCHES_PER_SCAN,
            "path_programs_per_scan": GDN_PATH_PROGRAMS_PER_SCAN,
            "padded_slots_per_scan": GDN_PADDED_SLOTS_PER_SCAN,
            "nodes_per_scan": GDN_NODES_PER_SCAN,
            "critical_path": GDN_CRITICAL_PATH,
            "grid_z": list(GDN_GRID_Z),
            "max_path_lengths": list(GDN_MAX_PATH_LENGTHS),
            "export_or_mask": GDN_EXPORT_OR_MASK,
        },
        "conv_pregather": {
            "route": CONV_PREGATHER_ROUTE,
            "stage_calls": 1,
            "stage_before_all_consumes": True,
            "layers": CONV_PREGATHER_LAYERS,
            "requests": batch_size,
            "row_elems": CONV_PREGATHER_ROW_ELEMS,
            "block": CONV_PREGATHER_BLOCK,
            "grid": [
                CONV_PREGATHER_LAYERS,
                batch_size,
                (
                    CONV_PREGATHER_ROW_ELEMS
                    + CONV_PREGATHER_BLOCK
                    - 1
                )
                // CONV_PREGATHER_BLOCK,
            ],
            "programs": conv_pregather_programs,
            "ssi_pointer_entries": CONV_PREGATHER_LAYERS,
            "ssi_groups": 3,
            "source_validations": CONV_PREGATHER_LAYERS,
            "staged_rows": CONV_PREGATHER_LAYERS * batch_size,
            "consume_calls": CONV_PREGATHER_LAYERS,
            "consume_hits": CONV_PREGATHER_LAYERS,
            "consume_fallbacks": 0,
            "freshness_matches": CONV_PREGATHER_LAYERS,
        },
    }


def forward_graph_structural_signature(
    batch_size: int, *, kernel_shape: str = UNFUSED_KERNEL_SHAPE
) -> str:
    """Hash the canonical structural facts for one arm's kernel shape.

    The result is checked against the pinned table: the canonical references
    are written down, so an accidental edit to a manifest fails here rather
    than silently reclassifying an arm.
    """
    _validated_kernel_shape(kernel_shape)
    manifest = forward_graph_structural_manifest(
        batch_size, kernel_shape=kernel_shape
    )
    canonical = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    pinned = FORWARD_GRAPH_STRUCTURAL_SIGNATURES[kernel_shape].get(batch_size)
    if pinned is not None and digest != pinned:
        raise CensusError(
            "forward graph structural manifest drifted from its pinned "
            f"signature: kernel_shape={kernel_shape} batch_size={batch_size} "
            f"computed={digest} pinned={pinned}"
        )
    return digest


def validate_event(raw: object, *, source: str) -> ValidatedEvent:
    """Validate one event and return its arm-independent work signature."""

    event = _mapping(raw, source)
    expected_event_keys = TOP_LEVEL_KEYS | (
        {"gdn_comparator"} if "gdn_comparator" in event else set()
    )
    _exact_keys(event, frozenset(expected_event_keys), source)

    _expect(_string(event["schema"], f"{source}.schema"), SCHEMA, f"{source}.schema")
    event_id = _string(event["event_id"], f"{source}.event_id")
    event_index = _integer(event["event_index"], f"{source}.event_index")
    forward_step_index = _integer(
        event["forward_step_index"], f"{source}.forward_step_index"
    )
    producer_pid = _integer(event["producer_pid"], f"{source}.producer_pid", minimum=1)
    if event["event_complete"] is not True:
        raise CensusError(f"{source}.event_complete: expected literal true")
    mode = _string(event["mode"], f"{source}.mode")
    if mode not in MODE_SEMANTICS:
        raise CensusError(
            f"{source}.mode: expected one of {sorted(MODE_SEMANTICS)}, got {mode!r}"
        )
    batch_size = _integer(event["batch_size"], f"{source}.batch_size", minimum=1)
    if batch_size not in SUPPORTED_BATCH_SIZES:
        raise CensusError(
            f"{source}.batch_size: expected an occupied batch in [1, 4], got {batch_size}"
        )

    physical_drafts = _integer(event["physical_drafts"], f"{source}.physical_drafts")
    verify_rows = _integer(event["verify_rows"], f"{source}.verify_rows")
    active_nodes = _integer(event["active_nodes"], f"{source}.active_nodes")
    valid_mask = _integer(event["valid_mask"], f"{source}.valid_mask")
    semantics = MODE_SEMANTICS[mode]
    _expect(
        physical_drafts,
        PHYSICAL_DRAFTS,
        f"{source}.physical_drafts",
    )
    _expect(
        verify_rows,
        VERIFY_ROWS_PER_REQUEST * batch_size,
        f"{source}.verify_rows",
    )
    _expect(active_nodes, semantics["active_nodes"], f"{source}.active_nodes")
    _expect(valid_mask, semantics["valid_mask"], f"{source}.valid_mask")

    batch_purity_raw = _mapping(
        event["batch_purity"], f"{source}.batch_purity"
    )
    _exact_keys(
        batch_purity_raw, BATCH_PURITY_KEYS, f"{source}.batch_purity"
    )
    batch_rows = _integer(
        batch_purity_raw["batch_rows"],
        f"{source}.batch_purity.batch_rows",
    )
    spec_rows = _integer(
        batch_purity_raw["spec_rows"],
        f"{source}.batch_purity.spec_rows",
    )
    physical_draft_counts = _integer_tuple(
        batch_purity_raw["physical_draft_counts"],
        f"{source}.batch_purity.physical_draft_counts",
        length=batch_rows,
    )
    mixed_pseudo_rows = _integer(
        batch_purity_raw["mixed_pseudo_rows"],
        f"{source}.batch_purity.mixed_pseudo_rows",
    )
    all_physical_31 = batch_purity_raw["all_physical_31"]
    if not isinstance(all_physical_31, bool):
        raise CensusError(
            f"{source}.batch_purity.all_physical_31: expected a boolean"
        )
    _expect(batch_rows, batch_size, f"{source}.batch_purity.batch_rows")
    _expect(spec_rows, batch_size, f"{source}.batch_purity.spec_rows")
    _expect(
        mixed_pseudo_rows,
        batch_rows - spec_rows,
        f"{source}.batch_purity.mixed_pseudo_rows",
    )
    _expect(
        all_physical_31,
        all(value == PHYSICAL_DRAFTS for value in physical_draft_counts),
        f"{source}.batch_purity.all_physical_31",
    )
    _expect(
        physical_draft_counts,
        (PHYSICAL_DRAFTS,) * batch_size,
        f"{source}.batch_purity.physical_draft_counts",
    )

    drafter = _mapping(event["drafter"], f"{source}.drafter")
    _exact_keys(drafter, DRAFTER_KEYS, f"{source}.drafter")
    mtp_forward_calls = _integer(
        drafter["mtp_forward_calls"], f"{source}.drafter.mtp_forward_calls"
    )
    mtp_forward_rows = _integer(
        drafter["mtp_forward_rows"], f"{source}.drafter.mtp_forward_rows"
    )
    arctic_lookup_calls = _integer(
        drafter["arctic_lookup_calls"], f"{source}.drafter.arctic_lookup_calls"
    )
    arctic_requested_tokens = _integer(
        drafter["arctic_requested_tokens"],
        f"{source}.drafter.arctic_requested_tokens",
    )
    main_tail_length = _integer(
        drafter["main_tail_length"], f"{source}.drafter.main_tail_length"
    )
    rescue_chains = _integer_pairs(
        drafter["rescue_chains"],
        f"{source}.drafter.rescue_chains",
        length=len(ARCTIC_LOOKUP_CHAINS),
    )
    carry_fill_slots = _integer(
        drafter["carry_fill_slots"], f"{source}.drafter.carry_fill_slots"
    )
    pack_columns = _integer(drafter["pack_columns"], f"{source}.drafter.pack_columns")
    packed_rows = _integer(drafter["packed_rows"], f"{source}.drafter.packed_rows")
    _expect(
        mtp_forward_calls,
        MTP_FORWARD_CALLS,
        f"{source}.drafter.mtp_forward_calls",
    )
    _expect(
        mtp_forward_rows,
        MTP_FORWARD_CALLS * batch_size,
        f"{source}.drafter.mtp_forward_rows",
    )
    _expect(
        arctic_lookup_calls,
        ARCTIC_LOOKUP_CALLS_PER_REQUEST * batch_size,
        f"{source}.drafter.arctic_lookup_calls",
    )
    _expect(
        arctic_requested_tokens,
        ARCTIC_LOOKUP_TOKENS_PER_REQUEST * batch_size,
        f"{source}.drafter.arctic_requested_tokens",
    )
    _expect(
        main_tail_length,
        ARCTIC_MAIN_TAIL_LENGTH,
        f"{source}.drafter.main_tail_length",
    )
    _expect(
        rescue_chains,
        ARCTIC_LOOKUP_CHAINS,
        f"{source}.drafter.rescue_chains",
    )
    _expect(
        carry_fill_slots,
        RESCUE_CARRY_SLOTS_PER_REQUEST * batch_size,
        f"{source}.drafter.carry_fill_slots",
    )
    _expect(pack_columns, PHYSICAL_DRAFTS, f"{source}.drafter.pack_columns")
    _expect(
        packed_rows,
        PHYSICAL_DRAFTS * batch_size,
        f"{source}.drafter.packed_rows",
    )

    runtime_label = f"{source}.drafter_runtime"
    drafter_runtime = _mapping(event["drafter_runtime"], runtime_label)
    _exact_keys(drafter_runtime, DRAFTER_RUNTIME_KEYS, runtime_label)
    _expect(
        _string(drafter_runtime["association"], f"{runtime_label}.association"),
        "same_runner_step",
        f"{runtime_label}.association",
    )
    runtime_forward = _integer(
        drafter_runtime["forward_step_index"],
        f"{runtime_label}.forward_step_index",
    )
    runtime_batch = _integer(
        drafter_runtime["batch_size"],
        f"{runtime_label}.batch_size",
        minimum=1,
    )
    _expect(runtime_forward, forward_step_index, f"{runtime_label}.forward_step_index")
    _expect(runtime_batch, batch_size, f"{runtime_label}.batch_size")
    _sha256(
        drafter_runtime["request_ids_sha256"],
        f"{runtime_label}.request_ids_sha256",
    )
    request_id_sha256s = _string_tuple(
        drafter_runtime["request_id_sha256s"],
        f"{runtime_label}.request_id_sha256s",
        length=batch_size,
    )
    normalized_request_id_sha256s = tuple(
        _sha256(value, f"{runtime_label}.request_id_sha256s[{index}]")
        for index, value in enumerate(request_id_sha256s)
    )
    if len(set(normalized_request_id_sha256s)) != batch_size:
        raise CensusError(
            f"{runtime_label}.request_id_sha256s: duplicate request digest"
        )
    if "gdn_comparator" in event:
        comparator_label = f"{source}.gdn_comparator"
        expected_comparator_event_id = (
            f"{mode}:{producer_pid}:{event_index}"
        )
        _expect(event_id, expected_comparator_event_id, f"{source}.event_id")
        comparator = _mapping(event["gdn_comparator"], comparator_label)
        _exact_keys(comparator, GDN_COMPARATOR_KEYS, comparator_label)
        _expect(
            _string(comparator["schema"], f"{comparator_label}.schema"),
            GDN_COMPARATOR_SCHEMA,
            f"{comparator_label}.schema",
        )
        _expect(
            _string(comparator["mode"], f"{comparator_label}.mode"),
            mode,
            f"{comparator_label}.mode",
        )
        _expect(
            _integer(
                comparator["batch_size"],
                f"{comparator_label}.batch_size",
                minimum=1,
            ),
            batch_size,
            f"{comparator_label}.batch_size",
        )
        _sha256(
            comparator["runtime_capture_manifest_sha256"],
            f"{comparator_label}.runtime_capture_manifest_sha256",
        )
        _expect(
            _sha256(
                comparator["structural_graph_signature"],
                f"{comparator_label}.structural_graph_signature",
            ),
            forward_graph_structural_signature(batch_size),
            f"{comparator_label}.structural_graph_signature",
        )
        _expect(
            _string(comparator["reference"], f"{comparator_label}.reference"),
            "fixed32_gdn_two_launch_reference_v1",
            f"{comparator_label}.reference",
        )
        comparator_candidate = _string(
            comparator["candidate"], f"{comparator_label}.candidate"
        )
        if comparator_candidate not in GDN_COMPARATOR_CANDIDATES:
            raise CensusError(
                f"{comparator_label}.candidate: unsupported ordered GDN identity"
            )
        for field, expected in (
            ("reference_physical_launches_per_request_layer", 2),
            ("candidate_physical_launches_per_request_layer", 1),
            ("records", 48),
        ):
            _expect(
                _integer(comparator[field], f"{comparator_label}.{field}"),
                expected,
                f"{comparator_label}.{field}",
            )
        _expect(
            _string_tuple(
                comparator["compared_byte_surfaces"],
                f"{comparator_label}.compared_byte_surfaces",
                length=len(GDN_COMPARATOR_SURFACES),
            ),
            GDN_COMPARATOR_SURFACES,
            f"{comparator_label}.compared_byte_surfaces",
        )
        for field, expected in (
            ("raw_byte_equal", True),
            ("state_restored", True),
            ("reference_served", True),
            ("candidate_served", False),
        ):
            if comparator[field] is not expected:
                raise CensusError(
                    f"{comparator_label}.{field}: expected literal {expected!r}"
                )
        _expect(
            _string_tuple(
                comparator["comparison_order"],
                f"{comparator_label}.comparison_order",
                length=4,
            ),
            (
                "reference",
                "restore_baseline",
                "candidate",
                "restore_baseline_in_finally",
            ),
            f"{comparator_label}.comparison_order",
        )
        _expect(
            _string(
                comparator["census_event_id"],
                f"{comparator_label}.census_event_id",
            ),
            event_id,
            f"{comparator_label}.census_event_id",
        )
        _expect(
            _integer(
                comparator["census_event_index"],
                f"{comparator_label}.census_event_index",
            ),
            event_index,
            f"{comparator_label}.census_event_index",
        )
        _expect(
            _integer(
                comparator["census_forward_step_index"],
                f"{comparator_label}.census_forward_step_index",
            ),
            forward_step_index,
            f"{comparator_label}.census_forward_step_index",
        )
        _expect(
            _string_tuple(
                comparator["request_id_sha256s"],
                f"{comparator_label}.request_id_sha256s",
                length=batch_size,
            ),
            normalized_request_id_sha256s,
            f"{comparator_label}.request_id_sha256s",
        )
        marker = _string(
            comparator["observed_task_marker"],
            f"{comparator_label}.observed_task_marker",
        )
        if re.fullmatch(r"swe_verified:[A-Za-z0-9._/-]+", marker) is None:
            raise CensusError(
                f"{comparator_label}.observed_task_marker: invalid marker"
            )
    for field in ("proposal_begins", "proposal_ends", "graph_replays"):
        _expect(
            _integer(drafter_runtime[field], f"{runtime_label}.{field}"),
            1,
            f"{runtime_label}.{field}",
        )
    _integer(
        drafter_runtime["graph_id"], f"{runtime_label}.graph_id", minimum=1
    )
    graph_signature = _sha256(
        drafter_runtime["graph_signature"],
        f"{runtime_label}.graph_signature",
    )
    _expect(
        graph_signature,
        _drafter_graph_signature(batch_size),
        f"{runtime_label}.graph_signature",
    )
    graph_captures = _integer(
        drafter_runtime["graph_captures"],
        f"{runtime_label}.graph_captures",
    )
    if graph_captures not in (0, 1):
        raise CensusError(
            f"{runtime_label}.graph_captures: expected 0 or 1, got {graph_captures}"
        )
    _expect(
        _string(
            drafter_runtime["mtp_observation"],
            f"{runtime_label}.mtp_observation",
        ),
        "capture_manifest_bound_replay",
        f"{runtime_label}.mtp_observation",
    )
    runtime_mtp_calls = _integer(
        drafter_runtime["mtp_forward_calls"],
        f"{runtime_label}.mtp_forward_calls",
    )
    runtime_mtp_rows = _integer(
        drafter_runtime["mtp_forward_rows"],
        f"{runtime_label}.mtp_forward_rows",
    )
    _expect(runtime_mtp_calls, MTP_FORWARD_CALLS, f"{runtime_label}.mtp_forward_calls")
    _expect(
        runtime_mtp_rows,
        MTP_FORWARD_CALLS * batch_size,
        f"{runtime_label}.mtp_forward_rows",
    )
    raw_ledger = drafter_runtime["arctic_ledger"]
    if not isinstance(raw_ledger, list) or len(raw_ledger) != 3:
        raise CensusError(f"{runtime_label}.arctic_ledger: expected three ordered rows")
    expected_ledger = (
        ("main", batch_size, 6 * batch_size),
        ("rank1", batch_size, 4 * batch_size),
        ("rank2", batch_size, 2 * batch_size),
    )
    normalized_ledger = []
    for index, (kind, calls, tokens) in enumerate(expected_ledger):
        ledger_label = f"{runtime_label}.arctic_ledger[{index}]"
        row = _mapping(raw_ledger[index], ledger_label)
        _exact_keys(row, DRAFTER_LEDGER_KEYS, ledger_label)
        actual = {
            "kind": _string(row["kind"], f"{ledger_label}.kind"),
            "calls": _integer(row["calls"], f"{ledger_label}.calls"),
            "tokens": _integer(row["tokens"], f"{ledger_label}.tokens"),
        }
        _expect(
            (actual["kind"], actual["calls"], actual["tokens"]),
            (kind, calls, tokens),
            ledger_label,
        )
        normalized_ledger.append(actual)
    runtime_arctic_calls = _integer(
        drafter_runtime["arctic_lookup_calls"],
        f"{runtime_label}.arctic_lookup_calls",
    )
    runtime_arctic_tokens = _integer(
        drafter_runtime["arctic_requested_tokens"],
        f"{runtime_label}.arctic_requested_tokens",
    )
    runtime_fill_calls = _integer(
        drafter_runtime["merge_fill_calls"],
        f"{runtime_label}.merge_fill_calls",
    )
    runtime_fill_columns = _integer(
        drafter_runtime["merge_fill_columns"],
        f"{runtime_label}.merge_fill_columns",
    )
    runtime_fill_rows = _integer(
        drafter_runtime["merge_fill_rows"],
        f"{runtime_label}.merge_fill_rows",
    )
    runtime_carry = _integer(
        drafter_runtime["rescue_carry_slots"],
        f"{runtime_label}.rescue_carry_slots",
    )
    runtime_publish_shape = _integer_tuple(
        drafter_runtime["publish_shape"],
        f"{runtime_label}.publish_shape",
        length=2,
    )
    runtime_parent_sha = _sha256(
        drafter_runtime["physical_parent_sha256"],
        f"{runtime_label}.physical_parent_sha256",
    )
    runtime_handoff_calls = _integer(
        drafter_runtime["outer_handoff_calls"],
        f"{runtime_label}.outer_handoff_calls",
    )
    for actual, expected, field in (
        (runtime_arctic_calls, ARCTIC_LOOKUP_CALLS_PER_REQUEST * batch_size, "arctic_lookup_calls"),
        (runtime_arctic_tokens, ARCTIC_LOOKUP_TOKENS_PER_REQUEST * batch_size, "arctic_requested_tokens"),
        (runtime_fill_calls, 1, "merge_fill_calls"),
        (runtime_fill_columns, 16, "merge_fill_columns"),
        (runtime_fill_rows, 16 * batch_size, "merge_fill_rows"),
        (runtime_carry, RESCUE_CARRY_SLOTS_PER_REQUEST * batch_size, "rescue_carry_slots"),
        (runtime_publish_shape, (batch_size, PHYSICAL_DRAFTS), "publish_shape"),
        (runtime_parent_sha, PHYSICAL_PARENT_SHA256, "physical_parent_sha256"),
        (runtime_handoff_calls, 1, "outer_handoff_calls"),
    ):
        _expect(actual, expected, f"{runtime_label}.{field}")
    runtime_projection = {
        "mtp_forward_calls": runtime_mtp_calls,
        "mtp_forward_rows": runtime_mtp_rows,
        "arctic_lookup_calls": runtime_arctic_calls,
        "arctic_requested_tokens": runtime_arctic_tokens,
        "main_tail_length": ARCTIC_MAIN_TAIL_LENGTH,
        "rescue_chains": [list(chain) for chain in ARCTIC_LOOKUP_CHAINS],
        "carry_fill_slots": runtime_carry,
        "pack_columns": runtime_publish_shape[1],
        "packed_rows": runtime_publish_shape[0] * runtime_publish_shape[1],
    }
    if dict(drafter) != runtime_projection:
        raise CensusError(
            f"{runtime_label}: legacy drafter projection mismatch "
            f"{dict(drafter)!r} != {runtime_projection!r}"
        )

    tree = _mapping(event["tree_attn"], f"{source}.tree_attn")
    _exact_keys(tree, TREE_KEYS, f"{source}.tree_attn")
    tree_calls = _integer(tree["calls"], f"{source}.tree_attn.calls", minimum=1)
    _expect(
        tree_calls,
        TREE_CALLS_PER_EVENT,
        f"{source}.tree_attn.calls",
    )
    tree_q_rows = _integer(tree["q_rows"], f"{source}.tree_attn.q_rows")
    _expect(
        tree_q_rows,
        tree_calls * batch_size * TREE_ROWS_PER_REQUEST,
        f"{source}.tree_attn.q_rows",
    )
    tree_bias_shape = _integer_tuple(
        tree["bias_shape"], f"{source}.tree_attn.bias_shape", length=2
    )
    _expect(
        tree_bias_shape,
        TREE_BIAS_SHAPE,
        f"{source}.tree_attn.bias_shape",
    )
    physical_parent_digest = _sha256(
        tree["physical_parent_digest"],
        f"{source}.tree_attn.physical_parent_digest",
    )
    bias_digest = _sha256(tree["bias_digest"], f"{source}.tree_attn.bias_digest")
    _expect(
        physical_parent_digest,
        PHYSICAL_PARENT_SHA256,
        f"{source}.tree_attn.physical_parent_digest",
    )
    _expect(
        bias_digest,
        TREE_ANCESTRY_SHA256,
        f"{source}.tree_attn.bias_digest",
    )

    gdn = _mapping(event["gdn"], f"{source}.gdn")
    _exact_keys(gdn, GDN_KEYS, f"{source}.gdn")
    scan_calls = _integer(gdn["scan_calls"], f"{source}.gdn.scan_calls", minimum=1)
    _expect(
        scan_calls,
        GDN_SCAN_CALLS_PER_REQUEST * batch_size,
        f"{source}.gdn.scan_calls",
    )
    gdn_launches = _integer(gdn["launches"], f"{source}.gdn.launches")
    gdn_path_programs = _integer(gdn["path_programs"], f"{source}.gdn.path_programs")
    gdn_padded_slots = _integer(gdn["padded_slots"], f"{source}.gdn.padded_slots")
    gdn_nodes = _integer(gdn["nodes"], f"{source}.gdn.nodes")
    gdn_critical_path = _integer(gdn["critical_path"], f"{source}.gdn.critical_path")
    gdn_grid_z = _integer_tuple(gdn["grid_z"], f"{source}.gdn.grid_z", length=2)
    gdn_max_path_lengths = _integer_tuple(
        gdn["max_path_lengths"],
        f"{source}.gdn.max_path_lengths",
        length=2,
    )
    gdn_export_or_mask = _integer(gdn["export_or_mask"], f"{source}.gdn.export_or_mask")
    _expect(
        gdn_launches,
        scan_calls * GDN_LAUNCHES_PER_SCAN,
        f"{source}.gdn.launches",
    )
    _expect(
        gdn_path_programs,
        scan_calls * GDN_PATH_PROGRAMS_PER_SCAN,
        f"{source}.gdn.path_programs",
    )
    _expect(
        gdn_padded_slots,
        scan_calls * GDN_PADDED_SLOTS_PER_SCAN,
        f"{source}.gdn.padded_slots",
    )
    _expect(
        gdn_nodes,
        scan_calls * GDN_NODES_PER_SCAN,
        f"{source}.gdn.nodes",
    )
    _expect(
        gdn_critical_path,
        GDN_CRITICAL_PATH,
        f"{source}.gdn.critical_path",
    )
    _expect(gdn_grid_z, GDN_GRID_Z, f"{source}.gdn.grid_z")
    _expect(
        gdn_max_path_lengths,
        GDN_MAX_PATH_LENGTHS,
        f"{source}.gdn.max_path_lengths",
    )
    _expect(
        gdn_export_or_mask,
        GDN_EXPORT_OR_MASK,
        f"{source}.gdn.export_or_mask",
    )

    taw = _mapping(event["taw"], f"{source}.taw")
    _exact_keys(taw, TAW_KEYS, f"{source}.taw")
    taw_preseeded_batches = _integer_tuple(
        taw["preseeded_batches"],
        f"{source}.taw.preseeded_batches",
        length=len(SUPPORTED_BATCH_SIZES),
    )
    if taw["topology_cache_hit"] is not True:
        raise CensusError(f"{source}.taw.topology_cache_hit: expected literal true")
    taw_cache_misses = _integer(taw["cache_misses"], f"{source}.taw.cache_misses")
    taw_table_shape = _integer_tuple(
        taw["table_shape"],
        f"{source}.taw.table_shape",
        length=3,
    )
    taw_buffer_capacity = _integer(
        taw["buffer_capacity"], f"{source}.taw.buffer_capacity"
    )
    taw_loop_iterations = _integer(
        taw["loop_iterations"], f"{source}.taw.loop_iterations"
    )
    taw_uniform_slots = _integer(taw["uniform_slots"], f"{source}.taw.uniform_slots")
    taw_child_lanes = _integer(taw["child_lanes"], f"{source}.taw.child_lanes")
    taw_target_rows = _integer(taw["target_rows"], f"{source}.taw.target_rows")
    taw_self_rows = _integer(taw["self_rows"], f"{source}.taw.self_rows")
    taw_self_cdf_rows = _integer(taw["self_cdf_rows"], f"{source}.taw.self_cdf_rows")
    taw_source_cdf_rows = _integer(
        taw["source_cdf_rows"], f"{source}.taw.source_cdf_rows"
    )
    taw_residual_cdf_rows = _integer(
        taw["residual_cdf_rows"], f"{source}.taw.residual_cdf_rows"
    )
    taw_qmix_rows = _integer(taw["qmix_rows"], f"{source}.taw.qmix_rows")
    taw_residual_rows = _integer(taw["residual_rows"], f"{source}.taw.residual_rows")
    taw_row_scatter_slots = _integer(
        taw["row_scatter_slots"], f"{source}.taw.row_scatter_slots"
    )
    taw_path_scatter_slots = _integer(
        taw["path_scatter_slots"], f"{source}.taw.path_scatter_slots"
    )
    taw_route = _string(taw["route"], f"{source}.taw.route")
    if taw_route not in (
        TAW_ROUTE,
        TAW_NATIVE_PRECOMPUTE_ROUTE,
        TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
        TAW_CFWD_LOGIT_DIRECT_PRODUCTION_ROUTE,
    ):
        raise CensusError(
            f"{source}.taw.route: expected a pinned fixed32 TAW route, "
            f"got {taw_route!r}"
        )
    if taw_route == TAW_NATIVE_PRECOMPUTE_ROUTE:
        expected_target_rows = (
            TAW_ROWS_PER_REQUEST + TAW_ALL_PARENT_TARGET_ROWS_PER_REQUEST
        )
        expected_self_rows = (
            TAW_ROWS_PER_REQUEST + TAW_ALL_PARENT_SELF_ROWS_PER_REQUEST
        )
        expected_product_write_multiplier = 2
        expected_exact_commit_launches = TAW_EXACT_COMMIT_LAUNCHES + 1
        expected_exact_commit_programs = (
            TAW_EXACT_COMMIT_PROGRAMS_PER_REQUEST + 1
        )
    elif taw_route in (
        TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
        TAW_CFWD_LOGIT_DIRECT_PRODUCTION_ROUTE,
    ):
        expected_target_rows = TAW_ALL_PARENT_TARGET_ROWS_PER_REQUEST
        expected_self_rows = TAW_ALL_PARENT_SELF_ROWS_PER_REQUEST
        expected_product_write_multiplier = 1
        expected_exact_commit_launches = 1
        expected_exact_commit_programs = 1
    else:
        expected_target_rows = TAW_ROWS_PER_REQUEST
        expected_self_rows = TAW_ROWS_PER_REQUEST
        expected_product_write_multiplier = 1
        expected_exact_commit_launches = TAW_EXACT_COMMIT_LAUNCHES
        expected_exact_commit_programs = TAW_EXACT_COMMIT_PROGRAMS_PER_REQUEST
    taw_exact_commit_launches = _integer(
        taw["exact_commit_launches"],
        f"{source}.taw.exact_commit_launches",
    )
    taw_exact_commit_programs = _integer(
        taw["exact_commit_programs"],
        f"{source}.taw.exact_commit_programs",
    )
    if taw["floating_sampling_reimplementation"] is not False:
        raise CensusError(
            f"{source}.taw.floating_sampling_reimplementation: "
            "expected literal false"
        )
    _expect(
        taw_preseeded_batches,
        SUPPORTED_BATCH_SIZES,
        f"{source}.taw.preseeded_batches",
    )
    _expect(taw_cache_misses, 0, f"{source}.taw.cache_misses")
    _expect(
        taw_table_shape,
        (batch_size, PHYSICAL_ROWS, SAMPLER_MAX_FANOUT),
        f"{source}.taw.table_shape",
    )
    _expect(
        taw_buffer_capacity,
        TAW_BUFFER_CAPACITY,
        f"{source}.taw.buffer_capacity",
    )
    _expect(
        taw_loop_iterations,
        TAW_LOOP_ITERATIONS,
        f"{source}.taw.loop_iterations",
    )
    _expect(
        taw_uniform_slots,
        TAW_UNIFORM_SLOTS * batch_size,
        f"{source}.taw.uniform_slots",
    )
    _expect(
        taw_child_lanes,
        expected_target_rows * SAMPLER_MAX_FANOUT * batch_size,
        f"{source}.taw.child_lanes",
    )
    for field_name, field_value, expected_rows in (
        ("target_rows", taw_target_rows, expected_target_rows),
        ("self_rows", taw_self_rows, expected_self_rows),
        ("self_cdf_rows", taw_self_cdf_rows, expected_self_rows),
        ("source_cdf_rows", taw_source_cdf_rows, expected_target_rows),
        ("residual_cdf_rows", taw_residual_cdf_rows, expected_target_rows),
        ("qmix_rows", taw_qmix_rows, expected_target_rows),
        ("residual_rows", taw_residual_rows, expected_target_rows),
    ):
        _expect(
            field_value,
            expected_rows * batch_size,
            f"{source}.taw.{field_name}",
        )
    _expect(
        taw_row_scatter_slots,
        TAW_ROW_SCATTER_SLOTS
        * batch_size
        * expected_product_write_multiplier,
        f"{source}.taw.row_scatter_slots",
    )
    _expect(
        taw_path_scatter_slots,
        TAW_PATH_SCATTER_SLOTS
        * batch_size
        * expected_product_write_multiplier,
        f"{source}.taw.path_scatter_slots",
    )
    _expect(
        taw_exact_commit_launches,
        expected_exact_commit_launches,
        f"{source}.taw.exact_commit_launches",
    )
    _expect(
        taw_exact_commit_programs,
        expected_exact_commit_programs * batch_size,
        f"{source}.taw.exact_commit_programs",
    )
    taw_source_schema = _string(
        taw["source_contract_schema"],
        f"{source}.taw.source_contract_schema",
    )
    taw_source_sha256 = _sha256(
        taw["source_contract_sha256"],
        f"{source}.taw.source_contract_sha256",
    )
    expected_source_schema = (
        TAW_CFWD_LOGIT_DIRECT_SOURCE_SCHEMA
        if taw_route == TAW_CFWD_LOGIT_DIRECT_PRODUCTION_ROUTE
        else TAW_SOURCE_CONTRACT_SCHEMA
    )
    expected_source_sha256 = (
        TAW_CFWD_LOGIT_DIRECT_SOURCE_SHA256
        if taw_route == TAW_CFWD_LOGIT_DIRECT_PRODUCTION_ROUTE
        else TAW_SOURCE_CONTRACT_SHA256
    )
    _expect(
        taw_source_schema,
        expected_source_schema,
        f"{source}.taw.source_contract_schema",
    )
    _expect(
        taw_source_sha256,
        expected_source_sha256,
        f"{source}.taw.source_contract_sha256",
    )
    raw_tensor_calls = _mapping(
        taw["tensor_call_census"],
        f"{source}.taw.tensor_call_census",
    )
    _exact_keys(
        raw_tensor_calls,
        TAW_TENSOR_CALL_CENSUS_KEYS,
        f"{source}.taw.tensor_call_census",
    )
    taw_tensor_calls = {}
    for name in sorted(TAW_TENSOR_CALL_CENSUS_KEYS):
        label = f"{source}.taw.tensor_call_census.{name}"
        if name == "floating_sampling_reimplementation":
            if raw_tensor_calls[name] is not False:
                raise CensusError(f"{label}: expected literal false")
            taw_tensor_calls[name] = False
        else:
            taw_tensor_calls[name] = _integer(raw_tensor_calls[name], label)
    if taw_route == TAW_NATIVE_PRECOMPUTE_ROUTE:
        expected_tensor_calls = TAW_NATIVE_PRECOMPUTE_TENSOR_CALL_CENSUS
    elif taw_route == TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE:
        expected_tensor_calls = (
            TAW_NATIVE_PRECOMPUTE_PRODUCTION_TENSOR_CALL_CENSUS
        )
    elif taw_route == TAW_CFWD_LOGIT_DIRECT_PRODUCTION_ROUTE:
        expected_tensor_calls = (
            TAW_CFWD_LOGIT_DIRECT_PRODUCTION_TENSOR_CALL_CENSUS
        )
    else:
        expected_tensor_calls = TAW_TENSOR_CALL_CENSUS
    _expect(
        taw_tensor_calls,
        {
            name: expected_tensor_calls[name]
            for name in sorted(expected_tensor_calls)
        },
        f"{source}.taw.tensor_call_census",
    )
    taw_count_route = _string(taw["count_route"], f"{source}.taw.count_route")
    taw_rng_route = _string(taw["rng_route"], f"{source}.taw.rng_route")
    taw_vocab_size = _integer(
        taw["vocab_size"],
        f"{source}.taw.vocab_size",
        minimum=1,
    )
    _expect(taw_count_route, TAW_COUNT_ROUTE, f"{source}.taw.count_route")
    _expect(taw_rng_route, TAW_RNG_ROUTE, f"{source}.taw.rng_route")
    _expect(taw_vocab_size, TAW_VOCAB_SIZE, f"{source}.taw.vocab_size")

    taw_live_layouts = {
        "count": _tensor_layout(
            taw,
            prefix="count",
            expected_shape=(batch_size,),
            expected_dtype="torch.int32",
            expected_stride=(1,),
            label=f"{source}.taw",
        ),
        "draft": _tensor_layout(
            taw,
            prefix="draft",
            expected_shape=(batch_size * PHYSICAL_DRAFTS,),
            expected_dtype="torch.int32",
            expected_stride=(1,),
            label=f"{source}.taw",
        ),
        "parent": _tensor_layout(
            taw,
            prefix="parent",
            expected_shape=(batch_size * PHYSICAL_DRAFTS,),
            expected_dtype="torch.int32",
            expected_stride=(1,),
            label=f"{source}.taw",
        ),
        "bonus": _tensor_layout(
            taw,
            prefix="bonus",
            expected_shape=(batch_size, 1),
            expected_dtype="torch.int32",
            expected_stride=(1, 1),
            label=f"{source}.taw",
        ),
        "target": _tensor_layout(
            taw,
            prefix="target",
            expected_shape=(batch_size * PHYSICAL_DRAFTS, TAW_VOCAB_SIZE),
            expected_dtype="torch.float32",
            expected_stride=(TAW_VOCAB_SIZE, 1),
            label=f"{source}.taw",
        ),
        "self": _tensor_layout(
            taw,
            prefix="self",
            expected_shape=(batch_size * PHYSICAL_DRAFTS, TAW_VOCAB_SIZE),
            expected_dtype="torch.float32",
            expected_stride=(TAW_VOCAB_SIZE, 1),
            label=f"{source}.taw",
        ),
        "uniform": _tensor_layout(
            taw,
            prefix="uniform",
            expected_shape=(batch_size, WALK_CAP, SAMPLER_MAX_FANOUT),
            expected_dtype="torch.float32",
            expected_stride=(WALK_CAP * SAMPLER_MAX_FANOUT, SAMPLER_MAX_FANOUT, 1),
            label=f"{source}.taw",
        ),
    }
    expected_taw_cache_shapes = {
        "child_table_shape": (batch_size, PHYSICAL_ROWS, SAMPLER_MAX_FANOUT),
        "child_counts_shape": (batch_size, PHYSICAL_ROWS),
        "output_shape": (batch_size, OUTPUT_PUBLISH_CAPACITY),
        "output_lens_shape": (batch_size,),
        "accepted_path_shape": (batch_size, ACCEPTED_PATH_CAPACITY),
        "accepted_lens_shape": (batch_size,),
        "last_row_shape": (batch_size,),
        "exact_current_shape": (batch_size,),
        "exact_alive_shape": (batch_size,),
    }
    taw_cache_shapes: dict[str, list[int]] = {}
    for field_name, expected_shape in expected_taw_cache_shapes.items():
        shape = _integer_tuple(
            taw[field_name],
            f"{source}.taw.{field_name}",
            length=len(expected_shape),
        )
        _expect(shape, expected_shape, f"{source}.taw.{field_name}")
        taw_cache_shapes[field_name] = list(shape)

    output_publish = _fixed_section(
        event["output_publish"],
        keys=OUTPUT_PUBLISH_KEYS,
        expected={
            "route": OUTPUT_PUBLISH_ROUTE,
            "capacity": OUTPUT_PUBLISH_CAPACITY,
            "requests": batch_size,
            "launches": 2,
            "slots_written": OUTPUT_PUBLISH_CAPACITY * batch_size,
            "accepted_rows_written": batch_size,
            "host_materializations": 0,
            "host_scalar_writes": 0,
            "dtoh": 0,
            "h2d": 0,
            "fallback": 0,
        },
        label=f"{source}.output_publish",
    )
    accepted_path_pack = _fixed_section(
        event["accepted_path_pack"],
        keys=ACCEPTED_PATH_PACK_KEYS,
        expected={
            "route": ACCEPTED_PATH_PACK_ROUTE,
            "capacity": ACCEPTED_PATH_CAPACITY,
            "requests": batch_size,
            "pack_launches": 2,
            "slots_written": ACCEPTED_PATH_CAPACITY * batch_size,
            "source_walk_slots": WALK_CAP * batch_size,
            "lens_written": batch_size,
            "host_path_items": 0,
            "overflow": 0,
            "fallback": 0,
        },
        label=f"{source}.accepted_path_pack",
    )
    request_key_pack = _fixed_section(
        event["request_key_pack"],
        keys=REQUEST_KEY_PACK_KEYS,
        expected={
            "route": REQUEST_KEY_PACK_ROUTE,
            "sampler_rows": batch_size,
            "spec_rows": batch_size,
            "map_passes": 2,
            "path_slots_gathered": 2 * REQUEST_KEY_PATH_CAPACITY * batch_size,
            "lens_gathered": 2 * batch_size,
            "zero_launches": 2,
            "gather_launches": 4,
            "host_dict_inserts": 0,
            "host_hash_lookups": 0,
            "missing": 0,
            "fallback": 0,
        },
        label=f"{source}.request_key_pack",
    )
    kv_pair_rows = KV_REMAP_CACHE_TENSORS * KV_REMAP_PATH_CAPACITY * batch_size
    kv_remap = _fixed_section(
        event["kv_remap"],
        keys=KV_REMAP_KEYS,
        expected={
            "route": KV_REMAP_ROUTE,
            "path_capacity": KV_REMAP_PATH_CAPACITY,
            "pair_slots": KV_REMAP_PAIR_SLOTS * batch_size,
            "target_pair_slots": KV_REMAP_TARGET_PAIR_SLOTS * batch_size,
            "drafter_pair_slots": KV_REMAP_DRAFTER_PAIR_SLOTS * batch_size,
            "kv_groups": KV_REMAP_GROUPS,
            "target_cache_tensors": KV_REMAP_TARGET_CACHE_TENSORS,
            "drafter_cache_tensors": KV_REMAP_DRAFTER_CACHE_TENSORS,
            "kv_cache_tensors": KV_REMAP_CACHE_TENSORS,
            "kv_planes": KV_REMAP_PLANES,
            "target_prepare_calls": KV_REMAP_TARGET_PREPARE_CALLS,
            "drafter_prepare_calls": KV_REMAP_DRAFTER_PREPARE_CALLS,
            "prepare_calls": (
                KV_REMAP_TARGET_PREPARE_CALLS
                + KV_REMAP_DRAFTER_PREPARE_CALLS
            ),
            "target_apply_cache_calls": KV_REMAP_TARGET_APPLY_CACHE_CALLS,
            "drafter_apply_cache_calls": KV_REMAP_DRAFTER_APPLY_CACHE_CALLS,
            "apply_cache_calls": KV_REMAP_CACHE_TENSORS,
            "src_pair_rows": kv_pair_rows,
            "dst_pair_rows": kv_pair_rows,
            "identity_safe_writes": kv_pair_rows,
            "host_syncs": 0,
            "skips": 0,
            "fallback": 0,
        },
        label=f"{source}.kv_remap",
    )
    conv_rows = CONV_COMMIT_LAYERS * batch_size
    conv_commit_programs = (
        CONV_COMMIT_LAYERS
        * batch_size
        * (
            (
                GDN_CONV_CHANNELS
                + CONV_PREGATHER_BLOCK
                - 1
            )
            // CONV_PREGATHER_BLOCK
        )
    )
    conv_commit = _fixed_section(
        event["conv_commit"],
        keys=CONV_COMMIT_KEYS,
        expected={
            "route": CONV_COMMIT_ROUTE,
            "layers": CONV_COMMIT_LAYERS,
            "requests": batch_size,
            "row_elems": CONV_PREGATHER_ROW_ELEMS,
            "channels": GDN_CONV_CHANNELS,
            "state_length": GDN_CONV_STATE_LENGTH,
            "source_rows_per_batch": CONV_COMMIT_SOURCE_ROWS,
            "block": CONV_PREGATHER_BLOCK,
            "direct_launches": 1,
            "gather_launches": 0,
            "scatter_launches": 0,
            "direct_programs": conv_commit_programs,
            "committed_rows": conv_rows,
            "source_staging_reused": True,
            "source_pointer_entries": 48,
            "row_guard_route": CONV_ROW_GUARD_ROUTE,
            "row_guard_kernel_launches": 1,
            "row_guard_programs": (
                CONV_ROW_GUARD_PROGRAMS_PER_REQUEST * batch_size
            ),
            "row_guard_physical_rows": PHYSICAL_ROWS,
            "row_guard_path_capacity": ACCEPTED_PATH_CAPACITY,
            "row_guard_alias_width": CONV_ROW_GUARD_ALIAS_WIDTH,
            "row_guard_compare_capacity": CONV_ROW_GUARD_COMPARE_CAPACITY,
            "row_guard_path_validation_programs": (
                CONV_ROW_GUARD_PATH_PROGRAMS_PER_REQUEST * batch_size
            ),
            "row_guard_path_vector_loads": (
                CONV_ROW_GUARD_PATH_VECTOR_LOADS_PER_REQUEST * batch_size
            ),
            "row_guard_alias_validation_programs": (
                CONV_ROW_GUARD_ALIAS_PROGRAMS_PER_EVENT
            ),
            "row_guard_alias_vector_loads": (
                CONV_ROW_GUARD_ALIAS_VECTOR_LOADS_PER_EVENT
            ),
            "row_guard_selected_row_loads": (
                CONV_ROW_GUARD_SELECTED_ROW_LOADS_PER_PROGRAM
                * CONV_ROW_GUARD_PROGRAMS_PER_REQUEST
                * batch_size
            ),
            "row_guard_peer_topology_proof": (
                CONV_ROW_GUARD_PEER_TOPOLOGY_PROOF
            ),
            "row_guard_torch_index_transforms": 0,
            "row_guard_async_scalar_reductions": 1,
            "row_guard_async_assertions": 1,
            "full_node_writebacks": 0,
            "conv_remaps": 0,
            "host_syncs": 0,
            "skips": 0,
            "fallback": 0,
        },
        label=f"{source}.conv_commit",
    )
    conv_pregather_raw = _mapping(event["conv_pregather"], f"{source}.conv_pregather")
    _exact_keys(
        conv_pregather_raw,
        CONV_PREGATHER_KEYS,
        f"{source}.conv_pregather",
    )
    conv_layout_sha256 = _sha256(
        conv_pregather_raw["layout_sha256"],
        f"{source}.conv_pregather.layout_sha256",
    )
    conv_row_elems = _integer(
        conv_pregather_raw["row_elems"],
        f"{source}.conv_pregather.row_elems",
        minimum=1,
    )
    _expect(
        conv_row_elems,
        CONV_PREGATHER_ROW_ELEMS,
        f"{source}.conv_pregather.row_elems",
    )
    conv_pregather_programs = (
        CONV_PREGATHER_LAYERS
        * batch_size
        * ((conv_row_elems + CONV_PREGATHER_BLOCK - 1) // CONV_PREGATHER_BLOCK)
    )
    conv_pregather = _fixed_section(
        conv_pregather_raw,
        keys=CONV_PREGATHER_KEYS,
        expected={
            "route": CONV_PREGATHER_ROUTE,
            "layout_sha256": conv_layout_sha256,
            "stage_calls": 1,
            "stage_before_all_consumes": True,
            "layers": CONV_PREGATHER_LAYERS,
            "requests": batch_size,
            "row_elems": conv_row_elems,
            "programs": conv_pregather_programs,
            "staged_rows": CONV_PREGATHER_LAYERS * batch_size,
            "consume_calls": CONV_PREGATHER_LAYERS,
            "consume_hits": CONV_PREGATHER_LAYERS,
            "consume_fallbacks": 0,
            "freshness_matches": CONV_PREGATHER_LAYERS,
        },
        label=f"{source}.conv_pregather",
    )
    committer_raw = _mapping(event["committer"], f"{source}.committer")
    _exact_keys(committer_raw, COMMITTER_KEYS, f"{source}.committer")
    committer_fused_calls = _integer(
        committer_raw["fused_layer_calls"],
        f"{source}.committer.fused_layer_calls",
        minimum=1,
    )
    if committer_fused_calls not in (1, GDN_LAYERS):
        raise CensusError(
            f"{source}.committer.fused_layer_calls: expected 1 or "
            f"{GDN_LAYERS}, got {committer_fused_calls}"
        )
    expected_committer_neutralizations = (
        0 if committer_fused_calls == 1 else COMMITTER_NEUTRALIZE_OPS
    )
    expected_committer_ring_gathers = (
        0 if committer_fused_calls == 1 else COMMITTER_RING_GATHER_OPS
    )
    expected_committer_ring_rows = (
        0
        if committer_fused_calls == 1
        else COMMITTER_RING_LAYER_PATH_ROWS_PER_REQUEST * batch_size
    )
    committer = _fixed_section(
        committer_raw,
        keys=COMMITTER_KEYS,
        expected={
            "route": COMMITTER_ROUTE,
            "layers": GDN_LAYERS,
            "requests": batch_size,
            "path_capacity": COMMITTER_PATH_CAPACITY,
            "layout_slots": COMMITTER_PATH_CAPACITY * batch_size,
            "ring_gather_ops": expected_committer_ring_gathers,
            "ring_layer_path_rows": expected_committer_ring_rows,
            "neutralize_ops": expected_committer_neutralizations,
            "fused_layer_calls": committer_fused_calls,
            "graph_replays": 1,
            "graph_captures": 0,
            "host_lens_readbacks": 0,
            "host_flag_readbacks": 0,
            "pointer_table_rebuilds": 0,
            "overflow": 0,
            "fallback": 0,
            "graph_dead": 0,
        },
        label=f"{source}.committer",
    )

    failures = _mapping(event["failures"], f"{source}.failures")
    _exact_keys(failures, FAILURE_KEYS, f"{source}.failures")
    parsed_failures = {
        failure_name: _integer(
            failures[failure_name], f"{source}.failures.{failure_name}"
        )
        for failure_name in sorted(FAILURE_KEYS)
    }
    route_mismatches = sum(
        section["route"] != route
        for section, route in (
            (output_publish, OUTPUT_PUBLISH_ROUTE),
            (accepted_path_pack, ACCEPTED_PATH_PACK_ROUTE),
            (request_key_pack, REQUEST_KEY_PACK_ROUTE),
            (kv_remap, KV_REMAP_ROUTE),
            (conv_commit, CONV_COMMIT_ROUTE),
            (conv_pregather, CONV_PREGATHER_ROUTE),
            (committer, COMMITTER_ROUTE),
        )
    )
    derived_failures = {
        "fallback": (
            int(output_publish["fallback"])
            + int(accepted_path_pack["fallback"])
            + int(request_key_pack["fallback"])
            + int(kv_remap["fallback"])
            + int(conv_commit["fallback"])
            + int(conv_pregather["consume_fallbacks"])
            + int(committer["fallback"])
            + route_mismatches
        ),
        "overflow": (
            int(accepted_path_pack["overflow"])
            + int(committer["overflow"])
            + int(taw_loop_iterations > int(accepted_path_pack["capacity"]))
            + int(taw_loop_iterations > int(output_publish["capacity"]))
        ),
        "graph_dead": (
            int(committer["graph_dead"])
            + int(committer["route"] != COMMITTER_ROUTE)
            + int(committer["graph_replays"] != 1)
            + int(committer["graph_captures"] != 0)
        ),
        "mixed_pseudo": (
            batch_rows
            - spec_rows
            + sum(
                value != PHYSICAL_DRAFTS
                for value in physical_draft_counts
            )
        ),
        "taw_cache_miss": (
            taw_cache_misses + int(taw["topology_cache_hit"] is not True)
        ),
    }
    _expect(parsed_failures, derived_failures, f"{source}.failures")
    for failure_name, failure_count in parsed_failures.items():
        _expect(failure_count, 0, f"{source}.failures.{failure_name}")

    normalized_work = {
        "physical_drafts": physical_drafts,
        "verify_rows_per_request": verify_rows // batch_size,
        "batch_purity": {
            "batch_rows_per_request": batch_rows // batch_size,
            "spec_rows_per_request": spec_rows // batch_size,
            "physical_drafts_per_request": physical_draft_counts[0],
            "mixed_pseudo_rows": mixed_pseudo_rows,
            "all_physical_31": all_physical_31,
        },
        "drafter": {
            "mtp_forward_calls": mtp_forward_calls,
            "mtp_forward_rows_per_request": mtp_forward_rows // batch_size,
            "arctic_lookup_calls_per_request": (arctic_lookup_calls // batch_size),
            "arctic_requested_tokens_per_request": (
                arctic_requested_tokens // batch_size
            ),
            "main_tail_length": main_tail_length,
            "rescue_chains": [list(chain) for chain in rescue_chains],
            "carry_fill_slots_per_request": carry_fill_slots // batch_size,
            "pack_columns": pack_columns,
            "packed_rows_per_request": packed_rows // batch_size,
        },
        "drafter_runtime": {
            "association": "same_runner_step",
            "proposal_begins": 1,
            "proposal_ends": 1,
            "graph_replays": 1,
            "mtp_observation": "capture_manifest_bound_replay",
            "mtp_forward_calls": runtime_mtp_calls,
            "mtp_forward_rows_per_request": runtime_mtp_rows // batch_size,
            "arctic_ledger": [
                {
                    "kind": row["kind"],
                    "calls_per_request": int(row["calls"]) // batch_size,
                    "tokens_per_request": int(row["tokens"]) // batch_size,
                }
                for row in normalized_ledger
            ],
            "arctic_lookup_calls_per_request": (
                runtime_arctic_calls // batch_size
            ),
            "arctic_requested_tokens_per_request": (
                runtime_arctic_tokens // batch_size
            ),
            "merge_fill_calls": runtime_fill_calls,
            "merge_fill_columns": runtime_fill_columns,
            "merge_fill_rows_per_request": runtime_fill_rows // batch_size,
            "rescue_carry_slots_per_request": runtime_carry // batch_size,
            "publish_columns": runtime_publish_shape[1],
            "publish_rows_per_request": runtime_publish_shape[0] // batch_size,
            "physical_parent_sha256": runtime_parent_sha,
            "outer_handoff_calls": runtime_handoff_calls,
        },
        "tree_attn": {
            "calls_per_event": tree_calls,
            "q_rows_per_call_per_request": (tree_q_rows // (tree_calls * batch_size)),
            "bias_shape": list(tree_bias_shape),
            "physical_parent_digest": physical_parent_digest,
            "bias_digest": bias_digest,
        },
        "gdn": {
            "scan_calls_per_request": scan_calls // batch_size,
            "launches_per_scan": gdn_launches // scan_calls,
            "path_programs_per_scan": gdn_path_programs // scan_calls,
            "padded_slots_per_scan": gdn_padded_slots // scan_calls,
            "nodes_per_scan": gdn_nodes // scan_calls,
            "critical_path": gdn_critical_path,
            "grid_z": list(gdn_grid_z),
            "max_path_lengths": list(gdn_max_path_lengths),
            "export_or_mask": gdn_export_or_mask,
        },
        "taw": {
            "route": taw_route,
            "preseeded_batches": list(taw_preseeded_batches),
            "topology_cache_hit": True,
            "cache_misses": taw_cache_misses,
            "table_rows_per_request": taw_table_shape[1],
            "table_child_width": taw_table_shape[2],
            "buffer_capacity": taw_buffer_capacity,
            "loop_iterations": taw_loop_iterations,
            "uniform_slots_per_request": taw_uniform_slots // batch_size,
            "child_lanes_per_request": taw_child_lanes // batch_size,
            "target_rows_per_request": taw_target_rows // batch_size,
            "self_rows_per_request": taw_self_rows // batch_size,
            "self_cdf_rows_per_request": taw_self_cdf_rows // batch_size,
            "source_cdf_rows_per_request": taw_source_cdf_rows // batch_size,
            "residual_cdf_rows_per_request": taw_residual_cdf_rows // batch_size,
            "qmix_rows_per_request": taw_qmix_rows // batch_size,
            "residual_rows_per_request": taw_residual_rows // batch_size,
            "row_scatter_slots_per_request": taw_row_scatter_slots // batch_size,
            "path_scatter_slots_per_request": taw_path_scatter_slots // batch_size,
            "exact_commit_launches": taw_exact_commit_launches,
            "exact_commit_programs_per_request": (
                taw_exact_commit_programs // batch_size
            ),
            "floating_sampling_reimplementation": False,
            "source_contract_schema": taw_source_schema,
            "source_contract_sha256": taw_source_sha256,
            "tensor_call_census": taw_tensor_calls,
            "count_route": taw_count_route,
            "rng_route": taw_rng_route,
            "vocab_size": taw_vocab_size,
            "live_layouts": {
                name: {
                    **layout,
                    "shape": [
                        int(layout["shape"][0]) // batch_size,
                        *layout["shape"][1:],
                    ],
                }
                for name, layout in sorted(taw_live_layouts.items())
            },
            "cache_shapes": {
                name: [
                    int(shape[0]) // batch_size,
                    *shape[1:],
                ]
                for name, shape in sorted(taw_cache_shapes.items())
            },
        },
        "output_publish": {
            "route": output_publish["route"],
            "capacity": output_publish["capacity"],
            "launches_per_event": output_publish["launches"],
            "slots_written_per_request": (
                int(output_publish["slots_written"]) // batch_size
            ),
            "accepted_rows_written_per_request": (
                int(output_publish["accepted_rows_written"]) // batch_size
            ),
            "host_materializations": output_publish["host_materializations"],
            "host_scalar_writes": output_publish["host_scalar_writes"],
            "dtoh": output_publish["dtoh"],
            "h2d": output_publish["h2d"],
            "fallback": output_publish["fallback"],
        },
        "accepted_path_pack": {
            "route": accepted_path_pack["route"],
            "capacity": accepted_path_pack["capacity"],
            "pack_launches_per_event": accepted_path_pack["pack_launches"],
            "slots_written_per_request": (
                int(accepted_path_pack["slots_written"]) // batch_size
            ),
            "source_walk_slots_per_request": (
                int(accepted_path_pack["source_walk_slots"]) // batch_size
            ),
            "lens_written_per_request": (
                int(accepted_path_pack["lens_written"]) // batch_size
            ),
            "host_path_items": accepted_path_pack["host_path_items"],
            "overflow": accepted_path_pack["overflow"],
            "fallback": accepted_path_pack["fallback"],
        },
        "request_key_pack": {
            "route": request_key_pack["route"],
            "sampler_rows_per_request": (
                int(request_key_pack["sampler_rows"]) // batch_size
            ),
            "spec_rows_per_request": (int(request_key_pack["spec_rows"]) // batch_size),
            "map_passes_per_event": request_key_pack["map_passes"],
            "path_slots_gathered_per_request": (
                int(request_key_pack["path_slots_gathered"]) // batch_size
            ),
            "lens_gathered_per_request": (
                int(request_key_pack["lens_gathered"]) // batch_size
            ),
            "zero_launches_per_event": request_key_pack["zero_launches"],
            "gather_launches_per_event": request_key_pack["gather_launches"],
            "host_dict_inserts": request_key_pack["host_dict_inserts"],
            "host_hash_lookups": request_key_pack["host_hash_lookups"],
            "missing": request_key_pack["missing"],
            "fallback": request_key_pack["fallback"],
        },
        "kv_remap": {
            "route": kv_remap["route"],
            "path_capacity": kv_remap["path_capacity"],
            "pair_slots_per_request": int(kv_remap["pair_slots"]) // batch_size,
            "target_pair_slots_per_request": (
                int(kv_remap["target_pair_slots"]) // batch_size
            ),
            "drafter_pair_slots_per_request": (
                int(kv_remap["drafter_pair_slots"]) // batch_size
            ),
            "kv_groups": kv_remap["kv_groups"],
            "target_cache_tensors": kv_remap["target_cache_tensors"],
            "drafter_cache_tensors": kv_remap["drafter_cache_tensors"],
            "kv_cache_tensors": kv_remap["kv_cache_tensors"],
            "kv_planes": kv_remap["kv_planes"],
            "target_prepare_calls_per_event": kv_remap[
                "target_prepare_calls"
            ],
            "drafter_prepare_calls_per_event": kv_remap[
                "drafter_prepare_calls"
            ],
            "prepare_calls_per_event": kv_remap["prepare_calls"],
            "target_apply_cache_calls_per_event": kv_remap[
                "target_apply_cache_calls"
            ],
            "drafter_apply_cache_calls_per_event": kv_remap[
                "drafter_apply_cache_calls"
            ],
            "apply_cache_calls_per_event": kv_remap["apply_cache_calls"],
            "src_pair_rows_per_request": (int(kv_remap["src_pair_rows"]) // batch_size),
            "dst_pair_rows_per_request": (int(kv_remap["dst_pair_rows"]) // batch_size),
            "identity_safe_writes_per_request": (
                int(kv_remap["identity_safe_writes"]) // batch_size
            ),
            "host_syncs": kv_remap["host_syncs"],
            "skips": kv_remap["skips"],
            "fallback": kv_remap["fallback"],
        },
        "conv_commit": {
            "route": conv_commit["route"],
            "layers": conv_commit["layers"],
            "row_elems": conv_commit["row_elems"],
            "channels": conv_commit["channels"],
            "state_length": conv_commit["state_length"],
            "source_rows_per_batch": conv_commit[
                "source_rows_per_batch"
            ],
            "block": conv_commit["block"],
            "direct_launches_per_event": conv_commit["direct_launches"],
            "gather_launches_per_event": conv_commit["gather_launches"],
            "scatter_launches_per_event": conv_commit[
                "scatter_launches"
            ],
            "direct_programs_per_request": (
                int(conv_commit["direct_programs"]) // batch_size
            ),
            "committed_rows_per_request": (
                int(conv_commit["committed_rows"]) // batch_size
            ),
            "source_staging_reused": conv_commit["source_staging_reused"],
            "source_pointer_entries": conv_commit["source_pointer_entries"],
            "row_guard_route": conv_commit["row_guard_route"],
            "row_guard_kernel_launches_per_event": conv_commit[
                "row_guard_kernel_launches"
            ],
            "row_guard_programs_per_request": (
                int(conv_commit["row_guard_programs"]) // batch_size
            ),
            "row_guard_physical_rows": conv_commit[
                "row_guard_physical_rows"
            ],
            "row_guard_path_capacity": conv_commit[
                "row_guard_path_capacity"
            ],
            "row_guard_alias_width": conv_commit[
                "row_guard_alias_width"
            ],
            "row_guard_compare_capacity": conv_commit[
                "row_guard_compare_capacity"
            ],
            "row_guard_path_validation_programs_per_request": (
                int(conv_commit["row_guard_path_validation_programs"])
                // batch_size
            ),
            "row_guard_path_vector_loads_per_request": (
                int(conv_commit["row_guard_path_vector_loads"])
                // batch_size
            ),
            "row_guard_alias_validation_programs_per_event": conv_commit[
                "row_guard_alias_validation_programs"
            ],
            "row_guard_alias_vector_loads_per_event": conv_commit[
                "row_guard_alias_vector_loads"
            ],
            "row_guard_selected_row_loads_per_program": (
                int(conv_commit["row_guard_selected_row_loads"])
                // int(conv_commit["row_guard_programs"])
            ),
            "row_guard_peer_topology_proof": conv_commit[
                "row_guard_peer_topology_proof"
            ],
            "row_guard_torch_index_transforms": conv_commit[
                "row_guard_torch_index_transforms"
            ],
            "row_guard_async_scalar_reductions_per_event": conv_commit[
                "row_guard_async_scalar_reductions"
            ],
            "row_guard_async_assertions_per_event": conv_commit[
                "row_guard_async_assertions"
            ],
            "full_node_writebacks": conv_commit["full_node_writebacks"],
            "conv_remaps": conv_commit["conv_remaps"],
            "host_syncs": conv_commit["host_syncs"],
            "skips": conv_commit["skips"],
            "fallback": conv_commit["fallback"],
        },
        "conv_pregather": {
            "route": conv_pregather["route"],
            "stage_calls_per_event": conv_pregather["stage_calls"],
            "stage_before_all_consumes": conv_pregather[
                "stage_before_all_consumes"
            ],
            "layers": conv_pregather["layers"],
            "row_elems": conv_pregather["row_elems"],
            "programs_per_request": (int(conv_pregather["programs"]) // batch_size),
            "staged_rows_per_request": (
                int(conv_pregather["staged_rows"]) // batch_size
            ),
            "consume_calls_per_event": conv_pregather["consume_calls"],
            "consume_hits_per_event": conv_pregather["consume_hits"],
            "consume_fallbacks": conv_pregather["consume_fallbacks"],
            "freshness_matches_per_event": conv_pregather["freshness_matches"],
        },
        "committer": {
            "route": committer["route"],
            "layers": committer["layers"],
            "path_capacity": committer["path_capacity"],
            "layout_slots_per_request": (int(committer["layout_slots"]) // batch_size),
            "ring_gather_ops": committer["ring_gather_ops"],
            "ring_layer_path_rows_per_request": (
                int(committer["ring_layer_path_rows"]) // batch_size
            ),
            "neutralize_ops": committer["neutralize_ops"],
            "fused_layer_calls_per_event": committer["fused_layer_calls"],
            "graph_replays_per_event": committer["graph_replays"],
            "graph_captures": committer["graph_captures"],
            "host_lens_readbacks": committer["host_lens_readbacks"],
            "host_flag_readbacks": committer["host_flag_readbacks"],
            "pointer_table_rebuilds": committer["pointer_table_rebuilds"],
            "overflow": committer["overflow"],
            "fallback": committer["fallback"],
            "graph_dead": committer["graph_dead"],
        },
        "failure_counts": dict(parsed_failures),
    }
    return ValidatedEvent(
        source=source,
        event_id=event_id,
        event_index=event_index,
        forward_step_index=forward_step_index,
        producer_pid=producer_pid,
        mode=mode,
        batch_size=batch_size,
        conv_layout_sha256=conv_layout_sha256,
        drafter_graph_signature=graph_signature,
        drafter_graph_captures=graph_captures,
        normalized_work=normalized_work,
    )


def _duplicate_checked_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_jsonl(path: Path) -> list[LocatedRecord]:
    """Load strict JSONL while preserving source locations for errors."""

    if not path.is_file():
        raise CensusError(f"missing census JSONL: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CensusError(f"cannot read census JSONL {path}: {error}") from error
    if not lines:
        raise CensusError(f"{path}: census JSONL is empty")

    records: list[LocatedRecord] = []
    for line_number, line in enumerate(lines, start=1):
        source = f"{path}:{line_number}"
        if not line.strip():
            raise CensusError(f"{source}: blank JSONL records are forbidden")
        try:
            raw = json.loads(line, object_pairs_hook=_duplicate_checked_object)
        except (json.JSONDecodeError, DuplicateJsonKey) as error:
            raise CensusError(f"{source}: invalid JSON: {error}") from error
        records.append((raw, source))
    return records


def load_jsonl_bytes(raw: bytes, *, source: str) -> list[LocatedRecord]:
    """Load strict JSONL from authenticated bytes without a path reread."""

    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise CensusError(f"cannot decode census JSONL {source}: {error}") from error
    if not lines:
        raise CensusError(f"{source}: census JSONL is empty")
    records: list[LocatedRecord] = []
    for line_number, line in enumerate(lines, start=1):
        location = f"{source}:{line_number}"
        if not line.strip():
            raise CensusError(f"{location}: blank JSONL records are forbidden")
        try:
            record = json.loads(line, object_pairs_hook=_duplicate_checked_object)
        except (json.JSONDecodeError, DuplicateJsonKey) as error:
            raise CensusError(f"{location}: invalid JSON: {error}") from error
        records.append((record, location))
    return records


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalized_work_sha256(value: object) -> str:
    """Return the canonical digest used for mode-neutral physical work."""

    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _events_sha256(events: Sequence[object]) -> str:
    body = _canonical_json(list(events)).encode("ascii")
    return hashlib.sha256(body).hexdigest()


def _split_terminal(
    records: Sequence[LocatedRecord], *, expected_mode: str
) -> tuple[list[LocatedRecord], LocatedRecord]:
    if len(records) < 2:
        raise CensusError(
            f"{expected_mode}: census requires at least one event and one terminal record"
        )
    event_records = list(records[:-1])
    terminal_record = records[-1]
    for raw, source in event_records:
        record = _mapping(raw, source)
        if record.get("schema") == TERMINAL_SCHEMA:
            raise CensusError(f"{source}: terminal record must be last")
    terminal = _mapping(terminal_record[0], terminal_record[1])
    if terminal.get("schema") != TERMINAL_SCHEMA:
        raise CensusError(
            f"{terminal_record[1]}: last JSONL record must use {TERMINAL_SCHEMA!r}"
        )
    return event_records, terminal_record


def _validate_terminal(
    raw: object,
    *,
    source: str,
    expected_mode: str,
    raw_events: Sequence[object],
    events: Sequence[ValidatedEvent],
) -> dict[str, Any]:
    terminal = _mapping(raw, source)
    _exact_keys(terminal, TERMINAL_KEYS, source)
    _expect(
        _string(terminal["schema"], f"{source}.schema"),
        TERMINAL_SCHEMA,
        f"{source}.schema",
    )
    mode = _string(terminal["mode"], f"{source}.mode")
    _expect(mode, expected_mode, f"{source}.mode")
    producer_pid = _integer(
        terminal["producer_pid"], f"{source}.producer_pid", minimum=1
    )
    if terminal["final"] is not True:
        raise CensusError(f"{source}.final: expected literal true")
    event_count = _integer(terminal["event_count"], f"{source}.event_count", minimum=1)
    first_event_index = _integer(
        terminal["first_event_index"], f"{source}.first_event_index"
    )
    last_event_index = _integer(
        terminal["last_event_index"], f"{source}.last_event_index"
    )
    first_forward_step_index = _integer(
        terminal["first_forward_step_index"],
        f"{source}.first_forward_step_index",
    )
    last_forward_step_index = _integer(
        terminal["last_forward_step_index"],
        f"{source}.last_forward_step_index",
    )
    events_sha256 = _sha256(terminal["events_sha256"], f"{source}.events_sha256")
    histogram_raw = _mapping(
        terminal["batch_histogram"], f"{source}.batch_histogram"
    )
    expected_histogram_keys = frozenset(str(batch) for batch in SUPPORTED_BATCH_SIZES)
    _exact_keys(histogram_raw, expected_histogram_keys, f"{source}.batch_histogram")
    batch_histogram = {
        str(batch): _integer(
            histogram_raw[str(batch)],
            f"{source}.batch_histogram.{batch}",
        )
        for batch in SUPPORTED_BATCH_SIZES
    }
    expected_histogram = {
        str(batch): sum(event.batch_size == batch for event in events)
        for batch in SUPPORTED_BATCH_SIZES
    }
    _expect(batch_histogram, expected_histogram, f"{source}.batch_histogram")
    scope = _mapping(terminal["scope"], f"{source}.scope")
    _exact_keys(scope, SCOPE_KEYS, f"{source}.scope")
    _expect(dict(scope), FIXED_WORK_SCOPE, f"{source}.scope")
    raw_registry = terminal["drafter_graph_registry"]
    if not isinstance(raw_registry, list) or not raw_registry:
        raise CensusError(
            f"{source}.drafter_graph_registry: expected a non-empty array"
        )
    registry: list[dict[str, Any]] = []
    seen_batches: set[int] = set()
    for index, raw_row in enumerate(raw_registry):
        row_label = f"{source}.drafter_graph_registry[{index}]"
        row = _mapping(raw_row, row_label)
        _exact_keys(row, DRAFTER_GRAPH_REGISTRY_KEYS, row_label)
        batch = _integer(row["batch_size"], f"{row_label}.batch_size", minimum=1)
        if batch not in SUPPORTED_BATCH_SIZES or batch in seen_batches:
            raise CensusError(
                f"{row_label}.batch_size: expected unique B in [1,4], got {batch}"
            )
        seen_batches.add(batch)
        signature = _sha256(
            row["graph_signature"], f"{row_label}.graph_signature"
        )
        _expect(
            signature,
            _drafter_graph_signature(batch),
            f"{row_label}.graph_signature",
        )
        captures = _integer(row["captures"], f"{row_label}.captures")
        _expect(captures, 1, f"{row_label}.captures")
        origin = _string(row["capture_origin"], f"{row_label}.capture_origin")
        if origin not in ("measured", "unmeasured"):
            raise CensusError(
                f"{row_label}.capture_origin: expected measured or unmeasured"
            )
        measured_replays = _integer(
            row["measured_replays"], f"{row_label}.measured_replays"
        )
        unmeasured_replays = _integer(
            row["unmeasured_replays"], f"{row_label}.unmeasured_replays"
        )
        expected_measured = expected_histogram[str(batch)]
        _expect(
            measured_replays,
            expected_measured,
            f"{row_label}.measured_replays",
        )
        measured_capture_count = sum(
            event.batch_size == batch and event.drafter_graph_captures == 1
            for event in events
        )
        if origin == "measured":
            if measured_capture_count != 1 or measured_replays < 1:
                raise CensusError(
                    f"{row_label}: measured origin requires exactly one "
                    "same-pending-event capture"
                )
        elif measured_capture_count != 0 or unmeasured_replays < 1:
            raise CensusError(
                f"{row_label}: unmeasured origin requires a prior mixed/prefill "
                "capture and its immediate replay"
            )
        registry.append(
            {
                "batch_size": batch,
                "graph_signature": signature,
                "captures": 1,
                "capture_origin": origin,
                "measured_replays": measured_replays,
                "unmeasured_replays": unmeasured_replays,
            }
        )
    if [row["batch_size"] for row in registry] != sorted(seen_batches):
        raise CensusError(
            f"{source}.drafter_graph_registry: rows must be sorted by batch_size"
        )
    event_batches = {event.batch_size for event in events}
    if not event_batches.issubset(seen_batches):
        raise CensusError(
            f"{source}.drafter_graph_registry: missing event batches "
            f"{sorted(event_batches - seen_batches)}"
        )
    raw_forward_registry = terminal["forward_graph_registry"]
    if not isinstance(raw_forward_registry, list) or not raw_forward_registry:
        raise CensusError(
            f"{source}.forward_graph_registry: expected a non-empty array"
        )
    forward_registry: list[dict[str, Any]] = []
    forward_batches: set[int] = set()
    forward_signatures: set[str] = set()
    for index, raw_row in enumerate(raw_forward_registry):
        row_label = f"{source}.forward_graph_registry[{index}]"
        row = _mapping(raw_row, row_label)
        _exact_keys(row, FORWARD_GRAPH_REGISTRY_KEYS, row_label)
        batch = _integer(row["batch_size"], f"{row_label}.batch_size", minimum=1)
        if batch not in SUPPORTED_BATCH_SIZES or batch in forward_batches:
            raise CensusError(
                f"{row_label}.batch_size: expected unique B in [1,4], got {batch}"
            )
        forward_batches.add(batch)
        signature = _sha256(
            row["graph_signature"], f"{row_label}.graph_signature"
        )
        conv_layout_sha256 = _sha256(
            row["conv_layout_sha256"],
            f"{row_label}.conv_layout_sha256",
        )
        _expect(
            signature,
            forward_graph_structural_signature(batch),
            f"{row_label}.graph_signature",
        )
        if signature in forward_signatures:
            raise CensusError(
                f"{row_label}.graph_signature: signatures must be unique by B"
            )
        forward_signatures.add(signature)
        event_layouts = {
            event.conv_layout_sha256
            for event in events
            if event.batch_size == batch
        }
        if event_layouts and event_layouts != {conv_layout_sha256}:
            raise CensusError(
                f"{row_label}.conv_layout_sha256: terminal layout does not "
                f"match same-B events {sorted(event_layouts)}"
            )
        row_elems = _integer(
            row["row_elems"], f"{row_label}.row_elems", minimum=1
        )
        programs = (
            CONV_PREGATHER_LAYERS
            * batch
            * ((row_elems + CONV_PREGATHER_BLOCK - 1) // CONV_PREGATHER_BLOCK)
        )
        checked = _fixed_section(
            row,
            keys=FORWARD_GRAPH_REGISTRY_KEYS,
            expected={
                "batch_size": batch,
                "graph_signature": signature,
                "conv_layout_sha256": conv_layout_sha256,
                "captures": 1,
                "capture_origin": "final_full",
                "stage_calls": 1,
                "stage_before_all_consumes": True,
                "layers": CONV_PREGATHER_LAYERS,
                "requests": batch,
                "row_elems": CONV_PREGATHER_ROW_ELEMS,
                "programs": programs,
                "ssi_pointer_entries": CONV_PREGATHER_LAYERS,
                "ssi_groups": 3,
                "source_validations": CONV_PREGATHER_LAYERS,
                "staged_rows": CONV_PREGATHER_LAYERS * batch,
                "consume_calls": CONV_PREGATHER_LAYERS,
                "consume_hits": CONV_PREGATHER_LAYERS,
                "consume_fallbacks": 0,
                "freshness_matches": CONV_PREGATHER_LAYERS,
                "measured_replays": expected_histogram[str(batch)],
            },
            label=row_label,
        )
        forward_registry.append(dict(checked))
    capacity = max(forward_batches)
    if capacity not in SUPPORTED_CAMPAIGN_CAPACITIES:
        raise CensusError(
            f"{source}.forward_graph_registry: unsupported capacity B{capacity}"
        )
    expected_forward_batches = set(range(1, capacity + 1))
    if forward_batches != expected_forward_batches:
        raise CensusError(
            f"{source}.forward_graph_registry: expected contiguous B1..B{capacity}"
        )
    if [row["batch_size"] for row in forward_registry] != list(
        range(1, capacity + 1)
    ):
        raise CensusError(
            f"{source}.forward_graph_registry: rows must be sorted by batch_size"
        )
    if not event_batches.issubset(forward_batches):
        raise CensusError(
            f"{source}.forward_graph_registry: missing event batches "
            f"{sorted(event_batches - forward_batches)}"
        )
    auxiliary = _mapping(
        terminal["conv_pregather_auxiliary"],
        f"{source}.conv_pregather_auxiliary",
    )
    _exact_keys(
        auxiliary,
        CONV_PREGATHER_AUXILIARY_KEYS,
        f"{source}.conv_pregather_auxiliary",
    )
    for key in (
        "profile_capture_stages",
        "aux_capture_stages",
        "host_actual_stages",
    ):
        _expect(
            _integer(auxiliary[key], f"{source}.conv_pregather_auxiliary.{key}"),
            0,
            f"{source}.conv_pregather_auxiliary.{key}",
        )
    host_by_batch_raw = _mapping(
        auxiliary["host_actual_stages_by_batch"],
        f"{source}.conv_pregather_auxiliary.host_actual_stages_by_batch",
    )
    expected_counter_keys = frozenset(
        str(batch) for batch in SUPPORTED_BATCH_SIZES
    )
    _exact_keys(
        host_by_batch_raw,
        expected_counter_keys,
        f"{source}.conv_pregather_auxiliary.host_actual_stages_by_batch",
    )
    host_by_batch = {
        str(batch): _integer(
            host_by_batch_raw[str(batch)],
            (
                f"{source}.conv_pregather_auxiliary."
                f"host_actual_stages_by_batch.{batch}"
            ),
        )
        for batch in SUPPORTED_BATCH_SIZES
    }
    _expect(
        host_by_batch,
        {str(batch): 0 for batch in SUPPORTED_BATCH_SIZES},
        f"{source}.conv_pregather_auxiliary.host_actual_stages_by_batch",
    )
    nonpure_raw = _mapping(
        terminal["nonpure_dispatch"], f"{source}.nonpure_dispatch"
    )
    _exact_keys(
        nonpure_raw, NONPURE_DISPATCH_KEYS, f"{source}.nonpure_dispatch"
    )
    nonpure_dispatch = {
        name: _integer(
            nonpure_raw[name], f"{source}.nonpure_dispatch.{name}"
        )
        for name in sorted(NONPURE_DISPATCH_KEYS)
    }
    _expect(
        nonpure_dispatch["guarded_steps"],
        (
            nonpure_dispatch["piecewise_steps"]
            + nonpure_dispatch["none_steps"]
            + nonpure_dispatch["forbidden_full_steps"]
        ),
        f"{source}.nonpure_dispatch.guarded_steps",
    )
    _expect(
        nonpure_dispatch["forbidden_full_steps"],
        0,
        f"{source}.nonpure_dispatch.forbidden_full_steps",
    )
    nonpure_commit_raw = _mapping(
        terminal["nonpure_committer_replays_by_batch"],
        f"{source}.nonpure_committer_replays_by_batch",
    )
    _exact_keys(
        nonpure_commit_raw,
        expected_counter_keys,
        f"{source}.nonpure_committer_replays_by_batch",
    )
    nonpure_committer_replays_by_batch = {
        str(batch): _integer(
            nonpure_commit_raw[str(batch)],
            f"{source}.nonpure_committer_replays_by_batch.{batch}",
        )
        for batch in SUPPORTED_BATCH_SIZES
    }
    if (
        sum(nonpure_committer_replays_by_batch.values())
        > nonpure_dispatch["guarded_steps"]
    ):
        raise CensusError(
            f"{source}.nonpure_committer_replays_by_batch: compact commits "
            "exceed guarded nonpure steps"
        )
    _expect(event_count, len(events), f"{source}.event_count")
    _expect(first_event_index, events[0].event_index, f"{source}.first_event_index")
    _expect(last_event_index, events[-1].event_index, f"{source}.last_event_index")
    _expect(
        first_forward_step_index,
        events[0].forward_step_index,
        f"{source}.first_forward_step_index",
    )
    _expect(
        last_forward_step_index,
        events[-1].forward_step_index,
        f"{source}.last_forward_step_index",
    )
    producer_pids = {event.producer_pid for event in events}
    if producer_pids != {producer_pid}:
        raise CensusError(
            f"{source}.producer_pid: terminal/event PID mismatch {producer_pid} "
            f"vs {sorted(producer_pids)}"
        )
    _expect(events_sha256, _events_sha256(raw_events), f"{source}.events_sha256")
    return {
        "source": source,
        "producer_pid": producer_pid,
        "event_count": event_count,
        "first_event_index": first_event_index,
        "last_event_index": last_event_index,
        "first_forward_step_index": first_forward_step_index,
        "last_forward_step_index": last_forward_step_index,
        "events_sha256": events_sha256,
        "batch_histogram": batch_histogram,
        "drafter_graph_registry": registry,
        "forward_graph_registry": forward_registry,
        "conv_pregather_auxiliary": {
            "profile_capture_stages": 0,
            "aux_capture_stages": 0,
            "host_actual_stages": 0,
            "host_actual_stages_by_batch": host_by_batch,
        },
        "nonpure_dispatch": nonpure_dispatch,
        "nonpure_committer_replays_by_batch": (
            nonpure_committer_replays_by_batch
        ),
        "scope": dict(scope),
        "final": True,
    }


def validate_arm(
    records: Sequence[LocatedRecord],
    *,
    expected_mode: str,
    expected_route: str,
    required_batches: Sequence[int],
) -> dict[str, Any]:
    """Validate one complete census and require one physical-work signature."""

    if expected_mode not in MODE_SEMANTICS:
        raise CensusError(f"unsupported fixed32 mode {expected_mode!r}")
    batches = tuple(required_batches)
    if (
        not batches
        or len(set(batches)) != len(batches)
        or any(batch not in SUPPORTED_BATCH_SIZES for batch in batches)
    ):
        raise CensusError(
            "required_batches must be a non-empty unique subset of (1, 2, 3, 4)"
        )

    event_records, (terminal_raw, terminal_source) = _split_terminal(
        records, expected_mode=expected_mode
    )
    events: list[ValidatedEvent] = []
    raw_events: list[object] = []
    for raw, source in event_records:
        event = validate_event(raw, source=source)
        raw_mapping = _mapping(raw, source)
        taw = _mapping(raw_mapping.get("taw"), f"{source}.taw")
        if event.mode != expected_mode:
            raise CensusError(
                f"{source}.mode: record was supplied as {expected_mode}, "
                f"but declares {event.mode}"
            )
        if taw.get("route") != expected_route:
            raise CensusError(
                f"{source}.taw.route: expected {expected_route!r}, "
                f"got {taw.get('route')!r}"
            )
        events.append(event)
        raw_events.append(raw)

    terminal = _validate_terminal(
        terminal_raw,
        source=terminal_source,
        expected_mode=expected_mode,
        raw_events=raw_events,
        events=events,
    )
    event_ids = [event.event_id for event in events]
    if len(event_ids) != len(set(event_ids)):
        raise CensusError(f"{expected_mode}: duplicate event_id within census")
    event_indices = [event.event_index for event in events]
    expected_indices = list(range(len(events)))
    if event_indices != expected_indices:
        raise CensusError(
            f"{expected_mode}: event_index sequence must be {expected_indices}, "
            f"got {event_indices}"
        )
    forward_step_indices = [event.forward_step_index for event in events]
    if any(
        current <= previous
        for previous, current in zip(
            forward_step_indices, forward_step_indices[1:]
        )
    ):
        raise CensusError(
            f"{expected_mode}: forward_step_index values must be strictly "
            f"increasing and unique, got {forward_step_indices}"
        )
    producer_pids = {event.producer_pid for event in events}
    if len(producer_pids) != 1:
        raise CensusError(
            f"{expected_mode}: census has multiple producer PIDs "
            f"{sorted(producer_pids)}"
        )
    event_counts = {
        str(batch): sum(event.batch_size == batch for event in events)
        for batch in SUPPORTED_BATCH_SIZES
    }
    for batch in batches:
        if event_counts[str(batch)] == 0:
            raise CensusError(f"{expected_mode}: missing required B{batch} events")

    signatures: dict[str, int] = {}
    for event in events:
        signature = normalized_work_sha256(event.normalized_work)
        signatures[signature] = signatures.get(signature, 0) + 1
    if len(signatures) != 1:
        raise CensusError(
            f"{expected_mode}: census has multiple normalized physical-work "
            f"signatures {sorted(signatures)}"
        )
    signature_sha256 = next(iter(signatures))
    baseline = events[0]
    return {
        "schema": ARM_REPORT_SCHEMA,
        "status": "PASS",
        "mode": expected_mode,
        "route": expected_route,
        "required_batch_sizes": list(batches),
        "event_count": len(events),
        "event_counts_by_batch": event_counts,
        "batch_size_sequence": [event.batch_size for event in events],
        "forward_step_indices": forward_step_indices,
        "producer_pid": next(iter(producer_pids)),
        "terminal_summary": terminal,
        "normalized_work_signature": baseline.normalized_work,
        "normalized_work_signature_sha256": signature_sha256,
    }


def validate_bound_arm_report(
    raw: object,
    *,
    census_raw: bytes,
    census_source: str,
    expected_mode: str,
    expected_route: str,
    required_batches: Sequence[int],
) -> dict[str, Any]:
    """Revalidate a persisted arm report against the live census bytes."""

    report = _mapping(raw, "arm_report")
    _exact_keys(
        report,
        frozenset(
            {
                "schema",
                "status",
                "mode",
                "route",
                "required_batch_sizes",
                "event_count",
                "event_counts_by_batch",
                "batch_size_sequence",
                "forward_step_indices",
                "producer_pid",
                "terminal_summary",
                "normalized_work_signature",
                "normalized_work_signature_sha256",
                "census_sha256",
                "census_bytes",
            }
        ),
        "arm_report",
    )
    if (
        report["schema"] != ARM_REPORT_SCHEMA
        or report["status"] != "PASS"
        or report["mode"] != expected_mode
        or report["route"] != expected_route
        or report["required_batch_sizes"] != list(required_batches)
    ):
        raise CensusError("arm_report identity or route contract drifted")
    event_count = _integer(report["event_count"], "arm_report.event_count", minimum=1)
    event_counts = _mapping(
        report["event_counts_by_batch"], "arm_report.event_counts_by_batch"
    )
    _exact_keys(
        event_counts,
        frozenset(str(batch) for batch in SUPPORTED_BATCH_SIZES),
        "arm_report.event_counts_by_batch",
    )
    parsed_counts = {
        str(batch): _integer(
            event_counts[str(batch)],
            f"arm_report.event_counts_by_batch.{batch}",
        )
        for batch in SUPPORTED_BATCH_SIZES
    }
    if sum(parsed_counts.values()) != event_count:
        raise CensusError("arm_report event counts do not sum to event_count")
    for batch in required_batches:
        if parsed_counts[str(batch)] == 0:
            raise CensusError(f"arm_report is missing required B{batch} events")
    terminal = _mapping(report["terminal_summary"], "arm_report.terminal_summary")
    if terminal.get("final") is not True or terminal.get("event_count") != event_count:
        raise CensusError("arm_report terminal summary is incomplete")
    normalized = _mapping(
        report["normalized_work_signature"],
        "arm_report.normalized_work_signature",
    )
    normalized_sha256 = _sha256(
        report["normalized_work_signature_sha256"],
        "arm_report.normalized_work_signature_sha256",
    )
    if normalized_work_sha256(normalized) != normalized_sha256:
        raise CensusError("arm_report normalized-work digest mismatch")
    census_sha256 = _sha256(report["census_sha256"], "arm_report.census_sha256")
    census_bytes = _integer(
        report["census_bytes"], "arm_report.census_bytes", minimum=1
    )
    if (
        census_bytes != len(census_raw)
        or census_sha256 != hashlib.sha256(census_raw).hexdigest()
    ):
        raise CensusError("arm_report no longer binds the live census bytes")
    derived = validate_arm(
        load_jsonl_bytes(census_raw, source=census_source),
        expected_mode=expected_mode,
        expected_route=expected_route,
        required_batches=required_batches,
    )
    derived.update(
        {
            "census_sha256": hashlib.sha256(census_raw).hexdigest(),
            "census_bytes": len(census_raw),
        }
    )
    if dict(report) != derived:
        raise CensusError("arm_report differs from canonical live-census derivation")
    return derived


def validate_campaign(
    tail_records: Sequence[LocatedRecord],
    hydra_records: Sequence[LocatedRecord],
    *,
    required_batches: Sequence[int] = SUPPORTED_CAMPAIGN_CAPACITIES,
) -> dict[str, Any]:
    """Validate complete arm records and compare normalized work exactly."""

    batches = tuple(required_batches)
    if (
        not batches
        or len(set(batches)) != len(batches)
        or any(batch not in SUPPORTED_CAMPAIGN_CAPACITIES for batch in batches)
    ):
        raise CensusError(
            "required_batches must be a non-empty unique subset of (1, 4)"
        )

    validated_by_mode: dict[str, list[ValidatedEvent]] = {}
    terminal_by_mode: dict[str, dict[str, Any]] = {}
    for expected_mode, located_records in (
        (TAIL_MODE, tail_records),
        (HYDRA_MODE, hydra_records),
    ):
        event_records, (terminal_raw, terminal_source) = _split_terminal(
            located_records, expected_mode=expected_mode
        )
        validated: list[ValidatedEvent] = []
        raw_events: list[object] = []
        for raw, source in event_records:
            event = validate_event(raw, source=source)
            if event.mode != expected_mode:
                raise CensusError(
                    f"{source}.mode: record was supplied as {expected_mode}, "
                    f"but declares {event.mode}"
                )
            validated.append(event)
            raw_events.append(raw)
        validated_by_mode[expected_mode] = validated
        terminal_by_mode[expected_mode] = _validate_terminal(
            terminal_raw,
            source=terminal_source,
            expected_mode=expected_mode,
            raw_events=raw_events,
            events=validated,
        )

    counts: dict[str, dict[str, int]] = {}
    batch_size_sequences: dict[str, list[int]] = {}
    event_ids_by_mode: dict[str, list[str]] = {}
    forward_step_indices_by_mode: dict[str, list[int]] = {}
    producer_pids_by_mode: dict[str, int] = {}
    all_events: list[ValidatedEvent] = []
    for mode, events in validated_by_mode.items():
        counts[mode] = {}
        ids = [event.event_id for event in events]
        if len(ids) != len(set(ids)):
            raise CensusError(f"{mode}: duplicate event_id within census")
        event_indices = [event.event_index for event in events]
        expected_indices = list(range(len(events)))
        if event_indices != expected_indices:
            raise CensusError(
                f"{mode}: event_index sequence must be {expected_indices}, "
                f"got {event_indices}"
            )
        forward_step_indices = [event.forward_step_index for event in events]
        if any(
            current <= previous
            for previous, current in zip(forward_step_indices, forward_step_indices[1:])
        ):
            raise CensusError(
                f"{mode}: forward_step_index values must be strictly increasing "
                f"and unique, got {forward_step_indices}"
            )
        batch_size_sequences[mode] = [event.batch_size for event in events]
        event_ids_by_mode[mode] = ids
        forward_step_indices_by_mode[mode] = forward_step_indices
        producer_pids = {event.producer_pid for event in events}
        if len(producer_pids) != 1:
            raise CensusError(
                f"{mode}: census has multiple producer PIDs {sorted(producer_pids)}"
            )
        producer_pids_by_mode[mode] = next(iter(producer_pids))
        for batch_size in batches:
            selected = [event for event in events if event.batch_size == batch_size]
            if not selected:
                raise CensusError(f"{mode}: missing required B{batch_size} events")
        for batch_size in SUPPORTED_BATCH_SIZES:
            selected = [event for event in events if event.batch_size == batch_size]
            if selected:
                counts[mode][str(batch_size)] = len(selected)
        all_events.extend(events)

    physical_work_histograms: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, events in validated_by_mode.items():
        physical_work_histograms[mode] = {}
        for batch_size in SUPPORTED_BATCH_SIZES:
            selected = [event for event in events if event.batch_size == batch_size]
            signatures: dict[str, int] = {}
            graph_signatures: set[str] = set()
            for event in selected:
                signature = normalized_work_sha256(event.normalized_work)
                signatures[signature] = signatures.get(signature, 0) + 1
                graph_signatures.add(event.drafter_graph_signature)
            if len(signatures) > 1:
                raise CensusError(
                    f"{mode}: B{batch_size} has multiple physical-work signatures"
                )
            if len(graph_signatures) > 1:
                raise CensusError(
                    f"{mode}: B{batch_size} has multiple drafter graph signatures"
                )
            physical_work_histograms[mode][str(batch_size)] = {
                "event_count": len(selected),
                "normalized_event_signatures": dict(sorted(signatures.items())),
            }
    shared_batches = {
        event.batch_size for event in validated_by_mode[TAIL_MODE]
    } & {
        event.batch_size for event in validated_by_mode[HYDRA_MODE]
    }
    for batch_size in shared_batches:
        tail_signatures = set(
            physical_work_histograms[TAIL_MODE][str(batch_size)][
                "normalized_event_signatures"
            ]
        )
        hydra_signatures = set(
            physical_work_histograms[HYDRA_MODE][str(batch_size)][
                "normalized_event_signatures"
            ]
        )
        if (
            len(tail_signatures) != 1
            or len(hydra_signatures) != 1
            or tail_signatures != hydra_signatures
        ):
            raise CensusError(
                f"B{batch_size}: Tail/Hydra physical-work signature mismatch"
            )
    baseline = all_events[0]
    signature_sha256 = normalized_work_sha256(baseline.normalized_work)
    tail_forward = {
        row["batch_size"]: (
            row["graph_signature"],
            row["conv_layout_sha256"],
        )
        for row in terminal_by_mode[TAIL_MODE]["forward_graph_registry"]
    }
    hydra_forward = {
        row["batch_size"]: (
            row["graph_signature"],
            row["conv_layout_sha256"],
        )
        for row in terminal_by_mode[HYDRA_MODE]["forward_graph_registry"]
    }
    if tail_forward != hydra_forward:
        raise CensusError(
            "Tail/Hydra final-FULL forward graph/layout signatures differ"
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "required_batch_sizes": list(batches),
        "event_counts": counts,
        "batch_size_sequences": batch_size_sequences,
        "forward_step_indices": forward_step_indices_by_mode,
        "event_ids": event_ids_by_mode,
        "producer_pids": producer_pids_by_mode,
        "terminal_summaries": terminal_by_mode,
        "drafter_graph_registries": {
            mode: terminal["drafter_graph_registry"]
            for mode, terminal in terminal_by_mode.items()
        },
        "forward_graph_registries": {
            mode: terminal["forward_graph_registry"]
            for mode, terminal in terminal_by_mode.items()
        },
        "conv_pregather_auxiliary": {
            mode: terminal["conv_pregather_auxiliary"]
            for mode, terminal in terminal_by_mode.items()
        },
        "physical_work_histograms": physical_work_histograms,
        "scope": json.loads(json.dumps(FIXED_WORK_SCOPE)),
        "semantic_modes": {
            mode: dict(semantics) for mode, semantics in MODE_SEMANTICS.items()
        },
        "normalized_work_signature": baseline.normalized_work,
        "normalized_work_signature_sha256": signature_sha256,
    }


def _reference_taw(batch_size: int) -> dict[str, Any]:
    rows = batch_size * PHYSICAL_DRAFTS
    return {
        "route": TAW_ROUTE,
        "preseeded_batches": list(SUPPORTED_BATCH_SIZES),
        "topology_cache_hit": True,
        "cache_misses": 0,
        "table_shape": [batch_size, PHYSICAL_ROWS, SAMPLER_MAX_FANOUT],
        "buffer_capacity": TAW_BUFFER_CAPACITY,
        "loop_iterations": TAW_LOOP_ITERATIONS,
        "uniform_slots": TAW_UNIFORM_SLOTS * batch_size,
        "child_lanes": TAW_CHILD_LANES_PER_REQUEST * batch_size,
        "target_rows": TAW_ROWS_PER_REQUEST * batch_size,
        "self_rows": TAW_ROWS_PER_REQUEST * batch_size,
        "self_cdf_rows": TAW_ROWS_PER_REQUEST * batch_size,
        "source_cdf_rows": TAW_ROWS_PER_REQUEST * batch_size,
        "residual_cdf_rows": TAW_ROWS_PER_REQUEST * batch_size,
        "qmix_rows": TAW_ROWS_PER_REQUEST * batch_size,
        "residual_rows": TAW_ROWS_PER_REQUEST * batch_size,
        "row_scatter_slots": TAW_ROW_SCATTER_SLOTS * batch_size,
        "path_scatter_slots": TAW_PATH_SCATTER_SLOTS * batch_size,
        "exact_commit_launches": TAW_EXACT_COMMIT_LAUNCHES,
        "exact_commit_programs": (
            TAW_EXACT_COMMIT_PROGRAMS_PER_REQUEST * batch_size
        ),
        "floating_sampling_reimplementation": False,
        "source_contract_schema": TAW_SOURCE_CONTRACT_SCHEMA,
        "source_contract_sha256": TAW_SOURCE_CONTRACT_SHA256,
        "tensor_call_census": dict(TAW_TENSOR_CALL_CENSUS),
        "count_route": TAW_COUNT_ROUTE,
        "rng_route": TAW_RNG_ROUTE,
        "vocab_size": TAW_VOCAB_SIZE,
        "count_shape": [batch_size],
        "count_dtype": "torch.int32",
        "count_stride": [1],
        "count_contiguous": True,
        "draft_shape": [rows],
        "draft_dtype": "torch.int32",
        "draft_stride": [1],
        "draft_contiguous": True,
        "parent_shape": [rows],
        "parent_dtype": "torch.int32",
        "parent_stride": [1],
        "parent_contiguous": True,
        "bonus_shape": [batch_size, 1],
        "bonus_dtype": "torch.int32",
        "bonus_stride": [1, 1],
        "bonus_contiguous": True,
        "target_shape": [rows, TAW_VOCAB_SIZE],
        "target_dtype": "torch.float32",
        "target_stride": [TAW_VOCAB_SIZE, 1],
        "target_contiguous": True,
        "self_shape": [rows, TAW_VOCAB_SIZE],
        "self_dtype": "torch.float32",
        "self_stride": [TAW_VOCAB_SIZE, 1],
        "self_contiguous": True,
        "uniform_shape": [batch_size, WALK_CAP, SAMPLER_MAX_FANOUT],
        "uniform_dtype": "torch.float32",
        "uniform_stride": [
            WALK_CAP * SAMPLER_MAX_FANOUT,
            SAMPLER_MAX_FANOUT,
            1,
        ],
        "uniform_contiguous": True,
        "child_table_shape": [batch_size, PHYSICAL_ROWS, SAMPLER_MAX_FANOUT],
        "child_counts_shape": [batch_size, PHYSICAL_ROWS],
        "output_shape": [batch_size, OUTPUT_PUBLISH_CAPACITY],
        "output_lens_shape": [batch_size],
        "accepted_path_shape": [batch_size, ACCEPTED_PATH_CAPACITY],
        "accepted_lens_shape": [batch_size],
        "last_row_shape": [batch_size],
        "exact_current_shape": [batch_size],
        "exact_alive_shape": [batch_size],
    }


def _native_production_taw(batch_size: int) -> dict[str, Any]:
    """Build the independently qualified all-parent production TAW census."""
    taw = _reference_taw(batch_size)
    taw.update(
        {
            "route": TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
            "child_lanes": (
                TAW_ALL_PARENT_TARGET_ROWS_PER_REQUEST
                * SAMPLER_MAX_FANOUT
                * batch_size
            ),
            "target_rows": TAW_ALL_PARENT_TARGET_ROWS_PER_REQUEST * batch_size,
            "self_rows": TAW_ALL_PARENT_SELF_ROWS_PER_REQUEST * batch_size,
            "self_cdf_rows": TAW_ALL_PARENT_SELF_ROWS_PER_REQUEST * batch_size,
            "source_cdf_rows": (
                TAW_ALL_PARENT_TARGET_ROWS_PER_REQUEST * batch_size
            ),
            "residual_cdf_rows": (
                TAW_ALL_PARENT_TARGET_ROWS_PER_REQUEST * batch_size
            ),
            "qmix_rows": TAW_ALL_PARENT_TARGET_ROWS_PER_REQUEST * batch_size,
            "residual_rows": TAW_ALL_PARENT_TARGET_ROWS_PER_REQUEST * batch_size,
            "exact_commit_launches": 1,
            "exact_commit_programs": batch_size,
            "tensor_call_census": dict(
                TAW_NATIVE_PRECOMPUTE_PRODUCTION_TENSOR_CALL_CENSUS
            ),
        }
    )
    return taw


def reference_event(
    mode: str,
    batch_size: int,
    event_id: str,
    *,
    event_index: int = 0,
    forward_step_index: int | None = None,
    request_ids: Sequence[str] | None = None,
    drafter_runtime: Mapping[str, Any] | None = None,
    taw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an exact synthetic event for reducer and validator self-tests."""
    semantics = MODE_SEMANTICS[mode]
    tree_calls = TREE_CALLS_PER_EVENT
    gdn_scan_calls = GDN_SCAN_CALLS_PER_REQUEST * batch_size
    if forward_step_index is None:
        forward_step_index = event_index
    if request_ids is None:
        request_ids = tuple(
            f"{event_id}:request:{index}" for index in range(batch_size)
        )
    request_ids = tuple(str(value) for value in request_ids)
    request_ids_sha256 = hashlib.sha256(
        json.dumps(
            list(request_ids),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    request_id_sha256s = [
        hashlib.sha256(request_id.encode("utf-8")).hexdigest()
        for request_id in request_ids
    ]
    default_drafter_runtime = {
        "association": "same_runner_step",
        "forward_step_index": forward_step_index,
        "batch_size": batch_size,
        "request_ids_sha256": request_ids_sha256,
        "request_id_sha256s": request_id_sha256s,
        "proposal_begins": 1,
        "proposal_ends": 1,
        "graph_id": 1_000_000 + batch_size,
        "graph_signature": _drafter_graph_signature(batch_size),
        "graph_captures": 0,
        "graph_replays": 1,
        "mtp_observation": "capture_manifest_bound_replay",
        "mtp_forward_calls": MTP_FORWARD_CALLS,
        "mtp_forward_rows": MTP_FORWARD_CALLS * batch_size,
        "arctic_ledger": [
            {"kind": "main", "calls": batch_size, "tokens": 6 * batch_size},
            {"kind": "rank1", "calls": batch_size, "tokens": 4 * batch_size},
            {"kind": "rank2", "calls": batch_size, "tokens": 2 * batch_size},
        ],
        "arctic_lookup_calls": ARCTIC_LOOKUP_CALLS_PER_REQUEST * batch_size,
        "arctic_requested_tokens": ARCTIC_LOOKUP_TOKENS_PER_REQUEST * batch_size,
        "merge_fill_calls": 1,
        "merge_fill_columns": 16,
        "merge_fill_rows": 16 * batch_size,
        "rescue_carry_slots": RESCUE_CARRY_SLOTS_PER_REQUEST * batch_size,
        "publish_shape": [batch_size, PHYSICAL_DRAFTS],
        "physical_parent_sha256": PHYSICAL_PARENT_SHA256,
        "outer_handoff_calls": 1,
    }
    if drafter_runtime is not None:
        default_drafter_runtime = dict(drafter_runtime)
    default_taw = _reference_taw(batch_size) if taw is None else dict(taw)
    conv_pregather_programs = (
        CONV_PREGATHER_LAYERS
        * batch_size
        * (
            (CONV_PREGATHER_ROW_ELEMS + CONV_PREGATHER_BLOCK - 1)
            // CONV_PREGATHER_BLOCK
        )
    )
    conv_direct_programs = (
        CONV_COMMIT_LAYERS
        * batch_size
        * (
            (GDN_CONV_CHANNELS + CONV_PREGATHER_BLOCK - 1)
            // CONV_PREGATHER_BLOCK
        )
    )
    return {
        "schema": SCHEMA,
        "event_id": event_id,
        "event_index": event_index,
        "forward_step_index": forward_step_index,
        "producer_pid": 4242 if mode == TAIL_MODE else 4343,
        "event_complete": True,
        "mode": mode,
        "batch_size": batch_size,
        "physical_drafts": PHYSICAL_DRAFTS,
        "verify_rows": VERIFY_ROWS_PER_REQUEST * batch_size,
        "active_nodes": semantics["active_nodes"],
        "valid_mask": semantics["valid_mask"],
        "batch_purity": {
            "batch_rows": batch_size,
            "spec_rows": batch_size,
            "physical_draft_counts": [PHYSICAL_DRAFTS] * batch_size,
            "mixed_pseudo_rows": 0,
            "all_physical_31": True,
        },
        "drafter": {
            "mtp_forward_calls": MTP_FORWARD_CALLS,
            "mtp_forward_rows": MTP_FORWARD_CALLS * batch_size,
            "arctic_lookup_calls": (ARCTIC_LOOKUP_CALLS_PER_REQUEST * batch_size),
            "arctic_requested_tokens": (ARCTIC_LOOKUP_TOKENS_PER_REQUEST * batch_size),
            "main_tail_length": ARCTIC_MAIN_TAIL_LENGTH,
            "rescue_chains": [list(chain) for chain in ARCTIC_LOOKUP_CHAINS],
            "carry_fill_slots": RESCUE_CARRY_SLOTS_PER_REQUEST * batch_size,
            "pack_columns": PHYSICAL_DRAFTS,
            "packed_rows": PHYSICAL_DRAFTS * batch_size,
        },
        "drafter_runtime": default_drafter_runtime,
        "tree_attn": {
            "calls": tree_calls,
            "q_rows": tree_calls * TREE_ROWS_PER_REQUEST * batch_size,
            "bias_shape": list(TREE_BIAS_SHAPE),
            "physical_parent_digest": PHYSICAL_PARENT_SHA256,
            "bias_digest": TREE_ANCESTRY_SHA256,
        },
        "gdn": {
            "scan_calls": gdn_scan_calls,
            "launches": gdn_scan_calls * GDN_LAUNCHES_PER_SCAN,
            "path_programs": (gdn_scan_calls * GDN_PATH_PROGRAMS_PER_SCAN),
            "padded_slots": gdn_scan_calls * GDN_PADDED_SLOTS_PER_SCAN,
            "nodes": gdn_scan_calls * GDN_NODES_PER_SCAN,
            "critical_path": GDN_CRITICAL_PATH,
            "grid_z": list(GDN_GRID_Z),
            "max_path_lengths": list(GDN_MAX_PATH_LENGTHS),
            "export_or_mask": GDN_EXPORT_OR_MASK,
        },
        "taw": default_taw,
        "output_publish": {
            "route": OUTPUT_PUBLISH_ROUTE,
            "capacity": OUTPUT_PUBLISH_CAPACITY,
            "requests": batch_size,
            "launches": 2,
            "slots_written": OUTPUT_PUBLISH_CAPACITY * batch_size,
            "accepted_rows_written": batch_size,
            "host_materializations": 0,
            "host_scalar_writes": 0,
            "dtoh": 0,
            "h2d": 0,
            "fallback": 0,
        },
        "accepted_path_pack": {
            "route": ACCEPTED_PATH_PACK_ROUTE,
            "capacity": ACCEPTED_PATH_CAPACITY,
            "requests": batch_size,
            "pack_launches": 2,
            "slots_written": ACCEPTED_PATH_CAPACITY * batch_size,
            "source_walk_slots": WALK_CAP * batch_size,
            "lens_written": batch_size,
            "host_path_items": 0,
            "overflow": 0,
            "fallback": 0,
        },
        "request_key_pack": {
            "route": REQUEST_KEY_PACK_ROUTE,
            "sampler_rows": batch_size,
            "spec_rows": batch_size,
            "map_passes": 2,
            "path_slots_gathered": 2 * REQUEST_KEY_PATH_CAPACITY * batch_size,
            "lens_gathered": 2 * batch_size,
            "zero_launches": 2,
            "gather_launches": 4,
            "host_dict_inserts": 0,
            "host_hash_lookups": 0,
            "missing": 0,
            "fallback": 0,
        },
        "kv_remap": {
            "route": KV_REMAP_ROUTE,
            "path_capacity": KV_REMAP_PATH_CAPACITY,
            "pair_slots": KV_REMAP_PAIR_SLOTS * batch_size,
            "target_pair_slots": KV_REMAP_TARGET_PAIR_SLOTS * batch_size,
            "drafter_pair_slots": KV_REMAP_DRAFTER_PAIR_SLOTS * batch_size,
            "kv_groups": KV_REMAP_GROUPS,
            "target_cache_tensors": KV_REMAP_TARGET_CACHE_TENSORS,
            "drafter_cache_tensors": KV_REMAP_DRAFTER_CACHE_TENSORS,
            "kv_cache_tensors": KV_REMAP_CACHE_TENSORS,
            "kv_planes": KV_REMAP_PLANES,
            "target_prepare_calls": KV_REMAP_TARGET_PREPARE_CALLS,
            "drafter_prepare_calls": KV_REMAP_DRAFTER_PREPARE_CALLS,
            "prepare_calls": (
                KV_REMAP_TARGET_PREPARE_CALLS
                + KV_REMAP_DRAFTER_PREPARE_CALLS
            ),
            "target_apply_cache_calls": KV_REMAP_TARGET_APPLY_CACHE_CALLS,
            "drafter_apply_cache_calls": KV_REMAP_DRAFTER_APPLY_CACHE_CALLS,
            "apply_cache_calls": KV_REMAP_CACHE_TENSORS,
            "src_pair_rows": (
                KV_REMAP_CACHE_TENSORS * KV_REMAP_PATH_CAPACITY * batch_size
            ),
            "dst_pair_rows": (
                KV_REMAP_CACHE_TENSORS * KV_REMAP_PATH_CAPACITY * batch_size
            ),
            "identity_safe_writes": (
                KV_REMAP_CACHE_TENSORS * KV_REMAP_PATH_CAPACITY * batch_size
            ),
            "host_syncs": 0,
            "skips": 0,
            "fallback": 0,
        },
        "conv_commit": {
            "route": CONV_COMMIT_ROUTE,
            "layers": CONV_COMMIT_LAYERS,
            "requests": batch_size,
            "row_elems": CONV_PREGATHER_ROW_ELEMS,
            "channels": GDN_CONV_CHANNELS,
            "state_length": GDN_CONV_STATE_LENGTH,
            "source_rows_per_batch": CONV_COMMIT_SOURCE_ROWS,
            "block": CONV_PREGATHER_BLOCK,
            "direct_launches": 1,
            "gather_launches": 0,
            "scatter_launches": 0,
            "direct_programs": conv_direct_programs,
            "committed_rows": CONV_COMMIT_LAYERS * batch_size,
            "source_staging_reused": True,
            "source_pointer_entries": 48,
            "row_guard_route": CONV_ROW_GUARD_ROUTE,
            "row_guard_kernel_launches": 1,
            "row_guard_programs": (
                CONV_ROW_GUARD_PROGRAMS_PER_REQUEST * batch_size
            ),
            "row_guard_physical_rows": PHYSICAL_ROWS,
            "row_guard_path_capacity": ACCEPTED_PATH_CAPACITY,
            "row_guard_alias_width": CONV_ROW_GUARD_ALIAS_WIDTH,
            "row_guard_compare_capacity": CONV_ROW_GUARD_COMPARE_CAPACITY,
            "row_guard_path_validation_programs": (
                CONV_ROW_GUARD_PATH_PROGRAMS_PER_REQUEST * batch_size
            ),
            "row_guard_path_vector_loads": (
                CONV_ROW_GUARD_PATH_VECTOR_LOADS_PER_REQUEST * batch_size
            ),
            "row_guard_alias_validation_programs": (
                CONV_ROW_GUARD_ALIAS_PROGRAMS_PER_EVENT
            ),
            "row_guard_alias_vector_loads": (
                CONV_ROW_GUARD_ALIAS_VECTOR_LOADS_PER_EVENT
            ),
            "row_guard_selected_row_loads": (
                CONV_ROW_GUARD_SELECTED_ROW_LOADS_PER_PROGRAM
                * CONV_ROW_GUARD_PROGRAMS_PER_REQUEST
                * batch_size
            ),
            "row_guard_peer_topology_proof": (
                CONV_ROW_GUARD_PEER_TOPOLOGY_PROOF
            ),
            "row_guard_torch_index_transforms": 0,
            "row_guard_async_scalar_reductions": 1,
            "row_guard_async_assertions": 1,
            "full_node_writebacks": 0,
            "conv_remaps": 0,
            "host_syncs": 0,
            "skips": 0,
            "fallback": 0,
        },
        "conv_pregather": {
            "route": CONV_PREGATHER_ROUTE,
            "layout_sha256": _fixture_conv_layout_signature(batch_size),
            "stage_calls": 1,
            "stage_before_all_consumes": True,
            "layers": CONV_PREGATHER_LAYERS,
            "requests": batch_size,
            "row_elems": CONV_PREGATHER_ROW_ELEMS,
            "programs": conv_pregather_programs,
            "staged_rows": CONV_PREGATHER_LAYERS * batch_size,
            "consume_calls": CONV_PREGATHER_LAYERS,
            "consume_hits": CONV_PREGATHER_LAYERS,
            "consume_fallbacks": 0,
            "freshness_matches": CONV_PREGATHER_LAYERS,
        },
        "committer": {
            "route": COMMITTER_ROUTE,
            "layers": GDN_LAYERS,
            "requests": batch_size,
            "path_capacity": COMMITTER_PATH_CAPACITY,
            "layout_slots": COMMITTER_PATH_CAPACITY * batch_size,
            "ring_gather_ops": COMMITTER_RING_GATHER_OPS,
            "ring_layer_path_rows": (
                COMMITTER_RING_LAYER_PATH_ROWS_PER_REQUEST * batch_size
            ),
            "neutralize_ops": COMMITTER_NEUTRALIZE_OPS,
            "fused_layer_calls": GDN_LAYERS,
            "graph_replays": 1,
            "graph_captures": 0,
            "host_lens_readbacks": 0,
            "host_flag_readbacks": 0,
            "pointer_table_rebuilds": 0,
            "overflow": 0,
            "fallback": 0,
            "graph_dead": 0,
        },
        "failures": {name: 0 for name in sorted(FAILURE_KEYS)},
    }


def reference_terminal_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    drafter_graph_registry: Sequence[Mapping[str, Any]] | None = None,
    forward_graph_registry: Sequence[Mapping[str, Any]] | None = None,
    conv_pregather_auxiliary: Mapping[str, Any] | None = None,
    nonpure_dispatch: Mapping[str, Any] | None = None,
    nonpure_committer_replays_by_batch: Mapping[str, Any] | None = None,
    fixture_synthetic_runtime_proof: bool = False,
) -> dict[str, Any]:
    """Build the exact terminal record for an ordered sequence of v5 events."""

    if not events:
        raise ValueError("terminal summary requires at least one event")
    modes = {event.get("mode") for event in events}
    producer_pids = {event.get("producer_pid") for event in events}
    if len(modes) != 1 or next(iter(modes)) not in MODE_SEMANTICS:
        raise ValueError(f"terminal events have invalid/mixed modes: {modes}")
    if len(producer_pids) != 1:
        raise ValueError(f"terminal events have mixed producer PIDs: {producer_pids}")
    event_indices = [event.get("event_index") for event in events]
    forward_step_indices = [event.get("forward_step_index") for event in events]
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in event_indices
    ):
        raise ValueError("terminal events have non-integral event indices")
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        for value in forward_step_indices
    ):
        raise ValueError("terminal events have non-integral forward-step indices")
    batch_histogram = {
        str(batch): sum(event.get("batch_size") == batch for event in events)
        for batch in SUPPORTED_BATCH_SIZES
    }
    if drafter_graph_registry is None:
        built_registry = []
        for batch in SUPPORTED_BATCH_SIZES:
            selected = [event for event in events if event.get("batch_size") == batch]
            if not selected:
                continue
            capture_count = sum(
                event.get("drafter_runtime", {}).get("graph_captures") == 1
                for event in selected
            )
            built_registry.append(
                {
                    "batch_size": batch,
                    "graph_signature": _drafter_graph_signature(batch),
                    "captures": 1,
                    "capture_origin": (
                        "measured" if capture_count == 1 else "unmeasured"
                    ),
                    "measured_replays": len(selected),
                    "unmeasured_replays": 0 if capture_count == 1 else 1,
                }
            )
        drafter_graph_registry = built_registry
    if (
        forward_graph_registry is None
        or conv_pregather_auxiliary is None
        or nonpure_dispatch is None
        or nonpure_committer_replays_by_batch is None
    ) and fixture_synthetic_runtime_proof is not True:
        raise ValueError(
            "explicit live forward_graph_registry and "
            "conv_pregather_auxiliary/nonpure dispatch/commit ledgers are required; "
            "synthetic runtime proof "
            "is fixture-only"
        )
    if forward_graph_registry is None:
        capacity = max(int(event["batch_size"]) for event in events)
        forward_graph_registry = [
            {
                "batch_size": batch,
                "graph_signature": forward_graph_structural_signature(batch),
                "conv_layout_sha256": next(
                    (
                        str(event["conv_pregather"]["layout_sha256"])
                        for event in events
                        if int(event["batch_size"]) == batch
                    ),
                    _fixture_conv_layout_signature(batch),
                ),
                "captures": 1,
                "capture_origin": "final_full",
                "stage_calls": 1,
                "stage_before_all_consumes": True,
                "layers": CONV_PREGATHER_LAYERS,
                "requests": batch,
                "row_elems": CONV_PREGATHER_ROW_ELEMS,
                "programs": (
                    CONV_PREGATHER_LAYERS
                    * batch
                    * (
                        (
                            CONV_PREGATHER_ROW_ELEMS
                            + CONV_PREGATHER_BLOCK
                            - 1
                        )
                        // CONV_PREGATHER_BLOCK
                    )
                ),
                "ssi_pointer_entries": CONV_PREGATHER_LAYERS,
                "ssi_groups": 3,
                "source_validations": CONV_PREGATHER_LAYERS,
                "staged_rows": CONV_PREGATHER_LAYERS * batch,
                "consume_calls": CONV_PREGATHER_LAYERS,
                "consume_hits": CONV_PREGATHER_LAYERS,
                "consume_fallbacks": 0,
                "freshness_matches": CONV_PREGATHER_LAYERS,
                "measured_replays": batch_histogram[str(batch)],
            }
            for batch in range(1, capacity + 1)
        ]
    if conv_pregather_auxiliary is None:
        conv_pregather_auxiliary = {
            "profile_capture_stages": 0,
            "aux_capture_stages": 0,
            "host_actual_stages": 0,
            "host_actual_stages_by_batch": {
                str(batch): 0 for batch in SUPPORTED_BATCH_SIZES
            },
        }
    if nonpure_dispatch is None:
        nonpure_dispatch = {
            "guarded_steps": 0,
            "piecewise_steps": 0,
            "none_steps": 0,
            "forbidden_full_steps": 0,
        }
    if nonpure_committer_replays_by_batch is None:
        nonpure_committer_replays_by_batch = {
            str(batch): 0 for batch in SUPPORTED_BATCH_SIZES
        }
    return {
        "schema": TERMINAL_SCHEMA,
        "mode": next(iter(modes)),
        "producer_pid": next(iter(producer_pids)),
        "final": True,
        "event_count": len(events),
        "first_event_index": event_indices[0],
        "last_event_index": event_indices[-1],
        "first_forward_step_index": forward_step_indices[0],
        "last_forward_step_index": forward_step_indices[-1],
        "events_sha256": _events_sha256(events),
        "batch_histogram": batch_histogram,
        "drafter_graph_registry": [
            dict(row) for row in drafter_graph_registry
        ],
        "forward_graph_registry": [
            dict(row) for row in forward_graph_registry
        ],
        "conv_pregather_auxiliary": dict(conv_pregather_auxiliary),
        "nonpure_dispatch": dict(nonpure_dispatch),
        "nonpure_committer_replays_by_batch": dict(
            nonpure_committer_replays_by_batch
        ),
        "scope": json.loads(json.dumps(FIXED_WORK_SCOPE)),
    }


def _located(records: Sequence[dict[str, Any]], prefix: str) -> list[LocatedRecord]:
    return [
        (record, f"{prefix}:{index}") for index, record in enumerate(records, start=1)
    ]


def _located_campaign(
    events: Sequence[dict[str, Any]], prefix: str
) -> list[LocatedRecord]:
    return _located(
        [
            *events,
            reference_terminal_summary(
                events,
                fixture_synthetic_runtime_proof=True,
            ),
        ],
        prefix,
    )


def _valid_fixture() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Unequal event counts are deliberate and must not affect the verdict.
    tail = [
        reference_event(TAIL_MODE, 1, "tail-b1-0", event_index=0, forward_step_index=5),
        reference_event(TAIL_MODE, 1, "tail-b1-1", event_index=1, forward_step_index=6),
        reference_event(TAIL_MODE, 2, "tail-b2-0", event_index=2, forward_step_index=9),
        reference_event(
            TAIL_MODE, 4, "tail-b4-0", event_index=3, forward_step_index=10
        ),
    ]
    hydra = [
        reference_event(
            HYDRA_MODE, 1, "hydra-b1-0", event_index=0, forward_step_index=20
        ),
        reference_event(
            HYDRA_MODE, 3, "hydra-b3-0", event_index=1, forward_step_index=21
        ),
        reference_event(
            HYDRA_MODE, 4, "hydra-b4-0", event_index=2, forward_step_index=25
        ),
        reference_event(
            HYDRA_MODE, 4, "hydra-b4-1", event_index=3, forward_step_index=26
        ),
        reference_event(
            HYDRA_MODE, 4, "hydra-b4-2", event_index=4, forward_step_index=30
        ),
    ]
    return tail, hydra


def _set_path(record: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    cursor: dict[str, Any] = record
    for key in path[:-1]:
        child = cursor[key]
        if not isinstance(child, dict):
            raise TypeError(f"fixture path is not an object: {path!r}")
        cursor = child
    cursor[path[-1]] = value


def _expect_census_failure(name: str, action: Callable[[], object]) -> None:
    try:
        action()
    except CensusError:
        return
    raise AssertionError(f"tamper test {name!r} unexpectedly passed")


def run_self_test() -> dict[str, Any]:
    """Exercise valid B1/B4 input and representative fail-closed tampers."""

    tail, hydra = _valid_fixture()
    valid_report = validate_campaign(
        _located_campaign(tail, "valid-tail"),
        _located_campaign(hydra, "valid-hydra"),
    )
    expected_counts = {
        TAIL_MODE: {"1": 2, "2": 1, "4": 1},
        HYDRA_MODE: {"1": 1, "3": 1, "4": 3},
    }
    if valid_report["event_counts"] != expected_counts:
        raise AssertionError("valid fixture did not preserve unequal event counts")
    if valid_report["forward_step_indices"][TAIL_MODE] != [5, 6, 9, 10]:
        raise AssertionError("valid fixture did not preserve global forward indices")

    # Regression for the B4 timing gate: the mandatory final record is not an
    # event and therefore has no TAW section. A complete event+terminal stream
    # must pass strict arm validation.
    single_b4 = reference_event(
        TAIL_MODE,
        4,
        "single-tail-b4",
        taw=_native_production_taw(4),
    )
    single_records = _located_campaign([single_b4], "single-tail-b4")
    single_arm_report = validate_arm(
        single_records,
        expected_mode=TAIL_MODE,
        expected_route=TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
        required_batches=(4,),
    )
    if (
        single_arm_report["event_count"] != 1
        or single_arm_report["event_counts_by_batch"]["4"] != 1
        or single_arm_report["terminal_summary"]["final"] is not True
    ):
        raise AssertionError("single-event arm did not preserve its terminal proof")
    single_census_raw = "".join(
        _canonical_json(record) + "\n" for record, _source in single_records
    ).encode("ascii")
    bound_arm_report = {
        **single_arm_report,
        "census_sha256": hashlib.sha256(single_census_raw).hexdigest(),
        "census_bytes": len(single_census_raw),
    }
    validate_bound_arm_report(
        bound_arm_report,
        census_raw=single_census_raw,
        census_source="single-tail-b4",
        expected_mode=TAIL_MODE,
        expected_route=TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
        required_batches=(4,),
    )

    mixed_tail, mixed_hydra = _valid_fixture()
    for record in (*mixed_tail, *mixed_hydra):
        batch_size = record["batch_size"]
        if batch_size in SUPPORTED_CAMPAIGN_CAPACITIES:
            record["taw"] = _native_production_taw(batch_size)
    mixed_report = validate_campaign(
        _located_campaign(mixed_tail, "mixed-route-tail"),
        _located_campaign(mixed_hydra, "mixed-route-hydra"),
    )
    for mode in (TAIL_MODE, HYDRA_MODE):
        for batch_size in SUPPORTED_CAMPAIGN_CAPACITIES:
            if not mixed_report["physical_work_histograms"][mode][
                str(batch_size)
            ]["normalized_event_signatures"]:
                raise AssertionError(
                    f"mixed-route fixture lost required B{batch_size} census"
                )
    for incomplete_kwargs in (
        {},
        {
            "forward_graph_registry": valid_report[
                "forward_graph_registries"
            ][TAIL_MODE],
        },
        {
            "conv_pregather_auxiliary": valid_report[
                "conv_pregather_auxiliary"
            ][TAIL_MODE],
        },
    ):
        try:
            reference_terminal_summary(tail, **incomplete_kwargs)
        except ValueError as error:
            if "explicit live forward_graph_registry" not in str(error):
                raise AssertionError(
                    "terminal proof omission failed with the wrong error"
                ) from error
        else:
            raise AssertionError(
                "terminal summary accepted synthetic runtime proof by default"
            )

    tamper_tests: list[tuple[str, Callable[[], object]]] = []

    bad_bound_signature = json.loads(json.dumps(bound_arm_report))
    bad_bound_signature["normalized_work_signature"]["physical_drafts"] = 30
    bad_bound_signature["normalized_work_signature_sha256"] = (
        normalized_work_sha256(bad_bound_signature["normalized_work_signature"])
    )
    tamper_tests.append(
        (
            "bound-arm-normalized-signature",
            lambda: validate_bound_arm_report(
                bad_bound_signature,
                census_raw=single_census_raw,
                census_source="single-tail-b4",
                expected_mode=TAIL_MODE,
                expected_route=TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
                required_batches=(4,),
            ),
        )
    )
    tamper_tests.append(
        (
            "bound-arm-census-substitution",
            lambda: validate_bound_arm_report(
                bound_arm_report,
                census_raw=single_census_raw + b"\n",
                census_source="single-tail-b4",
                expected_mode=TAIL_MODE,
                expected_route=TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
                required_batches=(4,),
            ),
        )
    )

    wrong_route = reference_event(TAIL_MODE, 4, "arm-wrong-route")
    tamper_tests.append(
        (
            "arm-route",
            lambda: validate_arm(
                _located_campaign([wrong_route], "arm-wrong-route"),
                expected_mode=TAIL_MODE,
                expected_route=TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
                required_batches=(4,),
            ),
        )
    )

    wrong_mode = reference_event(
        HYDRA_MODE,
        4,
        "arm-wrong-mode",
        taw=_native_production_taw(4),
    )
    wrong_mode_terminal = reference_terminal_summary(
        [wrong_mode], fixture_synthetic_runtime_proof=True
    )
    wrong_mode_terminal["mode"] = TAIL_MODE
    tamper_tests.append(
        (
            "arm-mode",
            lambda: validate_arm(
                _located(
                    [wrong_mode, wrong_mode_terminal],
                    "arm-wrong-mode",
                ),
                expected_mode=TAIL_MODE,
                expected_route=TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
                required_batches=(4,),
            ),
        )
    )

    bad_arm_event = reference_event(
        TAIL_MODE,
        4,
        "arm-bad-terminal",
        taw=_native_production_taw(4),
    )
    bad_arm_terminal = reference_terminal_summary(
        [bad_arm_event], fixture_synthetic_runtime_proof=True
    )
    bad_arm_terminal["events_sha256"] = "0" * 64
    tamper_tests.append(
        (
            "arm-terminal",
            lambda: validate_arm(
                _located(
                    [bad_arm_event, bad_arm_terminal],
                    "arm-bad-terminal",
                ),
                expected_mode=TAIL_MODE,
                expected_route=TAW_NATIVE_PRECOMPUTE_PRODUCTION_ROUTE,
                required_batches=(4,),
            ),
        )
    )

    def event_tamper(
        name: str,
        path: tuple[str, ...],
        value: object,
        *,
        mode: str = TAIL_MODE,
        batch_size: int = 1,
    ) -> None:
        record = reference_event(mode, batch_size, f"tamper-{name}")
        _set_path(record, path, value)
        tamper_tests.append(
            (
                name,
                lambda record=record, name=name: validate_event(
                    record, source=f"tamper-{name}"
                ),
            )
        )

    event_tamper("schema", ("schema",), "fr13-fixed32-work-census-v0")
    event_tamper("negative-event-index", ("event_index",), -1)
    event_tamper("negative-forward-step-index", ("forward_step_index",), -1)
    event_tamper("event-incomplete", ("event_complete",), False)
    event_tamper("producer-pid", ("producer_pid",), 0)
    event_tamper("tail-mask", ("valid_mask",), 0x7A9CE73E)
    event_tamper("active-nodes", ("active_nodes",), 20)
    event_tamper("verify-rows", ("verify_rows",), 22)
    event_tamper("mtp-forward-calls", ("drafter", "mtp_forward_calls"), 3)
    event_tamper(
        "b4-mtp-forward-rows",
        ("drafter", "mtp_forward_rows"),
        (MTP_FORWARD_CALLS * 4) - 1,
        batch_size=4,
    )
    event_tamper(
        "arctic-lookup-calls",
        ("drafter", "arctic_lookup_calls"),
        ARCTIC_LOOKUP_CALLS_PER_REQUEST - 1,
    )
    event_tamper(
        "arctic-requested-tokens",
        ("drafter", "arctic_requested_tokens"),
        ARCTIC_LOOKUP_TOKENS_PER_REQUEST - 1,
    )
    event_tamper(
        "arctic-rescue-chains",
        ("drafter", "rescue_chains"),
        [[1, 4], [2, 3]],
    )
    event_tamper(
        "rescue-carry-slots",
        ("drafter", "carry_fill_slots"),
        RESCUE_CARRY_SLOTS_PER_REQUEST - 1,
    )
    event_tamper(
        "physical-pack-columns",
        ("drafter", "pack_columns"),
        PHYSICAL_DRAFTS - 1,
    )
    event_tamper(
        "b4-packed-rows",
        ("drafter", "packed_rows"),
        (PHYSICAL_DRAFTS * 4) - 1,
        batch_size=4,
    )
    event_tamper(
        "runtime-association",
        ("drafter_runtime", "association"),
        "previous_runner_step",
    )
    event_tamper(
        "runtime-forward-index",
        ("drafter_runtime", "forward_step_index"),
        1,
    )
    event_tamper(
        "runtime-request-digest",
        ("drafter_runtime", "request_ids_sha256"),
        "not-a-digest",
    )
    event_tamper(
        "runtime-per-request-digest-count",
        ("drafter_runtime", "request_id_sha256s"),
        [],
    )
    event_tamper(
        "runtime-per-request-digest-value",
        ("drafter_runtime", "request_id_sha256s"),
        ["not-a-digest"],
    )
    event_tamper(
        "runtime-proposal-begin",
        ("drafter_runtime", "proposal_begins"),
        0,
    )
    event_tamper(
        "runtime-proposal-end",
        ("drafter_runtime", "proposal_ends"),
        0,
    )
    event_tamper(
        "runtime-graph-signature",
        ("drafter_runtime", "graph_signature"),
        "0" * 64,
    )
    event_tamper(
        "runtime-duplicate-capture",
        ("drafter_runtime", "graph_captures"),
        2,
    )
    event_tamper(
        "runtime-missing-replay",
        ("drafter_runtime", "graph_replays"),
        0,
    )
    event_tamper(
        "runtime-mtp-observation",
        ("drafter_runtime", "mtp_observation"),
        "claimed_call_count",
    )
    event_tamper(
        "runtime-mtp-rows",
        ("drafter_runtime", "mtp_forward_rows"),
        3,
    )
    event_tamper(
        "runtime-arctic-ledger-order",
        ("drafter_runtime", "arctic_ledger"),
        [
            {"kind": "rank1", "calls": 1, "tokens": 4},
            {"kind": "main", "calls": 1, "tokens": 6},
            {"kind": "rank2", "calls": 1, "tokens": 2},
        ],
    )
    event_tamper(
        "runtime-arctic-ledger-token",
        ("drafter_runtime", "arctic_ledger"),
        [
            {"kind": "main", "calls": 1, "tokens": 5},
            {"kind": "rank1", "calls": 1, "tokens": 4},
            {"kind": "rank2", "calls": 1, "tokens": 2},
        ],
    )
    event_tamper(
        "runtime-arctic-total",
        ("drafter_runtime", "arctic_requested_tokens"),
        11,
    )
    event_tamper(
        "runtime-fill-columns",
        ("drafter_runtime", "merge_fill_columns"),
        15,
    )
    event_tamper(
        "runtime-fill-rows",
        ("drafter_runtime", "merge_fill_rows"),
        15,
    )
    event_tamper(
        "runtime-carry",
        ("drafter_runtime", "rescue_carry_slots"),
        3,
    )
    event_tamper(
        "runtime-bx27-publish",
        ("drafter_runtime", "publish_shape"),
        [1, 27],
    )
    event_tamper(
        "runtime-parent",
        ("drafter_runtime", "physical_parent_sha256"),
        "0" * 64,
    )
    event_tamper(
        "runtime-outer-handoff",
        ("drafter_runtime", "outer_handoff_calls"),
        0,
    )
    event_tamper(
        "b4-tree-q-rows",
        ("tree_attn", "q_rows"),
        (TREE_CALLS_PER_EVENT - 4) * TREE_ROWS_PER_REQUEST,
        batch_size=4,
    )
    event_tamper(
        "digest-format",
        ("tree_attn", "physical_parent_digest"),
        "not-a-sha256",
    )
    event_tamper("gdn-launches", ("gdn", "launches"), 95)
    event_tamper("gdn-padded-slots", ("gdn", "padded_slots"), 3935)
    event_tamper("gdn-grid", ("gdn", "grid_z"), [1, 10])
    event_tamper("gdn-critical", ("gdn", "critical_path"), 19)
    event_tamper("taw-preseed", ("taw", "preseeded_batches"), [1, 2, 4, 5])
    event_tamper("taw-cache-miss", ("taw", "topology_cache_hit"), False)
    event_tamper("taw-cache-miss-count", ("taw", "cache_misses"), 1)
    event_tamper("taw-table-shape", ("taw", "table_shape"), [1, 28, 3])
    event_tamper("taw-iterations", ("taw", "loop_iterations"), 21)
    event_tamper("taw-child-lanes", ("taw", "child_lanes"), 63)
    event_tamper("taw-target-rows", ("taw", "target_rows"), 21)
    event_tamper("taw-source-cdf", ("taw", "source_cdf_rows"), 11)
    event_tamper("taw-route", ("taw", "route"), "pytorch_float_and_integer")
    event_tamper(
        "taw-exact-commit-launches",
        ("taw", "exact_commit_launches"),
        TAW_EXACT_COMMIT_LAUNCHES - 1,
    )
    event_tamper(
        "taw-exact-commit-programs",
        ("taw", "exact_commit_programs"),
        TAW_EXACT_COMMIT_PROGRAMS_PER_REQUEST - 1,
    )
    event_tamper(
        "taw-floating-reimplementation",
        ("taw", "floating_sampling_reimplementation"),
        True,
    )
    event_tamper(
        "taw-source-schema",
        ("taw", "source_contract_schema"),
        "fr13-fixed32-taw-source-v0",
    )
    event_tamper(
        "taw-source-sha",
        ("taw", "source_contract_sha256"),
        "0" * 64,
    )
    event_tamper(
        "taw-tensor-call-count",
        ("taw", "tensor_call_census", "full_vocab_softmax_calls"),
        23,
    )
    event_tamper(
        "taw-count-route",
        ("taw", "count_route"),
        "per_event_host_tensor",
    )
    event_tamper(
        "taw-rng-route",
        ("taw", "rng_route"),
        "provided_uniforms",
    )
    event_tamper("taw-vocab", ("taw", "vocab_size"), TAW_VOCAB_SIZE - 1)
    event_tamper("taw-count-shape", ("taw", "count_shape"), [2])
    event_tamper("taw-count-dtype", ("taw", "count_dtype"), "torch.int64")
    event_tamper("taw-count-stride", ("taw", "count_stride"), [2])
    event_tamper("taw-count-contiguous", ("taw", "count_contiguous"), False)
    event_tamper("taw-draft-shape", ("taw", "draft_shape"), [PHYSICAL_DRAFTS - 1])
    event_tamper("taw-parent-dtype", ("taw", "parent_dtype"), "torch.int64")
    event_tamper("taw-bonus-stride", ("taw", "bonus_stride"), [2, 1])
    event_tamper(
        "taw-target-shape",
        ("taw", "target_shape"),
        [PHYSICAL_DRAFTS - 1, TAW_VOCAB_SIZE],
    )
    event_tamper("taw-target-dtype", ("taw", "target_dtype"), "torch.float16")
    event_tamper(
        "taw-target-stride",
        ("taw", "target_stride"),
        [TAW_VOCAB_SIZE + 1, 1],
    )
    event_tamper("taw-target-contiguous", ("taw", "target_contiguous"), False)
    event_tamper(
        "taw-self-shape",
        ("taw", "self_shape"),
        [PHYSICAL_DRAFTS, TAW_VOCAB_SIZE - 1],
    )
    event_tamper("taw-self-dtype", ("taw", "self_dtype"), "torch.bfloat16")
    event_tamper(
        "taw-uniform-shape",
        ("taw", "uniform_shape"),
        [1, WALK_CAP - 1, SAMPLER_MAX_FANOUT],
    )
    event_tamper(
        "taw-uniform-stride",
        ("taw", "uniform_stride"),
        [WALK_CAP * SAMPLER_MAX_FANOUT + 1, SAMPLER_MAX_FANOUT, 1],
    )
    event_tamper(
        "taw-child-table-shape",
        ("taw", "child_table_shape"),
        [1, PHYSICAL_ROWS - 1, SAMPLER_MAX_FANOUT],
    )
    event_tamper(
        "taw-child-counts-shape",
        ("taw", "child_counts_shape"),
        [1, PHYSICAL_ROWS - 1],
    )
    event_tamper(
        "taw-output-shape",
        ("taw", "output_shape"),
        [1, OUTPUT_PUBLISH_CAPACITY - 1],
    )
    event_tamper(
        "taw-output-lens-shape",
        ("taw", "output_lens_shape"),
        [2],
    )
    event_tamper(
        "taw-accepted-path-shape",
        ("taw", "accepted_path_shape"),
        [1, ACCEPTED_PATH_CAPACITY - 1],
    )
    event_tamper(
        "taw-accepted-lens-shape",
        ("taw", "accepted_lens_shape"),
        [2],
    )
    event_tamper("taw-last-row-shape", ("taw", "last_row_shape"), [2])
    event_tamper("taw-exact-current-shape", ("taw", "exact_current_shape"), [2])
    event_tamper("taw-exact-alive-shape", ("taw", "exact_alive_shape"), [2])
    event_tamper("output-slots", ("output_publish", "slots_written"), 31)
    event_tamper(
        "output-copy-calls",
        ("output_publish", "launches"),
        1,
    )
    event_tamper(
        "output-accepted-rows",
        ("output_publish", "accepted_rows_written"),
        0,
    )
    event_tamper(
        "output-host-materialization",
        ("output_publish", "host_materializations"),
        1,
    )
    event_tamper("accepted-pack-capacity", ("accepted_path_pack", "capacity"), 12)
    event_tamper(
        "accepted-pack-copy-calls",
        ("accepted_path_pack", "pack_launches"),
        1,
    )
    event_tamper(
        "request-key-map-passes",
        ("request_key_pack", "map_passes"),
        1,
    )
    event_tamper(
        "request-key-path-slots",
        ("request_key_pack", "path_slots_gathered"),
        16,
    )
    event_tamper(
        "request-key-zero-calls",
        ("request_key_pack", "zero_launches"),
        1,
    )
    event_tamper("request-key-host-dict", ("request_key_pack", "host_dict_inserts"), 1)
    event_tamper("kv-remap-pairs", ("kv_remap", "pair_slots"), 15)
    event_tamper(
        "kv-remap-target-pairs",
        ("kv_remap", "target_pair_slots"),
        15,
    )
    event_tamper(
        "kv-remap-drafter-pairs",
        ("kv_remap", "drafter_pair_slots"),
        15,
    )
    event_tamper(
        "kv-remap-target-prepare",
        ("kv_remap", "target_prepare_calls"),
        0,
    )
    event_tamper(
        "kv-remap-drafter-prepare",
        ("kv_remap", "drafter_prepare_calls"),
        0,
    )
    event_tamper(
        "kv-remap-target-apply",
        ("kv_remap", "target_apply_cache_calls"),
        15,
    )
    event_tamper(
        "kv-remap-drafter-apply",
        ("kv_remap", "drafter_apply_cache_calls"),
        0,
    )
    event_tamper("kv-remap-skip", ("kv_remap", "skips"), 1)
    event_tamper(
        "conv-commit-direct-launches",
        ("conv_commit", "direct_launches"),
        0,
    )
    event_tamper(
        "conv-commit-direct-programs",
        ("conv_commit", "direct_programs"),
        47,
    )
    event_tamper(
        "conv-commit-source-staging",
        ("conv_commit", "source_staging_reused"),
        False,
    )
    event_tamper(
        "conv-commit-row-guard-programs",
        ("conv_commit", "row_guard_programs"),
        47,
    )
    event_tamper(
        "conv-commit-row-guard-index-transform",
        ("conv_commit", "row_guard_torch_index_transforms"),
        1,
    )
    event_tamper(
        "conv-commit-row-guard-alias-width",
        ("conv_commit", "row_guard_alias_width"),
        16,
    )
    event_tamper(
        "conv-commit-row-guard-compare-capacity",
        ("conv_commit", "row_guard_compare_capacity"),
        32,
    )
    event_tamper(
        "conv-commit-row-guard-path-programs",
        ("conv_commit", "row_guard_path_validation_programs"),
        48,
    )
    event_tamper(
        "conv-commit-row-guard-path-loads",
        ("conv_commit", "row_guard_path_vector_loads"),
        48,
    )
    event_tamper(
        "conv-commit-row-guard-alias-programs",
        ("conv_commit", "row_guard_alias_validation_programs"),
        48,
    )
    event_tamper(
        "conv-commit-row-guard-alias-loads",
        ("conv_commit", "row_guard_alias_vector_loads"),
        48,
    )
    event_tamper(
        "conv-commit-row-guard-selected-row-loads",
        ("conv_commit", "row_guard_selected_row_loads"),
        48,
    )
    event_tamper(
        "conv-commit-row-guard-peer-proof",
        ("conv_commit", "row_guard_peer_topology_proof"),
        "event_alias_reload",
    )
    event_tamper(
        "conv-commit-full-writeback",
        ("conv_commit", "full_node_writebacks"),
        1,
    )
    event_tamper(
        "conv-commit-host-sync",
        ("conv_commit", "host_syncs"),
        1,
    )
    event_tamper("conv-commit-skip", ("conv_commit", "skips"), 1)
    event_tamper(
        "conv-pregather-old-route",
        ("conv_pregather", "route"),
        "fixed_all_layers",
    )
    event_tamper(
        "conv-pregather-stage-calls",
        ("conv_pregather", "stage_calls"),
        0,
    )
    event_tamper(
        "conv-pregather-stage-order",
        ("conv_pregather", "stage_before_all_consumes"),
        False,
    )
    event_tamper(
        "conv-pregather-row-elems",
        ("conv_pregather", "row_elems"),
        CONV_PREGATHER_ROW_ELEMS - 1,
    )
    event_tamper(
        "conv-pregather-programs",
        ("conv_pregather", "programs"),
        (CONV_PREGATHER_LAYERS * 32) - 1,
    )
    event_tamper(
        "conv-pregather-fallback",
        ("conv_pregather", "consume_fallbacks"),
        1,
    )
    event_tamper("committer-capacity", ("committer", "path_capacity"), 12)
    event_tamper(
        "committer-gather-rows",
        ("committer", "ring_layer_path_rows"),
        48 * 4 * 12,
    )
    event_tamper("committer-capture", ("committer", "graph_captures"), 1)
    event_tamper("committer-flag-readback", ("committer", "host_flag_readbacks"), 1)
    event_tamper("fallback", ("failures", "fallback"), 1)
    event_tamper("graph-dead", ("failures", "graph_dead"), 1)
    event_tamper("mixed-pseudo", ("failures", "mixed_pseudo"), 1)
    event_tamper(
        "batch-purity-spec-rows",
        ("batch_purity", "spec_rows"),
        0,
    )
    event_tamper(
        "batch-purity-draft-count",
        ("batch_purity", "physical_draft_counts"),
        [PHYSICAL_DRAFTS - 1],
    )
    event_tamper(
        "conv-layout-digest",
        ("conv_pregather", "layout_sha256"),
        "not-a-sha256",
    )

    unknown = reference_event(TAIL_MODE, 1, "unknown-key")
    unknown["unexpected"] = 0
    tamper_tests.append(
        (
            "unknown-key",
            lambda: validate_event(unknown, source="tamper-unknown-key"),
        )
    )

    wrong_batch = reference_event(TAIL_MODE, 1, "wrong-batch")
    wrong_batch["batch_size"] = 2
    tamper_tests.append(
        (
            "incoherent-occupancy",
            lambda: validate_event(wrong_batch, source="tamper-wrong-batch"),
        )
    )
    unsupported_batch = reference_event(TAIL_MODE, 1, "unsupported-batch")
    unsupported_batch["batch_size"] = 5
    tamper_tests.append(
        (
            "unsupported-batch",
            lambda: validate_event(
                unsupported_batch, source="tamper-unsupported-batch"
            ),
        )
    )

    digest_tail, digest_hydra = _valid_fixture()
    digest_hydra[0]["tree_attn"]["bias_digest"] = "3" * 64
    tamper_tests.append(
        (
            "cross-arm-bias-digest",
            lambda: validate_campaign(
                _located_campaign(digest_tail, "digest-tail"),
                _located_campaign(digest_hydra, "digest-hydra"),
            ),
        )
    )

    scaling_tail, scaling_hydra = _valid_fixture()
    for record in scaling_hydra:
        batch_size = record["batch_size"]
        if not isinstance(batch_size, int):
            raise TypeError("fixture batch_size is not an integer")
        scan_calls = record["gdn"]["scan_calls"] + batch_size
        record["gdn"]["scan_calls"] = scan_calls
        record["gdn"]["launches"] = scan_calls * GDN_LAUNCHES_PER_SCAN
        record["gdn"]["path_programs"] = scan_calls * GDN_PATH_PROGRAMS_PER_SCAN
        record["gdn"]["padded_slots"] = scan_calls * GDN_PADDED_SLOTS_PER_SCAN
        record["gdn"]["nodes"] = scan_calls * GDN_NODES_PER_SCAN
    tamper_tests.append(
        (
            "coherent-cross-arm-gdn-scaling",
            lambda: validate_campaign(
                _located_campaign(scaling_tail, "scaling-tail"),
                _located_campaign(scaling_hydra, "scaling-hydra"),
            ),
        )
    )

    duplicate_tail, duplicate_hydra = _valid_fixture()
    duplicate_tail[1]["event_id"] = duplicate_tail[0]["event_id"]
    tamper_tests.append(
        (
            "duplicate-event-id",
            lambda: validate_campaign(
                _located_campaign(duplicate_tail, "duplicate-tail"),
                _located_campaign(duplicate_hydra, "duplicate-hydra"),
            ),
        )
    )

    index_tail, index_hydra = _valid_fixture()
    index_tail[1]["event_index"] = 2
    tamper_tests.append(
        (
            "nonconsecutive-event-index",
            lambda: validate_campaign(
                _located_campaign(index_tail, "index-tail"),
                _located_campaign(index_hydra, "index-hydra"),
            ),
        )
    )

    pid_tail, pid_hydra = _valid_fixture()
    pid_terminal = reference_terminal_summary(
        pid_tail,
        fixture_synthetic_runtime_proof=True,
    )
    pid_tail[1]["producer_pid"] = 9999
    tamper_tests.append(
        (
            "multiple-producer-pids",
            lambda: validate_campaign(
                _located([*pid_tail, pid_terminal], "pid-tail"),
                _located_campaign(pid_hydra, "pid-hydra"),
            ),
        )
    )

    missing_tail, missing_hydra = _valid_fixture()
    missing_tail = [record for record in missing_tail if record["batch_size"] == 1]
    tamper_tests.append(
        (
            "missing-required-batch",
            lambda: validate_campaign(
                _located_campaign(missing_tail, "missing-tail"),
                _located_campaign(missing_hydra, "missing-hydra"),
            ),
        )
    )

    wrong_mode_tail, wrong_mode_hydra = _valid_fixture()
    wrong_mode_terminal = reference_terminal_summary(
        wrong_mode_tail,
        fixture_synthetic_runtime_proof=True,
    )
    wrong_mode_tail[0] = reference_event(HYDRA_MODE, 1, "wrong-arm-mode")
    tamper_tests.append(
        (
            "wrong-arm-mode",
            lambda: validate_campaign(
                _located(
                    [*wrong_mode_tail, wrong_mode_terminal],
                    "wrong-mode-tail",
                ),
                _located_campaign(wrong_mode_hydra, "wrong-mode-hydra"),
            ),
        )
    )

    forward_tail, forward_hydra = _valid_fixture()
    forward_tail[2]["forward_step_index"] = forward_tail[1]["forward_step_index"]
    tamper_tests.append(
        (
            "nonincreasing-forward-step-index",
            lambda: validate_campaign(
                _located_campaign(forward_tail, "forward-tail"),
                _located_campaign(forward_hydra, "forward-hydra"),
            ),
        )
    )

    terminal_tail, terminal_hydra = _valid_fixture()
    tamper_tests.append(
        (
            "missing-terminal",
            lambda: validate_campaign(
                _located(terminal_tail, "missing-terminal-tail"),
                _located_campaign(terminal_hydra, "missing-terminal-hydra"),
            ),
        )
    )
    bad_terminal = reference_terminal_summary(
        terminal_tail,
        fixture_synthetic_runtime_proof=True,
    )
    bad_terminal["events_sha256"] = "0" * 64
    tamper_tests.append(
        (
            "terminal-body-digest",
            lambda: validate_campaign(
                _located([*terminal_tail, bad_terminal], "bad-terminal-tail"),
                _located_campaign(terminal_hydra, "bad-terminal-hydra"),
            ),
        )
    )
    incomplete_terminal = reference_terminal_summary(
        terminal_tail,
        fixture_synthetic_runtime_proof=True,
    )
    incomplete_terminal["final"] = False
    tamper_tests.append(
        (
            "terminal-not-final",
            lambda: validate_campaign(
                _located(
                    [*terminal_tail, incomplete_terminal],
                    "incomplete-terminal-tail",
                ),
                _located_campaign(terminal_hydra, "incomplete-terminal-hydra"),
            ),
        )
    )

    def terminal_tamper(name: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        bad_tail, good_hydra = _valid_fixture()
        terminal = reference_terminal_summary(
            bad_tail,
            fixture_synthetic_runtime_proof=True,
        )
        mutate(terminal)
        tamper_tests.append(
            (
                name,
                lambda bad_tail=bad_tail, terminal=terminal,
                good_hydra=good_hydra, name=name: validate_campaign(
                    _located([*bad_tail, terminal], f"{name}-tail"),
                    _located_campaign(good_hydra, f"{name}-hydra"),
                ),
            )
        )

    terminal_tamper(
        "terminal-batch-histogram",
        lambda terminal: terminal["batch_histogram"].__setitem__("1", 1),
    )
    terminal_tamper(
        "terminal-registry-signature",
        lambda terminal: terminal["drafter_graph_registry"][0].__setitem__(
            "graph_signature", "0" * 64
        ),
    )
    terminal_tamper(
        "terminal-registry-origin",
        lambda terminal: terminal["drafter_graph_registry"][0].__setitem__(
            "capture_origin", "measured"
        ),
    )
    terminal_tamper(
        "terminal-registry-replays",
        lambda terminal: terminal["drafter_graph_registry"][0].__setitem__(
            "measured_replays", 1
        ),
    )
    terminal_tamper(
        "terminal-registry-order",
        lambda terminal: terminal.__setitem__(
            "drafter_graph_registry",
            list(reversed(terminal["drafter_graph_registry"])),
        ),
    )
    terminal_tamper(
        "terminal-forward-capture-count",
        lambda terminal: terminal["forward_graph_registry"][0].__setitem__(
            "captures", 0
        ),
    )
    terminal_tamper(
        "terminal-forward-structural-signature",
        lambda terminal: terminal["forward_graph_registry"][0].__setitem__(
            "graph_signature", "0" * 64
        ),
    )
    terminal_tamper(
        "terminal-forward-conv-layout-signature",
        lambda terminal: terminal["forward_graph_registry"][0].__setitem__(
            "conv_layout_sha256", "0" * 64
        ),
    )
    terminal_tamper(
        "terminal-forward-stage-count",
        lambda terminal: terminal["forward_graph_registry"][0].__setitem__(
            "stage_calls", 0
        ),
    )
    terminal_tamper(
        "terminal-forward-stage-order",
        lambda terminal: terminal["forward_graph_registry"][0].__setitem__(
            "stage_before_all_consumes", False
        ),
    )
    for field, value in (
        ("ssi_pointer_entries", 47),
        ("ssi_groups", 2),
        ("source_validations", 47),
    ):
        terminal_tamper(
            "terminal-forward-" + field.replace("_", "-"),
            lambda terminal, key=field, bad=value: terminal[
                "forward_graph_registry"
            ][0].__setitem__(key, bad),
        )
    terminal_tamper(
        "terminal-forward-replay-histogram",
        lambda terminal: terminal["forward_graph_registry"][0].__setitem__(
            "measured_replays", 0
        ),
    )
    terminal_tamper(
        "terminal-forward-missing-capacity",
        lambda terminal: terminal["forward_graph_registry"].pop(),
    )
    terminal_tamper(
        "terminal-profile-capture-stage",
        lambda terminal: terminal["conv_pregather_auxiliary"].__setitem__(
            "profile_capture_stages", 1
        ),
    )
    terminal_tamper(
        "terminal-aux-capture-stage",
        lambda terminal: terminal["conv_pregather_auxiliary"].__setitem__(
            "aux_capture_stages", 1
        ),
    )
    terminal_tamper(
        "terminal-host-actual-stage",
        lambda terminal: terminal["conv_pregather_auxiliary"].__setitem__(
            "host_actual_stages", 1
        ),
    )
    terminal_tamper(
        "terminal-host-actual-stage-by-batch",
        lambda terminal: terminal["conv_pregather_auxiliary"][
            "host_actual_stages_by_batch"
        ].__setitem__("1", 1),
    )

    def nonpure_forbidden(terminal: dict[str, Any]) -> None:
        terminal["nonpure_dispatch"].update(
            {
                "guarded_steps": 1,
                "forbidden_full_steps": 1,
            }
        )

    terminal_tamper("terminal-nonpure-forbidden-full", nonpure_forbidden)
    terminal_tamper(
        "terminal-nonpure-incoherent",
        lambda terminal: terminal["nonpure_dispatch"].__setitem__(
            "guarded_steps", 1
        ),
    )
    terminal_tamper(
        "terminal-nonpure-committer-exceeds-guard",
        lambda terminal: terminal[
            "nonpure_committer_replays_by_batch"
        ].__setitem__("1", 1),
    )
    terminal_tamper(
        "terminal-scope-overclaim",
        lambda terminal: terminal["scope"]["data_dependent_unproven"].clear(),
    )

    for name, action in tamper_tests:
        _expect_census_failure(name, action)

    return {
        "schema": SELF_TEST_SCHEMA,
        "status": "PASS",
        "valid_batches": list(SUPPORTED_BATCH_SIZES),
        "unequal_event_counts_accepted": expected_counts,
        "tamper_tests_passed": len(tamper_tests),
        "explicit_live_terminal_proof_required": True,
        "normalized_work_signature_sha256": valid_report[
            "normalized_work_signature_sha256"
        ],
    }


def _write_report(report: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
        return
    try:
        output.write_text(text, encoding="utf-8")
    except OSError as error:
        raise CensusError(f"cannot write report {output}: {error}") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Tail/Hydra fixed-32 per-event work-census JSONL."
    )
    parser.add_argument("--tail", type=Path, help="Tail6 fixed32 census JSONL")
    parser.add_argument("--hydra", type=Path, help="Hydra27 fixed32 census JSONL")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=SUPPORTED_CAMPAIGN_CAPACITIES,
        action="append",
        dest="batch_sizes",
        help="required batch size; repeat as needed (default: require B1 and B4)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run valid and tampered in-memory fixtures",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.self_test:
            if args.tail is not None or args.hydra is not None or args.batch_sizes:
                raise CensusError(
                    "--self-test cannot be combined with census inputs or --batch-size"
                )
            report = run_self_test()
        else:
            if args.tail is None or args.hydra is None:
                raise CensusError(
                    "--tail and --hydra are required unless --self-test is used"
                )
            batches = (
                tuple(args.batch_sizes)
                if args.batch_sizes
                else SUPPORTED_CAMPAIGN_CAPACITIES
            )
            report = validate_campaign(
                load_jsonl(args.tail),
                load_jsonl(args.hydra),
                required_batches=batches,
            )
        _write_report(report, args.output)
        return 0
    except (CensusError, AssertionError, TypeError) as error:
        print(f"FAIL fixed32 work census: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
