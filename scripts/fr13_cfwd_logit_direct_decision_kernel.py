#!/usr/bin/env python3
"""Source-only fixed32 CFWD decisions directly from verifier logits.

The served all-parent committer first gathers and softmaxes 13 self rows plus
17 target rows per request. It then materializes normalized target, q_mix, and
residual rows. This candidate keeps only block log-sum-exp stats, applies the
sparse q_mix correction while selecting the final token, and scatters the 30
reachable decision rows into physical slots for an indirection-free walk.

Importing this module is inert. The candidate is deliberately not wired into
serving; SM121a resource and real SWE-Verified gates must qualify it first.
"""

from __future__ import annotations

from typing import Any, NamedTuple

try:
    import torch
except Exception:  # pragma: no cover - torch is present in serving/tests
    torch = None  # type: ignore

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only source gates
    triton = None
    tl = None


CANDIDATE = "fixed32_cfwd_logit_direct_packed_physical_slots_v3"
SOURCE_SCHEMA = "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3"
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))
PHYSICAL_DRAFTS = 31
PHYSICAL_ROWS = 32
SELF_ROWS = 13
TARGET_ROWS = 17
FANOUT = 3
WALK_CAP = 12
VOCAB_SIZE = 248_320
BLOCK_V = 4096
MAX_VOCAB = 262_144
MAX_BLOCKS = MAX_VOCAB // BLOCK_V
FP32_BYTES = 4
PACKED_EVENT_TOKEN_MASK = 0x3FFFF
PACKED_EVENT_ACCEPTED_ROW_SHIFT = 18
PACKED_EVENT_ACCEPTED_ROW_MASK = 0x1F
PACKED_EVENT_PARENT_MASK = 0x800000
SELF_SOURCE_NODES = (4, 5, 6, 7, 9, 10, 12, 14, 15, 19, 20, 21, 30)
TARGET_SOURCE_NODES = (0, 3, 6, 7, 8, 11, 12, 13, 16, 18, 21, 23, 25, 27, 28, 29, 30)
SELF_UNIFORM_LEVELS = (2, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 11)
TARGET_PARENT_SLOTS = (0, 1, 2, 3, 4, 7, 8, 9, 12, 14, 17, 19, 24, 26, 28, 29, 30)
TARGET_UNIFORM_LEVELS = (0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 10)
COMMON_CHILDREN = {
    0: (0, 1, 2),
    1: (3, 4, 5),
    2: (6,),
    3: (7,),
    4: (8, 9, 10),
    9: (13, 14, 15),
    14: (18, 19, 20),
    19: (23,),
    24: (25,),
    26: (27,),
    28: (28,),
    29: (29,),
    30: (30,),
}
MODE_CHILDREN = {
    "tail6_fixed32": COMMON_CHILDREN,
    "hydra27_fixed32": {
        **COMMON_CHILDREN,
        7: (11,),
        8: (12,),
        12: (16,),
        17: (21,),
    },
}
MODE_TOPOLOGY = {
    "tail6_fixed32": ("Tail23", 23, 0x7A9CE7FF),
    "hydra27_fixed32": ("Hydra27", 27, 0x7ABDFFFF),
}


class Fixed32CfwdMetadataBinding(NamedTuple):
    """One-time, exact-value attestation for immutable graph metadata."""

    mode: str
    batch_size: int
    identities: tuple[tuple[str, int, int], ...]


def _metadata_identity(name: str, value) -> tuple[str, int, int]:
    """Bind storage plus mutation version when PyTorch provides one."""
    # TAW metadata is created while vLLM runs under inference_mode. Inference
    # tensors deliberately have no version counter, so their exact contents are
    # attested in prepare_metadata_binding and their storage remains pointer
    # bound. Normal tensors retain the stronger in-place mutation check.
    version = -1 if torch.is_inference(value) else int(value._version)
    return name, int(value.data_ptr()), version


def fixed32_cfwd_logit_direct_contract(
    batch_size: int,
    *,
    mode: str,
    physical_rows: int = PHYSICAL_ROWS,
    vocab_size: int = VOCAB_SIZE,
) -> dict[str, object]:
    """Return the exact fixed32 materialization and launch work ledger."""
    batch = int(batch_size)
    rows = int(physical_rows)
    vocab = int(vocab_size)
    if batch not in (1, 4):
        raise ValueError("logit-direct CFWD qualification is B1 or B4 only")
    if mode not in FIXED32_MODES:
        raise ValueError("logit-direct CFWD requires an exact fixed32 mode")
    logical_topology, logical_drafts, valid_mask = MODE_TOPOLOGY[mode]
    if rows != PHYSICAL_ROWS or vocab != VOCAB_SIZE:
        raise ValueError(
            "logit-direct CFWD physical32/vocab geometry drift: "
            f"rows={rows} vocab={vocab}"
        )

    reachable_rows = SELF_ROWS + TARGET_ROWS
    vocab_blocks = (vocab + BLOCK_V - 1) // BLOCK_V
    # The fixed32 live-layout contract supplies FP32 logits. The two indexed
    # gathers and two softmaxes write 60 FP32 rows; normalization, q_mix, and
    # residual construction write another 132 full-vocabulary FP32 rows.
    incumbent_fp32_rows = 60 + 132
    incumbent_materialized_bytes = (
        batch * vocab * incumbent_fp32_rows * FP32_BYTES
    )
    candidate_block_stat_bytes = (
        batch * reachable_rows * vocab_blocks * 2 * FP32_BYTES
    )
    candidate_workspace_bytes = (
        batch * reachable_rows * MAX_BLOCKS * 2 * FP32_BYTES
    )
    compact_decision_bytes = batch * (
        SELF_ROWS * 8 + TARGET_ROWS * (3 * 8 + 1)
    )
    physical_decision_bytes = batch * (
        PHYSICAL_DRAFTS * 8 + PHYSICAL_ROWS * (3 * 8 + 1)
    )
    # The exact verifier vocabulary fits 18 bits. Pack the emitted token, the
    # accepted child row, and a parent-event bit into one int64 word. The fixed
    # walk no longer rereads tree topology or dead intermediate decisions.
    packed_physical_decision_bytes = batch * (
        PHYSICAL_DRAFTS * 8 + PHYSICAL_ROWS * 8
    )
    return {
        "candidate": CANDIDATE,
        "schema": SOURCE_SCHEMA,
        "mode": mode,
        "batch_size": batch,
        "physical_rows": rows,
        "logical_topology": logical_topology,
        "logical_drafts": logical_drafts,
        "valid_mask": valid_mask,
        "fixed_work_for_exact_bound_topology": True,
        "physical_drafts": PHYSICAL_DRAFTS,
        "self_rows_per_request": SELF_ROWS,
        "target_rows_per_request": TARGET_ROWS,
        "fanout": FANOUT,
        "walk_cap": WALK_CAP,
        "vocab_size": vocab,
        "vocab_blocks": vocab_blocks,
        "incumbent_probability_producer_tensor_ops": 4,
        "candidate_triton_launch_sites": 2,
        "producer_dispatch_sites_removed_static": 2,
        "physical_kernel_launches_removed": "pending_gpu_trace",
        "additional_dense_decision_launches_removed_unscored": True,
        "integer_commit_launches_before": 1,
        "integer_commit_launches_after": 1,
        "decision_programs_per_request_before": reachable_rows,
        "decision_programs_per_request_after": reachable_rows,
        "decision_values_stored_per_request_before": (
            SELF_ROWS + 4 * TARGET_ROWS
        ),
        "decision_values_stored_per_request_after": (
            SELF_ROWS + TARGET_ROWS
        ),
        "integer_walk_topology_index_loads_per_request_before": 2 * WALK_CAP,
        "integer_walk_topology_index_loads_per_request_after": 0,
        "compact_decision_workspace_bytes_before": compact_decision_bytes,
        "physical_decision_workspace_bytes_before": physical_decision_bytes,
        "packed_physical_decision_workspace_bytes_after": (
            packed_physical_decision_bytes
        ),
        "decision_workspace_bytes_removed": (
            physical_decision_bytes - packed_physical_decision_bytes
        ),
        "packed_event_token_mask": 0x3FFFF,
        "packed_event_accepted_row_shift": 18,
        "packed_event_parent_mask": 0x800000,
        "decision_workspace_zero_seeded_once": True,
        "decision_padding_initialization_stores_per_event": 0,
        "incumbent_full_vocab_fp32_rows_materialized": (
            incumbent_fp32_rows * batch
        ),
        "incumbent_full_vocab_materialized_bytes": (
            incumbent_materialized_bytes
        ),
        "candidate_block_stat_materialized_bytes": (
            candidate_block_stat_bytes
        ),
        "candidate_block_stat_workspace_bytes": candidate_workspace_bytes,
        "full_vocab_materialized_bytes_removed": (
            incumbent_materialized_bytes - candidate_block_stat_bytes
        ),
        "candidate_default_off": True,
    }


def _inverse_cdf_oracle(weights, uniform):
    cdf = torch.cumsum(weights, dim=-1)
    threshold = uniform.unsqueeze(-1).to(weights.dtype) * cdf[..., -1:]
    return (cdf <= threshold).sum(dim=-1).clamp(max=weights.shape[-1] - 1)


def logit_direct_decision_oracle(
    logits,
    kid_tokens,
    kid_mask,
    uniforms,
) -> tuple[Any, Any, Any, Any, Any]:
    """CPU oracle for the logit-space sparse residual identity."""
    if torch is None:
        raise RuntimeError("logit-direct CFWD oracle requires torch")
    shifted = torch.exp(
        logits.to(torch.float64) - logits.max(-1, keepdim=True).values
    )
    total = shifted.sum(-1, keepdim=True)
    safe_tokens = kid_tokens.clamp(min=0, max=logits.shape[-1] - 1)
    kid_raw = torch.gather(shifted, -1, safe_tokens) * kid_mask
    overlap = kid_raw.sum(-1, keepdim=True)
    source = _inverse_cdf_oracle(kid_raw, uniforms[..., 0])
    selected = torch.gather(kid_tokens, -1, source.unsqueeze(-1)).squeeze(-1)
    weights = kid_raw / overlap.clamp(min=1.0e-30)
    same = (kid_tokens == selected.unsqueeze(-1)) & kid_mask
    q_mix_selected = (weights * same).sum(-1)
    selected_raw = torch.gather(
        shifted, -1, selected.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    accept_probability = (
        (selected_raw / total.squeeze(-1))
        / q_mix_selected.clamp(min=1.0e-30)
    ).clamp(max=1.0)
    has_kids = kid_mask.any(-1)
    accepted = (
        has_kids
        & (overlap.squeeze(-1) > 0)
        & (uniforms[..., 1] < accept_probability)
    )

    q_mix = torch.zeros_like(shifted)
    q_mix.scatter_add_(-1, safe_tokens, weights * kid_mask)
    residual_scaled = (shifted - total * q_mix).clamp(min=0)
    residual_mass = residual_scaled.sum(-1, keepdim=True)
    sampling_weights = torch.where(
        residual_mass > 0,
        residual_scaled,
        shifted,
    )
    rejected = _inverse_cdf_oracle(sampling_weights, uniforms[..., 2])
    return source, selected, rejected, accepted, sampling_weights


def pack_physical_event_oracle(
    source,
    selected_token,
    rejected_token,
    accepted,
    child_table,
    child_counts,
):
    """Pack exactly the decision products consumed by the physical walk."""
    if torch is None:
        raise RuntimeError("packed CFWD event oracle requires torch")
    batch = int(source.shape[0]) if isinstance(source, torch.Tensor) else 0
    expected = (batch, PHYSICAL_ROWS)
    if (
        batch not in (1, 4)
        or any(
            not isinstance(value, torch.Tensor) or tuple(value.shape) != expected
            for value in (source, selected_token, rejected_token, accepted)
        )
        or tuple(child_table.shape) != (batch, PHYSICAL_ROWS, FANOUT)
        or tuple(child_counts.shape) != expected
        or any(
            value.dtype != torch.long
            for value in (
                source,
                selected_token,
                rejected_token,
                child_table,
                child_counts,
            )
        )
        or accepted.dtype != torch.bool
    ):
        raise ValueError("packed CFWD event oracle geometry drift")
    if (
        torch.any((source < 0) | (source >= FANOUT))
        or torch.any((selected_token < 0) | (selected_token >= VOCAB_SIZE))
        or torch.any((rejected_token < 0) | (rejected_token >= VOCAB_SIZE))
        or torch.any((child_counts < 0) | (child_counts > FANOUT))
    ):
        raise ValueError("packed CFWD event oracle domain drift")
    accepted_node = torch.gather(
        child_table,
        2,
        source.unsqueeze(2),
    ).squeeze(2)
    if torch.any(accepted & ((accepted_node < 0) | (accepted_node >= PHYSICAL_DRAFTS))):
        raise ValueError("packed CFWD accepted child drift")
    emitted = torch.where(accepted, selected_token, rejected_token)
    accepted_row = torch.where(accepted, accepted_node + 1, 0)
    event = (
        emitted
        | (accepted_row << PACKED_EVENT_ACCEPTED_ROW_SHIFT)
        | PACKED_EVENT_PARENT_MASK
    )
    return torch.where(child_counts > 0, event, 0)


def packed_physical_walk_oracle(self_token, event, bonus_token):
    """CPU oracle for the fixed-cap packed-event integer committer."""
    if torch is None:
        raise RuntimeError("packed CFWD walk oracle requires torch")
    batch = int(self_token.shape[0]) if isinstance(self_token, torch.Tensor) else 0
    if (
        batch not in (1, 4)
        or tuple(self_token.shape) != (batch, PHYSICAL_DRAFTS)
        or tuple(event.shape) != (batch, PHYSICAL_ROWS)
        or tuple(bonus_token.shape) != (batch,)
        or any(
            value.dtype != torch.long for value in (self_token, event, bonus_token)
        )
    ):
        raise ValueError("packed CFWD walk oracle geometry drift")

    output = torch.full((batch, PHYSICAL_ROWS), -1, dtype=torch.long)
    output_lens = torch.zeros(batch, dtype=torch.long)
    paths = torch.zeros((batch, 16), dtype=torch.long)
    path_lens = torch.zeros(batch, dtype=torch.long)
    last_row = torch.zeros(batch, dtype=torch.long)
    for request in range(batch):
        current = -1
        alive = True
        for _level in range(WALK_CAP):
            if not alive:
                continue
            packed = int(event[request, current + 1])
            if packed & PACKED_EVENT_PARENT_MASK:
                output[request, output_lens[request]] = (
                    packed & PACKED_EVENT_TOKEN_MASK
                )
                output_lens[request] += 1
                accepted_row = (
                    packed >> PACKED_EVENT_ACCEPTED_ROW_SHIFT
                ) & PACKED_EVENT_ACCEPTED_ROW_MASK
                if accepted_row:
                    paths[request, path_lens[request]] = accepted_row
                    path_lens[request] += 1
                    current = accepted_row - 1
                    last_row[request] = accepted_row
                else:
                    alive = False
            else:
                output[request, output_lens[request]] = (
                    self_token[request, current]
                    if current >= 0
                    else bonus_token[request]
                )
                output_lens[request] += 1
                alive = False
    return output, output_lens, paths, path_lens, last_row


if triton is not None:

    @triton.jit
    def _fr13_cfwd_logit_block_stats_kernel(
        self_logits,
        target_logits,
        self_source_indices,
        target_source_indices,
        block_maxima,
        block_sums,
        invalid_out,
        vocab_size,
        self_total_rows,
        source_rows,
        BLOCK_V: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        """Write one stable max/sum-exp pair per reachable logit block."""
        row = tl.program_id(0)
        block = tl.program_id(1)
        is_self = row < self_total_rows
        self_row = tl.minimum(row, self_total_rows - 1)
        target_row = tl.maximum(row - self_total_rows, 0)
        source_self_raw = tl.load(
            self_source_indices + self_row,
            mask=is_self,
            other=0,
        ).to(tl.int64)
        source_target_raw = tl.load(
            target_source_indices + target_row,
            mask=~is_self,
            other=0,
        ).to(tl.int64)
        source_self_valid = (source_self_raw >= 0) & (source_self_raw < source_rows)
        source_target_valid = (source_target_raw >= 0) & (
            source_target_raw < source_rows
        )
        source_valid = tl.where(is_self, source_self_valid, source_target_valid)
        tl.atomic_max(invalid_out, 1, mask=~source_valid)
        source_self = tl.maximum(0, tl.minimum(source_self_raw, source_rows - 1))
        source_target = tl.maximum(0, tl.minimum(source_target_raw, source_rows - 1))
        offsets = block * BLOCK_V + tl.arange(0, BLOCK_V)
        valid = offsets < vocab_size
        self_values = tl.load(
            self_logits + source_self * vocab_size + offsets,
            mask=is_self & valid,
            other=-float("inf"),
        ).to(tl.float32)
        target_values = tl.load(
            target_logits + source_target * vocab_size + offsets,
            mask=(~is_self) & valid,
            other=-float("inf"),
        ).to(tl.float32)
        values = tl.where(is_self, self_values, target_values)
        block_max = tl.max(values, axis=0)
        has_mass = block_max != -float("inf")
        stable_max = tl.where(has_mass, block_max, 0.0)
        block_sum = tl.sum(tl.exp(values - stable_max), axis=0)
        offset = row * MAX_BLOCKS + block
        tl.store(block_maxima + offset, block_max)
        tl.store(block_sums + offset, block_sum)


    @triton.jit
    def _fr13_cfwd_logit_direct_decision_kernel(
        self_logits,
        target_logits,
        self_source_indices,
        target_source_indices,
        block_maxima,
        block_sums,
        drafts,
        child_table,
        child_counts,
        self_uniform_levels,
        target_parent_slots,
        target_uniform_levels,
        uniforms,
        self_token_out,
        event_out,
        invalid_out,
        vocab_size,
        number_of_blocks,
        self_total_rows,
        source_rows,
        SELF_ROWS: tl.constexpr,
        TARGET_ROWS: tl.constexpr,
        PHYSICAL_DRAFTS: tl.constexpr,
        PHYSICAL_ROWS: tl.constexpr,
        FANOUT: tl.constexpr,
        WALK_CAP: tl.constexpr,
        BLOCK_V: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        """Fuse normalization, canonical accept, and sparse inverse CDF."""
        row = tl.program_id(0)
        is_self = row < self_total_rows
        self_row = tl.minimum(row, self_total_rows - 1)
        target_row = tl.maximum(row - self_total_rows, 0)
        self_request = self_row // SELF_ROWS
        target_request = target_row // TARGET_ROWS
        request = tl.where(is_self, self_request, target_request)
        self_local = self_row - self_request * SELF_ROWS
        target_local = target_row - target_request * TARGET_ROWS

        block_offsets = tl.arange(0, MAX_BLOCKS)
        valid_blocks = block_offsets < number_of_blocks
        maxima = tl.load(
            block_maxima + row * MAX_BLOCKS + block_offsets,
            mask=valid_blocks,
            other=-float("inf"),
        )
        sums = tl.load(
            block_sums + row * MAX_BLOCKS + block_offsets,
            mask=valid_blocks,
            other=0.0,
        )
        row_max = tl.max(maxima, axis=0)
        raw_block_mass = sums * tl.exp(maxima - row_max)
        raw_block_mass = tl.where(valid_blocks, raw_block_mass, 0.0)
        raw_block_cdf = tl.cumsum(raw_block_mass, axis=0)
        total = tl.sum(
            tl.where(block_offsets == MAX_BLOCKS - 1, raw_block_cdf, 0.0),
            axis=0,
        )

        source_self_raw = tl.load(
            self_source_indices + self_row,
            mask=is_self,
            other=0,
        ).to(tl.int64)
        source_target_raw = tl.load(
            target_source_indices + target_row,
            mask=~is_self,
            other=0,
        ).to(tl.int64)
        source_self_valid = (source_self_raw >= 0) & (source_self_raw < source_rows)
        source_target_valid = (source_target_raw >= 0) & (
            source_target_raw < source_rows
        )
        source_valid = tl.where(is_self, source_self_valid, source_target_valid)
        source_self = tl.maximum(0, tl.minimum(source_self_raw, source_rows - 1))
        source_target = tl.maximum(0, tl.minimum(source_target_raw, source_rows - 1))
        self_level_raw = tl.load(
            self_uniform_levels + self_local,
            mask=is_self,
            other=0,
        ).to(tl.int64)
        target_level_raw = tl.load(
            target_uniform_levels + target_local,
            mask=~is_self,
            other=0,
        ).to(tl.int64)
        self_level_valid = (self_level_raw >= 0) & (self_level_raw < WALK_CAP)
        target_level_valid = (target_level_raw >= 0) & (
            target_level_raw < WALK_CAP
        )
        level_valid = tl.where(is_self, self_level_valid, target_level_valid)
        level_raw = tl.where(is_self, self_level_raw, target_level_raw)
        level = tl.maximum(0, tl.minimum(level_raw, WALK_CAP - 1))
        uniform_base = (request * WALK_CAP + level) * 3

        parent_slot_raw = tl.load(
            target_parent_slots + target_local,
            mask=~is_self,
            other=0,
        ).to(tl.int64)
        parent_valid = is_self | (
            (parent_slot_raw >= 0) & (parent_slot_raw < PHYSICAL_ROWS)
        )
        parent_slot = tl.maximum(
            0,
            tl.minimum(parent_slot_raw, PHYSICAL_ROWS - 1),
        )
        child_count = tl.load(
            child_counts + request * PHYSICAL_ROWS + parent_slot,
            mask=(~is_self) & parent_valid,
            other=0,
        ).to(tl.int64)
        child_count_valid = is_self | (
            (child_count >= 0) & (child_count <= FANOUT)
        )
        child_lanes = tl.arange(0, 4)
        child_lane_mask = child_lanes < FANOUT
        child_nodes = tl.load(
            child_table
            + request * PHYSICAL_ROWS * FANOUT
            + parent_slot * FANOUT
            + child_lanes,
            mask=(~is_self) & parent_valid & child_lane_mask,
            other=-1,
        ).to(tl.int64)
        target_child_lane = (~is_self) & child_lane_mask
        expected_child = target_child_lane & (child_lanes < child_count)
        child_node_valid = (child_nodes >= 0) & (child_nodes < PHYSICAL_DRAFTS)
        child_packing_valid = (
            tl.sum(
                (
                    target_child_lane
                    & (expected_child != child_node_valid)
                ).to(tl.int32),
                axis=0,
            )
            == 0
        )
        valid_child_node = expected_child & child_node_valid
        safe_nodes = tl.maximum(0, tl.minimum(child_nodes, PHYSICAL_DRAFTS - 1))
        kid_tokens = tl.load(
            drafts + request * PHYSICAL_DRAFTS + safe_nodes,
            mask=valid_child_node,
            other=0,
        ).to(tl.int64)
        kid_token_valid = (kid_tokens >= 0) & (kid_tokens < vocab_size)
        valid_child = valid_child_node & kid_token_valid
        safe_tokens = tl.maximum(0, tl.minimum(kid_tokens, vocab_size - 1))
        source_uniform = tl.load(uniforms + uniform_base)
        accept_uniform = tl.load(uniforms + uniform_base + 1)
        token_uniform = tl.load(uniforms + uniform_base + 2)
        uniforms_valid = (
            (source_uniform >= 0.0)
            & (source_uniform < 1.0)
            & (accept_uniform >= 0.0)
            & (accept_uniform < 1.0)
            & (token_uniform >= 0.0)
            & (token_uniform < 1.0)
        )
        metadata_valid = (
            source_valid
            & level_valid
            & parent_valid
            & child_count_valid
            & child_packing_valid
            & (
                tl.sum(
                    (valid_child_node & ~kid_token_valid).to(tl.int32),
                    axis=0,
                )
                == 0
            )
            & uniforms_valid
        )
        tl.atomic_max(invalid_out, 1, mask=~metadata_valid)
        kid_logits = tl.load(
            target_logits + source_target * vocab_size + safe_tokens,
            mask=valid_child,
            other=-float("inf"),
        ).to(tl.float32)
        kid_raw = tl.where(valid_child, tl.exp(kid_logits - row_max), 0.0)
        source_cdf = tl.cumsum(kid_raw, axis=0)
        overlap_mass = tl.sum(
            tl.where(child_lanes == FANOUT - 1, source_cdf, 0.0),
            axis=0,
        )
        q_weights = kid_raw / tl.maximum(overlap_mass, 1.0e-30)
        source_threshold = source_uniform * overlap_mass
        sampled_source = tl.sum(
            ((source_cdf <= source_threshold) & child_lane_mask).to(tl.int32),
            axis=0,
        )
        sampled_source = tl.minimum(sampled_source, FANOUT - 1)
        selected_token = tl.sum(
            tl.where(child_lanes == sampled_source, kid_tokens, 0),
            axis=0,
        ).to(tl.int64)
        selected_raw = tl.sum(
            tl.where(child_lanes == sampled_source, kid_raw, 0.0),
            axis=0,
        )
        selected_q_mix = tl.sum(
            tl.where(kid_tokens == selected_token, q_weights, 0.0),
            axis=0,
        )
        accept_probability = tl.minimum(
            (selected_raw / tl.maximum(total, 1.0e-30))
            / tl.maximum(selected_q_mix, 1.0e-30),
            1.0,
        )
        accepted = (
            (~is_self)
            & (child_count > 0)
            & (overlap_mass > 0.0)
            & (accept_uniform < accept_probability)
        )

        same_tokens = kid_tokens[:, None] == kid_tokens[None, :]
        q_mix_by_lane = tl.sum(
            tl.where(same_tokens, q_weights[None, :], 0.0),
            axis=1,
        )
        kid0 = tl.sum(tl.where(child_lanes == 0, kid_tokens, 0), axis=0)
        kid1 = tl.sum(tl.where(child_lanes == 1, kid_tokens, 0), axis=0)
        first_occurrence = (
            (child_lanes == 0)
            | ((child_lanes == 1) & (kid_tokens != kid0))
            | (
                (child_lanes == 2)
                & (kid_tokens != kid0)
                & (kid_tokens != kid1)
            )
        )
        corrections = tl.where(
            valid_child & first_occurrence,
            tl.minimum(kid_raw, total * q_mix_by_lane),
            0.0,
        )
        kid_blocks = safe_tokens // BLOCK_V
        correction_by_block = tl.sum(
            tl.where(
                block_offsets[:, None] == kid_blocks[None, :],
                corrections[None, :],
                0.0,
            ),
            axis=1,
        )
        residual_block_mass = tl.maximum(
            raw_block_mass - correction_by_block,
            0.0,
        )
        residual_block_cdf = tl.cumsum(residual_block_mass, axis=0)
        residual_total = tl.sum(
            tl.where(
                block_offsets == MAX_BLOCKS - 1,
                residual_block_cdf,
                0.0,
            ),
            axis=0,
        )
        use_raw = is_self | (residual_total <= 0.0)
        sampling_block_mass = tl.where(
            use_raw,
            raw_block_mass,
            residual_block_mass,
        )
        block_cdf = tl.cumsum(sampling_block_mass, axis=0)
        sampling_total = tl.sum(
            tl.where(block_offsets == MAX_BLOCKS - 1, block_cdf, 0.0),
            axis=0,
        )
        token_threshold = (
            token_uniform * sampling_total
        )
        selected_block = tl.sum(
            ((block_cdf <= token_threshold) & valid_blocks).to(tl.int32),
            axis=0,
        )
        selected_block = tl.minimum(selected_block, number_of_blocks - 1)
        prior_mass = tl.sum(
            tl.where(
                (selected_block > 0)
                & (block_offsets == selected_block - 1),
                block_cdf,
                0.0,
            ),
            axis=0,
        )
        selected_block_mass = tl.sum(
            tl.where(
                block_offsets == selected_block,
                sampling_block_mass,
                0.0,
            ),
            axis=0,
        )

        local_offsets = tl.arange(0, BLOCK_V)
        token_offsets = selected_block * BLOCK_V + local_offsets
        valid_tokens = token_offsets < vocab_size
        self_values = tl.load(
            self_logits + source_self * vocab_size + token_offsets,
            mask=is_self & valid_tokens,
            other=-float("inf"),
        ).to(tl.float32)
        target_values = tl.load(
            target_logits + source_target * vocab_size + token_offsets,
            mask=(~is_self) & valid_tokens,
            other=-float("inf"),
        ).to(tl.float32)
        local_logits = tl.where(is_self, self_values, target_values)
        local_raw = tl.where(
            valid_tokens,
            tl.exp(local_logits - row_max),
            0.0,
        )
        local_q_mix = tl.sum(
            tl.where(
                token_offsets[:, None] == kid_tokens[None, :],
                q_weights[None, :],
                0.0,
            ),
            axis=1,
        )
        local_residual = tl.maximum(local_raw - total * local_q_mix, 0.0)
        local_weights = tl.where(use_raw, local_raw, local_residual)
        local_cdf = tl.cumsum(local_weights, axis=0)
        local_total = tl.sum(
            tl.where(local_offsets == BLOCK_V - 1, local_cdf, 0.0),
            axis=0,
        )
        local_threshold = (token_threshold - prior_mass) * (
            local_total / tl.maximum(selected_block_mass, 1.0e-30)
        )
        selected_local = tl.sum(
            ((local_cdf <= local_threshold) & valid_tokens).to(tl.int32),
            axis=0,
        )
        valid_count = tl.minimum(
            BLOCK_V,
            vocab_size - selected_block * BLOCK_V,
        )
        selected_local = tl.minimum(selected_local, valid_count - 1)
        sampled_token = selected_block * BLOCK_V + selected_local

        physical_self_offset = source_self.to(tl.int32)
        physical_target_offset = (
            request * PHYSICAL_ROWS + parent_slot.to(tl.int32)
        )
        tl.store(
            self_token_out + physical_self_offset,
            sampled_token,
            mask=is_self,
        )
        accepted_node = tl.sum(
            tl.where(child_lanes == sampled_source, child_nodes, 0),
            axis=0,
        ).to(tl.int64)
        emitted_token = tl.where(accepted, selected_token, sampled_token)
        accepted_row = tl.where(accepted, accepted_node + 1, 0)
        packed_event = (
            emitted_token | (accepted_row << 18) | 0x800000
        )
        tl.store(
            event_out + physical_target_offset,
            tl.where(child_count > 0, packed_event, 0),
            mask=~is_self,
        )


def _metadata_operands(
    *,
    self_source_indices,
    target_source_indices,
    child_table,
    child_counts,
    self_uniform_levels,
    target_parent_slots,
    target_uniform_levels,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("self_source_indices", self_source_indices),
        ("target_source_indices", target_source_indices),
        ("child_table", child_table),
        ("child_counts", child_counts),
        ("self_uniform_levels", self_uniform_levels),
        ("target_parent_slots", target_parent_slots),
        ("target_uniform_levels", target_uniform_levels),
    )


def prepare_metadata_binding(
    *,
    self_source_indices,
    target_source_indices,
    child_table,
    child_counts,
    self_uniform_levels,
    target_parent_slots,
    target_uniform_levels,
    batch_size: int,
    mode: str,
) -> Fixed32CfwdMetadataBinding:
    """Synchronously attest immutable metadata once, before graph capture."""
    if torch is None:
        raise RuntimeError("logit-direct CFWD metadata binding requires torch")
    fixed32_cfwd_logit_direct_contract(batch_size, mode=mode)
    batch = int(batch_size)
    operands = _metadata_operands(
        self_source_indices=self_source_indices,
        target_source_indices=target_source_indices,
        child_table=child_table,
        child_counts=child_counts,
        self_uniform_levels=self_uniform_levels,
        target_parent_slots=target_parent_slots,
        target_uniform_levels=target_uniform_levels,
    )
    if any(not isinstance(value, torch.Tensor) for _name, value in operands):
        raise TypeError("logit-direct CFWD metadata must be tensors")
    device = operands[0][1].device
    if any(value.device != device for _name, value in operands):
        raise ValueError("logit-direct CFWD metadata must share one device")
    if any(value.dtype != torch.long for _name, value in operands):
        raise ValueError("logit-direct CFWD metadata must be int64")
    if any(not value.is_contiguous() for _name, value in operands):
        raise ValueError("logit-direct CFWD metadata must be contiguous")

    self_sources = [
        request * PHYSICAL_DRAFTS + node
        for request in range(batch)
        for node in SELF_SOURCE_NODES
    ]
    target_sources = [
        request * PHYSICAL_DRAFTS + node
        for request in range(batch)
        for node in TARGET_SOURCE_NODES
    ]
    table = [[-1] * FANOUT for _ in range(PHYSICAL_ROWS)]
    counts = [0] * PHYSICAL_ROWS
    for parent_slot, children in MODE_CHILDREN[mode].items():
        counts[parent_slot] = len(children)
        table[parent_slot][: len(children)] = children
    expected_cpu = {
        "self_source_indices": self_sources,
        "target_source_indices": target_sources,
        "child_table": [table for _ in range(batch)],
        "child_counts": [counts for _ in range(batch)],
        "self_uniform_levels": list(SELF_UNIFORM_LEVELS),
        "target_parent_slots": list(TARGET_PARENT_SLOTS),
        "target_uniform_levels": list(TARGET_UNIFORM_LEVELS),
    }
    for name, value in operands:
        expected = torch.tensor(expected_cpu[name], dtype=torch.long, device=device)
        if tuple(value.shape) != tuple(expected.shape) or not torch.equal(
            value, expected
        ):
            raise ValueError(f"logit-direct CFWD exact metadata drift: {name}")
    return Fixed32CfwdMetadataBinding(
        mode=mode,
        batch_size=batch,
        identities=tuple(_metadata_identity(name, value) for name, value in operands),
    )


def _validate_metadata_binding(
    binding: Fixed32CfwdMetadataBinding,
    *,
    operands: tuple[tuple[str, Any], ...],
    batch_size: int,
    mode: str,
) -> None:
    if not isinstance(binding, Fixed32CfwdMetadataBinding):
        raise TypeError("logit-direct CFWD requires an exact metadata binding")
    observed = tuple(_metadata_identity(name, value) for name, value in operands)
    if (
        binding.mode != mode
        or binding.batch_size != int(batch_size)
        or binding.identities != observed
    ):
        raise ValueError("logit-direct CFWD metadata binding drift")


def workspace_spec(batch_size: int) -> dict[str, tuple[tuple[int, ...], Any]]:
    """Return the fixed persistent workspace for a B1/B4 specialization."""
    if torch is None:
        raise RuntimeError("logit-direct CFWD workspace requires torch")
    batch = int(batch_size)
    if batch not in (1, 4):
        raise ValueError("logit-direct CFWD workspace is B1 or B4 only")
    all_rows = batch * (SELF_ROWS + TARGET_ROWS)
    return {
        "block_maxima": ((all_rows, MAX_BLOCKS), torch.float32),
        "block_sums": ((all_rows, MAX_BLOCKS), torch.float32),
        "self_token": ((batch, PHYSICAL_DRAFTS), torch.long),
        "event": ((batch, PHYSICAL_ROWS), torch.long),
        "invalid": ((1,), torch.int32),
    }


def allocate_workspace(*, device, batch_size: int) -> dict[str, Any]:
    """Allocate zero-seeded candidate buffers once, before graph capture."""
    workspace = {
        name: torch.zeros(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in workspace_spec(batch_size).items()
    }
    return workspace


def _reject_workspace_aliases(
    operands: tuple[Any, ...], workspace: dict[str, Any]
) -> None:
    """Reject any overlap involving buffers written by the two-stage candidate."""
    readable = [(f"operand[{index}]", value) for index, value in enumerate(operands)]
    writable = list(workspace.items())
    intervals: list[tuple[str, Any, int, int, bool]] = []
    for name, value in readable:
        start = int(value.data_ptr())
        stop = start + int(value.numel()) * int(value.element_size())
        intervals.append((name, value.device, start, stop, False))
    for name, value in writable:
        start = int(value.data_ptr())
        stop = start + int(value.numel()) * int(value.element_size())
        intervals.append((name, value.device, start, stop, True))
    for left_index, (left_name, left_device, left_start, left_stop, left_write) in enumerate(
        intervals
    ):
        for right_name, right_device, right_start, right_stop, right_write in intervals[
            left_index + 1 :
        ]:
            if not (left_write or right_write) or left_device != right_device:
                continue
            if left_start < right_stop and right_start < left_stop:
                raise ValueError(
                    "logit-direct CFWD writable storage alias: "
                    f"{left_name} overlaps {right_name}"
                )


def launch_logit_direct_fixed32(
    *,
    self_logits,
    target_logits,
    self_source_indices,
    target_source_indices,
    drafts,
    child_table,
    child_counts,
    self_uniform_levels,
    target_parent_slots,
    target_uniform_levels,
    uniforms,
    workspace: dict[str, Any],
    batch_size: int,
    mode: str,
    metadata_binding: Fixed32CfwdMetadataBinding,
) -> tuple[Any, Any, Any, Any, Any]:
    """Launch the unserved two-stage candidate after exact qualification."""
    if triton is None or tl is None or torch is None:
        raise RuntimeError("logit-direct CFWD requires Triton and torch")
    contract = fixed32_cfwd_logit_direct_contract(batch_size, mode=mode)
    batch = int(batch_size)
    tensors = (
        self_logits,
        target_logits,
        self_source_indices,
        target_source_indices,
        drafts,
        child_table,
        child_counts,
        self_uniform_levels,
        target_parent_slots,
        target_uniform_levels,
        uniforms,
    )
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("logit-direct CFWD operands must be tensors")
    if any(not value.is_cuda for value in tensors):
        raise ValueError("logit-direct CFWD operands must be CUDA tensors")
    if any(value.device != self_logits.device for value in tensors):
        raise ValueError("logit-direct CFWD operands must share one device")
    expected_dtypes = (
        torch.float32,
        torch.float32,
        torch.long,
        torch.long,
        torch.long,
        torch.long,
        torch.long,
        torch.long,
        torch.long,
        torch.long,
        torch.float32,
    )
    if any(
        value.dtype != dtype
        for value, dtype in zip(tensors, expected_dtypes, strict=True)
    ):
        raise ValueError("logit-direct CFWD exact tensor dtype drift")
    if any(not value.is_contiguous() for value in tensors):
        raise ValueError("logit-direct CFWD operands must be contiguous")
    flat_rows = batch * PHYSICAL_DRAFTS
    if (
        self_logits.ndim != 2
        or target_logits.ndim != 2
        or int(self_logits.shape[0]) != flat_rows
        or int(self_logits.shape[1]) != VOCAB_SIZE
        or tuple(target_logits.shape) != tuple(self_logits.shape)
        or tuple(self_source_indices.shape) != (batch * SELF_ROWS,)
        or tuple(target_source_indices.shape) != (batch * TARGET_ROWS,)
        or tuple(drafts.shape) != (batch, PHYSICAL_DRAFTS)
        or tuple(child_table.shape) != (batch, PHYSICAL_ROWS, FANOUT)
        or tuple(child_counts.shape) != (batch, PHYSICAL_ROWS)
        or tuple(self_uniform_levels.shape) != (SELF_ROWS,)
        or tuple(target_parent_slots.shape) != (TARGET_ROWS,)
        or tuple(target_uniform_levels.shape) != (TARGET_ROWS,)
        or tuple(uniforms.shape) != (batch, WALK_CAP, 3)
    ):
        raise ValueError("logit-direct CFWD exact tensor geometry drift")
    metadata_operands = _metadata_operands(
        self_source_indices=self_source_indices,
        target_source_indices=target_source_indices,
        child_table=child_table,
        child_counts=child_counts,
        self_uniform_levels=self_uniform_levels,
        target_parent_slots=target_parent_slots,
        target_uniform_levels=target_uniform_levels,
    )
    _validate_metadata_binding(
        metadata_binding,
        operands=metadata_operands,
        batch_size=batch,
        mode=mode,
    )
    expected_workspace = workspace_spec(batch)
    if not isinstance(workspace, dict) or set(workspace) != set(expected_workspace):
        raise ValueError("logit-direct CFWD workspace key drift")
    for name, (shape, dtype) in expected_workspace.items():
        value = workspace.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != self_logits.device
            or not value.is_contiguous()
        ):
            raise ValueError(f"logit-direct CFWD workspace drift: {name}")
    _reject_workspace_aliases(tensors, workspace)

    all_rows = batch * (SELF_ROWS + TARGET_ROWS)
    self_total_rows = batch * SELF_ROWS
    number_of_blocks = int(contract["vocab_blocks"])
    _fr13_cfwd_logit_block_stats_kernel[(all_rows, number_of_blocks)](
        self_logits,
        target_logits,
        self_source_indices,
        target_source_indices,
        workspace["block_maxima"],
        workspace["block_sums"],
        workspace["invalid"],
        VOCAB_SIZE,
        self_total_rows,
        flat_rows,
        BLOCK_V=BLOCK_V,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=4,
        num_stages=3,
    )
    _fr13_cfwd_logit_direct_decision_kernel[(all_rows,)](
        self_logits,
        target_logits,
        self_source_indices,
        target_source_indices,
        workspace["block_maxima"],
        workspace["block_sums"],
        drafts,
        child_table,
        child_counts,
        self_uniform_levels,
        target_parent_slots,
        target_uniform_levels,
        uniforms,
        workspace["self_token"],
        workspace["event"],
        workspace["invalid"],
        VOCAB_SIZE,
        number_of_blocks,
        self_total_rows,
        flat_rows,
        SELF_ROWS=SELF_ROWS,
        TARGET_ROWS=TARGET_ROWS,
        PHYSICAL_DRAFTS=PHYSICAL_DRAFTS,
        PHYSICAL_ROWS=PHYSICAL_ROWS,
        FANOUT=FANOUT,
        WALK_CAP=WALK_CAP,
        BLOCK_V=BLOCK_V,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=8,
        num_stages=3,
    )
    return (
        workspace["self_token"],
        workspace["event"],
    )
