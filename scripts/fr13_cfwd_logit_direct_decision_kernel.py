#!/usr/bin/env python3
"""Source-only fixed32 CFWD decisions directly from verifier logits.

The served all-parent committer first gathers and softmaxes 13 self rows plus
17 target rows per request. It then materializes normalized target, q_mix, and
residual rows. This candidate keeps only block log-sum-exp stats and applies
the sparse q_mix correction while selecting the final token.

Importing this module is inert. The candidate is deliberately not wired into
serving; SM121a resource and real SWE-Verified gates must qualify it first.
"""

from __future__ import annotations

from typing import Any

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


CANDIDATE = "fixed32_cfwd_logit_direct_decisions_v1"
SOURCE_SCHEMA = "fr13.fixed32.cfwd_logit_direct_decisions.v1"
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))
PHYSICAL_DRAFTS = 31
PHYSICAL_ROWS = 32
SELF_ROWS = 13
TARGET_ROWS = 17
FANOUT = 3
WALK_CAP = 12
VOCAB_SIZE = 151_936
BLOCK_V = 256
MAX_VOCAB = 262_144
MAX_BLOCKS = MAX_VOCAB // BLOCK_V
FP32_BYTES = 4


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
    return {
        "candidate": CANDIDATE,
        "schema": SOURCE_SCHEMA,
        "mode": mode,
        "batch_size": batch,
        "physical_rows": rows,
        "logical_tree_limit": rows,
        "fixed_work_for_any_logical_tree_lte": rows,
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


if triton is not None:

    @triton.jit
    def _fr13_cfwd_logit_block_stats_kernel(
        self_logits,
        target_logits,
        self_source_indices,
        target_source_indices,
        block_maxima,
        block_sums,
        vocab_size,
        self_total_rows,
        BLOCK_V: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        """Write one stable max/sum-exp pair per reachable logit block."""
        row = tl.program_id(0)
        block = tl.program_id(1)
        is_self = row < self_total_rows
        self_row = tl.minimum(row, self_total_rows - 1)
        target_row = tl.maximum(row - self_total_rows, 0)
        source_self = tl.load(
            self_source_indices + self_row,
            mask=is_self,
            other=0,
        ).to(tl.int64)
        source_target = tl.load(
            target_source_indices + target_row,
            mask=~is_self,
            other=0,
        ).to(tl.int64)
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
        source_out,
        selected_token_out,
        rejected_token_out,
        accepted_out,
        vocab_size,
        number_of_blocks,
        self_total_rows,
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
        total = tl.sum(raw_block_mass, axis=0)

        source_self = tl.load(
            self_source_indices + self_row,
            mask=is_self,
            other=0,
        ).to(tl.int64)
        source_target = tl.load(
            target_source_indices + target_row,
            mask=~is_self,
            other=0,
        ).to(tl.int64)
        self_level = tl.load(
            self_uniform_levels + self_local,
            mask=is_self,
            other=0,
        ).to(tl.int64)
        target_level = tl.load(
            target_uniform_levels + target_local,
            mask=~is_self,
            other=0,
        ).to(tl.int64)
        level = tl.where(is_self, self_level, target_level)
        uniform_base = (request * WALK_CAP + level) * 3

        parent_slot = tl.load(
            target_parent_slots + target_local,
            mask=~is_self,
            other=0,
        ).to(tl.int64)
        child_count = tl.load(
            child_counts + request * PHYSICAL_ROWS + parent_slot,
            mask=~is_self,
            other=0,
        ).to(tl.int64)
        child_lanes = tl.arange(0, 4)
        child_lane_mask = child_lanes < FANOUT
        child_nodes = tl.load(
            child_table
            + request * PHYSICAL_ROWS * FANOUT
            + parent_slot * FANOUT
            + child_lanes,
            mask=(~is_self) & child_lane_mask,
            other=-1,
        ).to(tl.int64)
        target_child_lane = (~is_self) & child_lane_mask
        valid_child = target_child_lane & (child_nodes >= 0)
        safe_nodes = tl.maximum(0, tl.minimum(child_nodes, PHYSICAL_DRAFTS - 1))
        kid_tokens = tl.load(
            drafts + request * PHYSICAL_DRAFTS + safe_nodes,
            mask=target_child_lane,
            other=0,
        ).to(tl.int64)
        safe_tokens = tl.maximum(0, tl.minimum(kid_tokens, vocab_size - 1))
        kid_logits = tl.load(
            target_logits + source_target * vocab_size + safe_tokens,
            mask=valid_child,
            other=-float("inf"),
        ).to(tl.float32)
        kid_raw = tl.where(valid_child, tl.exp(kid_logits - row_max), 0.0)
        overlap_mass = tl.sum(kid_raw, axis=0)
        q_weights = kid_raw / tl.maximum(overlap_mass, 1.0e-30)

        source_threshold = tl.load(uniforms + uniform_base) * overlap_mass
        source_cdf = tl.cumsum(kid_raw, axis=0)
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
            & (tl.load(uniforms + uniform_base + 1) < accept_probability)
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
        residual_total = tl.sum(residual_block_mass, axis=0)
        use_raw = is_self | (residual_total <= 0.0)
        sampling_block_mass = tl.where(
            use_raw,
            raw_block_mass,
            residual_block_mass,
        )
        sampling_total = tl.sum(sampling_block_mass, axis=0)
        token_threshold = (
            tl.load(uniforms + uniform_base + 2) * sampling_total
        )
        block_cdf = tl.cumsum(sampling_block_mass, axis=0)
        selected_block = tl.sum(
            ((block_cdf <= token_threshold) & valid_blocks).to(tl.int32),
            axis=0,
        )
        selected_block = tl.minimum(selected_block, number_of_blocks - 1)
        prior_mass = tl.sum(
            tl.where(
                block_offsets < selected_block,
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
        local_threshold = token_threshold - prior_mass
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

        tl.store(self_token_out + self_row, sampled_token, mask=is_self)
        tl.store(source_out + target_row, sampled_source, mask=~is_self)
        tl.store(
            selected_token_out + target_row,
            selected_token,
            mask=~is_self,
        )
        tl.store(
            rejected_token_out + target_row,
            sampled_token,
            mask=~is_self,
        )
        tl.store(accepted_out + target_row, accepted, mask=~is_self)


def workspace_spec(batch_size: int) -> dict[str, tuple[tuple[int, ...], Any]]:
    """Return the fixed persistent workspace for a B1/B4 specialization."""
    if torch is None:
        raise RuntimeError("logit-direct CFWD workspace requires torch")
    batch = int(batch_size)
    if batch not in (1, 4):
        raise ValueError("logit-direct CFWD workspace is B1 or B4 only")
    all_rows = batch * (SELF_ROWS + TARGET_ROWS)
    target_rows = batch * TARGET_ROWS
    return {
        "block_maxima": ((all_rows, MAX_BLOCKS), torch.float32),
        "block_sums": ((all_rows, MAX_BLOCKS), torch.float32),
        "self_token": ((batch, SELF_ROWS), torch.long),
        "source": ((batch, TARGET_ROWS), torch.long),
        "selected_token": ((batch, TARGET_ROWS), torch.long),
        "rejected_token": ((batch, TARGET_ROWS), torch.long),
        "accepted": ((batch, TARGET_ROWS), torch.bool),
    }


def allocate_workspace(*, device, batch_size: int) -> dict[str, Any]:
    """Allocate candidate buffers before capture; never called at import."""
    return {
        name: torch.empty(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in workspace_spec(batch_size).items()
    }


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
    expected_workspace = workspace_spec(batch)
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
        VOCAB_SIZE,
        self_total_rows,
        BLOCK_V=BLOCK_V,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=4,
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
        workspace["source"],
        workspace["selected_token"],
        workspace["rejected_token"],
        workspace["accepted"],
        VOCAB_SIZE,
        number_of_blocks,
        self_total_rows,
        SELF_ROWS=SELF_ROWS,
        TARGET_ROWS=TARGET_ROWS,
        PHYSICAL_DRAFTS=PHYSICAL_DRAFTS,
        PHYSICAL_ROWS=PHYSICAL_ROWS,
        FANOUT=FANOUT,
        WALK_CAP=WALK_CAP,
        BLOCK_V=BLOCK_V,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=8,
    )
    return (
        workspace["self_token"],
        workspace["source"],
        workspace["selected_token"],
        workspace["rejected_token"],
        workspace["accepted"],
    )
