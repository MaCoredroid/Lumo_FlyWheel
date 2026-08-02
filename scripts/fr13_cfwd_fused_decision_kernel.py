#!/usr/bin/env python3
"""Default-off fixed32 sparse all-parent decision kernels.

The candidate replaces the dense q_mix/residual materialization in
``_fr13_fixed32_taw_all_parent_decisions``.  It keeps the canonical
distributional rule: every child occurrence contributes to q_mix, including
duplicate draft tokens.  Rejection sampling scans sparse-corrected target
probabilities in fixed vocabulary blocks.

Importing this module is inert.  ``launch`` requires CUDA tensors and is only
called by the separately gated fixed32 route.
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


CANDIDATE = "fixed32_cfwd_sparse_decisions_v1"
SOURCE_SCHEMA = "fr13.fixed32.cfwd_sparse_decisions.v1"
SELF_ROWS = 13
TARGET_ROWS = 17
FANOUT = 3
WALK_CAP = 12
BLOCK_V = 256
MAX_VOCAB = 262144
MAX_BLOCKS = MAX_VOCAB // BLOCK_V


if triton is not None:

    @triton.jit
    def _fr13_cfwd_probability_block_sums_kernel(
        self_probability,
        target_probability,
        block_sums,
        vocab_size,
        self_total_rows,
        SELF_ROWS: tl.constexpr,
        TARGET_ROWS: tl.constexpr,
        BLOCK_V: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        """Reduce self and target probability rows into fixed-size blocks."""
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_V + tl.arange(0, BLOCK_V)
        is_self = row < self_total_rows
        self_row = tl.minimum(row, self_total_rows - 1)
        target_row = tl.maximum(row - self_total_rows, 0)
        valid = offsets < vocab_size
        self_values = tl.load(
            self_probability + self_row * vocab_size + offsets,
            mask=is_self & valid,
            other=0.0,
        ).to(tl.float32)
        target_values = tl.load(
            target_probability + target_row * vocab_size + offsets,
            mask=(~is_self) & valid,
            other=0.0,
        ).to(tl.float32)
        block_mass = tl.sum(self_values + target_values, axis=0)
        tl.store(block_sums + row * MAX_BLOCKS + block, block_mass)


    @triton.jit
    def _fr13_cfwd_parent_setup_kernel(
        target_probability,
        probability_block_sums,
        drafts,
        child_table,
        child_counts,
        target_parent_slots,
        target_uniform_levels,
        uniforms,
        target_totals,
        kid_tokens_out,
        q_weights_out,
        source_out,
        selected_token_out,
        accepted_out,
        vocab_size,
        self_total_rows,
        SELF_ROWS: tl.constexpr,
        TARGET_ROWS: tl.constexpr,
        PHYSICAL_DRAFTS: tl.constexpr,
        PHYSICAL_ROWS: tl.constexpr,
        FANOUT: tl.constexpr,
        WALK_CAP: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        """Fuse child gather, source draw, duplicate q_mix, and accept test."""
        row = tl.program_id(0)
        request = row // TARGET_ROWS
        local_row = row - request * TARGET_ROWS

        block_offsets = tl.arange(0, MAX_BLOCKS)
        target_total = tl.sum(
            tl.load(
                probability_block_sums
                + (self_total_rows + row) * MAX_BLOCKS
                + block_offsets
            ),
            axis=0,
        )
        tl.store(target_totals + row, target_total)

        parent_slot = tl.load(target_parent_slots + local_row).to(tl.int64)
        child_count = tl.load(
            child_counts + request * PHYSICAL_ROWS + parent_slot
        ).to(tl.int64)
        child_lanes = tl.arange(0, 4)
        child_mask = child_lanes < FANOUT
        child_nodes = tl.load(
            child_table
            + request * PHYSICAL_ROWS * FANOUT
            + parent_slot * FANOUT
            + child_lanes,
            mask=child_mask,
            other=-1,
        ).to(tl.int64)
        valid_child = child_mask & (child_nodes >= 0)
        safe_nodes = tl.maximum(0, tl.minimum(child_nodes, PHYSICAL_DRAFTS - 1))
        kid_tokens = tl.load(
            drafts + request * PHYSICAL_DRAFTS + safe_nodes,
            mask=child_mask,
            other=0,
        ).to(tl.int64)
        safe_tokens = tl.maximum(0, tl.minimum(kid_tokens, vocab_size - 1))
        overlaps = tl.load(
            target_probability + row * vocab_size + safe_tokens,
            mask=valid_child,
            other=0.0,
        ).to(tl.float32)
        overlap_mass = tl.sum(overlaps, axis=0)
        safe_overlap_mass = tl.maximum(overlap_mass, 1.0e-30)
        q_weights = overlaps / safe_overlap_mass

        level = tl.load(target_uniform_levels + local_row).to(tl.int64)
        uniform_base = (request * WALK_CAP + level) * 3
        source_threshold = tl.load(uniforms + uniform_base) * overlap_mass
        overlap_cdf = tl.cumsum(overlaps, axis=0)
        source = tl.sum(
            ((overlap_cdf <= source_threshold) & child_mask).to(tl.int32),
            axis=0,
        )
        source = tl.minimum(source, FANOUT - 1)
        selected_token = tl.sum(
            tl.where(child_lanes == source, kid_tokens, 0),
            axis=0,
        ).to(tl.int64)
        selected_probability = tl.load(
            target_probability
            + row * vocab_size
            + tl.maximum(0, tl.minimum(selected_token, vocab_size - 1))
        ).to(tl.float32) / tl.maximum(target_total, 1.0e-30)
        selected_q_mix = tl.sum(
            tl.where(kid_tokens == selected_token, q_weights, 0.0),
            axis=0,
        )
        accept_probability = tl.minimum(
            selected_probability / tl.maximum(selected_q_mix, 1.0e-30),
            1.0,
        )
        accepted = (
            (child_count > 0)
            & (overlap_mass > 0.0)
            & (tl.load(uniforms + uniform_base + 1) < accept_probability)
        )

        tl.store(
            kid_tokens_out + row * FANOUT + child_lanes,
            kid_tokens,
            mask=child_mask,
        )
        tl.store(
            q_weights_out + row * FANOUT + child_lanes,
            q_weights,
            mask=child_mask,
        )
        tl.store(source_out + row, source)
        tl.store(selected_token_out + row, selected_token)
        tl.store(accepted_out + row, accepted)


    @triton.jit
    def _fr13_cfwd_residual_block_sums_kernel(
        target_probability,
        target_totals,
        kid_tokens,
        q_weights,
        residual_block_sums,
        vocab_size,
        TARGET_ROWS: tl.constexpr,
        FANOUT: tl.constexpr,
        BLOCK_V: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        """Reduce max(p - q_mix, 0) without materializing either dense row."""
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_V + tl.arange(0, BLOCK_V)
        valid = offsets < vocab_size
        probability = tl.load(
            target_probability + row * vocab_size + offsets,
            mask=valid,
            other=0.0,
        ).to(tl.float32)
        target_total = tl.load(target_totals + row)
        probability = probability / tl.maximum(target_total, 1.0e-30)

        kid0 = tl.load(kid_tokens + row * FANOUT)
        kid1 = tl.load(kid_tokens + row * FANOUT + 1)
        kid2 = tl.load(kid_tokens + row * FANOUT + 2)
        q0 = tl.load(q_weights + row * FANOUT)
        q1 = tl.load(q_weights + row * FANOUT + 1)
        q2 = tl.load(q_weights + row * FANOUT + 2)
        q_mix = (
            tl.where(offsets == kid0, q0, 0.0)
            + tl.where(offsets == kid1, q1, 0.0)
            + tl.where(offsets == kid2, q2, 0.0)
        )
        residual = tl.maximum(probability - q_mix, 0.0)
        tl.store(
            residual_block_sums + row * MAX_BLOCKS + block,
            tl.sum(residual, axis=0),
        )


    @triton.jit
    def _fr13_cfwd_inverse_cdf_kernel(
        self_probability,
        target_probability,
        probability_block_sums,
        residual_block_sums,
        target_totals,
        kid_tokens,
        q_weights,
        self_uniform_levels,
        target_uniform_levels,
        uniforms,
        self_token_out,
        rejected_token_out,
        vocab_size,
        self_total_rows,
        SELF_ROWS: tl.constexpr,
        TARGET_ROWS: tl.constexpr,
        FANOUT: tl.constexpr,
        WALK_CAP: tl.constexpr,
        BLOCK_V: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        """Select a probability block, then perform one in-block CDF scan."""
        row = tl.program_id(0)
        is_self = row < self_total_rows
        self_row = tl.minimum(row, self_total_rows - 1)
        target_row = tl.maximum(row - self_total_rows, 0)
        self_request = self_row // SELF_ROWS
        self_local = self_row - self_request * SELF_ROWS
        target_request = target_row // TARGET_ROWS
        target_local = target_row - target_request * TARGET_ROWS
        request = tl.where(is_self, self_request, target_request)
        self_level = tl.load(self_uniform_levels + self_local).to(tl.int64)
        target_level = tl.load(target_uniform_levels + target_local).to(tl.int64)
        level = tl.where(is_self, self_level, target_level)
        uniform = tl.load(uniforms + (request * WALK_CAP + level) * 3 + 2)

        block_offsets = tl.arange(0, MAX_BLOCKS)
        number_of_blocks = (vocab_size + BLOCK_V - 1) // BLOCK_V
        valid_blocks = block_offsets < number_of_blocks
        raw_block_sums = tl.load(
            probability_block_sums + row * MAX_BLOCKS + block_offsets
        )
        residual_sums = tl.load(
            residual_block_sums + target_row * MAX_BLOCKS + block_offsets
        )
        residual_total = tl.sum(
            tl.where(valid_blocks, residual_sums, 0.0), axis=0
        )
        use_raw_probability = is_self | (residual_total <= 0.0)
        selected_block_sums = tl.where(
            use_raw_probability,
            raw_block_sums,
            residual_sums,
        )
        selected_block_sums = tl.where(
            valid_blocks, selected_block_sums, 0.0
        )
        total = tl.sum(selected_block_sums, axis=0)
        threshold = uniform * total
        block_cdf = tl.cumsum(selected_block_sums, axis=0)
        selected_block = tl.sum(
            ((block_cdf <= threshold) & valid_blocks).to(tl.int32),
            axis=0,
        )
        selected_block = tl.minimum(selected_block, number_of_blocks - 1)
        prefix = tl.sum(
            tl.where(block_offsets < selected_block, selected_block_sums, 0.0),
            axis=0,
        )

        local_offsets = tl.arange(0, BLOCK_V)
        token_offsets = selected_block * BLOCK_V + local_offsets
        valid_tokens = token_offsets < vocab_size
        self_values = tl.load(
            self_probability + self_row * vocab_size + token_offsets,
            mask=is_self & valid_tokens,
            other=0.0,
        ).to(tl.float32)
        target_values = tl.load(
            target_probability + target_row * vocab_size + token_offsets,
            mask=(~is_self) & valid_tokens,
            other=0.0,
        ).to(tl.float32)
        raw_values = self_values + target_values

        target_total = tl.load(target_totals + target_row)
        target_values_normalized = target_values / tl.maximum(
            target_total, 1.0e-30
        )
        kid0 = tl.load(kid_tokens + target_row * FANOUT)
        kid1 = tl.load(kid_tokens + target_row * FANOUT + 1)
        kid2 = tl.load(kid_tokens + target_row * FANOUT + 2)
        q0 = tl.load(q_weights + target_row * FANOUT)
        q1 = tl.load(q_weights + target_row * FANOUT + 1)
        q2 = tl.load(q_weights + target_row * FANOUT + 2)
        q_mix = (
            tl.where(token_offsets == kid0, q0, 0.0)
            + tl.where(token_offsets == kid1, q1, 0.0)
            + tl.where(token_offsets == kid2, q2, 0.0)
        )
        residual_values = tl.maximum(target_values_normalized - q_mix, 0.0)
        values = tl.where(use_raw_probability, raw_values, residual_values)
        local_cdf = tl.cumsum(values, axis=0)
        local_threshold = threshold - prefix
        selected_local = tl.sum(
            ((local_cdf <= local_threshold) & valid_tokens).to(tl.int32),
            axis=0,
        )
        valid_count = tl.minimum(BLOCK_V, vocab_size - selected_block * BLOCK_V)
        selected_local = tl.minimum(selected_local, valid_count - 1)
        selected_token = selected_block * BLOCK_V + selected_local
        tl.store(self_token_out + self_row, selected_token, mask=is_self)
        tl.store(
            rejected_token_out + target_row,
            selected_token,
            mask=~is_self,
        )


def workspace_spec(batch_size: int) -> dict[str, tuple[tuple[int, ...], Any]]:
    """Return the persistent fixed32 workspace contract for one batch size."""
    if torch is None:
        raise RuntimeError("fixed32 fused decisions require torch")
    if batch_size not in (1, 2, 3, 4):
        raise ValueError(f"fixed32 fused decision batch must be 1..4: {batch_size}")
    all_rows = batch_size * (SELF_ROWS + TARGET_ROWS)
    target_rows = batch_size * TARGET_ROWS
    return {
        "cfwd_fused_probability_block_sums": (
            (all_rows, MAX_BLOCKS),
            torch.float32,
        ),
        "cfwd_fused_residual_block_sums": (
            (target_rows, MAX_BLOCKS),
            torch.float32,
        ),
        "cfwd_fused_target_totals": ((target_rows,), torch.float32),
        "cfwd_fused_kid_tokens": ((target_rows, FANOUT), torch.long),
        "cfwd_fused_q_weights": ((target_rows, FANOUT), torch.float32),
        "cfwd_fused_self_token": ((batch_size, SELF_ROWS), torch.long),
        "cfwd_fused_source": ((batch_size, TARGET_ROWS), torch.long),
        "cfwd_fused_selected_token": ((batch_size, TARGET_ROWS), torch.long),
        "cfwd_fused_rejected_token": ((batch_size, TARGET_ROWS), torch.long),
        "cfwd_fused_accepted": ((batch_size, TARGET_ROWS), torch.bool),
    }


def preseed_workspace(entry: dict[str, Any], *, device, batch_size: int) -> None:
    """Allocate every candidate output before graph capture."""
    for name, (shape, dtype) in workspace_spec(batch_size).items():
        entry[name] = torch.empty(shape, dtype=dtype, device=device)


def _validate_workspace(entry: dict[str, Any], *, device, batch_size: int) -> None:
    for name, (shape, dtype) in workspace_spec(batch_size).items():
        value = entry.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != device
            or not value.is_contiguous()
        ):
            raise RuntimeError(
                f"FR13 fixed32 fused decision workspace drift: {name}"
            )


def launch(
    entry: dict[str, Any],
    drafts,
    uniforms,
    probability_caches: tuple[Any, Any],
) -> tuple[Any, Any, Any, Any, Any]:
    """Launch the four-stage sparse decision pipeline."""
    if triton is None or tl is None or torch is None:
        raise RuntimeError("FR13 fixed32 fused decisions require Triton and torch")
    self_probability, target_probability = probability_caches
    if not isinstance(self_probability, torch.Tensor) or not self_probability.is_cuda:
        raise RuntimeError("FR13 fixed32 fused decisions require CUDA probabilities")
    device = self_probability.device
    batch_size = int(entry["batch_size"])
    vocab_size = int(self_probability.shape[-1])
    expected = (
        (self_probability, (batch_size * SELF_ROWS, vocab_size), torch.float32),
        (target_probability, (batch_size * TARGET_ROWS, vocab_size), torch.float32),
        (drafts, (batch_size, 31), torch.long),
        (uniforms, (batch_size, WALK_CAP, 3), torch.float32),
    )
    for value, shape, dtype in expected:
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != device
            or not value.is_contiguous()
        ):
            raise RuntimeError("FR13 fixed32 fused decision input layout drift")
    if vocab_size <= 0 or vocab_size > MAX_VOCAB:
        raise RuntimeError(
            f"FR13 fixed32 fused decision vocab {vocab_size} exceeds {MAX_VOCAB}"
        )
    _validate_workspace(entry, device=device, batch_size=batch_size)

    self_total_rows = batch_size * SELF_ROWS
    target_total_rows = batch_size * TARGET_ROWS
    all_rows = self_total_rows + target_total_rows
    _fr13_cfwd_probability_block_sums_kernel[(all_rows, MAX_BLOCKS)](
        self_probability,
        target_probability,
        entry["cfwd_fused_probability_block_sums"],
        vocab_size,
        self_total_rows,
        SELF_ROWS=SELF_ROWS,
        TARGET_ROWS=TARGET_ROWS,
        BLOCK_V=BLOCK_V,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=4,
    )
    _fr13_cfwd_parent_setup_kernel[(target_total_rows,)](
        target_probability,
        entry["cfwd_fused_probability_block_sums"],
        drafts,
        entry["child_table"],
        entry["child_counts"],
        entry["all_parent_target_parent_slots"],
        entry["all_parent_target_uniform_levels"],
        uniforms,
        entry["cfwd_fused_target_totals"],
        entry["cfwd_fused_kid_tokens"],
        entry["cfwd_fused_q_weights"],
        entry["cfwd_fused_source"],
        entry["cfwd_fused_selected_token"],
        entry["cfwd_fused_accepted"],
        vocab_size,
        self_total_rows,
        SELF_ROWS=SELF_ROWS,
        TARGET_ROWS=TARGET_ROWS,
        PHYSICAL_DRAFTS=31,
        PHYSICAL_ROWS=32,
        FANOUT=FANOUT,
        WALK_CAP=WALK_CAP,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=8,
    )
    _fr13_cfwd_residual_block_sums_kernel[(target_total_rows, MAX_BLOCKS)](
        target_probability,
        entry["cfwd_fused_target_totals"],
        entry["cfwd_fused_kid_tokens"],
        entry["cfwd_fused_q_weights"],
        entry["cfwd_fused_residual_block_sums"],
        vocab_size,
        TARGET_ROWS=TARGET_ROWS,
        FANOUT=FANOUT,
        BLOCK_V=BLOCK_V,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=4,
    )
    _fr13_cfwd_inverse_cdf_kernel[(all_rows,)](
        self_probability,
        target_probability,
        entry["cfwd_fused_probability_block_sums"],
        entry["cfwd_fused_residual_block_sums"],
        entry["cfwd_fused_target_totals"],
        entry["cfwd_fused_kid_tokens"],
        entry["cfwd_fused_q_weights"],
        entry["all_parent_self_uniform_levels"],
        entry["all_parent_target_uniform_levels"],
        uniforms,
        entry["cfwd_fused_self_token"],
        entry["cfwd_fused_rejected_token"],
        vocab_size,
        self_total_rows,
        SELF_ROWS=SELF_ROWS,
        TARGET_ROWS=TARGET_ROWS,
        FANOUT=FANOUT,
        WALK_CAP=WALK_CAP,
        BLOCK_V=BLOCK_V,
        MAX_BLOCKS=MAX_BLOCKS,
        num_warps=8,
    )
    return (
        entry["cfwd_fused_self_token"],
        entry["cfwd_fused_source"],
        entry["cfwd_fused_selected_token"],
        entry["cfwd_fused_rejected_token"],
        entry["cfwd_fused_accepted"],
    )


def sparse_residual_oracle(
    target_probability,
    kid_tokens,
    kid_mask,
) -> tuple[Any, Any, Any, Any]:
    """CPU oracle for duplicate-aware q_mix and normalized residual rows."""
    if torch is None:
        raise RuntimeError("fixed32 fused decision oracle requires torch")
    normalized = target_probability / target_probability.sum(
        dim=-1, keepdim=True
    )
    overlaps = torch.gather(normalized, -1, kid_tokens) * kid_mask
    overlap_mass = overlaps.sum(dim=-1, keepdim=True)
    q_weights = overlaps / overlap_mass.clamp(min=1.0e-30)
    vocab = int(normalized.shape[-1])
    token_columns = torch.arange(vocab, device=normalized.device)
    q_mix = (
        (kid_tokens.unsqueeze(-1) == token_columns)
        * q_weights.unsqueeze(-1)
        * kid_mask.unsqueeze(-1)
    ).sum(dim=-2)
    residual = (normalized - q_mix).clamp(min=0.0)
    residual_mass = residual.sum(dim=-1, keepdim=True)
    residual = torch.where(
        residual_mass > 0.0,
        residual / residual_mass.clamp(min=1.0e-30),
        normalized,
    )
    zero_mass = kid_mask.any(dim=-1, keepdim=True) & (overlap_mass <= 0.0)
    residual = torch.where(zero_mass, normalized, residual)
    return normalized, overlaps, q_weights, residual
