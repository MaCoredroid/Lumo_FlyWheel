#!/usr/bin/env python3
"""Default-off bounded active-depth packed CFWD committer.

The reviewed node-trust walk statically unrolls all 12 Hydra27 levels.  The
packed producer makes the next row and termination decision available in one
int64 load, so this candidate retains the exact 12-level bound but exits the
device loop as soon as the realized path rejects or reaches a leaf.

This module is deliberately not installed into serving.  It is bound to the
exact reviewed packed-v3 producer and node-trust source and requires a real
SWE-Verified byte gate before any production use.
"""

from __future__ import annotations

from collections.abc import Mapping
import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU source/test environments
    triton = None
    tl = None


CANDIDATE = "fixed32_cfwd_packed_walk_active_depth_v1"
CANDIDATE_SCHEMA = "fr13.fixed32.cfwd_packed_walk.active_depth.v1"
BASE_CANDIDATE = "fixed32_cfwd_packed_walk_node_trust_v1"
BASE_CANDIDATE_SCHEMA = "fr13.fixed32.cfwd_packed_walk.node_trust.v1"
BASE_CANDIDATE_SOURCE_SHA256 = (
    "07cd03173ab1a6e6b9aa597d9c912475034f5b8100c2c57d819b2b7bbcf3bc37"
)
PRODUCER_CANDIDATE = "fixed32_cfwd_logit_direct_packed_physical_slots_v3"
PRODUCER_SCHEMA = "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3"
PRODUCER_SOURCE_SHA256 = (
    "a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0"
)
FIXED32_MODES = frozenset(("tail6_fixed32", "hydra27_fixed32"))
PHYSICAL_DRAFTS = 31
PHYSICAL_ROWS = 32
WALK_CAP = 12
OUTPUT_CAPACITY = 32
PATH_CAPACITY = 16
PACKED_EVENT_TOKEN_MASK = 0x3FFFF
PACKED_EVENT_ACCEPTED_ROW_SHIFT = 18
PACKED_EVENT_ACCEPTED_ROW_MASK = 0x1F
PACKED_EVENT_PARENT_MASK = 0x800000


def active_depth_walk_contract(mode: str) -> dict[str, object]:
    """Return the exact source and physical32 bound for this candidate."""
    if mode not in FIXED32_MODES:
        raise ValueError(f"unsupported fixed32 mode {mode!r}")
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate": CANDIDATE,
        "mode": mode,
        "base_candidate": BASE_CANDIDATE,
        "base_candidate_schema": BASE_CANDIDATE_SCHEMA,
        "base_candidate_source_sha256": BASE_CANDIDATE_SOURCE_SHA256,
        "producer_candidate": PRODUCER_CANDIDATE,
        "producer_schema": PRODUCER_SCHEMA,
        "producer_source_sha256": PRODUCER_SOURCE_SHA256,
        "physical_drafts": PHYSICAL_DRAFTS,
        "physical_rows": PHYSICAL_ROWS,
        "maximum_walk_iterations": WALK_CAP,
        "walk_termination": "first_reject_or_leaf_or_fixed_cap",
        "topology_size_controls_loop_bound": False,
        "candidate_default_off": True,
    }


def validate_active_depth_base_contract(
    base: Mapping[str, object], *, mode: str
) -> None:
    """Fail closed unless the caller supplies the exact reviewed base."""
    expected = active_depth_walk_contract(mode)
    required = {
        "candidate": expected["base_candidate"],
        "candidate_schema": expected["base_candidate_schema"],
        "candidate_source_sha256": expected["base_candidate_source_sha256"],
        "producer_candidate": expected["producer_candidate"],
        "producer_schema": expected["producer_schema"],
        "producer_source_sha256": expected["producer_source_sha256"],
        "mode": mode,
        "physical_drafts": PHYSICAL_DRAFTS,
        "physical_rows": PHYSICAL_ROWS,
        "walk_levels": WALK_CAP,
    }
    if not isinstance(base, Mapping) or any(
        base.get(name) != value for name, value in required.items()
    ):
        raise RuntimeError(
            "FR13 active-depth packed walk requires the exact reviewed "
            "node-trust and packed-v3 source contracts"
        )


if triton is not None:

    @triton.jit
    def _fr13_fixed32_taw_packed_active_depth_commit_kernel(
        self_token,
        event,
        bonus_token,
        output_tokens,
        output_lens,
        accepted_path_rows,
        accepted_lens,
        last_row,
        PHYSICAL_DRAFTS: tl.constexpr,
        PHYSICAL_ROWS: tl.constexpr,
        WALK_CAP: tl.constexpr,
        OUTPUT_CAPACITY: tl.constexpr,
        PATH_CAPACITY: tl.constexpr,
    ):
        """Walk only the realized packed path, with a fixed hard cap."""
        request = tl.program_id(0)
        output_columns = tl.arange(0, OUTPUT_CAPACITY)
        path_columns = tl.arange(0, PATH_CAPACITY)
        tl.store(
            output_tokens + request * OUTPUT_CAPACITY + output_columns,
            -1,
        )
        tl.store(
            accepted_path_rows + request * PATH_CAPACITY + path_columns,
            0,
        )

        current = -1
        alive = True
        output_len = 0
        path_len = 0
        final_row = 0
        level = 0
        while alive & (level < WALK_CAP):
            # Bound source proves root row 0 and accepted rows in [1, 31].
            parent_slot = current + 1
            packed_event = tl.load(
                event + request * PHYSICAL_ROWS + parent_slot
            ).to(tl.int64)
            has_kids = (packed_event & 0x800000) != 0
            leaf = ~has_kids

            # Exact fixed32 root always has children, so leaf proves current >= 0.
            sampled_self = tl.load(
                self_token + request * PHYSICAL_DRAFTS + current,
                mask=leaf,
                other=0,
            ).to(tl.int64)
            tl.store(
                output_tokens + request * OUTPUT_CAPACITY + output_len,
                sampled_self,
                mask=leaf,
            )

            emitted_token = packed_event & 0x3FFFF
            tl.store(
                output_tokens + request * OUTPUT_CAPACITY + output_len,
                emitted_token,
                mask=has_kids,
            )
            output_len += 1

            accepted_row = (packed_event >> 18) & 0x1F
            is_accepted = has_kids & (accepted_row != 0)
            tl.store(
                accepted_path_rows + request * PATH_CAPACITY + path_len,
                accepted_row,
                mask=is_accepted,
            )
            path_len += is_accepted.to(tl.int64)
            current = tl.where(is_accepted, accepted_row - 1, current)
            alive = is_accepted
            final_row = tl.where(is_accepted, accepted_row, final_row)
            level += 1

        tl.store(output_lens + request, output_len)
        tl.store(accepted_lens + request, path_len)
        tl.store(last_row + request, final_row)


def active_depth_packed_walk_oracle(
    self_token: torch.Tensor,
    event: torch.Tensor,
    bonus_token: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """CPU model returning the five products plus executed iterations."""
    if (
        self_token.dtype != torch.long
        or event.dtype != torch.long
        or bonus_token.dtype != torch.long
        or self_token.ndim != 2
        or event.ndim != 2
        or bonus_token.ndim != 1
        or int(self_token.shape[1]) != PHYSICAL_DRAFTS
        or int(event.shape[1]) != PHYSICAL_ROWS
        or int(event.shape[0]) != int(self_token.shape[0])
        or int(bonus_token.shape[0]) != int(self_token.shape[0])
    ):
        raise ValueError("active-depth packed-walk oracle operand contract drift")
    batch = int(self_token.shape[0])
    if batch not in (1, 4):
        raise ValueError("active-depth packed walk is B1/B4 only")
    output = torch.full(
        (batch, OUTPUT_CAPACITY), -1, dtype=torch.long, device=self_token.device
    )
    output_lens = torch.zeros(batch, dtype=torch.long, device=self_token.device)
    paths = torch.zeros(
        (batch, PATH_CAPACITY), dtype=torch.long, device=self_token.device
    )
    path_lens = torch.zeros(batch, dtype=torch.long, device=self_token.device)
    last_rows = torch.zeros(batch, dtype=torch.long, device=self_token.device)
    iterations = torch.zeros(batch, dtype=torch.long, device=self_token.device)
    for request in range(batch):
        current = -1
        alive = True
        while alive and int(iterations[request]) < WALK_CAP:
            parent_slot = current + 1
            if not 0 <= parent_slot < PHYSICAL_ROWS:
                raise RuntimeError("trusted packed parent slot escaped physical32")
            packed = int(event[request, parent_slot])
            iterations[request] += 1
            has_kids = bool(packed & PACKED_EVENT_PARENT_MASK)
            if not has_kids:
                if current < 0:
                    raise RuntimeError("exact fixed32 root cannot be a leaf")
                output[request, output_lens[request]] = self_token[
                    request, current
                ]
                output_lens[request] += 1
                alive = False
                continue
            output[request, output_lens[request]] = (
                packed & PACKED_EVENT_TOKEN_MASK
            )
            output_lens[request] += 1
            accepted_row = (
                packed >> PACKED_EVENT_ACCEPTED_ROW_SHIFT
            ) & PACKED_EVENT_ACCEPTED_ROW_MASK
            if accepted_row == 0:
                alive = False
                continue
            if not 1 <= accepted_row <= PHYSICAL_DRAFTS:
                raise RuntimeError("trusted packed accepted row escaped physical32")
            paths[request, path_lens[request]] = accepted_row
            path_lens[request] += 1
            last_rows[request] = accepted_row
            current = accepted_row - 1
    return output, output_lens, paths, path_lens, last_rows, iterations


def launch_active_depth_packed_walk(
    *,
    base_contract: Mapping[str, object],
    mode: str,
    self_token: torch.Tensor,
    event: torch.Tensor,
    bonus_token: torch.Tensor,
    output_tokens: torch.Tensor,
    output_lens: torch.Tensor,
    accepted_path_rows: torch.Tensor,
    accepted_lens: torch.Tensor,
    last_row: torch.Tensor,
) -> None:
    """Launch the default-off candidate against persistent output buffers."""
    validate_active_depth_base_contract(base_contract, mode=mode)
    if triton is None or tl is None:
        raise RuntimeError("FR13 active-depth packed walk requires Triton")
    batch = int(self_token.shape[0])
    expected = (
        (self_token, (batch, PHYSICAL_DRAFTS)),
        (event, (batch, PHYSICAL_ROWS)),
        (bonus_token, (batch,)),
        (output_tokens, (batch, OUTPUT_CAPACITY)),
        (output_lens, (batch,)),
        (accepted_path_rows, (batch, PATH_CAPACITY)),
        (accepted_lens, (batch,)),
        (last_row, (batch,)),
    )
    if batch not in (1, 4) or any(
        not torch.is_tensor(value)
        or tuple(value.shape) != shape
        or value.dtype != torch.long
        or value.device != self_token.device
        or not value.is_contiguous()
        for value, shape in expected
    ):
        raise RuntimeError("FR13 active-depth packed-walk tensor contract drift")
    if self_token.device.type != "cuda":
        raise RuntimeError("FR13 active-depth packed walk requires CUDA tensors")
    _fr13_fixed32_taw_packed_active_depth_commit_kernel[(batch,)](
        self_token,
        event,
        bonus_token,
        output_tokens,
        output_lens,
        accepted_path_rows,
        accepted_lens,
        last_row,
        PHYSICAL_DRAFTS=PHYSICAL_DRAFTS,
        PHYSICAL_ROWS=PHYSICAL_ROWS,
        WALK_CAP=WALK_CAP,
        OUTPUT_CAPACITY=OUTPUT_CAPACITY,
        PATH_CAPACITY=PATH_CAPACITY,
        num_warps=1,
    )
