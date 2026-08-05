#!/usr/bin/env python3
"""Default-off physical32 packed-walk kernel with validated node domains.

The credentialed CFWD v3 producer packs the emitted token, accepted child row,
and parent-event bit into one int64 value per physical row. Its metadata binding
already proves that root is row zero and every accepted child is a draft row in
``[1, 31]``. This candidate consumes that proof instead of clamping the current
node twice in each of the fixed 12 walk levels.

This module is deliberately not installed into the served CFWD overlay. It is
an executable source candidate for the next authenticated real-task byte gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU source/test environments
    triton = None
    tl = None


CANDIDATE = "fixed32_cfwd_packed_walk_node_trust_v1"
CANDIDATE_SCHEMA = "fr13.fixed32.cfwd_packed_walk.node_trust.v1"
BASE_CANDIDATE = "fixed32_cfwd_logit_direct_packed_physical_slots_v3"
BASE_CANDIDATE_SCHEMA = (
    "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3"
)
BASE_CANDIDATE_SOURCE_SHA256 = (
    "a7a7b6582cdc11e930916f5e65583195fd31a3b664e8f567bb33a24ea1a64ee0"
)
BASE_INTEGRATION_SOURCE_SCHEMA = (
    "fr13.fixed32.cfwd_logit_direct.integration_source.v2"
)
BASE_INTEGRATION_SOURCE_SHA256 = (
    "5c30860712e9766fd397b3e90e2ea203ad4ee2a89302d4a3c3c0e412452e4e07"
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


def packed_walk_node_trust_contract(mode: str) -> dict[str, object]:
    """Return the exact source and geometry proof required by the candidate."""
    if mode not in FIXED32_MODES:
        raise ValueError(f"unsupported fixed32 mode {mode!r}")
    return {
        "schema": CANDIDATE_SCHEMA,
        "candidate": CANDIDATE,
        "mode": mode,
        "base_candidate": BASE_CANDIDATE,
        "base_candidate_schema": BASE_CANDIDATE_SCHEMA,
        "base_candidate_source_sha256": BASE_CANDIDATE_SOURCE_SHA256,
        "base_integration_source_schema": BASE_INTEGRATION_SOURCE_SCHEMA,
        "base_integration_source_sha256": BASE_INTEGRATION_SOURCE_SHA256,
        "physical_drafts": PHYSICAL_DRAFTS,
        "physical_rows": PHYSICAL_ROWS,
        "walk_levels": WALK_CAP,
        "output_capacity": OUTPUT_CAPACITY,
        "path_capacity": PATH_CAPACITY,
        "loop_bound_topology_constant": True,
        "root_parent_slot": 0,
        "accepted_row_domain": [1, 31],
        "node_domain_clamps_per_request_before": 48,
        "node_domain_clamps_per_request_after": 0,
        "leaf_domain_comparisons_per_request_before": 24,
        "leaf_domain_comparisons_per_request_after": 0,
        "unconditional_self_token_loads_per_request_before": 12,
        "unconditional_self_token_loads_per_request_after": 0,
        "leaf_self_token_loads_per_request_after_max": 1,
        "bonus_token_loads_per_request_after": 0,
    }


def validate_packed_walk_producer_contract(
    producer: Mapping[str, object], *, mode: str
) -> None:
    """Fail closed unless the caller supplies the reviewed exact v3 producer."""
    expected = packed_walk_node_trust_contract(mode)
    required = {
        "candidate": expected["base_candidate"],
        "candidate_schema": expected["base_candidate_schema"],
        "candidate_source_sha256": expected["base_candidate_source_sha256"],
        "integration_source_schema": expected["base_integration_source_schema"],
        "integration_source_sha256": expected["base_integration_source_sha256"],
        "mode": mode,
        "physical_drafts": PHYSICAL_DRAFTS,
        "physical_rows": PHYSICAL_ROWS,
    }
    if not isinstance(producer, Mapping) or any(
        producer.get(name) != value for name, value in required.items()
    ):
        raise RuntimeError(
            "FR13 packed-walk node trust requires the exact reviewed "
            "physical32 CFWD v3 producer contract"
        )


if triton is not None:

    @triton.jit
    def _fr13_fixed32_taw_packed_node_trust_commit_kernel(
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
        """Walk producer-validated physical slots without repeated clamps."""
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
        for _level in tl.static_range(0, WALK_CAP):
            # Source-bound invariant: current is root (-1) or a draft in [0, 30].
            parent_slot = current + 1
            packed_event = tl.load(
                event + request * PHYSICAL_ROWS + parent_slot,
                mask=alive,
                other=0,
            ).to(tl.int64)
            has_kids = alive & ((packed_event & 0x800000) != 0)
            leaf = alive & ((packed_event & 0x800000) == 0)

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
            output_len += leaf.to(tl.int64) + has_kids.to(tl.int64)

            accepted_row = (packed_event >> 18) & 0x1F
            is_accepted = has_kids & (accepted_row != 0)
            accepted_node = accepted_row - 1
            tl.store(
                accepted_path_rows + request * PATH_CAPACITY + path_len,
                accepted_row,
                mask=is_accepted,
            )
            path_len += is_accepted.to(tl.int64)
            current = tl.where(is_accepted, accepted_node, current)
            alive = is_accepted
            final_row = tl.where(is_accepted, accepted_row, final_row)

        tl.store(output_lens + request, output_len)
        tl.store(accepted_lens + request, path_len)
        tl.store(last_row + request, final_row)


def packed_walk_node_trust_oracle(
    self_token: torch.Tensor,
    event: torch.Tensor,
    bonus_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU model of the trust-bound kernel with explicit domain assertions."""
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
        raise ValueError("packed-walk node-trust oracle operand contract drift")
    batch = int(self_token.shape[0])
    output = torch.full(
        (batch, OUTPUT_CAPACITY), -1, dtype=torch.long, device=self_token.device
    )
    output_lens = torch.zeros(batch, dtype=torch.long, device=self_token.device)
    paths = torch.zeros(
        (batch, PATH_CAPACITY), dtype=torch.long, device=self_token.device
    )
    path_lens = torch.zeros(batch, dtype=torch.long, device=self_token.device)
    last_rows = torch.zeros(batch, dtype=torch.long, device=self_token.device)
    for request in range(batch):
        current = -1
        alive = True
        for _level in range(WALK_CAP):
            if not alive:
                continue
            parent_slot = current + 1
            if not 0 <= parent_slot < PHYSICAL_ROWS:
                raise RuntimeError("trusted packed parent slot escaped physical32")
            packed = int(event[request, parent_slot])
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
    return output, output_lens, paths, path_lens, last_rows


def launch_packed_walk_node_trust(
    *,
    producer_contract: Mapping[str, object],
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
    """Launch the source candidate against persistent caller-owned workspace."""
    validate_packed_walk_producer_contract(producer_contract, mode=mode)
    if triton is None or tl is None:
        raise RuntimeError("FR13 packed-walk node trust requires Triton")
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
        raise RuntimeError("FR13 packed-walk node-trust tensor contract drift")
    if self_token.device.type != "cuda":
        raise RuntimeError("FR13 packed-walk node trust requires CUDA tensors")
    _fr13_fixed32_taw_packed_node_trust_commit_kernel[(batch,)](
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
