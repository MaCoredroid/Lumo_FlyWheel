#!/usr/bin/env python3
"""Native packed-decode reference arm for the FR13 GDN scan A/B gate.

This builds a per-node reference for the tree-GDN scan by running the GDN
recurrent update with the SAME Triton kernel that the live no-spec B=1 decode
oracle dispatches, confirmed from the live dispatch:

    vllm/model_executor/layers/mamba/gdn_linear_attn.py
      _forward_core (L845-855): when
        enable_packed_recurrent_decode (env VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE,
        default True) AND spec_sequence_masks is None AND num_prefills == 0
        AND num_decodes > 0
      => _forward_core_decode_non_spec (L1261)
      => fused_recurrent_gated_delta_rule_packed_decode  (L1295)
         (vllm/model_executor/layers/fla/ops/fused_recurrent.py:338, kernel L256)

This is bug-class #10 (codegen identity / shared-source != shared-SASS): the
deployed tree scan was previously only A/B'd vs native_update_serial_per_path,
a SERIAL TORCH ref using fused_sigmoid_gating_delta_rule_update -- NOT the
packed-decode SASS the decode oracle actually runs. This module supplies the
RIGHT reference.

The packed-decode kernel is a single-step decoder (one token per sequence). To
build the per-node "native-on-path-to-root" branch oracle
(reference_gdn_tree_branch_oracle_losslessness: the target run on the branch's
path-to-root), we run the kernel SEQUENTIALLY along each node's path, feeding
the recurrent state forward, and take the output/state at the final step. The
spine path is a special case where every node's path is a prefix of the spine.

The kernel internals (read from source, fused_recurrent.py L256-335):
  * mixed_qkv is PACKED [B, q(H*K) | k(H*K) | v(HV*V)], head-major, contiguous.
  * q/k get l2norm-in-kernel (USE_QK_L2NORM_IN_KERNEL) then q *= scale.
  * a/b are RAW [B, HV]; the kernel computes softplus(a + dt_bias),
    g = -exp(A_log)*softplus, beta = sigmoid(b) internally.
  * initial_state [num_slots, HV, V, K]; state is updated IN PLACE (h0 == ht).
  * out shape [B, 1, HV, V].
"""

from __future__ import annotations

from typing import Any

import torch


def build_packed_mixed_qkv(
    query_spec: torch.Tensor,
    key_spec: torch.Tensor,
    value_spec: torch.Tensor,
) -> torch.Tensor:
    """Pack (q, k, v) rows into the native packed-decode mixed_qkv layout.

    query_spec/key_spec: [B, H, K] (pre-l2norm post-conv q/k)
    value_spec:          [B, HV, V]
    returns:             [B, H*K + H*K + HV*V], contiguous, head-major.
    """
    if query_spec.ndim != 3 or key_spec.ndim != 3 or value_spec.ndim != 3:
        raise ValueError(
            "query/key/value_spec must be 3D [B,H,K]/[B,HV,V]; got "
            f"{tuple(query_spec.shape)}/{tuple(key_spec.shape)}/{tuple(value_spec.shape)}"
        )
    b = query_spec.shape[0]
    if key_spec.shape[0] != b or value_spec.shape[0] != b:
        raise ValueError("q/k/v batch sizes differ")
    q_flat = query_spec.reshape(b, -1)
    k_flat = key_spec.reshape(b, -1)
    v_flat = value_spec.reshape(b, -1)
    return torch.cat([q_flat, k_flat, v_flat], dim=1).contiguous()


def native_packed_decode_per_path(
    tree: Any,
    query_spec: torch.Tensor,
    key_spec: torch.Tensor,
    value_spec: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    h0: torch.Tensor,
    scale: float,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node reference using the native packed-decode SASS along each path.

    For each node, walk root..node and run the packed-decode kernel one token
    at a time (B=1), feeding state forward. The output/state at the final step
    is that node's reference. This reproduces the native-on-path-to-root branch
    oracle for off-spine nodes and the native sequential decode for the spine.

    tree.path(node) must return the ordered list of node indices root..node.
    All tensors must be on CUDA. Returns (outputs[n, HV, V], states[n, HV, V, K]).
    """
    from vllm.model_executor.layers.fla.ops import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )

    device = query_spec.device
    hv = value_spec.shape[1]
    v_dim = value_spec.shape[2]
    k_dim = query_spec.shape[2]

    A_log = A_log.contiguous()
    dt_bias = dt_bias.contiguous()

    outputs: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    # CRITICAL (read from the pinned-image kernel, fused_recurrent.py L292-296):
    #   state_idx = tl.load(ssm_state_indices + ...)
    #   if state_idx <= 0:   # NULL_BLOCK_ID == 0
    #       store ZEROS to out; RETURN  (state is NOT updated)
    # The prior ref passed ssm_state_indices = zeros(1) -> state_idx == 0 -> the
    # kernel SHORT-CIRCUITED: it wrote zeros to `out` (the all-zeros-output bug)
    # AND never reached `tl.store(p_ht, b_h)` (the state stayed at the un-updated
    # h0). Both the durable STATE and the output were vacuous. The live decode
    # path (gdn_linear_attn._forward_core_decode_non_spec L1085) passes the REAL
    # cache slot indices (non_spec_state_indices_tensor, all >= 1), indexing into
    # the multi-slot ssm_state cache `initial_state=ssm_state` [num_slots,HV,V,K].
    # We reproduce that exactly: a >=2-slot buffer with the real h0 at SLOT 1 and
    # ssm_state_indices=[1]. The kernel reads slot 1, runs the rank-1 update, and
    # writes the UPDATED state back to slot 1 in place (h0 == ht). We extract that
    # durable STATE for the A/B (bug-class #10: A/B the incumbent decode SASS, not
    # a serial ref; bug-class #9: the comparand is now the live STATE, not zeros).
    state_slot = 1
    ssm_state_indices = torch.full(
        (1,), state_slot, dtype=torch.int32, device=device
    )

    for node in range(tree.n):
        path = list(tree.path(node))
        # Per-path fresh initial state. Native ssm_state cache is fp32 and 4D
        # [num_slots, HV, V, K]; slot 0 is NULL_BLOCK_ID (never used), slot 1
        # carries the real h0 and is updated IN PLACE by the kernel along the
        # path. zeros() for slot 0 so a stray slot-0 read would surface loudly.
        state = h0.new_zeros((state_slot + 1, *h0.shape), dtype=torch.float32)
        state[state_slot].copy_(h0.to(torch.float32))
        state = state.contiguous()
        out_buf = torch.empty(
            (1, 1, hv, v_dim), device=device, dtype=query_spec.dtype
        )
        for tok in path:
            mixed_qkv = build_packed_mixed_qkv(
                query_spec[tok : tok + 1],
                key_spec[tok : tok + 1],
                value_spec[tok : tok + 1],
            )
            a_row = a[tok : tok + 1].contiguous()
            b_row = b[tok : tok + 1].contiguous()
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv,
                a=a_row,
                b=b_row,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=state,  # slot 1 updated IN PLACE -> next h0
                out=out_buf,
                ssm_state_indices=ssm_state_indices,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        # out_buf holds the output of the LAST token on this path; state[slot]
        # holds the durable STATE after the last token (the A/B comparand).
        outputs.append(out_buf[0, 0].clone())
        states.append(state[state_slot].clone())  # [HV, V, K]

    return torch.stack(outputs, dim=0), torch.stack(states, dim=0)
