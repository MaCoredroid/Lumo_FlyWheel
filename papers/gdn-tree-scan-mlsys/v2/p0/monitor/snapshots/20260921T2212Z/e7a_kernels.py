#!/usr/bin/env python3
"""E7a device realizations of mechanisms B and C (LOCAL reimplementations, Triton).

  B-fs : ancestor-masked triangular system solved by FORWARD SUBSTITUTION (WY / TreeWY family)
  C-nm : the same system solved by the FINITE NEUMANN series  (Bole family)
  commit: COMPACT reconstruction S_a = P_a S0 + sum_{j<=a} (P_a/P_j) U_j k_j^T  (shared by B and C)

Both verifier kernels take the SAME operands as the production scan (bf16 q/k/v, raw gates a/b,
A_log, dt_bias, fp32 h0) and compute gates/l2norm in-kernel exactly like the native fused
sigmoid-gating recurrence (fp32; rsqrt l2norm; softplus threshold 20; no beta round-trip).
They store: out (any float dtype; production is bf16), U factors (fp32), cum_g (fp32).

Precision knobs (recorded in every manifest):
  DOT_PREC: "ieee" (true fp32 dots; default for the correctness arm) or "tf32" (tensor-core MMA,
            the Bole-paper-style fragment path; sensitivity arm).
  D_TERMS : number of Neumann terms; d = max strict-ancestor count makes the series exact (G^{d+1}=0).

Layout: q,k (B, N_PAD, KH, DK); v (B, N_PAD, VH, DV); a,b (B, N_PAD, VH); h0 (B, VH, DV, DK) or a
bank (rows, VH, DV, DK) addressed via h0_rows (B,). Padded node rows must be finite (zero-fill).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_tree_operands(
    q, k, v, a, b, A_log, dt_bias, h0,
    pid_b, pid_kh, pid_vh, offs_n, offs_k, offs_v, n_mask, v_mask,
    N_PAD: tl.constexpr, NUM_KH: tl.constexpr, NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr, DIM_V: tl.constexpr, OUTPUT_SCALE: tl.constexpr,
):
    """Load one program's operands and compute native-form gates/norms in fp32."""
    q_ptr = q + ((pid_b * N_PAD + offs_n[:, None]) * NUM_KH + pid_kh) * DIM_K + offs_k[None, :]
    k_ptr = k + ((pid_b * N_PAD + offs_n[:, None]) * NUM_KH + pid_kh) * DIM_K + offs_k[None, :]
    b_q = tl.load(q_ptr, mask=n_mask[:, None], other=0.0).to(tl.float32)
    b_k = tl.load(k_ptr, mask=n_mask[:, None], other=0.0).to(tl.float32)
    v_ptr = v + ((pid_b * N_PAD + offs_n[:, None]) * NUM_VH + pid_vh) * DIM_V + offs_v[None, :]
    b_v = tl.load(v_ptr, mask=n_mask[:, None] & v_mask[None, :], other=0.0).to(tl.float32)
    b_a = tl.load(a + (pid_b * N_PAD + offs_n) * NUM_VH + pid_vh, mask=n_mask, other=0.0).to(tl.float32)
    b_b = tl.load(b + (pid_b * N_PAD + offs_n) * NUM_VH + pid_vh, mask=n_mask, other=0.0).to(tl.float32)
    b_a_log = tl.load(A_log + pid_vh).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + pid_vh).to(tl.float32)
    # native gating (fused_sigmoid_gating: threshold 20, beta=1)
    x = b_a + b_dt_bias
    softplus_x = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    b_g = -tl.exp(b_a_log) * softplus_x
    b_g = tl.where(n_mask, b_g, 0.0)
    b_beta = tl.where(n_mask, tl.sigmoid(b_b), 0.0)
    # native l2norm (rsqrt form) + scale on q
    b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q, axis=1)[:, None] + 1e-6)
    b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, axis=1)[:, None] + 1e-6)
    b_q = b_q * OUTPUT_SCALE
    b_q = tl.where(n_mask[:, None], b_q, 0.0)
    b_k = tl.where(n_mask[:, None], b_k, 0.0)
    return b_q, b_k, b_v, b_g, b_beta


@triton.jit
def _tree_gdn_factor_kernel(
    q, k, v, a, b, A_log, dt_bias, h0, h0_rows,
    strict_mask, visible_mask,
    out, u_out, cumg_out,
    N_ACTUAL: tl.constexpr, N_PAD: tl.constexpr,
    NUM_KH: tl.constexpr, NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr, DIM_V: tl.constexpr, BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
    H0_BANK_STRIDE: tl.constexpr,
    SOLVER: tl.constexpr,      # 0 = forward substitution (B-fs), 1 = finite Neumann (C-nm)
    D_TERMS: tl.constexpr,     # Neumann terms (ignored for SOLVER=0)
    DOT_PREC: tl.constexpr,    # "ieee" | "tf32"
    STORE_STATE: tl.constexpr, # also materialize every node's state (diagnostic; NOT the compact route)
    state_out,
):
    pid_b = tl.program_id(0)
    pid_vh = tl.program_id(1)
    pid_v = tl.program_id(2)
    head_group: tl.constexpr = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_n = tl.arange(0, N_PAD)
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    n_mask = offs_n < N_ACTUAL
    v_mask = offs_v < DIM_V

    b_q, b_k, b_v, b_g, b_beta = _gdn_tree_operands(
        q, k, v, a, b, A_log, dt_bias, h0, pid_b, pid_kh, pid_vh, offs_n, offs_k, offs_v, n_mask, v_mask,
        N_PAD, NUM_KH, NUM_VH, DIM_K, DIM_V, OUTPUT_SCALE)

    h0_row = tl.load(h0_rows + pid_b).to(tl.int64)
    b_h0 = tl.load(
        h0 + h0_row * H0_BANK_STRIDE + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
        mask=v_mask[:, None], other=0.0,
    ).to(tl.float32)  # (BLOCK_V, DIM_K)

    m_strict = tl.load(strict_mask + offs_n[:, None] * N_PAD + offs_n[None, :]) != 0
    m_visible = tl.load(visible_mask + offs_n[:, None] * N_PAD + offs_n[None, :]) != 0
    # cumulative log-decay over root..i (inclusive); P_i = exp(cum_i)
    cum_g = tl.sum(tl.where(m_visible, b_g[None, :], 0.0), axis=1)
    decay = tl.where(m_visible, tl.exp(cum_g[:, None] - cum_g[None, :]), 0.0)  # P_i/P_j (expdiff form)
    P = tl.exp(cum_g)

    # G = beta_i (P_i/P_j) <k_i,k_j> on strict ancestors
    kk = tl.dot(b_k, tl.trans(b_k), input_precision=DOT_PREC)
    G = tl.where(m_strict, b_beta[:, None] * decay * kk, 0.0)
    # R = beta (v - P S0 k)   with (S0 k_i) restricted to this value tile
    s0k = tl.dot(b_k, tl.trans(b_h0), input_precision=DOT_PREC)  # (N_PAD, BLOCK_V)
    R = b_beta[:, None] * (b_v - P[:, None] * s0k)

    if SOLVER == 0:
        # forward substitution in topological order: U_i = R_i - sum_{j<i} G[i,j] U_j
        U = R
        for i in tl.static_range(1, N_PAD):
            g_row = tl.sum(tl.where((offs_n == i)[:, None], G, 0.0), axis=0)  # (N_PAD,) row i of G
            contrib = tl.sum(g_row[:, None] * U, axis=0)  # (BLOCK_V,)  uses rows j<i only (others x0)
            r_i = tl.sum(tl.where((offs_n == i)[:, None], R, 0.0), axis=0)
            U = tl.where((offs_n == i)[:, None], (r_i - contrib)[None, :], U)
    else:
        # finite Neumann: U = sum_{m=0}^{D} (-G)^m R, evaluated as Z <- -G Z, U += Z
        U = R
        Z = R
        for _m in tl.static_range(0, D_TERMS):
            Z = -tl.dot(G, Z, input_precision=DOT_PREC)
            U = U + Z

    # verifier output without forming states: o_i = P_i (S0 q_i) + sum_{j<=i} (P_i/P_j) <q_i,k_j> U_j
    s0q = tl.dot(b_q, tl.trans(b_h0), input_precision=DOT_PREC)  # (N_PAD, BLOCK_V)
    qk = tl.dot(b_q, tl.trans(b_k), input_precision=DOT_PREC)  # (N_PAD, N_PAD)
    Cm = decay * qk  # visible-masked by decay
    O = P[:, None] * s0q + tl.dot(Cm, U, input_precision=DOT_PREC)

    o_ptr = out + ((pid_b * N_PAD + offs_n[:, None]) * NUM_VH + pid_vh) * DIM_V + offs_v[None, :]
    tl.store(o_ptr, O.to(o_ptr.dtype.element_ty), mask=n_mask[:, None] & v_mask[None, :])
    u_ptr = u_out + ((pid_b * N_PAD + offs_n[:, None]) * NUM_VH + pid_vh) * DIM_V + offs_v[None, :]
    tl.store(u_ptr, U, mask=n_mask[:, None] & v_mask[None, :])
    tl.store(cumg_out + (pid_b * N_PAD + offs_n) * NUM_VH + pid_vh, cum_g, mask=n_mask & (pid_v == 0))

    if STORE_STATE:
        # diagnostic materialization of every node state (same formula as the compact commit)
        for i in tl.static_range(0, N_PAD):
            row_i = offs_n == i
            P_i = tl.sum(tl.where(row_i, P, 0.0), axis=0)
            dec_i = tl.sum(tl.where(row_i[:, None], decay, 0.0), axis=0)  # (N_PAD,) P_i/P_j on path
            W = dec_i[:, None] * U  # (N_PAD, BLOCK_V)
            S_i = P_i * b_h0 + tl.dot(tl.trans(W), b_k, input_precision=DOT_PREC)  # (BLOCK_V, DIM_K)
            s_ptr = state_out + (((pid_b * N_PAD + i) * NUM_VH + pid_vh) * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :]
            tl.store(s_ptr, S_i, mask=v_mask[:, None] & (i < N_ACTUAL))


@triton.jit
def _tree_gdn_compact_commit_kernel(
    k, u, cumg, visible_mask, h0, h0_rows, target_node, dst, dst_rows,
    N_PAD: tl.constexpr, NUM_KH: tl.constexpr, NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr, DIM_V: tl.constexpr, BLOCK_V: tl.constexpr,
    H0_BANK_STRIDE: tl.constexpr, DST_BANK_STRIDE: tl.constexpr,
    DOT_PREC: tl.constexpr,
):
    """S_a = P_a S0 + sum_{j <= a} (P_a/P_j) U_j k_j^T, using stored factors (U, cum_g) and the raw k
    ring (re-normalized in-kernel exactly as the verifier did). One program per (b, vh, v-tile)."""
    pid_b = tl.program_id(0)
    pid_vh = tl.program_id(1)
    pid_v = tl.program_id(2)
    head_group: tl.constexpr = NUM_VH // NUM_KH
    pid_kh = pid_vh // head_group
    offs_n = tl.arange(0, N_PAD)
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    a_node = tl.load(target_node + pid_b).to(tl.int64)
    on_path = tl.load(visible_mask + a_node * N_PAD + offs_n) != 0  # j <= a
    k_ptr = k + ((pid_b * N_PAD + offs_n[:, None]) * NUM_KH + pid_kh) * DIM_K + offs_k[None, :]
    b_k = tl.load(k_ptr, mask=on_path[:, None], other=0.0).to(tl.float32)
    b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k, axis=1)[:, None] + 1e-6)
    b_k = tl.where(on_path[:, None], b_k, 0.0)
    cum = tl.load(cumg + (pid_b * N_PAD + offs_n) * NUM_VH + pid_vh, mask=on_path, other=0.0)
    cum_a = tl.sum(tl.where(offs_n == a_node, cum, 0.0), axis=0)
    w = tl.where(on_path, tl.exp(cum_a - cum), 0.0)  # P_a/P_j
    P_a = tl.exp(cum_a)
    u_ptr = u + ((pid_b * N_PAD + offs_n[:, None]) * NUM_VH + pid_vh) * DIM_V + offs_v[None, :]
    b_u = tl.load(u_ptr, mask=on_path[:, None] & v_mask[None, :], other=0.0)
    W = w[:, None] * b_u  # (N_PAD, BLOCK_V)
    h0_row = tl.load(h0_rows + pid_b).to(tl.int64)
    b_h0 = tl.load(
        h0 + h0_row * H0_BANK_STRIDE + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
        mask=v_mask[:, None], other=0.0,
    ).to(tl.float32)
    S = P_a * b_h0 + tl.dot(tl.trans(W), b_k, input_precision=DOT_PREC)
    dst_row = tl.load(dst_rows + pid_b).to(tl.int64)
    tl.store(
        dst + dst_row * DST_BANK_STRIDE + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
        S, mask=v_mask[:, None],
    )


# ----------------------------------------------------------------------------- python launchers
def _check_pow2(n: int, name: str) -> None:
    if n < 16 or n & (n - 1):
        raise ValueError(f"{name} must be a power of two >= 16 (tl.dot tile constraint), got {n}")


def launch_factor_verify(
    *, q, k, v, a, b, A_log, dt_bias, h0_bank, h0_rows, strict_mask, visible_mask,
    n_actual: int, n_pad: int, output_scale: float, solver: str, d_terms: int | None = None,
    dot_prec: str = "ieee", block_v: int = 16, num_warps: int = 4, out_dtype=torch.bfloat16,
    store_state: bool = False, out=None, u_out=None, cumg_out=None, state_out=None,
):
    """Run B-fs (solver='fs') or C-nm (solver='neumann') on a batch of trees.
    q,k: (B,N_PAD,KH,DK) k in bf16/fp32 ; v: (B,N_PAD,VH,DV); a,b: (B,N_PAD,VH); h0_bank: (rows,VH,DV,DK) fp32.
    Returns (out, U, cum_g, state_or_None)."""
    B = q.shape[0]
    KH, DK = q.shape[2], q.shape[3]
    VH, DV = v.shape[2], v.shape[3]
    _check_pow2(n_pad, "n_pad")
    _check_pow2(block_v, "block_v")
    if not (q.shape[1] == k.shape[1] == v.shape[1] == a.shape[1] == b.shape[1] == n_pad):
        raise ValueError("all node-major operands must be padded to n_pad rows")
    if h0_bank.dtype != torch.float32 or h0_bank.ndim != 4:
        raise ValueError("h0_bank must be fp32 (rows, VH, DV, DK)")
    for t in (q, k, v, a, b, h0_bank, strict_mask, visible_mask):
        if not t.is_contiguous():
            raise ValueError("operands must be contiguous")
    dev = q.device
    if out is None:
        out = torch.zeros((B, n_pad, VH, DV), device=dev, dtype=out_dtype)
    if u_out is None:
        u_out = torch.zeros((B, n_pad, VH, DV), device=dev, dtype=torch.float32)
    if cumg_out is None:
        cumg_out = torch.zeros((B, n_pad, VH), device=dev, dtype=torch.float32)
    if store_state and state_out is None:
        state_out = torch.zeros((B, n_pad, VH, DV, DK), device=dev, dtype=torch.float32)
    if solver == "fs":
        solver_id, d = 0, 0
    elif solver == "neumann":
        if d_terms is None:
            raise ValueError("neumann needs d_terms (= max strict-ancestor depth for exactness)")
        solver_id, d = 1, int(d_terms)
    else:
        raise ValueError(solver)
    grid = (B, VH, triton.cdiv(DV, block_v))
    _tree_gdn_factor_kernel[grid](
        q, k, v, a, b, A_log, dt_bias, h0_bank, h0_rows, strict_mask, visible_mask,
        out, u_out, cumg_out,
        N_ACTUAL=n_actual, N_PAD=n_pad, NUM_KH=KH, NUM_VH=VH, DIM_K=DK, DIM_V=DV, BLOCK_V=block_v,
        OUTPUT_SCALE=output_scale, H0_BANK_STRIDE=h0_bank.stride(0),
        SOLVER=solver_id, D_TERMS=d, DOT_PREC=dot_prec, STORE_STATE=store_state,
        state_out=state_out if store_state else strict_mask,
        num_warps=num_warps,
    )
    return out, u_out, cumg_out, (state_out if store_state else None)


def launch_compact_commit(
    *, k, u, cumg, visible_mask, h0_bank, h0_rows, target_node, dst_bank, dst_rows,
    n_pad: int, dot_prec: str = "ieee", block_v: int = 16, num_warps: int = 4,
):
    """Commit the accepted leaf `target_node[b]` for each batch row into dst_bank[dst_rows[b]]."""
    B = k.shape[0]
    KH, DK = k.shape[2], k.shape[3]
    VH, DV = u.shape[2], u.shape[3]
    _check_pow2(n_pad, "n_pad")
    _check_pow2(block_v, "block_v")
    grid = (B, VH, triton.cdiv(DV, block_v))
    _tree_gdn_compact_commit_kernel[grid](
        k, u, cumg, visible_mask, h0_bank, h0_rows, target_node, dst_bank, dst_rows,
        N_PAD=n_pad, NUM_KH=KH, NUM_VH=VH, DIM_K=DK, DIM_V=DV, BLOCK_V=block_v,
        H0_BANK_STRIDE=h0_bank.stride(0), DST_BANK_STRIDE=dst_bank.stride(0), DOT_PREC=dot_prec,
        num_warps=num_warps,
    )
    return dst_bank
