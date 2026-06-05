#!/usr/bin/env python3
"""FR12 corrected WY-tree Triton probe.

This probe keeps the FR10 tree launch shape but replaces the value-wise dense
forward substitution with an explicit tree-ancestry WY T factor:

    T = solve_tril(I + A)^{-1}
    A[i, j] = beta_i <k_i, k_j> exp(G_i - G_j), j strict ancestor of i

This is the same triangular factorization family used by vLLM FLA's chunk path,
with the flat lower triangle generalized to the tree ancestor mask.
"""

from __future__ import annotations

import json
import statistics

import torch
import triton
import triton.language as tl

KH, VH, DK, DV = 16, 48, 128, 128
BV = 16
SCALE = DK ** -0.5

PARENT_14 = (-1, 0, 1, 2, 3, 4, 4, 3, 2, 2, 1, 1, 0, 0)


def ancestors(parent: tuple[int, ...], node: int) -> tuple[int, ...]:
    out: list[int] = []
    cur = parent[node]
    while cur >= 0:
        out.append(cur)
        cur = parent[cur]
    return tuple(reversed(out))


def path_of(parent: tuple[int, ...], node: int) -> tuple[int, ...]:
    return ancestors(parent, node) + (node,)


def build_masks(parent: tuple[int, ...], n_pad: int, device: torch.device):
    n = len(parent)
    strict = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)
    visible = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)
    for i in range(n):
        visible[i, i] = 1
        for j in ancestors(parent, i):
            strict[i, j] = 1
            visible[i, j] = 1
    return strict, visible


@triton.jit
def _wy_tree_solve_kernel(
    q,
    k,
    v,
    g,
    beta,
    h0,
    strict_mask,
    visible_mask,
    out,
    state,
    N_ACTUAL: tl.constexpr,
    N_PAD: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
):
    pid_vh = tl.program_id(0)
    pid_vblk = tl.program_id(1)
    group: tl.constexpr = NUM_VH // NUM_KH
    pid_kh = pid_vh // group
    offs_n = tl.arange(0, N_PAD)
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_vblk * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    b_g = tl.load(g + offs_n * NUM_VH + pid_vh).to(tl.float32)
    b_beta = tl.load(beta + offs_n * NUM_VH + pid_vh).to(tl.float32)
    b_q = tl.load(
        q + (offs_n[:, None] * NUM_KH + pid_kh) * DIM_K + offs_k[None, :]
    ).to(tl.float32)
    b_k = tl.load(
        k + (offs_n[:, None] * NUM_KH + pid_kh) * DIM_K + offs_k[None, :]
    ).to(tl.float32)
    b_q *= tl.rsqrt(tl.sum(b_q * b_q, axis=1)[:, None] + 1e-6)
    b_k *= tl.rsqrt(tl.sum(b_k * b_k, axis=1)[:, None] + 1e-6)
    b_v = tl.load(
        v + (offs_n[:, None] * NUM_VH + pid_vh) * DIM_V + offs_v[None, :],
        mask=v_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    b_h0 = tl.load(
        h0 + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    m_strict = tl.load(strict_mask + offs_n[:, None] * N_PAD + offs_n[None, :]) != 0
    m_visible = tl.load(visible_mask + offs_n[:, None] * N_PAD + offs_n[None, :]) != 0
    cum_g = tl.sum(tl.where(m_visible, b_g[None, :], 0.0), axis=1)

    kk = tl.dot(b_k, tl.trans(b_k), input_precision="tf32")
    decay = tl.exp(cum_g[:, None] - cum_g[None, :])
    system = tl.where(m_strict, kk * b_beta[:, None] * decay, 0.0)

    # Tree-ancestry solve_tril factor: T = inv(I + system).
    tmat = tl.zeros((N_PAD, N_PAD), dtype=tl.float32)
    for i in tl.static_range(0, N_PAD):
        row_i = offs_n == i
        coeff = tl.sum(tl.where(row_i[:, None], system, 0.0), axis=0)
        t_i = tl.where(offs_n == i, 1.0, 0.0)
        for j in tl.static_range(0, i):
            row_j = offs_n == j
            coeff_j = tl.sum(tl.where(row_j, coeff, 0.0), axis=0)
            t_j = tl.sum(tl.where(row_j[:, None], tmat, 0.0), axis=0)
            t_i -= coeff_j * t_j
        tmat = tl.where(row_i[:, None], t_i[None, :], tmat)

    src_v = b_beta[:, None] * b_v
    src_k = b_beta[:, None] * b_k * tl.exp(cum_g)[:, None]
    solved_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)
    solved_k = tl.zeros((N_PAD, DIM_K), dtype=tl.float32)
    trans_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)

    for i in tl.static_range(0, N_PAD):
        row_i = offs_n == i
        t_i = tl.sum(tl.where(row_i[:, None], tmat, 0.0), axis=0)
        y_i = tl.sum(t_i[:, None] * src_v, axis=0)
        sk_i = tl.sum(t_i[:, None] * src_k, axis=0)
        incoming_i = tl.sum(b_h0 * sk_i[None, :], axis=1)
        solved_v = tl.where(row_i[:, None], y_i[None, :], solved_v)
        solved_k = tl.where(row_i[:, None], sk_i[None, :], solved_k)
        trans_v = tl.where(row_i[:, None], (y_i - incoming_i)[None, :], trans_v)

    for i in tl.static_range(0, N_PAD):
        row_i = offs_n == i
        cumg_i = tl.sum(tl.where(row_i, cum_g, 0.0), axis=0)
        q_i = tl.sum(tl.where(row_i[:, None], b_q, 0.0), axis=0)
        state_i = b_h0 * tl.exp(cumg_i)
        for j in tl.static_range(0, N_PAD):
            vis = tl.load(visible_mask + i * N_PAD + j) != 0
            row_j = offs_n == j
            trans_j = tl.sum(tl.where(row_j[:, None], trans_v, 0.0), axis=0)
            k_j = tl.sum(tl.where(row_j[:, None], b_k, 0.0), axis=0)
            cumg_j = tl.sum(tl.where(row_j, cum_g, 0.0), axis=0)
            state_i += tl.where(
                vis,
                trans_j[:, None] * k_j[None, :] * tl.exp(cumg_i - cumg_j),
                0.0,
            )
        out_i = tl.sum(state_i * q_i[None, :], axis=1) * OUTPUT_SCALE
        tl.store(
            out + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
            out_i,
            mask=v_mask & (i < N_ACTUAL),
        )
        tl.store(
            state
            + ((i * NUM_VH + pid_vh) * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            state_i,
            mask=v_mask[:, None] & (i < N_ACTUAL),
        )


@triton.jit
def _dense_tree_kernel(
    q,
    k,
    v,
    g,
    beta,
    h0,
    strict_mask,
    visible_mask,
    out,
    state,
    N_ACTUAL: tl.constexpr,
    N_PAD: tl.constexpr,
    NUM_KH: tl.constexpr,
    NUM_VH: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
):
    pid_vh = tl.program_id(0)
    pid_vblk = tl.program_id(1)
    group: tl.constexpr = NUM_VH // NUM_KH
    pid_kh = pid_vh // group
    offs_n = tl.arange(0, N_PAD)
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_vblk * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V
    b_g = tl.load(g + offs_n * NUM_VH + pid_vh).to(tl.float32)
    b_beta = tl.load(beta + offs_n * NUM_VH + pid_vh).to(tl.float32)
    b_q = tl.load(
        q + (offs_n[:, None] * NUM_KH + pid_kh) * DIM_K + offs_k[None, :]
    ).to(tl.float32)
    b_k = tl.load(
        k + (offs_n[:, None] * NUM_KH + pid_kh) * DIM_K + offs_k[None, :]
    ).to(tl.float32)
    b_q *= tl.rsqrt(tl.sum(b_q * b_q, axis=1)[:, None] + 1e-6)
    b_k *= tl.rsqrt(tl.sum(b_k * b_k, axis=1)[:, None] + 1e-6)
    b_v = tl.load(
        v + (offs_n[:, None] * NUM_VH + pid_vh) * DIM_V + offs_v[None, :],
        mask=v_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    b_h0 = tl.load(
        h0 + (pid_vh * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
        mask=v_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    m_strict = tl.load(strict_mask + offs_n[:, None] * N_PAD + offs_n[None, :]) != 0
    m_visible = tl.load(visible_mask + offs_n[:, None] * N_PAD + offs_n[None, :]) != 0
    cum_g = tl.sum(tl.where(m_visible, b_g[None, :], 0.0), axis=1)
    kk = tl.dot(b_k, tl.trans(b_k), input_precision="tf32")
    decay = tl.exp(cum_g[:, None] - cum_g[None, :])
    system = tl.where(m_strict, kk * b_beta[:, None] * decay, 0.0)
    solved_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)
    solved_k = tl.zeros((N_PAD, DIM_K), dtype=tl.float32)
    trans_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)
    for i in tl.static_range(0, N_PAD):
        row_i = offs_n == i
        coeff = tl.sum(tl.where(row_i[:, None], system, 0.0), axis=0)
        beta_i = tl.sum(tl.where(row_i, b_beta, 0.0), axis=0)
        cumg_i = tl.sum(tl.where(row_i, cum_g, 0.0), axis=0)
        v_i = tl.sum(tl.where(row_i[:, None], b_v, 0.0), axis=0)
        k_i = tl.sum(tl.where(row_i[:, None], b_k, 0.0), axis=0)
        y_i = beta_i * v_i
        sk_i = beta_i * k_i * tl.exp(cumg_i)
        for j in tl.static_range(0, i):
            row_j = offs_n == j
            coeff_j = tl.sum(tl.where(row_j, coeff, 0.0), axis=0)
            solved_v_j = tl.sum(tl.where(row_j[:, None], solved_v, 0.0), axis=0)
            solved_k_j = tl.sum(tl.where(row_j[:, None], solved_k, 0.0), axis=0)
            y_i -= coeff_j * solved_v_j
            sk_i -= coeff_j * solved_k_j
        solved_v = tl.where(row_i[:, None], y_i[None, :], solved_v)
        solved_k = tl.where(row_i[:, None], sk_i[None, :], solved_k)
        incoming_i = tl.sum(b_h0 * sk_i[None, :], axis=1)
        trans_v = tl.where(row_i[:, None], (y_i - incoming_i)[None, :], trans_v)
    for i in tl.static_range(0, N_PAD):
        row_i = offs_n == i
        cumg_i = tl.sum(tl.where(row_i, cum_g, 0.0), axis=0)
        q_i = tl.sum(tl.where(row_i[:, None], b_q, 0.0), axis=0)
        state_i = b_h0 * tl.exp(cumg_i)
        for j in tl.static_range(0, N_PAD):
            vis = tl.load(visible_mask + i * N_PAD + j) != 0
            row_j = offs_n == j
            trans_j = tl.sum(tl.where(row_j[:, None], trans_v, 0.0), axis=0)
            k_j = tl.sum(tl.where(row_j[:, None], b_k, 0.0), axis=0)
            cumg_j = tl.sum(tl.where(row_j, cum_g, 0.0), axis=0)
            state_i += tl.where(
                vis,
                trans_j[:, None] * k_j[None, :] * tl.exp(cumg_i - cumg_j),
                0.0,
            )
        out_i = tl.sum(state_i * q_i[None, :], axis=1) * OUTPUT_SCALE
        tl.store(
            out + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
            out_i,
            mask=v_mask & (i < N_ACTUAL),
        )
        tl.store(
            state
            + ((i * NUM_VH + pid_vh) * DIM_V + offs_v[:, None]) * DIM_K
            + offs_k[None, :],
            state_i,
            mask=v_mask[:, None] & (i < N_ACTUAL),
        )


def median_us(samples: list[float]) -> float:
    return float(statistics.median(samples))


def time_kernel(launch, iters: int = 200, repeats: int = 5) -> float:
    launch()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            launch()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / iters)
    return median_us(out)


def make_inputs(n_pad: int, seed: int, device: torch.device):
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((n_pad, KH, DK), device=device, generator=gen).to(torch.bfloat16)
    k = torch.randn((n_pad, KH, DK), device=device, generator=gen).to(torch.bfloat16)
    v = (torch.randn((n_pad, VH, DV), device=device, generator=gen) * 0.2).to(
        torch.bfloat16
    )
    g = (torch.randn((n_pad, VH), device=device, generator=gen) * 0.05 - 0.1).to(
        torch.bfloat16
    )
    beta = torch.sigmoid(torch.randn((n_pad, VH), device=device, generator=gen)).to(
        torch.bfloat16
    )
    h0 = (torch.randn((VH, DV, DK), device=device, generator=gen) * 0.05).contiguous()
    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "g": g.contiguous(),
        "beta": beta.contiguous(),
        "h0": h0,
    }


def serial_tree_reference(x, parent: tuple[int, ...], n: int):
    q = x["q"][:n].float()
    k = x["k"][:n].float()
    v = x["v"][:n].float()
    g = x["g"][:n].float()
    beta = x["beta"][:n].float()
    h0 = x["h0"].float()

    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)

    out = torch.empty((n, VH, DV), device=q.device, dtype=torch.float32)
    state = torch.empty((n, VH, DV, DK), device=q.device, dtype=torch.float32)
    group = VH // KH
    for i in range(n):
        path = path_of(parent, i)
        for hv in range(VH):
            kh = hv // group
            s = h0[hv].clone()
            for j in path:
                s = s * torch.exp(g[j, hv])
                kv = s @ k[j, kh]
                s = s + (beta[j, hv] * (v[j, hv] - kv))[:, None] * k[j, kh][None, :]
            state[i, hv] = s
            out[i, hv] = (s @ q[i, kh]) * SCALE
    return out, state


def main():
    dev = torch.device("cuda")
    parent = PARENT_14
    n = len(parent)
    n_pad = 16
    strict, visible = build_masks(parent, n_pad, dev)
    x = make_inputs(n_pad, 12345, dev)
    for key in ("q", "k", "v", "g", "beta"):
        x[key][n:] = 0

    out_w = torch.empty((n_pad, VH, DV), device=dev, dtype=torch.float32)
    state_w = torch.empty((n_pad, VH, DV, DK), device=dev, dtype=torch.float32)
    out_d = torch.empty((n_pad, VH, DV), device=dev, dtype=torch.float32)
    state_d = torch.empty((n_pad, VH, DV, DK), device=dev, dtype=torch.float32)
    grid = (VH, triton.cdiv(DV, BV))

    def launch_wy():
        _wy_tree_solve_kernel[grid](
            x["q"],
            x["k"],
            x["v"],
            x["g"],
            x["beta"],
            x["h0"],
            strict,
            visible,
            out_w,
            state_w,
            N_ACTUAL=n,
            N_PAD=n_pad,
            NUM_KH=KH,
            NUM_VH=VH,
            DIM_K=DK,
            DIM_V=DV,
            BLOCK_V=BV,
            OUTPUT_SCALE=SCALE,
        )

    def launch_dense():
        _dense_tree_kernel[grid](
            x["q"],
            x["k"],
            x["v"],
            x["g"],
            x["beta"],
            x["h0"],
            strict,
            visible,
            out_d,
            state_d,
            N_ACTUAL=n,
            N_PAD=n_pad,
            NUM_KH=KH,
            NUM_VH=VH,
            DIM_K=DK,
            DIM_V=DV,
            BLOCK_V=BV,
            OUTPUT_SCALE=SCALE,
        )

    launch_wy()
    launch_dense()
    torch.cuda.synchronize()
    ref_out, ref_state = serial_tree_reference(x, parent, n)

    result = {
        "device": torch.cuda.get_device_name(0),
        "tree_nodes": n,
        "n_pad": n_pad,
        "wy_solve_us": time_kernel(launch_wy),
        "dense_fused_us": time_kernel(launch_dense),
        "wy_vs_dense_out_maxabs": float((out_w[:n] - out_d[:n]).abs().max()),
        "wy_vs_dense_state_maxabs": float((state_w[:n] - state_d[:n]).abs().max()),
        "wy_vs_serial_out_maxabs": float((out_w[:n] - ref_out).abs().max()),
        "wy_vs_serial_state_maxabs": float((state_w[:n] - ref_state).abs().max()),
        "dense_vs_serial_out_maxabs": float((out_d[:n] - ref_out).abs().max()),
        "dense_vs_serial_state_maxabs": float((state_d[:n] - ref_state).abs().max()),
    }

    torch.cuda.empty_cache()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
