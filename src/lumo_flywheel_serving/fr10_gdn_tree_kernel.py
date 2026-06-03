from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


NODE_FAMILIES = (2, 3, 6, 8, 14)
H = 48
K = 128
V = 128
BV = 16


@dataclass(frozen=True)
class Tree:
    parent: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.parent)

    def ancestors(self, node: int) -> tuple[int, ...]:
        out = []
        cur = self.parent[node]
        while cur >= 0:
            out.append(cur)
            cur = self.parent[cur]
        return tuple(reversed(out))

    def path(self, node: int) -> tuple[int, ...]:
        return (*self.ancestors(node), node)

    def is_single_spine(self) -> bool:
        return self.parent == tuple([-1, *range(0, self.n - 1)])

    def masks(self, device: torch.device, n_pad: int) -> tuple[torch.Tensor, torch.Tensor]:
        strict = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)
        visible = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)
        for i in range(self.n):
            visible[i, i] = 1
            for j in self.ancestors(i):
                strict[i, j] = 1
                visible[i, j] = 1
        return strict, visible


def make_tree(n: int) -> Tree:
    if n not in NODE_FAMILIES:
        raise ValueError(f"unwarmed FR10 tree family {n}; allowed={NODE_FAMILIES}")
    if n == 2:
        return Tree((-1, 0))
    if n == 3:
        return Tree((-1, 0, 0))
    if n == 6:
        return Tree((-1, 0, 1, 2, 2, 1))
    if n == 8:
        return Tree((-1, 0, 1, 2, 3, 3, 2, 1))
    return Tree((-1, 0, 1, 2, 3, 4, 4, 3, 2, 2, 1, 1, 0, 0))


def make_spine_tree(n: int) -> Tree:
    if n not in NODE_FAMILIES:
        raise ValueError(f"unwarmed FR10 tree family {n}; allowed={NODE_FAMILIES}")
    return Tree(tuple([-1, *range(0, n - 1)]))


def padded_nodes(n: int) -> int:
    n_pad = 1 << (n - 1).bit_length()
    if n_pad > 16:
        raise ValueError(f"FR10 tree kernel only warms padded node blocks up to 16, got {n}")
    return n_pad


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


@triton.jit
def _tree_gdn_kernel(
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
    NUM_H: tl.constexpr,
    DIM_K: tl.constexpr,
    DIM_V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    OUTPUT_SCALE: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_n = tl.arange(0, N_PAD)
    offs_k = tl.arange(0, DIM_K)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = offs_v < DIM_V

    b_g = tl.load(g + offs_n * NUM_H + pid_h).to(tl.float32)
    b_beta = tl.load(beta + offs_n * NUM_H + pid_h).to(tl.float32)
    b_q = tl.load(q + (offs_n[:, None] * NUM_H + pid_h) * DIM_K + offs_k[None, :]).to(
        tl.float32
    )
    b_k = tl.load(k + (offs_n[:, None] * NUM_H + pid_h) * DIM_K + offs_k[None, :]).to(
        tl.float32
    )
    b_v = tl.load(
        v + (offs_n[:, None] * NUM_H + pid_h) * DIM_V + offs_v[None, :],
        mask=v_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    b_h0 = tl.load(
        h0 + (pid_h * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
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
        solved_v = tl.where((offs_n == i)[:, None], y_i[None, :], solved_v)
        solved_k = tl.where((offs_n == i)[:, None], sk_i[None, :], solved_k)
        incoming_i = tl.sum(b_h0 * sk_i[None, :], axis=1)
        tv_i = y_i - incoming_i
        trans_v = tl.where((offs_n == i)[:, None], tv_i[None, :], trans_v)

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
            out + (i * NUM_H + pid_h) * DIM_V + offs_v,
            out_i,
            mask=v_mask & (i < N_ACTUAL),
        )
        tl.store(
            state + ((i * NUM_H + pid_h) * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
            state_i,
            mask=v_mask[:, None] & (i < N_ACTUAL),
        )


def launch_tree_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    tree: Tree,
    *,
    strict_mask: torch.Tensor | None = None,
    visible_mask: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
    output_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the FR10 dense tree verifier.

    For CUDA graph capture, pass preallocated masks, output, and state buffers.
    The allocation path is only for probes and offline validation.
    """
    n = tree.n
    n_pad = padded_nodes(n)
    if strict_mask is None or visible_mask is None:
        strict_mask, visible_mask = tree.masks(q.device, n_pad)
    if out is None:
        out = torch.empty((n_pad, H, V), device=q.device, dtype=q.dtype)
    if state is None:
        state = torch.empty((n_pad, H, V, K), device=q.device, dtype=torch.float32)
    grid = (H, triton.cdiv(V, BV))
    _tree_gdn_kernel[grid](
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
        N_ACTUAL=n,
        N_PAD=n_pad,
        NUM_H=H,
        DIM_K=K,
        DIM_V=V,
        BLOCK_V=BV,
        OUTPUT_SCALE=output_scale,
    )
    return out, state
