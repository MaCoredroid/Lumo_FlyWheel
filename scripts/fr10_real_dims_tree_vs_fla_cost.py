#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl


NODE_FAMILIES = (2, 3, 6, 8, 14)
KH = 16
VH = 48
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
    if n == 2:
        return Tree((-1, 0))
    if n == 3:
        return Tree((-1, 0, 0))
    if n == 6:
        return Tree((-1, 0, 1, 2, 2, 1))
    if n == 8:
        return Tree((-1, 0, 1, 2, 3, 3, 2, 1))
    if n == 14:
        return Tree((-1, 0, 1, 2, 3, 4, 4, 3, 2, 2, 1, 1, 0, 0))
    raise ValueError(f"unwarmed FR10 tree family {n}; allowed={NODE_FAMILIES}")


def make_spine(n: int) -> Tree:
    if n not in NODE_FAMILIES:
        raise ValueError(f"unwarmed FR10 tree family {n}; allowed={NODE_FAMILIES}")
    return Tree(tuple([-1, *range(0, n - 1)]))


def padded_nodes(n: int) -> int:
    n_pad = 1 << (n - 1).bit_length()
    if n_pad > 16:
        raise ValueError(f"FR10 cost script only warms padded node blocks up to 16, got {n}")
    return n_pad


@triton.jit
def _tree_gdn_gqa_kernel(
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
    b_q = tl.load(q + (offs_n[:, None] * NUM_KH + pid_kh) * DIM_K + offs_k[None, :]).to(tl.float32)
    b_k = tl.load(k + (offs_n[:, None] * NUM_KH + pid_kh) * DIM_K + offs_k[None, :]).to(tl.float32)
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
            out + (i * NUM_VH + pid_vh) * DIM_V + offs_v,
            out_i,
            mask=v_mask & (i < N_ACTUAL),
        )
        tl.store(
            state + ((i * NUM_VH + pid_vh) * DIM_V + offs_v[:, None]) * DIM_K + offs_k[None, :],
            state_i,
            mask=v_mask[:, None] & (i < N_ACTUAL),
        )


def make_inputs(n_pad: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((n_pad, KH, K), device=device, generator=gen, dtype=torch.float32).to(torch.bfloat16).contiguous()
    k = torch.randn((n_pad, KH, K), device=device, generator=gen, dtype=torch.float32).to(torch.bfloat16).contiguous()
    v = (torch.randn((n_pad, VH, V), device=device, generator=gen, dtype=torch.float32) * 0.2).to(torch.bfloat16).contiguous()
    g = (torch.randn((n_pad, VH), device=device, generator=gen, dtype=torch.float32) * 0.05 - 0.1).to(torch.bfloat16).contiguous()
    beta = torch.sigmoid(torch.randn((n_pad, VH), device=device, generator=gen, dtype=torch.float32)).to(torch.bfloat16).contiguous()
    h0 = (torch.randn((VH, V, K), device=device, generator=gen, dtype=torch.float32) * 0.05).contiguous()
    return {"q": q, "k": k, "v": v, "g": g, "beta": beta, "h0": h0}


def _median_us(samples: list[float]) -> float:
    return float(statistics.median(samples))


def bench_tree(tree: Tree, capture: bool, seed: int, iters: int, repeats: int) -> dict[str, Any]:
    device = torch.device("cuda")
    n_pad = padded_nodes(tree.n)
    strict, visible = tree.masks(device, n_pad)
    x = make_inputs(n_pad, seed, device)
    for key in ("q", "k", "v", "g", "beta"):
        x[key][tree.n :] = 0
    out = torch.empty((n_pad, VH, V), device=device, dtype=torch.bfloat16)
    state = torch.empty((n_pad, VH, V, K), device=device, dtype=torch.float32)
    grid = (VH, triton.cdiv(V, BV))
    scale = K**-0.5

    _tree_gdn_gqa_kernel[grid](
        x["q"], x["k"], x["v"], x["g"], x["beta"], x["h0"], strict, visible, out, state,
        N_ACTUAL=tree.n, N_PAD=n_pad, NUM_KH=KH, NUM_VH=VH, DIM_K=K, DIM_V=V,
        BLOCK_V=BV, OUTPUT_SCALE=scale,
    )
    torch.cuda.synchronize()
    eager_out = out[: tree.n].clone()
    eager_state = state[: tree.n].clone()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    eager_samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            _tree_gdn_gqa_kernel[grid](
                x["q"], x["k"], x["v"], x["g"], x["beta"], x["h0"], strict, visible, out, state,
                N_ACTUAL=tree.n, N_PAD=n_pad, NUM_KH=KH, NUM_VH=VH, DIM_K=K, DIM_V=V,
                BLOCK_V=BV, OUTPUT_SCALE=scale,
            )
        end.record()
        torch.cuda.synchronize()
        eager_samples.append(start.elapsed_time(end) * 1000.0 / iters)
    eager_us = _median_us(eager_samples)
    graph_us = None
    graph_samples = None
    graph_bit_exact = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            _tree_gdn_gqa_kernel[grid](
                x["q"], x["k"], x["v"], x["g"], x["beta"], x["h0"], strict, visible, out, state,
                N_ACTUAL=tree.n, N_PAD=n_pad, NUM_KH=KH, NUM_VH=VH, DIM_K=K, DIM_V=V,
                BLOCK_V=BV, OUTPUT_SCALE=scale,
            )
        out.zero_()
        state.zero_()
        graph.replay()
        torch.cuda.synchronize()
        graph_bit_exact = bool(torch.equal(out[: tree.n], eager_out) and torch.equal(state[: tree.n], eager_state))
        graph_samples = []
        for _ in range(repeats):
            start.record()
            for _ in range(iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            graph_samples.append(start.elapsed_time(end) * 1000.0 / iters)
        graph_us = _median_us(graph_samples)
    state_bytes_per_node = VH * V * K * 4
    return {
        "nodes": tree.n,
        "padded_nodes": n_pad,
        "tree_eager_us": eager_us,
        "tree_eager_us_samples": eager_samples,
        "tree_graph_us": graph_us,
        "tree_graph_us_samples": graph_samples,
        "tree_graph_bit_exact": graph_bit_exact,
        "tree_state_read_bytes": tree.n * state_bytes_per_node,
        "tree_state_write_bytes": tree.n * state_bytes_per_node,
    }


def bench_fla_chunk(tokens: int, seed: int, iters: int, repeats: int) -> dict[str, Any]:
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule

    device = torch.device("cuda")
    x = make_inputs(tokens, seed, device)
    q = x["q"].unsqueeze(0)
    k = x["k"].unsqueeze(0)
    v = x["v"].unsqueeze(0)
    g = x["g"].unsqueeze(0)
    beta = x["beta"].unsqueeze(0)
    h0 = x["h0"].unsqueeze(0).contiguous()
    scale = K**-0.5
    out, final_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            out, final_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=h0,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    eager_us = _median_us(samples)
    return {"tokens": tokens, "fla_chunk_eager_us": eager_us, "fla_chunk_eager_us_samples": samples}


def run(nodes: list[int], capture: bool, iters: int, repeats: int) -> dict[str, Any]:
    rows = []
    for n in nodes:
        tree = make_tree(n)
        tree_row = bench_tree(tree, capture=capture, seed=50_000 + n, iters=iters, repeats=repeats)
        fla_row = bench_fla_chunk(n, seed=50_000 + n, iters=iters, repeats=repeats)
        row = {**tree_row, **fla_row}
        row["tree_vs_fla_us"] = row["tree_graph_us" if capture else "tree_eager_us"] - row["fla_chunk_eager_us"]
        rows.append(row)

    base = Tree((-1, 0, 1, 2, 3))
    marginal_rows = []
    for depth in (0, 1, 2):
        ext = Tree((*base.parent, depth))
        base_row = bench_tree(base, capture=capture, seed=60_000, iters=iters, repeats=repeats)
        ext_row = bench_tree(ext, capture=capture, seed=60_000, iters=iters, repeats=repeats)
        base_us = base_row["tree_graph_us" if capture else "tree_eager_us"]
        ext_us = ext_row["tree_graph_us" if capture else "tree_eager_us"]
        state_bytes = VH * V * K * 4
        marginal_rows.append(
            {
                "tree_shape": f"{base.n}->{ext.n} padded {base_row['padded_nodes']}->{ext_row['padded_nodes']}",
                "branch_depth": depth,
                "base_us": base_us,
                "extended_us": ext_us,
                "marginal_leaf_us": ext_us - base_us,
                "marginal_state_read_bytes": state_bytes,
                "marginal_state_write_bytes": state_bytes,
                "graph_bit_exact": ext_row["tree_graph_bit_exact"],
            }
        )
    return {
        "schema": "fr10.real_dims_tree_vs_fla_cost.v1",
        "device": torch.cuda.get_device_name(0),
        "dims": {"key_heads": KH, "value_heads": VH, "head_dim_k": K, "head_dim_v": V},
        "capture": capture,
        "timing": {"iters": iters, "repeats": repeats, "summary": "median_us"},
        "rows": rows,
        "marginal_leaf_rows": marginal_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs="*", default=list(NODE_FAMILIES))
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(json.dumps(run(args.nodes, args.capture, args.iters, args.repeats), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
