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

from fr10_real_dims_tree_vs_fla_cost import (
    BV,
    K,
    V,
    VH,
    make_inputs,
    make_tree,
    padded_nodes,
)


@dataclass(frozen=True)
class Stage:
    name: str
    mode: int


STAGES = (
    Stage("setup_load_norm_cumg", 0),
    Stage("dense_kkt_system", 1),
    Stage("dense_triangular_solve", 2),
    Stage("state_output_only", 3),
    Stage("full_dense_tree_kernel", 4),
)


@triton.jit
def _stage_profile_kernel(
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
    MODE: tl.constexpr,
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

    if MODE == 0:
        checksum = tl.sum(cum_g) + tl.sum(b_q) + tl.sum(b_k) + tl.sum(b_v) + tl.sum(b_h0)
        tl.store(out + (pid_vh * DIM_V + offs_v), checksum, mask=v_mask)
        return

    kk = tl.dot(b_k, tl.trans(b_k), input_precision="tf32")
    decay = tl.exp(cum_g[:, None] - cum_g[None, :])
    system = tl.where(m_strict, kk * b_beta[:, None] * decay, 0.0)

    if MODE == 1:
        checksum = tl.sum(kk) + tl.sum(system)
        tl.store(out + (pid_vh * DIM_V + offs_v), checksum, mask=v_mask)
        return

    solved_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)
    solved_k = tl.zeros((N_PAD, DIM_K), dtype=tl.float32)
    trans_v = tl.zeros((N_PAD, BLOCK_V), dtype=tl.float32)

    if MODE != 3:
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
    else:
        trans_v = b_beta[:, None] * b_v

    if MODE == 2:
        checksum = tl.sum(trans_v) + tl.sum(solved_k)
        tl.store(out + (pid_vh * DIM_V + offs_v), checksum, mask=v_mask)
        return

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


def median_us(samples: list[float]) -> float:
    return float(statistics.median(samples))


def ancestor_stats(tree: Any, n_pad: int) -> dict[str, int | float]:
    strict_pairs = 0
    visible_pairs = 0
    depths = []
    for i in range(tree.n):
        depth = len(tree.ancestors(i))
        depths.append(depth)
        strict_pairs += depth
        visible_pairs += depth + 1
    return {
        "actual_nodes": tree.n,
        "padded_nodes": n_pad,
        "strict_ancestor_pairs": strict_pairs,
        "visible_ancestor_pairs": visible_pairs,
        "dense_square_pairs": n_pad * n_pad,
        "dense_lower_pairs": n_pad * (n_pad - 1) // 2,
        "strict_vs_dense_lower_fraction": strict_pairs / (n_pad * (n_pad - 1) / 2),
        "visible_vs_dense_square_fraction": visible_pairs / (n_pad * n_pad),
        "max_depth": max(depths),
        "mean_depth": sum(depths) / len(depths),
    }


def bench_stage(stage: Stage, nodes: int, iters: int, repeats: int, capture: bool) -> dict[str, Any]:
    device = torch.device("cuda")
    tree = make_tree(nodes)
    n_pad = padded_nodes(nodes)
    strict, visible = tree.masks(device, n_pad)
    x = make_inputs(n_pad, 70_000 + nodes, device)
    for key in ("q", "k", "v", "g", "beta"):
        x[key][nodes:] = 0
    out = torch.empty((n_pad, VH, V), device=device, dtype=torch.bfloat16)
    state = torch.empty((n_pad, VH, V, K), device=device, dtype=torch.float32)
    grid = (VH, triton.cdiv(V, BV))
    scale = K**-0.5

    def launch() -> None:
        _stage_profile_kernel[grid](
            x["q"],
            x["k"],
            x["v"],
            x["g"],
            x["beta"],
            x["h0"],
            strict,
            visible,
            out,
            state,
            N_ACTUAL=nodes,
            N_PAD=n_pad,
            NUM_KH=16,
            NUM_VH=VH,
            DIM_K=K,
            DIM_V=V,
            BLOCK_V=BV,
            OUTPUT_SCALE=scale,
            MODE=stage.mode,
        )

    out.zero_()
    state.zero_()
    launch()
    torch.cuda.synchronize()
    eager_out = out.clone()
    eager_state = state.clone()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    eager_samples = []
    for _ in range(repeats):
        start.record()
        for _ in range(iters):
            launch()
        end.record()
        torch.cuda.synchronize()
        eager_samples.append(start.elapsed_time(end) * 1000.0 / iters)

    graph_samples = None
    graph_bit_exact = None
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        out.zero_()
        state.zero_()
        graph.replay()
        torch.cuda.synchronize()
        graph_bit_exact = bool(torch.equal(out, eager_out) and torch.equal(state, eager_state))
        graph_samples = []
        for _ in range(repeats):
            start.record()
            for _ in range(iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            graph_samples.append(start.elapsed_time(end) * 1000.0 / iters)

    return {
        "stage": stage.name,
        "mode": stage.mode,
        "eager_us": median_us(eager_samples),
        "eager_us_samples": eager_samples,
        "graph_us": None if graph_samples is None else median_us(graph_samples),
        "graph_us_samples": graph_samples,
        "graph_bit_exact": graph_bit_exact,
    }


def run(nodes: int, iters: int, repeats: int, capture: bool) -> dict[str, Any]:
    tree = make_tree(nodes)
    n_pad = padded_nodes(nodes)
    rows = [bench_stage(stage, nodes, iters, repeats, capture) for stage in STAGES]
    by_name = {row["stage"]: row for row in rows}
    time_key = "graph_us" if capture else "eager_us"
    setup = by_name["setup_load_norm_cumg"][time_key]
    kkt = by_name["dense_kkt_system"][time_key]
    solve = by_name["dense_triangular_solve"][time_key]
    state_only = by_name["state_output_only"][time_key]
    full = by_name["full_dense_tree_kernel"][time_key]
    return {
        "schema": "fr10.tree_kernel_stage_profile.v1",
        "device": torch.cuda.get_device_name(0),
        "nodes": nodes,
        "padded_nodes": n_pad,
        "timing": {"iters": iters, "repeats": repeats, "summary": "median_us", "time_key": time_key},
        "ancestor_stats": ancestor_stats(tree, n_pad),
        "rows": rows,
        "derived": {
            "setup_us": setup,
            "dense_kkt_increment_us": kkt - setup,
            "dense_solve_increment_us": solve - kkt,
            "dense_solve_variant_us": solve,
            "state_output_only_variant_us": state_only,
            "full_us": full,
            "solve_variant_fraction_of_full": solve / full,
            "state_output_only_fraction_of_full": state_only / full,
            "strict_ancestor_pairs_vs_dense_lower": ancestor_stats(tree, n_pad)["strict_vs_dense_lower_fraction"],
            "visible_pairs_vs_dense_square": ancestor_stats(tree, n_pad)["visible_vs_dense_square_fraction"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=14)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--capture", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    print(json.dumps(run(args.nodes, args.iters, args.repeats, args.capture), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
