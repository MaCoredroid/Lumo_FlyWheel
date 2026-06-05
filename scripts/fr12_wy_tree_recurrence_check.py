#!/usr/bin/env python3
"""Validate FR12 tree-ancestry compact-WY recurrence against serial GDN.

This is the pre-kernel algebra gate.  It uses FR10's tree descriptor generator
and mirrors the per-path GDN recurrence in float64 so the target tolerance is
the algebraic floor, not vLLM's fp32 CPU implementation floor.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from fr10_gdn_tree_algebra_reference import make_tree  # noqa: E402

DTYPE = torch.float64
DK = 128
DV = 128
SCALE = DK**-0.5


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def _serial_path(
    path: tuple[int, ...],
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    state0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = state0.clone()
    out = None
    for node_id in path:
        gt = math.exp(float(g[node_id]))
        state = state * gt
        kv = state @ k[node_id]
        delta = (v[node_id] - kv) * beta[node_id]
        state = state + torch.outer(delta, k[node_id])
        out = state @ q[node_id]
    assert out is not None
    return out, state


def _wy_t(k_path: torch.Tensor, beta_path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build K,T where prod(I - beta k kT) = I - K T KT."""
    n = k_path.shape[0]
    basis = k_path.transpose(0, 1).contiguous()
    tri = torch.zeros((n, n), dtype=DTYPE)
    for j in range(n):
        bj = beta_path[j]
        if j:
            w = basis[:, :j].transpose(0, 1) @ k_path[j]
            u = tri[:j, :j] @ w
            tri[:j, j] = -bj * u
        tri[j, j] = bj
    return basis, tri


def _gated_wy_path(
    path: tuple[int, ...],
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    g_path = g[list(path)]
    cum_g = torch.cumsum(g_path, dim=0)
    k_path = k[list(path)]
    beta_path = beta[list(path)]
    ktil = k_path * torch.exp(-cum_g).unsqueeze(-1)
    betap = beta_path * torch.exp(2.0 * cum_g)
    basis, tri = _wy_t(ktil, betap)
    return basis, tri, float(cum_g[-1])


def _path_operator(
    path: tuple[int, ...],
    k: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
) -> torch.Tensor:
    if not path:
        return torch.eye(DK, dtype=DTYPE)
    basis, tri, cum_g = _gated_wy_path(path, k, beta, g)
    return math.exp(cum_g) * (torch.eye(DK, dtype=DTYPE) - basis @ tri @ basis.T)


def _tree_append_wy(tree, k: torch.Tensor, beta: torch.Tensor, g: torch.Tensor):
    """Parent-inherit + append one reflector per node."""
    basis_by_node: dict[int, torch.Tensor] = {}
    tri_by_node: dict[int, torch.Tensor] = {}
    cum_g_by_node: dict[int, float] = {}
    max_append_vs_rebuild_t = 0.0
    max_append_vs_rebuild_p = 0.0

    eye = torch.eye(DK, dtype=DTYPE)
    for node in tree.nodes:
        node_id = node.node_id
        parent_id = node.parent_id
        parent_g = 0.0 if parent_id < 0 else cum_g_by_node[parent_id]
        cum_g = parent_g + float(g[node_id])
        cum_g_by_node[node_id] = cum_g
        ktil = k[node_id] * math.exp(-cum_g)
        betap = float(beta[node_id]) * math.exp(2.0 * cum_g)

        if parent_id < 0:
            basis = ktil.unsqueeze(1)
            tri = torch.tensor([[betap]], dtype=DTYPE)
        else:
            parent_basis = basis_by_node[parent_id]
            parent_tri = tri_by_node[parent_id]
            width = parent_basis.shape[1]
            w = parent_basis.T @ ktil
            u = parent_tri @ w
            tri = torch.zeros((width + 1, width + 1), dtype=DTYPE)
            tri[:width, :width] = parent_tri
            tri[:width, width] = -betap * u
            tri[width, width] = betap
            basis = torch.cat((parent_basis, ktil.unsqueeze(1)), dim=1)

        basis_by_node[node_id] = basis
        tri_by_node[node_id] = tri

        path = tree.path_to(node_id)
        rebuild_basis, rebuild_tri, rebuild_g = _gated_wy_path(path, k, beta, g)
        max_append_vs_rebuild_t = max(
            max_append_vs_rebuild_t,
            float((tri - rebuild_tri).abs().max()),
            float((basis - rebuild_basis).abs().max()),
            abs(cum_g - rebuild_g),
        )
        p_append = math.exp(cum_g) * (eye - basis @ tri @ basis.T)
        p_rebuild = math.exp(rebuild_g) * (eye - rebuild_basis @ rebuild_tri @ rebuild_basis.T)
        max_append_vs_rebuild_p = max(
            max_append_vs_rebuild_p,
            float((p_append - p_rebuild).abs().max()),
        )

    return basis_by_node, tri_by_node, cum_g_by_node, max_append_vs_rebuild_t, max_append_vs_rebuild_p


def _wy_state_for_node(
    tree,
    node_id: int,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    state0: torch.Tensor,
) -> torch.Tensor:
    path = tree.path_to(node_id)
    state = state0 @ _path_operator(path, k, beta, g)
    for pos, src_node in enumerate(path):
        suffix = path[pos + 1 :]
        source = torch.outer(beta[src_node] * v[src_node], k[src_node])
        state = state + source @ _path_operator(suffix, k, beta, g)
    return state


def run_case(total_nodes: int, spine_depth: int, seed: int) -> dict[str, Any]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tree = make_tree(total_nodes=total_nodes, spine_depth=spine_depth, seed=seed + 1000)
    q = _normalize(torch.randn((total_nodes, DK), dtype=DTYPE, generator=gen)) * SCALE
    k = _normalize(torch.randn((total_nodes, DK), dtype=DTYPE, generator=gen))
    v = torch.randn((total_nodes, DV), dtype=DTYPE, generator=gen) * 0.2
    beta = torch.sigmoid(torch.randn((total_nodes,), dtype=DTYPE, generator=gen))
    g = -(torch.rand((total_nodes,), dtype=DTYPE, generator=gen) * 0.12)
    state0 = torch.randn((DV, DK), dtype=DTYPE, generator=gen) * 0.05

    basis_by_node, tri_by_node, cum_g_by_node, max_t, max_p = _tree_append_wy(
        tree, k, beta, g
    )
    eye = torch.eye(DK, dtype=DTYPE)
    max_s0map = 0.0
    max_state = 0.0
    max_out = 0.0
    for node in tree.nodes:
        path = tree.path_to(node.node_id)
        serial_out, serial_state = _serial_path(path, q, k, v, beta, g, state0)

        basis = basis_by_node[node.node_id]
        tri = tri_by_node[node.node_id]
        cum_g = cum_g_by_node[node.node_id]
        p_node = math.exp(cum_g) * (eye - basis @ tri @ basis.T)

        s0_only = state0.clone()
        for path_node in path:
            gt = math.exp(float(g[path_node]))
            s0_only = s0_only * gt
            kv = s0_only @ k[path_node]
            s0_only = s0_only - torch.outer(beta[path_node] * kv, k[path_node])
        max_s0map = max(max_s0map, float((state0 @ p_node - s0_only).abs().max()))

        wy_state = _wy_state_for_node(tree, node.node_id, k, v, beta, g, state0)
        wy_out = wy_state @ q[node.node_id]
        max_state = max(max_state, float((wy_state - serial_state).abs().max()))
        max_out = max(max_out, float((wy_out - serial_out).abs().max()))

    return {
        "total_nodes": total_nodes,
        "spine_depth": spine_depth,
        "seed": seed,
        "max_append_vs_rebuild_t": max_t,
        "max_append_vs_rebuild_operator": max_p,
        "max_homogeneous_s0_map": max_s0map,
        "max_state_vs_serial": max_state,
        "max_out_vs_serial": max_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("FR12_RESULTS.md"))
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    cases = [
        run_case(total_nodes=8, spine_depth=4, seed=1201),
        run_case(total_nodes=11, spine_depth=5, seed=1202),
        run_case(total_nodes=14, spine_depth=6, seed=1203),
        run_case(total_nodes=16, spine_depth=6, seed=1204),
    ]
    maxes = {
        key: max(float(case[key]) for case in cases)
        for key in (
            "max_append_vs_rebuild_t",
            "max_append_vs_rebuild_operator",
            "max_homogeneous_s0_map",
            "max_state_vs_serial",
            "max_out_vs_serial",
        )
    }
    verdict = "PASS" if max(maxes.values()) < 1e-8 else "FAIL"
    result = {
        "schema": "fr12.wy_tree_recurrence_check.v1",
        "dtype": str(DTYPE),
        "dk": DK,
        "dv": DV,
        "cases": cases,
        "maxes": maxes,
        "threshold": 1e-8,
        "verdict": verdict,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if verdict != "PASS":
        return 1

    section = [
        "# FR12 Results",
        "",
        "## WY Tree Recurrence Gate",
        "",
        "Command:",
        "",
        "```bash",
        "python3 scripts/fr12_wy_tree_recurrence_check.py --json-out output/fr12_wy_tree_recurrence_check.json",
        "```",
        "",
        "Scope:",
        "- Uses FR10 tree descriptors from `scripts/fr10_gdn_tree_algebra_reference.py`.",
        "- Runs the gated delta recurrence in float64 to avoid the vLLM CPU oracle's fp32 floor.",
        "- Validates parent-inherit plus one-reflector append T against rebuilding WY on each path.",
        "- Validates full per-node state/output against serial per-path GDN semantics.",
        "",
        "Result:",
        f"- Verdict: `{verdict}` at threshold `1e-8`.",
        f"- Max append-vs-rebuild T/basis error: `{maxes['max_append_vs_rebuild_t']}`.",
        f"- Max append-vs-rebuild operator error: `{maxes['max_append_vs_rebuild_operator']}`.",
        f"- Max homogeneous S0 map error: `{maxes['max_homogeneous_s0_map']}`.",
        f"- Max full state vs serial error: `{maxes['max_state_vs_serial']}`.",
        f"- Max output vs serial error: `{maxes['max_out_vs_serial']}`.",
        "",
        "Interpretation:",
        "",
        "`TREE_ANCESTRY_T_RECURRENCE_CONFIRMED`",
        "",
        "The FR12 WY tree recurrence is algebraically exact at float64 floor for the tested FR10 tree shapes. This validates the parent T inheritance plus append rule before building the Triton kernel.",
        "",
    ]
    args.out.write_text("\n".join(section))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
