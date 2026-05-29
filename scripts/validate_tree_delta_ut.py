#!/usr/bin/env python3
"""Validate the tree-ancestor WY/UT transform for Gated DeltaNet updates.

This is an offline algebra check for Round F.  It compares a sequential
parent-to-child reference against a single topo-ordered masked triangular solve.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Case:
    name: str
    parents: tuple[int, ...]
    gated: bool
    dtype: torch.dtype


def build_ancestor_mask(parents: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Return L[i, j] = 1 iff j is a proper ancestor of i."""

    n = len(parents)
    mask = torch.zeros((n, n), dtype=torch.bool, device=device)
    for i, parent in enumerate(parents):
        while parent >= 0:
            mask[i, parent] = True
            parent = parents[parent]
    return mask


def path_gammas(alpha: torch.Tensor, parents: tuple[int, ...]) -> torch.Tensor:
    gamma = torch.empty_like(alpha)
    for i, parent in enumerate(parents):
        gamma[i] = alpha[i] if parent < 0 else gamma[parent] * alpha[i]
    return gamma


def sequential_tree_delta(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    parents: tuple[int, ...],
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(parents)
    d_v, d_k = initial_state.shape
    states = torch.empty((n, d_v, d_k), dtype=k.dtype, device=k.device)
    writes = torch.empty((n, d_v), dtype=k.dtype, device=k.device)
    outputs = torch.empty((n, d_v), dtype=k.dtype, device=k.device)

    for i, parent in enumerate(parents):
        parent_state = initial_state if parent < 0 else states[parent]
        projected = parent_state @ k[i]
        write = beta[i] * (v[i] - alpha[i] * projected)
        states[i] = alpha[i] * parent_state + torch.outer(write, k[i])
        writes[i] = write
        outputs[i] = states[i] @ k[i]
    return states, writes, outputs


def tree_ut_delta(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    parents: tuple[int, ...],
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(parents)
    eye = torch.eye(n, dtype=k.dtype, device=k.device)
    ancestor = build_ancestor_mask(parents, k.device).to(k.dtype)
    gamma = path_gammas(alpha, parents)

    kk = k @ k.T
    gamma_ratio = gamma[:, None] / gamma[None, :]
    lower = ancestor * beta[:, None] * gamma_ratio * kk
    system = eye + lower

    initial_projection = k @ initial_state.T
    rhs = beta[:, None] * (v - gamma[:, None] * initial_projection)
    writes = torch.linalg.solve_triangular(system, rhs, upper=False)

    ancestor_or_self = ancestor + eye
    states = torch.empty((n, initial_state.shape[0], initial_state.shape[1]), dtype=k.dtype, device=k.device)
    for i in range(n):
        state = gamma[i] * initial_state.clone()
        for j in range(n):
            if ancestor_or_self[i, j] != 0:
                state = state + (gamma[i] / gamma[j]) * torch.outer(writes[j], k[j])
        states[i] = state
    outputs = torch.einsum("nvk,nk->nv", states, k)
    return states, writes, outputs


def sequential_multihead_tree_delta(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    parents: tuple[int, ...],
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, h, d_k = k.shape
    d_v = v.shape[-1]
    states = torch.empty((n, h, d_v, d_k), dtype=k.dtype, device=k.device)
    outputs = torch.empty((n, h, d_v), dtype=k.dtype, device=k.device)
    for i, parent in enumerate(parents):
        parent_state = initial_state if parent < 0 else states[parent]
        projected = torch.einsum("hvk,hk->hv", parent_state, k[i])
        write = beta[i, :, None] * (v[i] - alpha[i, :, None] * projected)
        states[i] = alpha[i, :, None, None] * parent_state + torch.einsum("hv,hk->hvk", write, k[i])
        outputs[i] = torch.einsum("hvk,hk->hv", states[i], k[i])
    return states, outputs


def tree_ut_multihead_delta(
    k_key_heads: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    alpha: torch.Tensor,
    parents: tuple[int, ...],
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(parents)
    h_v = v.shape[1]
    h_k = k_key_heads.shape[1]
    if h_v % h_k != 0:
        raise ValueError("value heads must be divisible by key heads")
    k = k_key_heads.repeat_interleave(h_v // h_k, dim=1)
    ancestor = build_ancestor_mask(parents, k.device).to(k.dtype)
    eye = torch.eye(n, dtype=k.dtype, device=k.device)

    gamma = torch.empty_like(alpha)
    for i, parent in enumerate(parents):
        gamma[i] = alpha[i] if parent < 0 else gamma[parent] * alpha[i]

    kk = torch.einsum("nhd,mhd->hnm", k, k)
    gamma_hn = gamma.transpose(0, 1)
    beta_hn = beta.transpose(0, 1)
    ratio = gamma_hn[:, :, None] / gamma_hn[:, None, :]
    system = eye.unsqueeze(0) + ancestor.unsqueeze(0) * beta_hn[:, :, None] * ratio * kk

    initial_projection = torch.einsum("hvk,nhk->nhv", initial_state, k)
    rhs = beta[:, :, None] * (v - gamma[:, :, None] * initial_projection)
    writes = torch.linalg.solve_triangular(system, rhs.permute(1, 0, 2).contiguous(), upper=False)

    coeff = (ancestor + eye).unsqueeze(0) * ratio
    states = (
        gamma[:, :, None, None] * initial_state.unsqueeze(0)
        + torch.einsum("hij,hjv,hjk->ihvk", coeff, writes, k.permute(1, 0, 2))
    )
    outputs = torch.einsum("ihvk,ihk->ihv", states, k)
    return states, outputs


def make_inputs(
    *,
    n: int,
    d_k: int,
    d_v: int,
    seed: int,
    gated: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    k = torch.randn((n, d_k), generator=generator, dtype=dtype, device=device)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = torch.randn((n, d_v), generator=generator, dtype=dtype, device=device)
    beta = torch.rand((n,), generator=generator, dtype=dtype, device=device) * 0.65 + 0.05
    if gated:
        alpha = torch.rand((n,), generator=generator, dtype=dtype, device=device) * 0.25 + 0.72
    else:
        alpha = torch.ones((n,), dtype=dtype, device=device)
    initial_state = torch.randn((d_v, d_k), generator=generator, dtype=dtype, device=device) * 0.2
    return k, v, beta, alpha, initial_state


def run_case(case: Case, *, seed: int, d_k: int, d_v: int, device: torch.device) -> dict[str, float | str]:
    k, v, beta, alpha, initial_state = make_inputs(
        n=len(case.parents),
        d_k=d_k,
        d_v=d_v,
        seed=seed,
        gated=case.gated,
        dtype=case.dtype,
        device=device,
    )
    ref_states, ref_writes, ref_outputs = sequential_tree_delta(k, v, beta, alpha, case.parents, initial_state)
    got_states, got_writes, got_outputs = tree_ut_delta(k, v, beta, alpha, case.parents, initial_state)

    state_abs = (got_states - ref_states).abs().max().item()
    write_abs = (got_writes - ref_writes).abs().max().item()
    output_abs = (got_outputs - ref_outputs).abs().max().item()
    state_rel = (got_states - ref_states).abs().max().div(ref_states.abs().max().clamp_min(1e-12)).item()
    write_rel = (got_writes - ref_writes).abs().max().div(ref_writes.abs().max().clamp_min(1e-12)).item()
    output_rel = (got_outputs - ref_outputs).abs().max().div(ref_outputs.abs().max().clamp_min(1e-12)).item()
    return {
        "case": case.name,
        "dtype": str(case.dtype).replace("torch.", ""),
        "state_max_abs": state_abs,
        "state_max_rel": state_rel,
        "write_max_abs": write_abs,
        "write_max_rel": write_rel,
        "output_max_abs": output_abs,
        "output_max_rel": output_rel,
    }


def run_multihead_case(*, seed: int, device: torch.device) -> dict[str, float | str]:
    dtype = torch.float64
    parents = (-1, 0, 0, 1, 1, 2, 4, 4)
    n = len(parents)
    h_k = 2
    h_v = 6
    d_k = 8
    d_v = 5
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    k = torch.randn((n, h_k, d_k), generator=generator, dtype=dtype, device=device)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v = torch.randn((n, h_v, d_v), generator=generator, dtype=dtype, device=device)
    beta = torch.rand((n, h_v), generator=generator, dtype=dtype, device=device) * 0.65 + 0.05
    alpha = torch.rand((n, h_v), generator=generator, dtype=dtype, device=device) * 0.25 + 0.72
    initial_state = torch.randn((h_v, d_v, d_k), generator=generator, dtype=dtype, device=device) * 0.2

    ref_states, ref_outputs = sequential_multihead_tree_delta(
        k.repeat_interleave(h_v // h_k, dim=1), v, beta, alpha, parents, initial_state
    )
    got_states, got_outputs = tree_ut_multihead_delta(k, v, beta, alpha, parents, initial_state)
    return {
        "case": "branched_gated_gqa_multihead_fp64",
        "dtype": "float64",
        "state_max_abs": (got_states - ref_states).abs().max().item(),
        "state_max_rel": (got_states - ref_states).abs().max().div(ref_states.abs().max().clamp_min(1e-12)).item(),
        "output_max_abs": (got_outputs - ref_outputs).abs().max().item(),
        "output_max_rel": (got_outputs - ref_outputs).abs().max().div(ref_outputs.abs().max().clamp_min(1e-12)).item(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--d-k", type=int, default=8)
    parser.add_argument("--d-v", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cases = [
        Case("branched_ungated_fp64", (-1, 0, 0, 1, 1, 2, 4, 4), False, torch.float64),
        Case("branched_gated_fp64", (-1, 0, 0, 1, 1, 2, 4, 4), True, torch.float64),
        Case("spine_gated_fp64", (-1, 0, 1, 2, 3, 4, 5, 6), True, torch.float64),
        Case("branched_gated_fp32", (-1, 0, 0, 1, 1, 2, 4, 4), True, torch.float32),
    ]

    tolerances = {torch.float64: 1e-10, torch.float32: 5e-5}
    failed = False
    for offset, case in enumerate(cases):
        result = run_case(case, seed=args.seed + offset, d_k=args.d_k, d_v=args.d_v, device=device)
        tolerance = tolerances[case.dtype]
        ok = (
            result["state_max_abs"] <= tolerance
            and result["write_max_abs"] <= tolerance
            and result["output_max_abs"] <= tolerance
        )
        failed = failed or not ok
        print(
            f"{'PASS' if ok else 'FAIL'} {result['case']} dtype={result['dtype']} "
            f"state_max_abs={result['state_max_abs']:.3e} "
            f"state_max_rel={result['state_max_rel']:.3e} "
            f"write_max_abs={result['write_max_abs']:.3e} "
            f"write_max_rel={result['write_max_rel']:.3e} "
            f"output_max_abs={result['output_max_abs']:.3e} "
            f"output_max_rel={result['output_max_rel']:.3e}"
        )
    result = run_multihead_case(seed=args.seed + 100, device=device)
    ok = result["state_max_abs"] <= 1e-10 and result["output_max_abs"] <= 1e-10
    failed = failed or not ok
    print(
        f"{'PASS' if ok else 'FAIL'} {result['case']} dtype={result['dtype']} "
        f"state_max_abs={result['state_max_abs']:.3e} "
        f"state_max_rel={result['state_max_rel']:.3e} "
        f"output_max_abs={result['output_max_abs']:.3e} "
        f"output_max_rel={result['output_max_rel']:.3e}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
