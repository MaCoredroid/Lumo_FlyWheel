#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl


NODE_FAMILIES = (2, 3, 6, 8, 14)
H = 48
K = 128
V = 128
BV = 16
VLLM_CPU_RULE = Path(
    "/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/mamba/ops/cpu/"
    "recurrent_gated_delta_rule.py"
)
VLLM_FLA_OPS = Path(
    "/tmp/vllm-0.22-src/vllm-0.22.0/vllm/model_executor/layers/fla/ops"
)


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
        raise ValueError(f"FR10 Phase 2 microbench only warms padded node blocks up to 16, got {n}")
    return n_pad


def load_cpu_rule():
    spec = importlib.util.spec_from_file_location("fr10_vllm_cpu_gdn_rule", VLLM_CPU_RULE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import vLLM CPU rule from {VLLM_CPU_RULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_native_chunk_module():
    # Diagnostic fallback only. Direct vLLM imports in the CUDA-13 audit image
    # currently fail through vllm._C -> missing libcudart.so.12, so this source
    # load is not a substitute for a production-wrapper Gate D comparison.
    if "fr10_vllm_fla_ops.chunk" in sys.modules:
        return sys.modules["fr10_vllm_fla_ops.chunk"]
    try:
        return importlib.import_module("vllm.model_executor.layers.fla.ops.chunk")
    except ImportError as first_error:
        if "libcudart.so.12" not in str(first_error):
            raise
    fake_triton_utils = types.ModuleType("vllm.triton_utils")
    fake_triton_utils.triton = triton
    fake_triton_utils.tl = tl
    fake_triton_utils.tldevice = tl.extra.libdevice
    fake_platforms = types.ModuleType("vllm.platforms")

    class _CurrentPlatform:
        @staticmethod
        def is_cuda_alike() -> bool:
            return True

    fake_platforms.current_platform = _CurrentPlatform()
    sys.modules["vllm.triton_utils"] = fake_triton_utils
    sys.modules["vllm.platforms"] = fake_platforms

    pkg_name = "fr10_vllm_fla_ops"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(VLLM_FLA_OPS)]
    sys.modules[pkg_name] = pkg
    module_order = [
        "utils",
        "op",
        "index",
        "l2norm",
        "wy_fast",
        "chunk_o",
        "chunk_scaled_dot_kkt",
        "solve_tril",
        "cumsum",
        "chunk_delta_h",
        "chunk",
    ]
    for name in module_order:
        full_name = f"{pkg_name}.{name}"
        if full_name in sys.modules:
            continue
        path = VLLM_FLA_OPS / f"{name}.py"
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load native FLA source module {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[f"{pkg_name}.chunk"]


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


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def production_l2norm_fwd(x: torch.Tensor) -> torch.Tensor:
    from vllm.model_executor.layers.fla.ops.chunk import l2norm_fwd

    return l2norm_fwd(x)


def torch_tree_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    visible: torch.Tensor,
    strict: torch.Tensor,
    output_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()
    n = q.shape[0]
    cum_g = (visible.to(g.dtype).unsqueeze(-1) * g.unsqueeze(0)).sum(dim=1)
    out = torch.empty((n, H, V), device=q.device, dtype=torch.float32)
    state = torch.empty((n, H, V, K), device=q.device, dtype=torch.float32)
    for h in range(H):
        kk = k[:, h] @ k[:, h].T
        decay = torch.exp(cum_g[:, h].unsqueeze(1) - cum_g[:, h].unsqueeze(0))
        system = torch.eye(n, device=q.device) + strict.to(torch.float32) * kk * beta[:, h].unsqueeze(1) * decay
        solved_v = torch.linalg.solve_triangular(system, beta[:, h].unsqueeze(1) * v[:, h], upper=False)
        solved_k = torch.linalg.solve_triangular(
            system,
            beta[:, h].unsqueeze(1) * k[:, h] * torch.exp(cum_g[:, h]).unsqueeze(1),
            upper=False,
        )
        incoming = h0[h] @ solved_k.T
        trans = solved_v - incoming.T
        for i in range(n):
            state_i = h0[h] * torch.exp(cum_g[i, h])
            for j in range(n):
                if visible[i, j]:
                    state_i = state_i + trans[j].unsqueeze(1) * k[j, h].unsqueeze(0) * torch.exp(
                        cum_g[i, h] - cum_g[j, h]
                    )
            state[i, h] = state_i
            out[i, h] = (state_i @ q[i, h]) * output_scale
    return out, state


def cpu_serial_oracle(
    tree: Tree,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    output_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rule = load_cpu_rule()
    q_cpu = q[: tree.n].detach().cpu().unsqueeze(0)
    k_cpu = k[: tree.n].detach().cpu().unsqueeze(0)
    v_cpu = v[: tree.n].detach().cpu().unsqueeze(0)
    g_cpu = g[: tree.n].detach().cpu().unsqueeze(0)
    beta_cpu = beta[: tree.n].detach().cpu().unsqueeze(0)
    h0_cpu = h0.detach().cpu().unsqueeze(0)
    outputs = []
    states = []
    for node in range(tree.n):
        path = torch.tensor(tree.path(node), dtype=torch.long)
        out, state = rule.recurrent_gated_delta_rule(
            query=q_cpu.index_select(1, path),
            key=k_cpu.index_select(1, path),
            value=v_cpu.index_select(1, path),
            g=g_cpu.index_select(1, path),
            beta=beta_cpu.index_select(1, path),
            initial_state=h0_cpu.clone(),
            scale=output_scale,
            use_qk_l2norm_in_kernel=False,
        )
        outputs.append(out[:, -1].squeeze(0))
        states.append(state.squeeze(0))
    return torch.stack(outputs, dim=0), torch.stack(states, dim=0)


def native_gpu_oracle(
    tree: Tree,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    output_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Source-loaded native FLA diagnostic, not the final production wrapper."""
    chunk = load_native_chunk_module()
    chunk_gated_delta_rule = chunk.chunk_gated_delta_rule
    q_native = q[: tree.n].to(torch.bfloat16).unsqueeze(0)
    k_native = k[: tree.n].to(torch.bfloat16).unsqueeze(0)
    v_native = v[: tree.n].to(torch.bfloat16).unsqueeze(0)
    g_native = g[: tree.n].to(torch.bfloat16).unsqueeze(0)
    beta_native = beta[: tree.n].to(torch.bfloat16).unsqueeze(0)
    outputs = []
    states = []
    for node in range(tree.n):
        path = torch.tensor(tree.path(node), dtype=torch.long, device=q.device)
        out, state = chunk_gated_delta_rule(
            q_native.index_select(1, path),
            k_native.index_select(1, path),
            v_native.index_select(1, path),
            g_native.index_select(1, path),
            beta_native.index_select(1, path),
            scale=output_scale,
            initial_state=h0.unsqueeze(0).contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        outputs.append(out[:, -1].squeeze(0).float())
        states.append(state.squeeze(0).float())
    return torch.stack(outputs, dim=0), torch.stack(states, dim=0)


def native_gpu_full_spine_oracle(
    tree: Tree,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    output_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-spine source-loaded FLA diagnostic for reduction-order drift."""
    if not tree.is_single_spine():
        raise ValueError("native full-path chunk oracle is only defined for single-spine trees")
    chunk = load_native_chunk_module()
    chunk_gated_delta_rule = chunk.chunk_gated_delta_rule
    out, state = chunk_gated_delta_rule(
        q[: tree.n].to(torch.bfloat16).unsqueeze(0),
        k[: tree.n].to(torch.bfloat16).unsqueeze(0),
        v[: tree.n].to(torch.bfloat16).unsqueeze(0),
        g[: tree.n].to(torch.bfloat16).unsqueeze(0),
        beta[: tree.n].to(torch.bfloat16).unsqueeze(0),
        scale=output_scale,
        initial_state=h0.unsqueeze(0).contiguous(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )
    return out.squeeze(0).float(), state.squeeze(0).float()


def production_gdn_full_spine_oracle(
    tree: Tree,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    backend: str = "auto",
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Call the installed vLLM production GDN wrapper under a config context."""
    if not tree.is_single_spine():
        raise ValueError("production GDN full-spine oracle is only defined for single-spine trees")
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mamba.gdn_linear_attn import ChunkGatedDeltaRule

    additional_config = {} if backend == "default" else {"gdn_prefill_backend": backend}
    cfg = VllmConfig(additional_config=additional_config)
    with set_current_vllm_config(cfg):
        op = ChunkGatedDeltaRule()
        method = getattr(getattr(op, "_forward_method", None), "__name__", repr(getattr(op, "_forward_method", None)))
        out, state = op(
            q=q[: tree.n].to(torch.bfloat16).unsqueeze(0),
            k=k[: tree.n].to(torch.bfloat16).unsqueeze(0),
            v=v[: tree.n].to(torch.bfloat16).unsqueeze(0),
            g=g[: tree.n].to(torch.bfloat16).unsqueeze(0),
            beta=beta[: tree.n].to(torch.bfloat16).unsqueeze(0),
            initial_state=h0.unsqueeze(0).contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    return out.squeeze(0).float(), state.squeeze(0).float(), method


def run_tree(
    tree: Tree,
    capture: bool,
    cpu_oracle: bool,
    native_gpu: bool,
    *,
    seed: int,
    input_dtype: torch.dtype = torch.float32,
    output_scale: float = 1.0,
    production_gdn: bool = False,
    production_gdn_backend: str = "auto",
) -> dict[str, float | int | bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 2 Triton microbench")
    if production_gdn and input_dtype == torch.float32:
        raise RuntimeError("cu130 production GDN path requires bf16/fp16 activation inputs, not fp32")
    device = torch.device("cuda")
    n = tree.n
    n_pad = padded_nodes(n)
    strict, visible = tree.masks(device, n_pad)
    gen = torch.Generator(device=device).manual_seed(seed)
    raw_q = torch.randn((n_pad, H, K), device=device, generator=gen, dtype=torch.float32).contiguous()
    raw_k = torch.randn((n_pad, H, K), device=device, generator=gen, dtype=torch.float32).contiguous()
    q = l2norm(raw_q).contiguous()
    k = l2norm(raw_k).contiguous()
    v = (torch.randn((n_pad, H, V), device=device, generator=gen, dtype=torch.float32) * 0.2).contiguous()
    g = (torch.randn((n_pad, H), device=device, generator=gen, dtype=torch.float32) * 0.05 - 0.1).contiguous()
    beta = torch.sigmoid(torch.randn((n_pad, H), device=device, generator=gen, dtype=torch.float32)).contiguous()
    if input_dtype != torch.float32:
        raw_q = raw_q.to(input_dtype).contiguous()
        raw_k = raw_k.to(input_dtype).contiguous()
        q = q.to(input_dtype).contiguous()
        k = k.to(input_dtype).contiguous()
        v = v.to(input_dtype).contiguous()
        g = g.to(input_dtype).contiguous()
        beta = beta.to(input_dtype).contiguous()
    if production_gdn:
        # cu130 forward_native receives raw bf16 q/k and applies vLLM l2norm_fwd
        # inside the wrapper. Precompute exactly that normalized representation
        # for the tree kernel while still passing raw q/k to the production op.
        q = production_l2norm_fwd(raw_q).contiguous()
        k = production_l2norm_fwd(raw_k).contiguous()
    q[n:] = 0
    k[n:] = 0
    raw_q[n:] = 0
    raw_k[n:] = 0
    v[n:] = 0
    g[n:] = 0
    beta[n:] = 0
    h0 = (torch.randn((H, V, K), device=device, generator=gen, dtype=torch.float32) * 0.05).contiguous()
    # vLLM FLA forward_native returns o.to(q.dtype) and fp32 final_state.
    # Keep fp32 output only for the fp32 algebra diagnostic path.
    out_dtype = input_dtype if input_dtype != torch.float32 else torch.float32
    out = torch.empty((n_pad, H, V), device=device, dtype=out_dtype)
    state = torch.empty((n_pad, H, V, K), device=device, dtype=torch.float32)

    grid = (H, triton.cdiv(V, BV))
    _tree_gdn_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        h0,
        strict,
        visible,
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
    torch.cuda.synchronize()
    ref_out, ref_state = torch_tree_reference(q[:n], k[:n], v[:n], g[:n], beta[:n], h0, visible[:n, :n], strict[:n, :n], output_scale=output_scale)
    max_out = (out[:n].float() - ref_out).abs().max().item()
    max_state = (state[:n] - ref_state).abs().max().item()
    cpu_max_out = None
    cpu_max_state = None
    native_max_out = None
    native_max_state = None
    native_full_out = None
    native_full_final_state = None
    production_gdn_out = None
    production_gdn_final_state = None
    production_gdn_method = None
    if cpu_oracle:
        cpu_out, cpu_state = cpu_serial_oracle(tree, q, k, v, g, beta, h0, output_scale=output_scale)
        cpu_max_out = (out[:n].detach().cpu().float() - cpu_out).abs().max().item()
        cpu_max_state = (state[:n].detach().cpu() - cpu_state).abs().max().item()
    if native_gpu:
        native_out, native_state = native_gpu_oracle(tree, q, k, v, g, beta, h0, output_scale=output_scale)
        native_max_out = (out[:n].float() - native_out).abs().max().item()
        native_max_state = (state[:n] - native_state).abs().max().item()
        if tree.is_single_spine():
            full_out, full_final_state = native_gpu_full_spine_oracle(tree, q, k, v, g, beta, h0, output_scale=output_scale)
            native_full_out = (out[:n].float() - full_out).abs().max().item()
            native_full_final_state = (state[n - 1] - full_final_state).abs().max().item()
    if production_gdn:
        prod_out, prod_state, production_gdn_method = production_gdn_full_spine_oracle(
            tree, raw_q, raw_k, v, g, beta, h0, backend=production_gdn_backend, use_qk_l2norm_in_kernel=True
        )
        production_gdn_out = (out[:n].float() - prod_out).abs().max().item()
        production_gdn_final_state = (state[n - 1] - prod_state).abs().max().item()

    eager_out = out[:n].clone()
    eager_state = state[:n].clone()

    iters = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _tree_gdn_kernel[grid](q, k, v, g, beta, h0, strict, visible, out, state, N_ACTUAL=n, N_PAD=n_pad, NUM_H=H, DIM_K=K, DIM_V=V, BLOCK_V=BV, OUTPUT_SCALE=output_scale)
    end.record()
    torch.cuda.synchronize()
    eager_us = start.elapsed_time(end) * 1000.0 / iters

    graph_ok = False
    graph_bit_exact = False
    graph_us = 0.0
    if capture:
        graph = torch.cuda.CUDAGraph()
        _tree_gdn_kernel[grid](q, k, v, g, beta, h0, strict, visible, out, state, N_ACTUAL=n, N_PAD=n_pad, NUM_H=H, DIM_K=K, DIM_V=V, BLOCK_V=BV, OUTPUT_SCALE=output_scale)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            _tree_gdn_kernel[grid](q, k, v, g, beta, h0, strict, visible, out, state, N_ACTUAL=n, N_PAD=n_pad, NUM_H=H, DIM_K=K, DIM_V=V, BLOCK_V=BV, OUTPUT_SCALE=output_scale)
        out.zero_()
        state.zero_()
        graph.replay()
        torch.cuda.synchronize()
        graph_bit_exact = bool(torch.equal(out[:n], eager_out) and torch.equal(state[:n], eager_state))
        if not graph_bit_exact:
            out_diff = (out[:n] - eager_out).abs().max().item()
            state_diff = (state[:n] - eager_state).abs().max().item()
            raise AssertionError(
                f"CUDA graph replay differed from eager for n={n}: "
                f"out_diff={out_diff} state_diff={state_diff}"
            )
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        graph_us = start.elapsed_time(end) * 1000.0 / iters
        graph_ok = graph_bit_exact

    return {
        "nodes": n,
        "padded_nodes": n_pad,
        "input_dtype": str(input_dtype).replace("torch.", ""),
        "output_dtype": str(out_dtype).replace("torch.", ""),
        "state_dtype": "float32",
        "output_scale": output_scale,
        "max_out_abs": max_out,
        "max_state_abs": max_state,
        "cpu_oracle_max_out_abs": cpu_max_out,
        "cpu_oracle_max_state_abs": cpu_max_state,
        "native_gpu_max_out_abs": native_max_out,
        "native_gpu_max_state_abs": native_max_state,
        "native_gpu_full_spine_max_out_abs": native_full_out,
        "native_gpu_full_spine_final_state_abs": native_full_final_state,
        "production_gdn_backend_request": production_gdn_backend if production_gdn else None,
        "production_gdn_forward_method": production_gdn_method,
        "production_gdn_full_spine_max_out_abs": production_gdn_out,
        "production_gdn_full_spine_final_state_abs": production_gdn_final_state,
        "eager_us": eager_us,
        "graph_us": graph_us,
        "graph_ok": graph_ok,
        "graph_bit_exact": graph_bit_exact,
    }


def run_shape(
    n: int,
    capture: bool,
    cpu_oracle: bool,
    native_gpu: bool,
    input_dtype: torch.dtype,
    output_scale: float,
    production_gdn: bool,
    production_gdn_backend: str,
) -> dict[str, float | int | bool]:
    return run_tree(
        make_tree(n),
        capture=capture,
        cpu_oracle=cpu_oracle,
        native_gpu=native_gpu,
        seed=10_000 + n,
        input_dtype=input_dtype,
        output_scale=output_scale,
        production_gdn=production_gdn,
        production_gdn_backend=production_gdn_backend,
    )


def run_single_spine_table(
    nodes: list[int],
    capture: bool,
    cpu_oracle: bool,
    native_gpu: bool,
    input_dtype: torch.dtype,
    output_scale: float,
    production_gdn: bool,
    production_gdn_backend: str,
) -> list[dict[str, float | int | bool]]:
    rows = []
    for n in nodes:
        row = run_tree(
            make_spine_tree(n),
            capture=capture,
            cpu_oracle=cpu_oracle,
            native_gpu=native_gpu,
            seed=30_000 + n,
            input_dtype=input_dtype,
            output_scale=output_scale,
            production_gdn=production_gdn,
            production_gdn_backend=production_gdn_backend,
        )
        row["tree_kind"] = "single_spine"
        rows.append(row)
    return rows


def run_branch_depth_table(
    capture: bool,
    cpu_oracle: bool,
    native_gpu: bool,
    input_dtype: torch.dtype,
    output_scale: float,
) -> list[dict[str, float | int | bool | str]]:
    # Fixed base tree removes the previous base-shape confound. Each row appends
    # one sibling leaf to a parent at the requested depth, staying in padded 8.
    base = Tree((-1, 0, 1, 2, 3))
    pairs = [
        (0, base, Tree((*base.parent, 0))),
        (1, base, Tree((*base.parent, 1))),
        (2, base, Tree((*base.parent, 2))),
    ]
    rows = []
    for branch_depth, base, extended in pairs:
        seed = 20_000
        base_row = run_tree(base, capture=capture, cpu_oracle=cpu_oracle, native_gpu=native_gpu, seed=seed, input_dtype=input_dtype, output_scale=output_scale)
        ext_row = run_tree(extended, capture=capture, cpu_oracle=cpu_oracle, native_gpu=native_gpu, seed=seed, input_dtype=input_dtype, output_scale=output_scale)
        kernel_us = float(ext_row["graph_us"] if capture else ext_row["eager_us"])
        base_us = float(base_row["graph_us"] if capture else base_row["eager_us"])
        nodes_delta = int(ext_row["nodes"]) - int(base_row["nodes"])
        state_bytes = H * V * K * 4
        rows.append(
            {
                "tree_shape": f"{base.n}->{extended.n} padded {base_row['padded_nodes']}->{ext_row['padded_nodes']}",
                "branch_depth": branch_depth,
                "nodes": int(ext_row["nodes"]),
                "shared_trunk_tokens": branch_depth + 1,
                "new_leaf_tokens": nodes_delta,
                "kernel_us": kernel_us,
                "base_kernel_us": base_us,
                "marginal_leaf_us": kernel_us - base_us,
                "memory_bytes": state_bytes * int(ext_row["nodes"]),
                "state_reads": int(ext_row["nodes"]),
                "state_writes": int(ext_row["nodes"]),
                "equivalent_serial_us": base_us + nodes_delta * float(base_row["eager_us"]),
                "graph_ok": bool(ext_row["graph_ok"]),
                "cpu_oracle_max_out_abs": ext_row["cpu_oracle_max_out_abs"],
                "cpu_oracle_max_state_abs": ext_row["cpu_oracle_max_state_abs"],
                "native_gpu_max_out_abs": ext_row["native_gpu_max_out_abs"],
                "native_gpu_max_state_abs": ext_row["native_gpu_max_state_abs"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, nargs="*", default=list(NODE_FAMILIES))
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--cpu-oracle", action="store_true")
    parser.add_argument("--native-gpu", action="store_true")
    parser.add_argument("--branch-depth-table", action="store_true")
    parser.add_argument("--single-spine-table", action="store_true")
    parser.add_argument("--input-dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--production-scale", action="store_true", help="Use vLLM production chunk_gated_delta_rule scale 1/sqrt(128).")
    parser.add_argument("--production-gdn", action="store_true", help="Compare single-spine output/state against installed vLLM ChunkGatedDeltaRule.")
    parser.add_argument("--production-gdn-backend", choices=["default", "auto", "triton", "flashinfer"], default="auto")
    args = parser.parse_args()
    input_dtype = torch.float32 if args.input_dtype == "fp32" else torch.bfloat16
    output_scale = K ** -0.5 if args.production_scale else 1.0
    torch.cuda.init()
    if args.cpu_oracle and not VLLM_CPU_RULE.exists():
        raise SystemExit(f"CPU oracle path missing: {VLLM_CPU_RULE}")
    if args.branch_depth_table and args.single_spine_table:
        raise SystemExit("--branch-depth-table and --single-spine-table are mutually exclusive")
    if args.branch_depth_table:
        rows = run_branch_depth_table(capture=args.capture, cpu_oracle=args.cpu_oracle, native_gpu=args.native_gpu, input_dtype=input_dtype, output_scale=output_scale)
        print(json.dumps({"branch_depth_rows": rows, "device": torch.cuda.get_device_name(0)}, indent=2, sort_keys=True))
    elif args.single_spine_table:
        bad = sorted(set(args.nodes) - set(NODE_FAMILIES))
        if bad:
            raise SystemExit(f"fail-closed: unwarmed node families requested: {bad}")
        rows = run_single_spine_table(
            nodes=args.nodes,
            capture=args.capture,
            cpu_oracle=args.cpu_oracle,
            native_gpu=args.native_gpu,
            input_dtype=input_dtype,
            output_scale=output_scale,
            production_gdn=args.production_gdn,
            production_gdn_backend=args.production_gdn_backend,
        )
        print(json.dumps({"single_spine_rows": rows, "device": torch.cuda.get_device_name(0), "production_scale": args.production_scale}, indent=2, sort_keys=True))
    else:
        bad = sorted(set(args.nodes) - set(NODE_FAMILIES))
        if bad:
            raise SystemExit(f"fail-closed: unwarmed node families requested: {bad}")
        rows = [
            run_shape(
                n,
                capture=args.capture,
                cpu_oracle=args.cpu_oracle,
                native_gpu=args.native_gpu,
                input_dtype=input_dtype,
                output_scale=output_scale,
                production_gdn=args.production_gdn,
                production_gdn_backend=args.production_gdn_backend,
            )
            for n in args.nodes
        ]
        print(json.dumps({"rows": rows, "device": torch.cuda.get_device_name(0), "production_scale": args.production_scale}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
