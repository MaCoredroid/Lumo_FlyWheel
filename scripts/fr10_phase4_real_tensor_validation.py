#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.fr10_gdn_tree_kernel import Tree, launch_tree_gdn


DEFAULT_TREE = (
    "[(0,), (1,), (0, 0), (1, 0), (0, 0, 0), (1, 0, 0), "
    "(0, 0, 0, 0), (1, 0, 0, 0), (0, 0, 0, 0, 0), (1, 0, 0, 0, 0)]"
)


def tree_from_choices(tree_literal: str) -> Tree:
    choices = ast.literal_eval(tree_literal)
    index = {choice: i + 1 for i, choice in enumerate(choices)}
    parent = [-1]
    for choice in choices:
        if len(choice) == 1:
            parent.append(0)
        else:
            parent.append(index[choice[:-1]])
    return Tree(tuple(parent))


def gqa_tree_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    visible: torch.Tensor,
    strict: torch.Tensor,
    output_scale: float,
    *,
    mapping: str = "consecutive",
) -> tuple[torch.Tensor, torch.Tensor]:
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()
    n, num_kh, dim_k = q.shape
    num_vh = v.shape[1]
    dim_v = v.shape[2]
    head_group = num_vh // num_kh
    cum_g = (visible.to(g.dtype).unsqueeze(-1) * g.unsqueeze(0)).sum(dim=1)
    out = torch.empty((n, num_vh, dim_v), device=q.device, dtype=torch.float32)
    state = torch.empty((n, num_vh, dim_v, dim_k), device=q.device, dtype=torch.float32)
    eye = torch.eye(n, device=q.device)
    strict_f = strict.to(torch.float32)
    for vh in range(num_vh):
        if mapping == "consecutive":
            kh = vh // head_group
        elif mapping == "strided":
            kh = vh % num_kh
        else:
            raise ValueError(f"unknown GQA mapping {mapping}")
        kk = k[:, kh] @ k[:, kh].T
        decay = torch.exp(cum_g[:, vh].unsqueeze(1) - cum_g[:, vh].unsqueeze(0))
        system = eye + strict_f * kk * beta[:, vh].unsqueeze(1) * decay
        solved_v = torch.linalg.solve_triangular(
            system,
            beta[:, vh].unsqueeze(1) * v[:, vh],
            upper=False,
        )
        solved_k = torch.linalg.solve_triangular(
            system,
            beta[:, vh].unsqueeze(1) * k[:, kh] * torch.exp(cum_g[:, vh]).unsqueeze(1),
            upper=False,
        )
        incoming = h0[vh] @ solved_k.T
        trans = solved_v - incoming.T
        for i in range(n):
            state_i = h0[vh] * torch.exp(cum_g[i, vh])
            for j in range(n):
                if visible[i, j]:
                    state_i = state_i + trans[j].unsqueeze(1) * k[j, kh].unsqueeze(0) * torch.exp(
                        cum_g[i, vh] - cum_g[j, vh]
                    )
            state[i, vh] = state_i
            out[i, vh] = (state_i @ q[i, kh]) * output_scale
    return out, state


def native_serial_per_path(
    tree: Tree,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    h0: torch.Tensor,
    output_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.fla.ops import chunk_gated_delta_rule

    outputs = []
    states = []
    for node in range(tree.n):
        path = torch.tensor(tree.path(node), device=q.device, dtype=torch.long)
        out, state = chunk_gated_delta_rule(
            q=q.index_select(0, path).unsqueeze(0).contiguous(),
            k=k.index_select(0, path).unsqueeze(0).contiguous(),
            v=v.index_select(0, path).unsqueeze(0).contiguous(),
            g=g.index_select(0, path).unsqueeze(0).contiguous(),
            beta=beta.index_select(0, path).unsqueeze(0).contiguous(),
            scale=output_scale,
            initial_state=h0.unsqueeze(0).contiguous(),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
        )
        outputs.append(out[0, -1].float())
        states.append(state[0].float())
    return torch.stack(outputs, dim=0), torch.stack(states, dim=0)


def native_update_serial_per_path(
    tree: Tree,
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    h0: torch.Tensor,
    output_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating_delta_rule_update

    outputs = []
    states = []
    for node in range(tree.n):
        path = torch.tensor(tree.path(node), device=q_raw.device, dtype=torch.long)
        initial = h0.unsqueeze(0).contiguous().clone()
        out, state = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a.index_select(0, path).contiguous(),
            b=b.index_select(0, path).contiguous(),
            dt_bias=dt_bias,
            q=q_raw.index_select(0, path).unsqueeze(0).contiguous(),
            k=k_raw.index_select(0, path).unsqueeze(0).contiguous(),
            v=v_raw.index_select(0, path).unsqueeze(0).contiguous(),
            scale=output_scale,
            initial_state=initial,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        outputs.append(out[0, -1].float())
        states.append(state[0].float())
    return torch.stack(outputs, dim=0), torch.stack(states, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tree", default=DEFAULT_TREE)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for real tensor validation")

    from vllm.model_executor.layers.fla.ops import fused_post_conv_prep

    payload_path = Path(args.payload)
    payload = torch.load(payload_path, map_location="cpu")
    tree = tree_from_choices(args.tree)
    n = tree.n
    strict, visible = tree.masks(torch.device("cuda"), 1 << (n - 1).bit_length())
    strict_n = strict[:n, :n]
    visible_n = visible[:n, :n]

    mixed_qkv = payload["mixed_qkv_spec"].to("cuda")
    a = payload["a"].to("cuda")
    b = payload["b"].to("cuda")
    A_log = payload["A_log"].to("cuda")
    dt_bias = payload["dt_bias"].to("cuda")
    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        num_k_heads=16,
        head_k_dim=128,
        head_v_dim=128,
        apply_l2norm=True,
        output_g_exp=False,
    )
    q_raw = payload["query_spec"].squeeze(0).to("cuda")
    k_raw = payload["key_spec"].squeeze(0).to("cuda")
    v_raw = payload["value_spec"].squeeze(0).to("cuda")
    state_indices = payload["spec_state_indices_tensor"][0]
    h0 = payload["initial_state_before_spec"][int(state_indices[0].item())].to("cuda")
    output_scale = 128**-0.5

    out, state = launch_tree_gdn(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        h0.contiguous(),
        tree,
        strict_mask=strict,
        visible_mask=visible,
        output_scale=output_scale,
    )
    torch.cuda.synchronize()
    ref_out, ref_state = gqa_tree_reference(
        q[:n],
        k[:n],
        v[:n],
        g[:n],
        beta[:n],
        h0,
        visible_n,
        strict_n,
        output_scale,
        mapping="consecutive",
    )
    strided_ref_out, strided_ref_state = gqa_tree_reference(
        q[:n],
        k[:n],
        v[:n],
        g[:n],
        beta[:n],
        h0,
        visible_n,
        strict_n,
        output_scale,
        mapping="strided",
    )
    native_path_out, native_path_state = native_serial_per_path(
        tree,
        q[:n],
        k[:n],
        v[:n],
        g[:n],
        beta[:n],
        h0,
        output_scale,
    )
    native_update_out, native_update_state = native_update_serial_per_path(
        tree,
        q_raw[:n],
        k_raw[:n],
        v_raw[:n],
        a[:n],
        b[:n],
        A_log,
        dt_bias,
        h0,
        output_scale,
    )
    native_linear = payload["core_attn_out_spec_native"].squeeze(0).to("cuda").float()
    non_linear_nodes = torch.tensor(
        [i for i, p in enumerate(tree.parent) if i > 0 and p != i - 1],
        device="cuda",
        dtype=torch.long,
    )
    native_consecutive_out = (native_path_out - ref_out).abs().max()
    native_strided_out = (native_path_out - strided_ref_out).abs().max()
    native_consecutive_state = (native_path_state - ref_state).abs().max()
    native_strided_state = (native_path_state - strided_ref_state).abs().max()
    native_consecutive_state_transposed = (
        native_path_state.transpose(-1, -2) - ref_state
    ).abs().max()
    tree_native_state_transposed = (
        state[:n].transpose(-1, -2) - native_path_state
    ).abs().max()
    result = {
        "payload": str(payload_path),
        "tree_parent": list(tree.parent),
        "num_nodes": n,
        "q_shape": list(q.shape),
        "k_shape": list(k.shape),
        "v_shape": list(v.shape),
        "g_shape": list(g.shape),
        "beta_shape": list(beta.shape),
        "h0_state_index": int(state_indices[0].item()),
        "tree_kernel_vs_gqa_ref_out_abs": float((out[:n].float() - ref_out).abs().max().item()),
        "tree_kernel_vs_gqa_ref_state_abs": float((state[:n] - ref_state).abs().max().item()),
        "tree_kernel_vs_native_serial_path_out_abs": float(
            (out[:n].float() - native_path_out).abs().max().item()
        ),
        "tree_kernel_vs_native_serial_path_state_abs": float(
            (state[:n] - native_path_state).abs().max().item()
        ),
        "native_serial_path_vs_gqa_consecutive_ref_out_abs": float(
            native_consecutive_out.item()
        ),
        "native_serial_path_vs_gqa_consecutive_ref_state_abs": float(
            native_consecutive_state.item()
        ),
        "native_update_path_vs_gqa_consecutive_ref_out_abs": float(
            (native_update_out - ref_out).abs().max().item()
        ),
        "native_update_path_vs_gqa_consecutive_ref_state_abs": float(
            (native_update_state - ref_state).abs().max().item()
        ),
        "tree_kernel_vs_native_update_path_out_abs": float(
            (out[:n].float() - native_update_out).abs().max().item()
        ),
        "tree_kernel_vs_native_update_path_state_abs": float(
            (state[:n] - native_update_state).abs().max().item()
        ),
        "native_serial_path_transposed_vs_gqa_consecutive_ref_state_abs": float(
            native_consecutive_state_transposed.item()
        ),
        "native_serial_path_vs_gqa_strided_ref_out_abs": float(
            native_strided_out.item()
        ),
        "native_serial_path_vs_gqa_strided_ref_state_abs": float(
            native_strided_state.item()
        ),
        "tree_kernel_transposed_state_vs_native_serial_path_state_abs": float(
            tree_native_state_transposed.item()
        ),
        "gqa_mapping_confirmed": "consecutive"
        if native_consecutive_out < native_strided_out
        else "strided_or_inconclusive",
        "native_linear_vs_tree_ref_out_abs": float((native_linear[:n] - ref_out).abs().max().item()),
        "native_linear_vs_tree_ref_non_linear_nodes_abs": float(
            (native_linear.index_select(0, non_linear_nodes) - ref_out.index_select(0, non_linear_nodes))
            .abs()
            .max()
            .item()
        ),
        "non_linear_nodes": non_linear_nodes.cpu().tolist(),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
