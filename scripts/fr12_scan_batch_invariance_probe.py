#!/usr/bin/env python3
"""Probe GDN tree-scan batch dependence on captured tensors.

This is boot-free and uses one captured FR10 tree-GDN payload. It replays our
tree scan with identical tensors under different co-resident row layouts, then
maps outputs back to the original spine node ids. The native reference is the
vLLM FLA recurrent update on the same path0 chain.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lumo_flywheel_serving.fr10_gdn_tree_kernel import (  # noqa: E402
    Tree,
    launch_tree_gdn_prepared,
    padded_nodes,
)


def _leftmost_spine(parent: list[int]) -> list[int]:
    out = [0]
    cur = 0
    while True:
        children = [idx for idx, par in enumerate(parent) if int(par) == cur]
        if not children:
            return out
        cur = min(children)
        out.append(cur)


def _children(parent: list[int]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = defaultdict(list)
    for idx, par in enumerate(parent):
        if par >= 0:
            children[int(par)].append(int(idx))
    for vals in children.values():
        vals.sort()
    return children


def _dfs_order(parent: list[int], *, reverse_children: bool) -> list[int]:
    children = _children(parent)
    order: list[int] = []

    def visit(node: int) -> None:
        order.append(node)
        vals = children.get(node, [])
        if reverse_children:
            vals = list(reversed(vals))
        for child in vals:
            visit(child)

    visit(0)
    if len(order) != len(parent):
        raise RuntimeError(f"DFS only visited {len(order)} of {len(parent)} nodes")
    return order


def _spine_first_order(parent: list[int], spine: list[int]) -> list[int]:
    children = _children(parent)
    spine_set = set(spine)
    order = list(spine)
    seen = set(order)

    def visit(node: int) -> None:
        if node in seen:
            return
        par = parent[node]
        if par >= 0 and par not in seen:
            visit(par)
        seen.add(node)
        order.append(node)
        for child in children.get(node, []):
            visit(child)

    for spine_node in reversed(spine):
        for child in reversed(children.get(spine_node, [])):
            if child not in spine_set:
                visit(child)
    for node in range(len(parent)):
        visit(node)
    if sorted(order) != list(range(len(parent))):
        raise RuntimeError(f"invalid spine-first order: {order}")
    return order


def _remap_parent(parent: list[int], order: list[int]) -> list[int]:
    old_to_new = {old: new for new, old in enumerate(order)}
    remapped: list[int] = []
    for old in order:
        par = parent[old]
        remapped.append(-1 if par < 0 else old_to_new[par])
    return remapped


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def _nonzero(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(((a.float() - b.float()).abs() != 0).sum().item())


def _first_mismatch(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    diff = (a.float() - b.float()).abs()
    flat = int(torch.argmax(diff).item())
    val = float(diff.reshape(-1)[flat].item())
    if val == 0.0:
        return None
    idx = list(torch.unravel_index(torch.tensor(flat), diff.shape))
    idx_i = [int(x.item()) for x in idx]
    return {
        "index": idx_i,
        "a": float(a[tuple(idx_i)].float().item()),
        "b": float(b[tuple(idx_i)].float().item()),
        "abs": val,
    }


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    return {
        "max_abs": _max_abs(a, b),
        "mean_abs": _mean_abs(a, b),
        "nonzero": _nonzero(a, b),
        "first_mismatch": _first_mismatch(a, b),
    }


def _bf16_bank_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    neq = a_bf16 != b_bf16
    mismatch_count = int(neq.sum().item())
    first = None
    if mismatch_count:
        flat = int(torch.argmax(neq.reshape(-1).to(torch.int32)).item())
        idx = [int(x.item()) for x in torch.unravel_index(torch.tensor(flat), neq.shape)]
        first = {
            "index": idx,
            "a_bf16_as_float": float(a_bf16[tuple(idx)].float().item()),
            "b_bf16_as_float": float(b_bf16[tuple(idx)].float().item()),
            "pre_round_a": float(a[tuple(idx)].float().item()),
            "pre_round_b": float(b[tuple(idx)].float().item()),
            "pre_round_abs": float((a.float() - b.float()).abs()[tuple(idx)].item()),
        }
    return {
        "torch_equal": bool(torch.equal(a_bf16, b_bf16)),
        "mismatch_count": mismatch_count,
        "numel": int(a_bf16.numel()),
        "first_mismatch": first,
    }


def _load_node_major(payload: dict[str, Any], key: str, rows: torch.Tensor) -> torch.Tensor:
    tensor = payload[key]
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    return tensor.index_select(0, rows.cpu()).contiguous()


def _run_tree_context(
    payload: dict[str, Any],
    *,
    parent: list[int],
    order: list[int],
    device: torch.device,
    output_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    rows = torch.tensor(order, dtype=torch.long)
    remapped_parent = _remap_parent(parent, order)
    tree = Tree(tuple(remapped_parent))
    n_actual = len(order)
    n_pad = padded_nodes(n_actual)
    strict, visible = tree.masks(device, n_pad)
    out, state = launch_tree_gdn_prepared(
        q=_load_node_major(payload, "query_spec", rows).to(device),
        k=_load_node_major(payload, "key_spec", rows).to(device),
        v=_load_node_major(payload, "value_tree", rows).to(device),
        g=_load_node_major(payload, "g_tree", rows).to(device),
        beta=_load_node_major(payload, "beta_tree", rows).to(device),
        raw_a=_load_node_major(payload, "a", rows).to(device),
        raw_b=_load_node_major(payload, "b", rows).to(device),
        A_log=payload["A_log"].to(device).contiguous(),
        dt_bias=payload["dt_bias"].to(device).contiguous(),
        h0=payload["h0"].to(device).contiguous(),
        n_actual=n_actual,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=visible,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    return out[:n_actual].contiguous(), state[:n_actual].contiguous(), remapped_parent


def _select_original_nodes(
    tensor: torch.Tensor,
    *,
    order: list[int],
    original_nodes: list[int],
) -> torch.Tensor:
    old_to_new = {old: new for new, old in enumerate(order)}
    idx = torch.tensor([old_to_new[old] for old in original_nodes], device=tensor.device)
    return tensor.index_select(0, idx).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    payload = torch.load(args.payload, map_location="cpu")
    parent = [int(x) for x in payload["tree_parent"]]
    n_actual = int(payload["n_actual"])
    if n_actual != len(parent):
        raise RuntimeError(f"n_actual={n_actual} but parent has {len(parent)} rows")
    output_scale = float(payload["output_scale"])
    full_order = list(range(n_actual))
    spine_all = _leftmost_spine(parent)
    spine_rows = spine_all[: args.max_depth]
    if not spine_rows:
        raise RuntimeError("empty spine")

    contexts = {
        "original_full": full_order,
        "spine_first_full": _spine_first_order(parent, spine_all),
        "reverse_sibling_dfs_full": _dfs_order(parent, reverse_children=True),
    }
    spine_only_order = spine_rows

    device = torch.device("cuda")
    context_results: dict[str, dict[str, Any]] = {}
    context_outputs: dict[str, torch.Tensor] = {}
    context_states: dict[str, torch.Tensor] = {}
    context_orders: dict[str, list[int]] = {}

    for name, order in contexts.items():
        out, state, remapped_parent = _run_tree_context(
            payload,
            parent=parent,
            order=order,
            device=device,
            output_scale=output_scale,
        )
        context_outputs[name] = out
        context_states[name] = state
        context_orders[name] = order
        context_results[name] = {
            "order": [int(x) for x in order],
            "remapped_parent": [int(x) for x in remapped_parent],
        }

    spine_parent = [-1, *range(len(spine_only_order) - 1)]
    spine_tree = Tree(tuple(spine_parent))
    spine_n_pad = padded_nodes(len(spine_rows))
    strict, visible = spine_tree.masks(device, spine_n_pad)
    spine_row_tensor = torch.tensor(spine_rows, dtype=torch.long)
    spine_out, spine_state = launch_tree_gdn_prepared(
        q=_load_node_major(payload, "query_spec", spine_row_tensor).to(device),
        k=_load_node_major(payload, "key_spec", spine_row_tensor).to(device),
        v=_load_node_major(payload, "value_tree", spine_row_tensor).to(device),
        g=_load_node_major(payload, "g_tree", spine_row_tensor).to(device),
        beta=_load_node_major(payload, "beta_tree", spine_row_tensor).to(device),
        raw_a=_load_node_major(payload, "a", spine_row_tensor).to(device),
        raw_b=_load_node_major(payload, "b", spine_row_tensor).to(device),
        A_log=payload["A_log"].to(device).contiguous(),
        dt_bias=payload["dt_bias"].to(device).contiguous(),
        h0=payload["h0"].to(device).contiguous(),
        n_actual=len(spine_rows),
        n_pad=spine_n_pad,
        strict_mask=strict,
        visible_mask=visible,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    context_outputs["spine_only"] = spine_out[: len(spine_rows)].contiguous()
    context_states["spine_only"] = spine_state[: len(spine_rows)].contiguous()
    context_orders["spine_only"] = spine_rows
    context_results["spine_only"] = {
        "order": [int(x) for x in spine_rows],
        "remapped_parent": [int(x) for x in spine_parent],
    }

    native_rows = torch.tensor(spine_rows, dtype=torch.long)
    native_out, native_state = fused_sigmoid_gating_delta_rule_update(
        A_log=payload["A_log"].to(device).contiguous(),
        a=_load_node_major(payload, "a", native_rows).to(device),
        b=_load_node_major(payload, "b", native_rows).to(device),
        dt_bias=payload["dt_bias"].to(device).contiguous(),
        q=_load_node_major(payload, "query_spec", native_rows).to(device).unsqueeze(0),
        k=_load_node_major(payload, "key_spec", native_rows).to(device).unsqueeze(0),
        v=_load_node_major(payload, "value_spec", native_rows).to(device).unsqueeze(0),
        scale=output_scale,
        initial_state=payload["h0"].to(device).unsqueeze(0).contiguous(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    native_out = native_out.squeeze(0).contiguous()

    serving_out = payload["serving_out"].to(device)
    serving_state = payload["serving_state"].to(device)
    original_spine_out = _select_original_nodes(
        context_outputs["original_full"], order=full_order, original_nodes=spine_rows
    )
    original_spine_state = _select_original_nodes(
        context_states["original_full"], order=full_order, original_nodes=spine_rows
    )

    comparisons: dict[str, Any] = {
        "original_full_replay_vs_payload": {
            "out": _metrics(context_outputs["original_full"], serving_out),
            "state": _metrics(context_states["original_full"], serving_state),
        },
        "original_spine_vs_native_fla": {
            "out": _metrics(original_spine_out, native_out),
            "state": _metrics(original_spine_state, native_state),
            "state_bf16_bank": _bf16_bank_metrics(
                original_spine_state, native_state
            ),
        },
    }

    for name in ("spine_only", "spine_first_full", "reverse_sibling_dfs_full"):
        if name == "spine_only":
            out = context_outputs[name]
            state = context_states[name]
        else:
            out = _select_original_nodes(
                context_outputs[name],
                order=context_orders[name],
                original_nodes=spine_rows,
            )
            state = _select_original_nodes(
                context_states[name],
                order=context_orders[name],
                original_nodes=spine_rows,
            )
        comparisons[f"{name}_spine_vs_original_full_spine"] = {
            "out": _metrics(out, original_spine_out),
            "state": _metrics(state, original_spine_state),
            "state_bf16_bank": _bf16_bank_metrics(state, original_spine_state),
        }
        comparisons[f"{name}_spine_vs_native_fla"] = {
            "out": _metrics(out, native_out),
            "state": _metrics(state, native_state),
            "state_bf16_bank": _bf16_bank_metrics(state, native_state),
        }

    by_depth: list[dict[str, Any]] = []
    for depth, original_node in enumerate(spine_rows):
        row: dict[str, Any] = {"depth": int(depth), "original_node": int(original_node)}
        native_row = native_out[depth : depth + 1]
        orig_row = original_spine_out[depth : depth + 1]
        row["original_vs_native_out_max_abs"] = _max_abs(orig_row, native_row)
        for name in ("spine_only", "spine_first_full", "reverse_sibling_dfs_full"):
            if name == "spine_only":
                out_row = context_outputs[name][depth : depth + 1]
            else:
                out_row = _select_original_nodes(
                    context_outputs[name],
                    order=context_orders[name],
                    original_nodes=[original_node],
                )
            row[f"{name}_vs_original_out_max_abs"] = _max_abs(out_row, orig_row)
            row[f"{name}_vs_native_out_max_abs"] = _max_abs(out_row, native_row)
        by_depth.append(row)

    result = {
        "schema": "fr12.scan_batch_invariance_probe.v1",
        "payload": str(args.payload),
        "native_reference": "vllm.fused_sigmoid_gating_delta_rule_update",
        "our_kernel": "lumo_flywheel_serving.fr10_gdn_tree_kernel.launch_tree_gdn_prepared",
        "tree_parent": parent,
        "spine_rows": [int(x) for x in spine_rows],
        "spine_all_rows": [int(x) for x in spine_all],
        "n_actual": n_actual,
        "contexts": context_results,
        "comparisons": comparisons,
        "by_depth": by_depth,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
