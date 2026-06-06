#!/usr/bin/env python3
"""Compare our tree scan against native vLLM scan on one linear spine.

This is a boot-free GPU probe. It uses a captured FR10 tree-GDN payload, extracts
the leftmost spine, runs our tree kernel with only those rows, and compares to
vLLM's native speculative recurrent update on the same row sequence.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


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
        "tree": float(a[tuple(idx_i)].float().item()),
        "native": float(b[tuple(idx_i)].float().item()),
        "abs": val,
    }


def _load_node_major(payload: dict[str, Any], key: str, rows: torch.Tensor) -> torch.Tensor:
    tensor = payload[key]
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    return tensor.index_select(0, rows.cpu()).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--fla-bf16-boundaries", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    payload = torch.load(args.payload, map_location="cpu")
    parent = [int(x) for x in payload["tree_parent"]]
    spine_rows = _leftmost_spine(parent)[: args.max_depth]
    if not spine_rows:
        raise RuntimeError("empty spine")

    device = torch.device("cuda")
    row_index = torch.tensor(spine_rows, dtype=torch.long)
    n = int(row_index.numel())
    n_pad = padded_nodes(n)
    spine_tree = Tree(tuple([-1, *range(0, n - 1)]))
    strict, visible = spine_tree.masks(device, n_pad)

    q = _load_node_major(payload, "query_spec", row_index).to(device)
    k = _load_node_major(payload, "key_spec", row_index).to(device)
    value_spec = _load_node_major(payload, "value_spec", row_index).to(device)
    value_tree = _load_node_major(payload, "value_tree", row_index).to(device)
    g_tree = _load_node_major(payload, "g_tree", row_index).to(device)
    beta_tree = _load_node_major(payload, "beta_tree", row_index).to(device)
    a = _load_node_major(payload, "a", row_index).to(device)
    b = _load_node_major(payload, "b", row_index).to(device)
    h0 = payload["h0"].to(device).contiguous()
    A_log = payload["A_log"].to(device).contiguous()
    dt_bias = payload["dt_bias"].to(device).contiguous()
    output_scale = float(payload["output_scale"])

    tree_out, tree_state = launch_tree_gdn_prepared(
        q=q.contiguous(),
        k=k.contiguous(),
        v=value_tree.contiguous(),
        g=g_tree.contiguous(),
        beta=beta_tree.contiguous(),
        h0=h0,
        n_actual=n,
        n_pad=n_pad,
        strict_mask=strict,
        visible_mask=visible,
        output_scale=output_scale,
        use_qk_l2norm_in_kernel=True,
        fla_bf16_boundaries=args.fla_bf16_boundaries,
    )

    native_out, native_state = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        q=q.unsqueeze(0).contiguous(),
        k=k.unsqueeze(0).contiguous(),
        v=value_spec.unsqueeze(0).contiguous(),
        scale=output_scale,
        initial_state=h0.unsqueeze(0).contiguous(),
        inplace_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    native_out = native_out.squeeze(0)

    depth_rows = []
    for depth in range(n):
        depth_rows.append(
            {
                "depth": int(depth),
                "source_row": int(spine_rows[depth]),
                "out_max_abs": _max_abs(tree_out[depth], native_out[depth]),
                "out_mean_abs": _mean_abs(tree_out[depth], native_out[depth]),
                "state_max_abs": _max_abs(tree_state[depth], native_state[depth]),
                "state_mean_abs": _mean_abs(tree_state[depth], native_state[depth]),
            }
        )

    result = {
        "schema": "fr12.spine_scan_rounding_probe.v1",
        "payload": str(args.payload),
        "native_reference": "vllm.fused_sigmoid_gating_delta_rule_update",
        "our_kernel": "lumo_flywheel_serving.fr10_gdn_tree_kernel.launch_tree_gdn_prepared",
        "fla_bf16_boundaries": bool(args.fla_bf16_boundaries),
        "spine_rows": spine_rows,
        "n": n,
        "n_pad": n_pad,
        "out_max_abs": _max_abs(tree_out[:n], native_out[:n]),
        "state_max_abs": _max_abs(tree_state[:n], native_state[:n]),
        "out_first_mismatch": _first_mismatch(tree_out[:n], native_out[:n]),
        "state_first_mismatch": _first_mismatch(tree_state[:n], native_state[:n]),
        "by_depth": depth_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
