#!/usr/bin/env python3
"""Replay clean-input GDN conv with Triton's native ex2.approx SiLU path.

This is boot-free: it does not call vLLM's native causal_conv1d_update.  It
uses captured bf16 tap products, performs the same fp32 accumulation order, and
uses a small Triton kernel for the activation so `tl.exp` lowers to NVIDIA
`ex2.approx.f32`, matching the PTX seen in causal_conv1d_update.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_ex2_silu_kernel(taps, out, n_cols: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_cols
    x0 = tl.load(taps + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(taps + n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(taps + 2 * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(taps + 3 * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    acc = ((x0 + x1) + x2) + x3
    y = acc / (1.0 + tl.exp(0.0 - acc))
    tl.store(out + offs, y, mask=mask)


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _layer_idx(payload: dict[str, Any]) -> int:
    match = re.search(r"layers\.(\d+)\.", str(payload.get("layer_prefix", "")))
    if not match:
        raise ValueError(f"cannot parse layer from {payload.get('layer_prefix')!r}")
    return int(match.group(1))


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    d = (a.float() - b.float()).abs()
    out: dict[str, Any] = {
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
        "nonzero": int((d != 0).sum().item()) if d.numel() else 0,
    }
    if out["nonzero"]:
        flat = int(torch.argmax(d).item())
        idx = [int(x.item()) for x in torch.unravel_index(torch.tensor(flat), d.shape)]
        out["first_mismatch"] = {
            "index": idx,
            "lhs": float(a[tuple(idx)].float().item()),
            "rhs": float(b[tuple(idx)].float().item()),
            "abs": float(d[tuple(idx)].item()),
        }
    else:
        out["first_mismatch"] = None
    return out


def _clean_alignment(
    tree_detail: dict[str, Any],
    native_detail: dict[str, Any],
    rows: int,
) -> dict[str, dict[str, Any]]:
    pairs = {
        "pre_conv_path0": ("pre_conv_path0", "pre_conv_rows"),
        "window_path0": ("window_path0", "window"),
        "tap_products_bf16_path0": ("tap_products_bf16_path0", "tap_products_bf16"),
        "tap_products_fp32_path0": ("tap_products_fp32_path0", "tap_products_fp32"),
    }
    return {
        name: _metrics(tree_detail[tree_key][:rows], native_detail[native_key][:rows])
        for name, (tree_key, native_key) in pairs.items()
        if tree_key in tree_detail and native_key in native_detail
    }


def _replay_one(tree_path: Path, native_path: Path) -> dict[str, Any]:
    tree = _load(tree_path)
    native = _load(native_path)
    tree_detail = tree["meta"]["tree_conv_detail"]
    native_detail = native["meta"]["native_conv_detail"]
    path0_nodes = tree_detail["path0_nodes"].reshape(-1).to(torch.long)
    rows = min(
        int(path0_nodes.numel()),
        int(native["stages"]["conv1d_out"]["tensor"].shape[0]),
    )
    path0_nodes = path0_nodes[:rows]

    taps = tree_detail["tap_products_bf16_path0"][:rows].contiguous()
    if taps.ndim != 3 or taps.shape[1] != 4:
        raise ValueError(f"expected [rows,4,cols] taps, got {tuple(taps.shape)}")
    rows_i, _, cols_i = [int(x) for x in taps.shape]
    flat_taps = taps.to(torch.float32).reshape(rows_i, 4 * cols_i).cuda()
    out = torch.empty((rows_i, cols_i), device="cuda", dtype=torch.bfloat16)
    for row in range(rows_i):
        _conv_ex2_silu_kernel[(triton.cdiv(cols_i, 256),)](
            flat_taps[row],
            out[row],
            cols_i,
            BLOCK=256,
        )
    replay = out.cpu()

    tree_conv = tree["stages"]["conv1d_out"]["tensor"]
    if tree_conv.shape[0] >= int(path0_nodes.max().item()) + 1:
        captured_tree = tree_conv.index_select(0, path0_nodes)
    else:
        captured_tree = tree_conv[:rows]
    native_target = native["stages"]["conv1d_out"]["tensor"][:rows].to(torch.bfloat16)

    alignment = _clean_alignment(tree_detail, native_detail, rows)
    clean_input = alignment.get("pre_conv_path0", {}).get("max_abs") == 0.0
    return {
        "layer": _layer_idx(tree),
        "tree_capture": str(tree_path),
        "native_capture": str(native_path),
        "rows": rows,
        "clean_input": bool(clean_input),
        "input_alignment": alignment,
        "captured_tree_vs_native": _metrics(captured_tree, native_target),
        "triton_ex2_silu_vs_native": _metrics(replay, native_target),
        "triton_ex2_silu_vs_captured_tree": _metrics(replay, captured_tree),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", nargs="+", type=Path, required=True)
    parser.add_argument("--native", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.tree) != len(args.native):
        raise SystemExit("--tree and --native must have the same length")

    layers = [_replay_one(t, n) for t, n in zip(args.tree, args.native)]
    clean_layers = [row for row in layers if row["clean_input"]]
    if clean_layers:
        lhs = torch.cat(
            [
                torch.tensor([row["triton_ex2_silu_vs_native"]["max_abs"]])
                for row in clean_layers
            ]
        )
        aggregate_max = float(lhs.max().item())
        aggregate_nonzero = int(
            sum(row["triton_ex2_silu_vs_native"]["nonzero"] for row in clean_layers)
        )
    else:
        aggregate_max = None
        aggregate_nonzero = None
    payload = {
        "schema": "fr13.ex2_silu_replay.v1",
        "source": "Triton tl.exp lowering to NVIDIA ex2.approx.f32; no native causal_conv1d_update call",
        "layers": layers,
        "clean_aggregate": {
            "layers": [row["layer"] for row in clean_layers],
            "max_abs": aggregate_max,
            "nonzero": aggregate_nonzero,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["clean_aggregate"], indent=2, sort_keys=True))
    for row in layers:
        print(
            json.dumps(
                {
                    "layer": row["layer"],
                    "clean_input": row["clean_input"],
                    "captured": row["captured_tree_vs_native"],
                    "triton_ex2": row["triton_ex2_silu_vs_native"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
