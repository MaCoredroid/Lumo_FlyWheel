#!/usr/bin/env python3
"""Replay clean spine GDN conv across a spread of layers with PTX-style SiLU.

The live native causal_conv1d_update kernel exposes only final conv output, not
its internal ex2/denom/div registers.  This script therefore checks two things:

1. Captured tree/native operand alignment plus reconstructed PTX intermediates
   for diagnosis. Native CUDA internal ex2/div registers are not captured.
2. The authoritative check: final bf16 output from that exact sequence vs the
   native captured conv1d_out, restricted to clean-input layers when deciding
   whether the conv kernel itself matches.

This is boot-free and does not call native causal_conv1d_update.
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


STAGES = ["acc", "neg", "arg", "ex2", "denom", "div", "out_bf16"]


@triton.jit
def _ptx_silu_kernel(
    taps,
    acc_out,
    neg_out,
    arg_out,
    ex2_out,
    denom_out,
    div_out,
    bf16_out,
    n: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x0 = tl.load(taps + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(taps + n + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(taps + 2 * n + offs, mask=mask, other=0.0).to(tl.float32)
    x3 = tl.load(taps + 3 * n + offs, mask=mask, other=0.0).to(tl.float32)
    acc0 = x0 + 0.0
    acc1 = acc0 + x1
    acc2 = acc1 + x2
    acc = acc2 + x3
    neg = 0.0 - acc
    arg = neg * 1.4426950216293335
    ex2 = tl.exp2(arg)
    denom = ex2 + 1.0
    div = acc / denom
    tl.store(acc_out + offs, acc, mask=mask)
    tl.store(neg_out + offs, neg, mask=mask)
    tl.store(arg_out + offs, arg, mask=mask)
    tl.store(ex2_out + offs, ex2, mask=mask)
    tl.store(denom_out + offs, denom, mask=mask)
    tl.store(div_out + offs, div, mask=mask)
    tl.store(bf16_out + offs, div, mask=mask)


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
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "torch_equal": bool(torch.equal(a, b)),
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


def _by_layer(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out = {}
    for path in paths:
        payload = _load(path)
        out[_layer_idx(payload)] = payload | {"_path": str(path)}
    return out


def _detail(payload: dict[str, Any], key: str) -> dict[str, Any]:
    meta = payload.get("meta", {})
    if key not in meta:
        raise KeyError(f"{payload.get('_path')}: missing {key}")
    return meta[key]


def _run_ptx_sequence(taps_bf16_products: torch.Tensor) -> dict[str, torch.Tensor]:
    taps = taps_bf16_products.contiguous()
    if taps.ndim != 3 or taps.shape[1] != 4:
        raise ValueError(f"expected [rows,4,cols] taps, got {tuple(taps.shape)}")
    rows, _, cols = [int(x) for x in taps.shape]
    n = rows * cols
    flat_taps = taps.to(torch.float32).permute(1, 0, 2).contiguous().reshape(-1).cuda()
    outs = {
        "acc": torch.empty(n, device="cuda", dtype=torch.float32),
        "neg": torch.empty(n, device="cuda", dtype=torch.float32),
        "arg": torch.empty(n, device="cuda", dtype=torch.float32),
        "ex2": torch.empty(n, device="cuda", dtype=torch.float32),
        "denom": torch.empty(n, device="cuda", dtype=torch.float32),
        "div": torch.empty(n, device="cuda", dtype=torch.float32),
        "out_bf16": torch.empty(n, device="cuda", dtype=torch.bfloat16),
    }
    _ptx_silu_kernel[(triton.cdiv(n, 256),)](
        flat_taps,
        outs["acc"],
        outs["neg"],
        outs["arg"],
        outs["ex2"],
        outs["denom"],
        outs["div"],
        outs["out_bf16"],
        n,
        BLOCK=256,
    )
    torch.cuda.synchronize()
    return {
        name: value.reshape(rows, cols).detach().cpu()
        for name, value in outs.items()
    }


def _alignment(tree_detail: dict[str, Any], native_detail: dict[str, Any], rows: int) -> dict[str, Any]:
    pairs = {
        "pre_conv": ("pre_conv_path0", "pre_conv_rows"),
        "window": ("window_path0", "window"),
        "tap_products_bf16": ("tap_products_bf16_path0", "tap_products_bf16"),
        "tap_products_fp32": ("tap_products_fp32_path0", "tap_products_fp32"),
    }
    out = {}
    for name, (tk, nk) in pairs.items():
        if tk in tree_detail and nk in native_detail:
            out[name] = _metrics(tree_detail[tk][:rows], native_detail[nk][:rows])
    return out


def _replay_layer(tree_payload: dict[str, Any], native_payload: dict[str, Any]) -> dict[str, Any]:
    layer = _layer_idx(tree_payload)
    tree_detail = _detail(tree_payload, "tree_conv_detail")
    native_detail = _detail(native_payload, "native_conv_detail")
    path0 = tree_detail["path0_nodes"].reshape(-1).to(torch.long)
    native_conv = native_payload["stages"]["conv1d_out"]["tensor"].to(torch.bfloat16)
    rows = min(int(path0.numel()), int(native_conv.shape[0]))
    path0 = path0[:rows]
    tree_conv_all = tree_payload["stages"]["conv1d_out"]["tensor"].to(torch.bfloat16)
    captured_tree = tree_conv_all.index_select(0, path0)
    native_target = native_conv[:rows]

    tree_taps = tree_detail["tap_products_bf16_path0"][:rows]
    native_taps = native_detail["tap_products_bf16"][:rows]
    tree_seq = _run_ptx_sequence(tree_taps)
    native_seq = _run_ptx_sequence(native_taps)
    alignment = _alignment(tree_detail, native_detail, rows)
    intermediate = {
        name: _metrics(tree_seq[name], native_seq[name])
        for name in STAGES
    }
    return {
        "layer": int(layer),
        "tree_capture": tree_payload["_path"],
        "native_capture": native_payload["_path"],
        "rows": int(rows),
        "clean_input": bool(alignment.get("pre_conv", {}).get("max_abs") == 0.0),
        "alignment": alignment,
        "captured_tree_vs_native": _metrics(captured_tree, native_target),
        "ptx_tree_vs_native_output": _metrics(tree_seq["out_bf16"], native_target),
        "ptx_native_operand_vs_native_output": _metrics(
            native_seq["out_bf16"], native_target
        ),
        "intermediate_tree_operand_vs_native_operand": intermediate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", nargs="+", type=Path, required=True)
    parser.add_argument("--native", nargs="+", type=Path, required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tree_layers = _by_layer(args.tree)
    native_layers = _by_layer(args.native)
    wanted = None
    if args.layers.strip():
        wanted = {int(x.strip()) for x in args.layers.split(",") if x.strip()}
    layers = sorted(set(tree_layers) & set(native_layers))
    if wanted is not None:
        layers = [layer for layer in layers if layer in wanted]
    rows = [_replay_layer(tree_layers[layer], native_layers[layer]) for layer in layers]
    aggregate = {
        "layers": [row["layer"] for row in rows],
        "all_clean_input": all(row["clean_input"] for row in rows),
        "clean_layers": [row["layer"] for row in rows if row["clean_input"]],
        "contaminated_layers": [
            row["layer"] for row in rows if not row["clean_input"]
        ],
        "max_ptx_tree_vs_native_output": max(
            (row["ptx_tree_vs_native_output"]["max_abs"] for row in rows),
            default=0.0,
        ),
        "max_clean_ptx_tree_vs_native_output": max(
            (
                row["ptx_tree_vs_native_output"]["max_abs"]
                for row in rows
                if row["clean_input"]
            ),
            default=0.0,
        ),
        "nonzero_clean_ptx_tree_vs_native_output": sum(
            row["ptx_tree_vs_native_output"]["nonzero"]
            for row in rows
            if row["clean_input"]
        ),
        "nonzero_ptx_tree_vs_native_output": sum(
            row["ptx_tree_vs_native_output"]["nonzero"] for row in rows
        ),
        "max_captured_tree_vs_native": max(
            (row["captured_tree_vs_native"]["max_abs"] for row in rows),
            default=0.0,
        ),
    }
    payload = {
        "schema": "fr13.conv_spread_ex2_replay.v1",
        "source": "PTX-style bf16 taps -> f32 adds -> -acc*0x3FB8AA3B -> exp2 -> +1 -> div -> bf16 store",
        "tree_captures": [str(p) for p in args.tree],
        "native_captures": [str(p) for p in args.native],
        "aggregate": aggregate,
        "layers": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    for row in rows:
        print(
            json.dumps(
                {
                    "layer": row["layer"],
                    "clean_input": row["clean_input"],
                    "captured": row["captured_tree_vs_native"],
                    "ptx": row["ptx_tree_vs_native_output"],
                    "ptx_native_operand": row[
                        "ptx_native_operand_vs_native_output"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if aggregate["max_ptx_tree_vs_native_output"] == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
