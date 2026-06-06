#!/usr/bin/env python3
"""Replay captured FR12 GDN conv arithmetic variants.

This is boot-free: it compares reconstructed tree-spine conv outputs against a
captured native causal_conv1d_update output using the saved window/tap tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().mean().item())


def _nonzero(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(((a.to(torch.float32) - b.to(torch.float32)) != 0).sum().item())


def _first_mismatch(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any] | None:
    diff = (a.to(torch.float32) - b.to(torch.float32)).abs()
    flat = int(torch.argmax(diff).item())
    val = float(diff.reshape(-1)[flat].item())
    if val == 0.0:
        return None
    idx = [int(x.item()) for x in torch.unravel_index(torch.tensor(flat), diff.shape)]
    return {
        "index": idx,
        "tree": float(a[tuple(idx)].to(torch.float32).item()),
        "native": float(b[tuple(idx)].to(torch.float32).item()),
        "abs": val,
    }


def _order_sum(taps: torch.Tensor, order: str) -> torch.Tensor:
    if order == "0123":
        return taps[:, 0] + taps[:, 1] + taps[:, 2] + taps[:, 3]
    if order == "3210":
        return taps[:, 3] + taps[:, 2] + taps[:, 1] + taps[:, 0]
    if order == "pair01_23":
        return (taps[:, 0] + taps[:, 1]) + (taps[:, 2] + taps[:, 3])
    if order == "pair03_12":
        return (taps[:, 0] + taps[:, 3]) + (taps[:, 1] + taps[:, 2])
    raise ValueError(order)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-capture", required=True, type=Path)
    parser.add_argument("--native-capture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()

    tree = _load(args.tree_capture)
    native = _load(args.native_capture)
    tree_detail = tree["meta"]["tree_conv_detail"]
    native_detail = native["meta"]["native_conv_detail"]

    path0 = [int(x) for x in tree_detail["path0_nodes"].reshape(-1).tolist()]
    depth = min(args.depth, len(path0), native["stages"]["conv1d_out"]["tensor"].shape[0])
    path0 = path0[:depth]
    native_out = native["stages"]["conv1d_out"]["tensor"][:depth].to(torch.float32)
    captured_tree_out = tree["stages"]["conv1d_out"]["tensor"][path0].to(torch.float32)

    variants = []
    for tap_name, tap_key in [
        ("fp32_product", "tap_products_fp32_path0"),
        ("bf16_product", "tap_products_bf16_path0"),
    ]:
        taps = tree_detail[tap_key][:depth].to(torch.float32)
        for order in ["0123", "3210", "pair01_23", "pair03_12"]:
            acc = _order_sum(taps, order)
            for silu_input in ["fp32", "bf16"]:
                silu_arg = acc if silu_input == "fp32" else acc.to(torch.bfloat16)
                out = torch.nn.functional.silu(silu_arg.to(torch.float32)).to(
                    torch.bfloat16
                )
                variants.append(
                    {
                        "tap_product": tap_name,
                        "accumulation_order": order,
                        "silu_input": silu_input,
                        "store_dtype": "bf16",
                        "max_abs": _max_abs(out, native_out),
                        "mean_abs": _mean_abs(out, native_out),
                        "nonzero": _nonzero(out, native_out),
                        "captured_tree_max_abs": _max_abs(out, captured_tree_out),
                        "first_mismatch": _first_mismatch(out, native_out),
                    }
                )

    best = min(variants, key=lambda row: (row["max_abs"], row["mean_abs"], row["nonzero"]))
    result = {
        "schema": "fr12.conv_rounding_replay.v1",
        "tree_capture": str(args.tree_capture),
        "native_capture": str(args.native_capture),
        "tree_rows": path0,
        "native_rows": list(range(depth)),
        "window_max_abs": _max_abs(
            tree_detail["window_path0"][:depth], native_detail["window"][:depth]
        ),
        "tap_products_fp32_max_abs": _max_abs(
            tree_detail["tap_products_fp32_path0"][:depth],
            native_detail["tap_products_fp32"][:depth],
        ),
        "tap_products_bf16_max_abs": _max_abs(
            tree_detail["tap_products_bf16_path0"][:depth],
            native_detail["tap_products_bf16"][:depth],
        ),
        "captured_tree_vs_native": {
            "max_abs": _max_abs(captured_tree_out, native_out),
            "mean_abs": _mean_abs(captured_tree_out, native_out),
            "nonzero": _nonzero(captured_tree_out, native_out),
            "first_mismatch": _first_mismatch(captured_tree_out, native_out),
        },
        "best_variant": best,
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
