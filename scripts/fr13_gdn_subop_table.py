#!/usr/bin/env python3
"""Reduce FR13 spine-only GDN sub-op captures into a top-down drift table."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


FULL_ATTN_STAGES = [
    "input_hidden",
    "qkv_proj",
    "q_norm_out",
    "k_norm_out",
    "v",
    "positions",
    "q_after_rope",
    "k_after_rope",
    "attn_out_raw",
    "gate_sigmoid",
    "attn_out_gated",
    "o_proj_out",
]
GDN_STAGES = [
    "input_hidden",
    "pre_conv",
    "conv1d_out",
    "h0_state_in",
    "gdn_scan_out",
    "gate_z",
    "gate_out",
    "o_proj_out",
]


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _layer_idx(payload: dict[str, Any]) -> int:
    prefix = str(payload.get("layer_prefix", ""))
    match = re.search(r"layers\.(\d+)\.", prefix)
    if not match:
        raise ValueError(f"cannot parse layer index from {prefix!r}")
    return int(match.group(1))


def _stage(payload: dict[str, Any], name: str) -> torch.Tensor | None:
    item = payload.get("stages", {}).get(name)
    if item is None:
        return None
    return item["tensor"]


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    if a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        a = a.reshape(-1)
        b = b.reshape(-1)
    d = (a.float() - b.float()).abs()
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "torch_equal": bool(torch.equal(a, b)),
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
        "nonzero": int((d != 0).sum().item()) if d.numel() else 0,
    }


def _by_row(a: torch.Tensor, b: torch.Tensor) -> list[dict[str, Any]]:
    if a.ndim == 0 or b.ndim == 0:
        return []
    rows = min(int(a.shape[0]), int(b.shape[0]))
    out = []
    for row in range(rows):
        stats = _metrics(a[row], b[row])
        out.append({"row": int(row), **stats})
    return out


def _captures_by_layer(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out = {}
    for path in paths:
        payload = _load(path)
        out[_layer_idx(payload)] = payload | {"_path": str(path)}
    return out


def _compare_layers(
    tree_caps: dict[int, dict[str, Any]],
    native_caps: dict[int, dict[str, Any]],
    stages: list[str],
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for layer in sorted(set(tree_caps) & set(native_caps)):
        tcap = tree_caps[layer]
        ncap = native_caps[layer]
        stage_rows = []
        first = None
        for stage in stages:
            t = _stage(tcap, stage)
            n = _stage(ncap, stage)
            if t is None or n is None:
                continue
            stats = _metrics(t, n)
            row = {
                "stage": stage,
                **stats,
                "by_row": _by_row(t, n),
            }
            stage_rows.append(row)
            if first is None and stats["max_abs"] > threshold:
                first = {"stage": stage, "max_abs": stats["max_abs"]}
        rows.append(
            {
                "layer": int(layer),
                "tree_capture": tcap["_path"],
                "native_capture": ncap["_path"],
                "first_nonzero_stage": first,
                "stages": stage_rows,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-full", nargs="*", type=Path, default=[])
    parser.add_argument("--native-full", nargs="*", type=Path, default=[])
    parser.add_argument("--tree-gdn", nargs="*", type=Path, default=[])
    parser.add_argument("--native-gdn", nargs="*", type=Path, default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.00390625)
    args = parser.parse_args()

    full_layers = _compare_layers(
        _captures_by_layer(args.tree_full),
        _captures_by_layer(args.native_full),
        FULL_ATTN_STAGES,
        args.threshold,
    )
    gdn_layers = _compare_layers(
        _captures_by_layer(args.tree_gdn),
        _captures_by_layer(args.native_gdn),
        GDN_STAGES,
        args.threshold,
    )
    all_layers = sorted(full_layers + gdn_layers, key=lambda row: row["layer"])
    first = next((row for row in all_layers if row["first_nonzero_stage"]), None)
    payload = {
        "schema": "fr13.gdn_subop_table.v1",
        "threshold": float(args.threshold),
        "first_nonzero_layer": None
        if first is None
        else {
            "layer": first["layer"],
            **first["first_nonzero_stage"],
        },
        "layers": all_layers,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "first_nonzero_layer": payload["first_nonzero_layer"],
                "layers": [
                    {
                        "layer": row["layer"],
                        "first_nonzero_stage": row["first_nonzero_stage"],
                        "stage_max_abs": {
                            stage["stage"]: stage["max_abs"]
                            for stage in row["stages"]
                        },
                    }
                    for row in all_layers
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
