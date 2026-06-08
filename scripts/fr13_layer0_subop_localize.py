#!/usr/bin/env python3
"""Localize layer-0 GDN sub-op drift with mapped tree/native rows.

This reducer compares FR12 subkernel captures where the tree arm has tree rows
and the native arm has packed MTP spine rows. It reports stage-level metrics,
per-depth metrics, and the largest element locations decoded into
head/channel coordinates for 40x128 and 48x128 shaped GDN surfaces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

STAGE_ORDER = [
    "input_hidden",
    "pre_conv",
    "conv1d_out",
    "h0_state_in",
    "gdn_scan_out",
    "gate_z",
    "gate_out",
    "o_proj_out",
]


def _csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _stage(payload: dict[str, Any], name: str) -> torch.Tensor:
    return payload["stages"][name]["tensor"].to(torch.float32)


def _decode_index(shape: list[int], idx: list[int]) -> dict[str, int]:
    out: dict[str, int] = {"row": idx[0] if idx else 0}
    if len(shape) == 2 and len(idx) == 2:
        width = int(shape[1])
        col = int(idx[1])
        out["channel"] = col
        if width == 5120:
            out["head"] = col // 128
            out["head_dim"] = col % 128
        elif width == 10240:
            out["proj_channel"] = col
            out["proj_block"] = col // 5120
            out["proj_block_channel"] = col % 5120
            out["head"] = (col % 5120) // 128
            out["head_dim"] = col % 128
        elif width == 6144:
            out["head"] = col // 128
            out["head_dim"] = col % 128
    elif len(shape) == 3 and len(idx) == 3:
        out["head"] = int(idx[1])
        out["head_dim"] = int(idx[2])
    elif len(shape) == 4 and len(idx) == 4:
        out["head"] = int(idx[1])
        out["value_dim"] = int(idx[2])
        out["key_dim"] = int(idx[3])
    return out


def _topk(a: torch.Tensor, b: torch.Tensor, k: int) -> list[dict[str, Any]]:
    diff = (a - b).abs()
    flat = diff.reshape(-1)
    if flat.numel() == 0:
        return []
    k = min(int(k), int(flat.numel()))
    vals, inds = torch.topk(flat, k=k)
    rows: list[dict[str, Any]] = []
    for val, ind in zip(vals.tolist(), inds.tolist()):
        if float(val) == 0.0:
            break
        idx = [int(x.item()) for x in torch.unravel_index(torch.tensor(ind), diff.shape)]
        loc = _decode_index(list(diff.shape), idx)
        rows.append(
            {
                "index": idx,
                "location": loc,
                "tree": float(a[tuple(idx)].item()),
                "native": float(b[tuple(idx)].item()),
                "abs": float(val),
            }
        )
    return rows


def _metrics(a: torch.Tensor, b: torch.Tensor, *, topk: int) -> dict[str, Any]:
    diff = (a - b).abs()
    return {
        "shape": list(a.shape),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "nonzero": int((diff != 0).sum().item()) if diff.numel() else 0,
        "top": _topk(a, b, topk),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--tree-rows", required=True)
    parser.add_argument("--native-rows", required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tree = _load(args.tree)
    native = _load(args.native)
    tree_rows = _csv_ints(args.tree_rows)
    native_rows = _csv_ints(args.native_rows)
    if len(tree_rows) != len(native_rows):
        raise ValueError("tree/native row lists must have the same length")

    report: dict[str, Any] = {
        "schema": "fr13.layer0_subop_localize.v1",
        "tree": str(args.tree),
        "native": str(args.native),
        "tree_layer_prefix": tree.get("layer_prefix"),
        "native_layer_prefix": native.get("layer_prefix"),
        "tree_rows": tree_rows,
        "native_rows": native_rows,
        "threshold": float(args.threshold),
        "stages": [],
        "first_diverging_stage": None,
    }

    tree_idx = torch.tensor(tree_rows, dtype=torch.long)
    native_idx = torch.tensor(native_rows, dtype=torch.long)
    for name in STAGE_ORDER:
        if name not in tree["stages"] or name not in native["stages"]:
            report["stages"].append({"stage": name, "missing": True})
            continue
        a_all = _stage(tree, name)
        b_all = _stage(native, name)
        if max(tree_rows) >= a_all.shape[0] or max(native_rows) >= b_all.shape[0]:
            report["stages"].append(
                {
                    "stage": name,
                    "row_range_error": {
                        "tree_shape": list(a_all.shape),
                        "native_shape": list(b_all.shape),
                    },
                }
            )
            continue
        a = a_all.index_select(0, tree_idx)
        b = b_all.index_select(0, native_idx)
        if a.shape != b.shape:
            report["stages"].append(
                {
                    "stage": name,
                    "shape_mismatch": [list(a.shape), list(b.shape)],
                    "tree_extra": tree["stages"][name].get("extra"),
                    "native_extra": native["stages"][name].get("extra"),
                }
            )
            if report["first_diverging_stage"] is None:
                report["first_diverging_stage"] = name
            continue
        stage = {
            "stage": name,
            **_metrics(a, b, topk=args.topk),
            "by_depth": [],
            "tree_extra": tree["stages"][name].get("extra"),
            "native_extra": native["stages"][name].get("extra"),
        }
        for depth, (tr, nr) in enumerate(zip(tree_rows, native_rows)):
            row_a = a_all[tr : tr + 1]
            row_b = b_all[nr : nr + 1]
            stage["by_depth"].append(
                {
                    "depth": int(depth),
                    "tree_row": int(tr),
                    "native_row": int(nr),
                    **_metrics(row_a, row_b, topk=min(args.topk, 3)),
                }
            )
        if (
            report["first_diverging_stage"] is None
            and float(stage["max_abs"]) > float(args.threshold)
        ):
            report["first_diverging_stage"] = name
        report["stages"].append(stage)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "first_diverging_stage": report["first_diverging_stage"],
                "stages": [
                    {
                        "stage": s["stage"],
                        "max_abs": s.get("max_abs"),
                        "mean_abs": s.get("mean_abs"),
                        "nonzero": s.get("nonzero"),
                    }
                    for s in report["stages"]
                    if not s.get("missing")
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
