#!/usr/bin/env python3
"""Compare FR12 layer-3 full-attention captures on aligned spine rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


TREE_PARENT = [-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]
TREE_SPINE_ROWS = [0, 1, 2, 4, 6]
NATIVE_SPINE_ROWS = [0, 1, 2, 3, 4]
STAGE_ORDER = [
    "input_hidden",
    "qkv_proj",
    "q_norm_out",
    "k_norm_out",
    "v",
    "position_ids",
    "q_after_rope",
    "k_after_rope",
    "attn_out_raw",
    "gate_sigmoid",
    "attn_out_gated",
    "o_proj_out",
]


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _stage(capture: dict[str, Any], name: str) -> torch.Tensor | None:
    item = capture.get("stages", {}).get(name)
    if item is None:
        return None
    return item["tensor"]


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().mean().item())


def _depths(parent: list[int]) -> list[int]:
    out = []
    for node in range(len(parent)):
        depth = 0
        cur = parent[node]
        while cur >= 0:
            depth += 1
            cur = parent[cur]
        out.append(depth)
    return out


def _token_positions(pos: torch.Tensor) -> torch.Tensor:
    return pos[0] if pos.ndim == 2 else pos.reshape(-1)


def _row_compare(
    tree_t: torch.Tensor,
    native_t: torch.Tensor,
    tree_rows: list[int],
    native_rows: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for depth, (tree_row, native_row) in enumerate(zip(tree_rows, native_rows)):
        rows.append(
            {
                "depth": int(depth),
                "tree_row": int(tree_row),
                "native_row": int(native_row),
                "max_abs": _max_abs(tree_t[tree_row], native_t[native_row]),
                "mean_abs": _mean_abs(tree_t[tree_row], native_t[native_row]),
            }
        )
    return rows


def _position_compare(
    tree_pos: torch.Tensor,
    native_pos: torch.Tensor,
    tree_rows: list[int],
    native_rows: list[int],
    parent: list[int],
) -> dict[str, Any]:
    tree_tok = _token_positions(tree_pos)
    native_tok = _token_positions(native_pos)
    by_depth = []
    for depth, (tree_row, native_row) in enumerate(zip(tree_rows, native_rows)):
        tree_value = int(tree_tok[tree_row].item())
        native_value = int(native_tok[native_row].item())
        by_depth.append(
            {
                "depth": int(depth),
                "tree_row": int(tree_row),
                "native_row": int(native_row),
                "tree_position": tree_value,
                "native_position": native_value,
                "delta": int(tree_value - native_value),
                "max_abs": float(abs(tree_value - native_value)),
                "mean_abs": float(abs(tree_value - native_value)),
            }
        )
    depths = _depths(parent)
    base = int(tree_tok[0].item()) if tree_tok.numel() else 0
    expected = [base + int(depths[row]) for row in range(min(len(parent), tree_tok.numel()))]
    actual = [int(x) for x in tree_tok[: len(expected)].tolist()]
    return {
        "stage": "position_ids",
        "tree_shape": list(tree_pos.shape),
        "native_shape": list(native_pos.shape),
        "max_abs": float(max((row["max_abs"] for row in by_depth), default=0.0)),
        "mean_abs_max_depth": float(
            max((row["mean_abs"] for row in by_depth), default=0.0)
        ),
        "by_depth": by_depth,
        "tree_depth_expected_positions": expected,
        "tree_actual_token_positions": actual,
        "tree_depth_positions_match": actual == expected,
        "tree_parent": parent,
        "tree_depths": depths[: len(expected)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-capture", required=True, type=Path)
    parser.add_argument("--native-capture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--tree-rows", default=",".join(str(x) for x in TREE_SPINE_ROWS))
    parser.add_argument(
        "--native-rows", default=",".join(str(x) for x in NATIVE_SPINE_ROWS)
    )
    parser.add_argument("--tree-parent", default=",".join(str(x) for x in TREE_PARENT))
    args = parser.parse_args()

    tree_cap = _load(args.tree_capture)
    native_cap = _load(args.native_capture)
    tree_rows = [int(x) for x in args.tree_rows.split(",") if x.strip()]
    native_rows = [int(x) for x in args.native_rows.split(",") if x.strip()]
    parent = [int(x) for x in args.tree_parent.split(",") if x.strip()]

    rows: list[dict[str, Any]] = []
    for stage in STAGE_ORDER:
        if stage == "position_ids":
            tree_pos = _stage(tree_cap, "positions")
            native_pos = _stage(native_cap, "positions")
            if tree_pos is not None and native_pos is not None:
                rows.append(
                    _position_compare(tree_pos, native_pos, tree_rows, native_rows, parent)
                )
            continue
        tree_t = _stage(tree_cap, stage)
        native_t = _stage(native_cap, stage)
        if tree_t is None or native_t is None:
            continue
        by_depth = _row_compare(tree_t, native_t, tree_rows, native_rows)
        rows.append(
            {
                "stage": stage,
                "tree_shape": list(tree_t.shape),
                "native_shape": list(native_t.shape),
                "max_abs": float(max((row["max_abs"] for row in by_depth), default=0.0)),
                "mean_abs_max_depth": float(
                    max((row["mean_abs"] for row in by_depth), default=0.0)
                ),
                "by_depth": by_depth,
            }
        )

    first = next((row for row in rows if row["max_abs"] > args.threshold), None)
    if first is None:
        first = {"stage": "none", "max_abs": 0.0}

    out = {
        "schema": "fr12.full_attn_l3_spine_compare.v1",
        "tree_capture": str(args.tree_capture),
        "native_capture": str(args.native_capture),
        "tree_layer_prefix": tree_cap.get("layer_prefix"),
        "native_layer_prefix": native_cap.get("layer_prefix"),
        "tree_rows": tree_rows,
        "native_rows": native_rows,
        "stages": rows,
        "first_origin": first,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "first_origin": {
                    "stage": first["stage"],
                    "max_abs": first["max_abs"],
                },
                "stages": [
                    {
                        "stage": row["stage"],
                        "max_abs": row["max_abs"],
                        "mean_abs_max_depth": row["mean_abs_max_depth"],
                    }
                    for row in rows
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
