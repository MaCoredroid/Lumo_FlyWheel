#!/usr/bin/env python3
"""Compare FR12 layer-0 GDN subkernel captures on aligned spine rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


TREE_SPINE_NODES = [0, 1, 3, 5, 7]
SPEC_ORDER_STAGES = {"conv1d_out", "gdn_scan_out"}
FULL_ROW_STAGES = {"gate_out", "o_proj_out"}


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _row_start(counts: list[int], req: int) -> int:
    return int(sum(int(x) for x in counts[:req]))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().mean().item())


def _stage_tensor(capture: dict[str, Any], stage: str) -> torch.Tensor:
    try:
        return capture["stages"][stage]["tensor"]
    except KeyError as exc:
        raise KeyError(f"capture {capture.get('call_path')} missing stage {stage}") from exc


def _spec_indices(
    capture: dict[str, Any], stage: str, hidden_rows: list[int], target_rows: list[int]
) -> list[int]:
    extra = capture["stages"].get(stage, {}).get("extra", {})
    spec_token_indx = extra.get("spec_token_indx")
    if spec_token_indx:
        pos_by_hidden = {int(row): int(i) for i, row in enumerate(spec_token_indx)}
        return [pos_by_hidden[int(row)] for row in hidden_rows]
    return [int(row) for row in target_rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-capture", required=True, type=Path)
    parser.add_argument("--native-capture", required=True, type=Path)
    parser.add_argument("--tree-logits", required=True, type=Path)
    parser.add_argument("--native-logits", required=True, type=Path)
    parser.add_argument("--tree-req", type=int, default=0)
    parser.add_argument("--native-req", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="First-origin threshold.",
    )
    args = parser.parse_args()

    tree_cap = _load(args.tree_capture)
    native_cap = _load(args.native_capture)
    tree_logits = _load(args.tree_logits)
    native_logits = _load(args.native_logits)

    tree_counts = [int(x) for x in tree_logits["num_draft_tokens"]]
    native_counts = [int(x) for x in native_logits["num_draft_tokens"]]
    tree_start = _row_start(tree_counts, args.tree_req)
    native_start = _row_start(native_counts, args.native_req)
    tree_target_rows = [tree_start + node for node in TREE_SPINE_NODES]
    native_target_rows = [native_start + depth for depth in range(5)]
    tree_hidden_rows = [
        int(tree_logits["target_logits_indices"][row].item())
        for row in tree_target_rows
    ]
    native_hidden_rows = [
        int(native_logits["target_logits_indices"][row].item())
        for row in native_target_rows
    ]
    tree_tokens = [
        int(tree_logits["draft_token_ids"][row].item()) for row in tree_target_rows
    ]
    native_tokens = [
        int(native_logits["draft_token_ids"][row].item()) for row in native_target_rows
    ]

    rows = []
    for stage in ["conv1d_out", "gdn_scan_out", "gate_out", "o_proj_out"]:
        tree_t = _stage_tensor(tree_cap, stage)
        native_t = _stage_tensor(native_cap, stage)
        if stage in SPEC_ORDER_STAGES:
            tree_rows = _spec_indices(tree_cap, stage, tree_hidden_rows, tree_target_rows)
            native_rows = _spec_indices(
                native_cap, stage, native_hidden_rows, native_target_rows
            )
        elif stage in FULL_ROW_STAGES:
            tree_rows = tree_hidden_rows
            native_rows = native_hidden_rows
        else:
            raise AssertionError(stage)
        depth = []
        for d, (tr, nr) in enumerate(zip(tree_rows, native_rows)):
            depth.append(
                {
                    "depth": int(d),
                    "tree_row": int(tr),
                    "native_row": int(nr),
                    "max_abs": _max_abs(tree_t[tr], native_t[nr]),
                    "mean_abs": _mean_abs(tree_t[tr], native_t[nr]),
                }
            )
        rows.append(
            {
                "stage": stage,
                "tree_shape": list(tree_t.shape),
                "native_shape": list(native_t.shape),
                "max_abs": float(max(x["max_abs"] for x in depth)),
                "mean_abs_max_depth": float(max(x["mean_abs"] for x in depth)),
                "by_depth": depth,
                "tree_extra": tree_cap["stages"].get(stage, {}).get("extra", {}),
                "native_extra": native_cap["stages"].get(stage, {}).get("extra", {}),
            }
        )

    first = next((row for row in rows if row["max_abs"] > args.threshold), None)
    if first is None:
        first = {"stage": "none", "max_abs": 0.0}

    out = {
        "schema": "fr12.gdn_l0_subkernel_spine_compare.v1",
        "tree_capture": str(args.tree_capture),
        "native_capture": str(args.native_capture),
        "tree_logits": str(args.tree_logits),
        "native_logits": str(args.native_logits),
        "tree_req": int(args.tree_req),
        "native_req": int(args.native_req),
        "tree_target_rows": tree_target_rows,
        "native_target_rows": native_target_rows,
        "tree_hidden_rows": tree_hidden_rows,
        "native_hidden_rows": native_hidden_rows,
        "tree_spine_tokens": tree_tokens,
        "native_spine_tokens": native_tokens,
        "spine_tokens_match": tree_tokens == native_tokens,
        "stages": rows,
        "first_origin": first,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                "spine_tokens_match": out["spine_tokens_match"],
                "tree_spine_tokens": tree_tokens,
                "native_spine_tokens": native_tokens,
                "first_origin": first,
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


if __name__ == "__main__":
    main()
