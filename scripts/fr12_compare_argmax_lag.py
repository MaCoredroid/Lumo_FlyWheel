#!/usr/bin/env python3
"""Compare tree/native spine argmax and detect one-depth lag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _counts(capture: dict[str, Any]) -> list[int]:
    return [int(x) for x in capture["num_draft_tokens"]]


def _start(capture: dict[str, Any], req: int) -> int:
    return int(sum(_counts(capture)[:req]))


def _target_rows(capture: dict[str, Any], req: int, *, tree: bool, depth: int) -> list[int]:
    start = _start(capture, req)
    end = start + _counts(capture)[req]
    rows = [int(x) for x in capture["target_logits_indices"][start:end].tolist()]
    if tree:
        ordered: list[int] = []
        for row in rows:
            if row not in ordered:
                ordered.append(row)
        return ordered[:depth]
    return rows[:depth]


def _entry_for_target_row(capture: dict[str, Any], req: int, target_row: int) -> int:
    start = _start(capture, req)
    end = start + _counts(capture)[req]
    target_logits_indices = capture["target_logits_indices"][start:end]
    matches = (target_logits_indices == int(target_row)).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        raise RuntimeError(
            f"target row {target_row} missing in req {req}; "
            f"available={target_logits_indices.tolist()}"
        )
    return int(start + int(matches[0].item()))


def _row(capture: dict[str, Any], req: int, target_row: int) -> dict[str, Any]:
    entry = _entry_for_target_row(capture, req, target_row)
    logits = capture["target_logits"][entry].to(torch.float32)
    probs = torch.softmax(logits, dim=-1)
    draft = int(capture["draft_token_ids"][entry].item())
    argmax = int(torch.argmax(logits).item())
    return {
        "entry": int(entry),
        "target_row": int(target_row),
        "draft_token": draft,
        "argmax": argmax,
        "prob_draft": float(probs[draft].item()),
        "max_logit": float(logits[argmax].item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-logits", required=True, type=Path)
    parser.add_argument("--native-logits", required=True, type=Path)
    parser.add_argument("--tree-req", type=int, default=0)
    parser.add_argument("--native-req", type=int, default=0)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tree = _load(args.tree_logits)
    native = _load(args.native_logits)
    tree_rows = _target_rows(tree, args.tree_req, tree=True, depth=args.depth)
    native_rows = _target_rows(native, args.native_req, tree=False, depth=args.depth)
    depth = min(args.depth, len(tree_rows), len(native_rows))
    tree_rows = tree_rows[:depth]
    native_rows = native_rows[:depth]

    per_depth = []
    for d in range(depth):
        tree_row = _row(tree, args.tree_req, tree_rows[d])
        native_row = _row(native, args.native_req, native_rows[d])
        lag_match = False
        if d > 0:
            prev_native_row = _row(native, args.native_req, native_rows[d - 1])
            lag_match = tree_row["argmax"] == prev_native_row["argmax"]
        per_depth.append(
            {
                "depth": int(d),
                "tree": tree_row,
                "native": native_row,
                "argmax_match": tree_row["argmax"] == native_row["argmax"],
                "tree_argmax_equals_native_prev_depth": bool(lag_match),
                "draft_match": tree_row["draft_token"] == native_row["draft_token"],
            }
        )

    lag_depths = [
        row["depth"]
        for row in per_depth
        if row["depth"] > 0 and row["tree_argmax_equals_native_prev_depth"]
    ]
    mismatched = [row["depth"] for row in per_depth if not row["argmax_match"]]
    result = {
        "schema": "fr12.argmax_lag_compare.v1",
        "tree_logits": str(args.tree_logits),
        "native_logits": str(args.native_logits),
        "tree_req": int(args.tree_req),
        "native_req": int(args.native_req),
        "tree_mode": tree.get("mode"),
        "native_mode": native.get("mode"),
        "tree_has_tree_parent_indices": bool(tree.get("has_tree_parent_indices")),
        "tree_num_draft_tokens": _counts(tree),
        "native_num_draft_tokens": _counts(native),
        "tree_rows": tree_rows,
        "native_rows": native_rows,
        "per_depth": per_depth,
        "argmax_mismatch_depths": mismatched,
        "lag_match_depths": lag_depths,
        "lag_persists_from_first_branch_gap": lag_depths[:3] == [2, 3, 4],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
