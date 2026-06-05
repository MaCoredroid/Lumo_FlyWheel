#!/usr/bin/env python3
"""Compare FR10 tree/native spine hidden captures layer by layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


TREE_SPINE_NODES = [0, 1, 3, 5, 7]


def _load(path: Path):
    return torch.load(path, map_location="cpu")


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.to(torch.float32), dim=-1)


def _row_start(counts: list[int], req: int) -> int:
    return int(sum(int(x) for x in counts[:req]))


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree-hidden", required=True, type=Path)
    parser.add_argument("--tree-logits", required=True, type=Path)
    parser.add_argument("--tree-req", type=int, default=0)
    parser.add_argument("--native-hidden", required=True, type=Path)
    parser.add_argument("--native-logits", required=True, type=Path)
    parser.add_argument("--native-req", type=int, default=0)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1e-4)
    args = parser.parse_args()

    tree_h = _load(args.tree_hidden)
    native_h = _load(args.native_hidden)
    tree_l = _load(args.tree_logits)
    native_l = _load(args.native_logits)

    tree_counts = [int(x) for x in tree_l["num_draft_tokens"]]
    native_counts = [int(x) for x in native_l["num_draft_tokens"]]
    tree_start = _row_start(tree_counts, args.tree_req)
    native_start = _row_start(native_counts, args.native_req)
    tree_rows_abs = [tree_start + node for node in TREE_SPINE_NODES]
    native_rows_abs = [native_start + depth for depth in range(5)]
    tree_hidden_rows_abs = [
        int(tree_l["target_logits_indices"][row].item()) for row in tree_rows_abs
    ]
    native_hidden_rows_abs = [
        int(native_l["target_logits_indices"][row].item()) for row in native_rows_abs
    ]

    tree_capture_rows = [int(x) for x in tree_h["rows"]]
    native_capture_rows = [int(x) for x in native_h["rows"]]
    tree_row_pos = [tree_capture_rows.index(row) for row in tree_hidden_rows_abs]
    native_row_pos = [native_capture_rows.index(row) for row in native_hidden_rows_abs]

    tree_drafts = [int(x) for x in tree_l["draft_token_ids"][tree_rows_abs].tolist()]
    native_drafts = [int(x) for x in native_l["draft_token_ids"][native_rows_abs].tolist()]
    tree_probs = _softmax(tree_l["target_logits"])
    native_probs = _softmax(native_l["target_logits"])

    per_depth_probs = []
    for depth, (tree_row, native_row, token) in enumerate(zip(tree_rows_abs, native_rows_abs, tree_drafts)):
        per_depth_probs.append({
            "depth": int(depth),
            "draft_token": int(token),
            "native_draft_token": int(native_drafts[depth]),
            "tree_prob_draft": float(tree_probs[tree_row, token].item()),
            "native_prob_draft": float(native_probs[native_row, token].item()),
            "tree_argmax": int(torch.argmax(tree_probs[tree_row]).item()),
            "native_argmax": int(torch.argmax(native_probs[native_row]).item()),
        })

    rows = []
    input_by_depth = []
    for depth in range(5):
        input_by_depth.append({
            "depth": int(depth),
            "max_abs": _max_abs(
                tree_h["input_hidden"][tree_row_pos[depth]],
                native_h["input_hidden"][native_row_pos[depth]],
            ),
        })

    layer_count = min(len(tree_h["layers"]), len(native_h["layers"]))
    for layer_idx in range(layer_count):
        tl = tree_h["layers"][layer_idx]
        nl = native_h["layers"][layer_idx]
        depth_deltas = []
        for depth in range(5):
            depth_deltas.append(_max_abs(
                tl["hidden"][tree_row_pos[depth]],
                nl["hidden"][native_row_pos[depth]],
            ))
        rows.append({
            "layer_idx": int(layer_idx),
            "tree_layer_type": tl.get("layer_type"),
            "native_layer_type": nl.get("layer_type"),
            "max_abs_by_depth": depth_deltas,
            "max_abs": float(max(depth_deltas)),
        })

    final_by_depth = []
    for depth in range(5):
        final_by_depth.append({
            "depth": int(depth),
            "max_abs": _max_abs(
                tree_h["final_norm_hidden"][tree_row_pos[depth]],
                native_h["final_norm_hidden"][native_row_pos[depth]],
            ),
        })

    first = None
    input_max = max(row["max_abs"] for row in input_by_depth)
    if input_max > args.threshold:
        first = {"where": "input", "max_abs": float(input_max)}
    else:
        for row in rows:
            if float(row["max_abs"]) > args.threshold:
                first = row
                break
        if first is None:
            final_max = max(row["max_abs"] for row in final_by_depth)
            if final_max > args.threshold:
                first = {"where": "final_norm", "max_abs": float(final_max)}
            else:
                first = {"where": "none", "max_abs": float(final_max)}

    out = {
        "schema": "fr10.layer_hidden_spine_compare.v1",
        "tree_hidden": str(args.tree_hidden),
        "native_hidden": str(args.native_hidden),
        "tree_logits": str(args.tree_logits),
        "native_logits": str(args.native_logits),
        "tree_req": int(args.tree_req),
        "native_req": int(args.native_req),
        "tree_rows_abs": tree_rows_abs,
        "native_rows_abs": native_rows_abs,
        "tree_hidden_rows_abs": tree_hidden_rows_abs,
        "native_hidden_rows_abs": native_hidden_rows_abs,
        "tree_spine_tokens": tree_drafts,
        "native_spine_tokens": native_drafts,
        "spine_tokens_match": tree_drafts == native_drafts,
        "input_max_abs_by_depth": input_by_depth,
        "layers": rows,
        "final_norm_max_abs_by_depth": final_by_depth,
        "probs": per_depth_probs,
        "first_divergence": first,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({
        "spine_tokens_match": out["spine_tokens_match"],
        "tree_spine_tokens": tree_drafts,
        "native_spine_tokens": native_drafts,
        "first_divergence": first,
        "probs": per_depth_probs,
    }, indent=2))


if __name__ == "__main__":
    main()
