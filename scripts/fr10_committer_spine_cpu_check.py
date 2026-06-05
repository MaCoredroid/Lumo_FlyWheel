#!/usr/bin/env python3
"""Replay native-linear vs tree deterministic committers on captured logits.

This is a diagnostic, not a serving path. It consumes the spine-logit
comparison artifact plus the paired capture .pt files and asks whether the
tree multi-draft committer shortens the path0/spine acceptance when logits are
held fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


SPINE_NODES = [0, 1, 3, 5, 7]


def _load_pt(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _softmax_rows(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.to(torch.float32), dim=-1).cpu().numpy()


def _native_expected_len(probs_by_depth: list[np.ndarray], spine_tokens: list[int]) -> float:
    survival = 1.0
    expected = 0.0
    for p, tok in zip(probs_by_depth, spine_tokens):
        survival *= float(p[int(tok)])
        expected += survival
    return expected


def _tree_depth_continue_probs(
    tree_probs: np.ndarray,
    tree_drafts: list[int],
    parents: list[int],
    req_start: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    current_parent = -1
    continue_probs: list[float] = []
    rows: list[dict[str, Any]] = []
    for _ in range(len(SPINE_NODES)):
        children = [i for i, parent in enumerate(parents) if int(parent) == current_parent]
        if not children:
            break
        spine_child = SPINE_NODES[len(continue_probs)]
        if spine_child not in children:
            raise RuntimeError(
                f"expected spine child {spine_child} under parent {current_parent}, got {children}"
            )
        target_row = req_start + children[0]
        child_tokens = [int(tree_drafts[c]) for c in children]
        p = tree_probs[target_row]
        overlaps = np.asarray([float(p[tok]) for tok in child_tokens], dtype=np.float64)
        overlap_mass = float(overlaps.sum())
        spine_source = int(children.index(spine_child))
        spine_token = int(tree_drafts[spine_child])
        if overlap_mass <= 0.0:
            continue_prob = 0.0
            q_mix_token = 0.0
            accept_prob_if_spine_selected = 0.0
        else:
            weights = overlaps / overlap_mass
            q_mix_token = float(
                sum(
                    float(weights[i])
                    for i, tok in enumerate(child_tokens)
                    if int(tok) == spine_token
                )
            )
            accept_prob_if_spine_selected = min(
                1.0, float(p[spine_token]) / q_mix_token
            ) if q_mix_token > 0.0 else 0.0
            continue_prob = float(weights[spine_source]) * accept_prob_if_spine_selected
        continue_probs.append(continue_prob)
        rows.append(
            {
                "depth": len(continue_probs) - 1,
                "parent": int(current_parent),
                "children": [int(x) for x in children],
                "child_tokens": child_tokens,
                "spine_child": int(spine_child),
                "spine_token": int(spine_token),
                "spine_target_prob": float(p[spine_token]),
                "sibling_overlap_mass": float(overlap_mass),
                "q_mix_spine_token": float(q_mix_token),
                "accept_prob_if_spine_selected": float(accept_prob_if_spine_selected),
                "tree_continue_prob": float(continue_prob),
                "duplicate_spine_token_count": int(
                    sum(1 for tok in child_tokens if int(tok) == spine_token)
                ),
            }
        )
        current_parent = spine_child
    return continue_probs, rows


def _expected_len_from_continue_probs(probs: list[float]) -> float:
    survival = 1.0
    expected = 0.0
    for p in probs:
        survival *= float(p)
        expected += survival
    return expected


def _capture_path(run: Path, capture_name: str) -> Path:
    return run / "logs" / capture_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with args.comparison.open() as f:
        comparison = json.load(f)
    tree_run = Path(comparison["tree_run"])
    native_run = Path(comparison["native_run"])

    tree_cache: dict[str, dict[str, Any]] = {}
    native_cache: dict[str, dict[str, Any]] = {}
    rows = []
    for match in comparison["matches"]:
        tree_capture = match["tree_capture"]
        native_capture = match["native_capture"]
        if tree_capture not in tree_cache:
            tree_cache[tree_capture] = _load_pt(_capture_path(tree_run, tree_capture))
        if native_capture not in native_cache:
            native_cache[native_capture] = _load_pt(_capture_path(native_run, native_capture))
        tree_cap = tree_cache[tree_capture]
        native_cap = native_cache[native_capture]

        tree_req = int(match["tree_req"])
        native_req = int(match["native_req"])
        seq = [int(x) for x in match["seq"]]

        tree_counts = [int(x) for x in tree_cap["num_draft_tokens"]]
        native_counts = [int(x) for x in native_cap["num_draft_tokens"]]
        tree_start = int(sum(tree_counts[:tree_req]))
        native_start = int(sum(native_counts[:native_req]))
        tree_n = int(tree_counts[tree_req])
        native_n = int(native_counts[native_req])
        if tree_n < 9 or native_n < 5:
            raise RuntimeError(f"unexpected draft widths in match {match['match_index']}")

        parents = [int(x) for x in tree_cap["tree_parent_indices"][tree_start:tree_start + tree_n].tolist()]
        tree_drafts = [int(x) for x in tree_cap["draft_token_ids"][tree_start:tree_start + tree_n].tolist()]
        tree_spine = [tree_drafts[i] for i in SPINE_NODES]
        if tree_spine != seq:
            raise RuntimeError(
                f"tree spine mismatch in match {match['match_index']}: {tree_spine} != {seq}"
            )
        native_drafts = [int(x) for x in native_cap["draft_token_ids"][native_start:native_start + 5].tolist()]
        if native_drafts != seq:
            raise RuntimeError(
                f"native spine mismatch in match {match['match_index']}: {native_drafts} != {seq}"
            )

        tree_probs = _softmax_rows(tree_cap["target_logits"])
        native_probs_all = _softmax_rows(native_cap["target_logits"])
        native_probs_by_depth = [native_probs_all[native_start + d] for d in range(5)]

        tree_spine_probs_by_depth = [
            tree_probs[tree_start + node] for node in SPINE_NODES
        ]
        native_on_native = _native_expected_len(native_probs_by_depth, seq)
        native_on_tree = _native_expected_len(tree_spine_probs_by_depth, seq)
        tree_continue_probs, depth_rows = _tree_depth_continue_probs(
            tree_probs, tree_drafts, parents, tree_start
        )
        tree_expected = _expected_len_from_continue_probs(tree_continue_probs)

        rows.append({
            "match_index": int(match["match_index"]),
            "seq": seq,
            "tree_capture": tree_capture,
            "tree_req": tree_req,
            "native_capture": native_capture,
            "native_req": native_req,
            "native_linear_expected_on_native_logits": float(native_on_native),
            "native_linear_expected_on_tree_logits": float(native_on_tree),
            "tree_spine_expected_on_tree_logits": float(tree_expected),
            "committer_delta_tree_minus_native_same_tree_logits": float(
                tree_expected - native_on_tree
            ),
            "verify_delta_native_logits_minus_tree_logits": float(
                native_on_native - native_on_tree
            ),
            "depth_rows": depth_rows,
        })

    native_same_logits_all = np.array(
        [r["native_linear_expected_on_tree_logits"] for r in rows], dtype=np.float64
    )
    tree_all = np.array([r["tree_spine_expected_on_tree_logits"] for r in rows], dtype=np.float64)
    native_native_all = np.array(
        [r["native_linear_expected_on_native_logits"] for r in rows], dtype=np.float64
    )
    result = {
        "schema": "fr10.committer_spine_cpu_check.v1",
        "comparison": str(args.comparison),
        "rows": rows,
        "summary": {
            "matches": len(rows),
            "native_linear_on_native_logits_mean": float(native_native_all.mean()) if len(rows) else None,
            "native_linear_on_tree_logits_mean": float(native_same_logits_all.mean()) if len(rows) else None,
            "tree_spine_mean_of_means": float(tree_all.mean()) if len(rows) else None,
            "committer_delta_tree_minus_native_same_tree_logits_mean": float((tree_all - native_same_logits_all).mean()) if len(rows) else None,
            "verify_delta_native_logits_minus_tree_logits_mean": float((native_native_all - native_same_logits_all).mean()) if len(rows) else None,
            "tree_less_than_native_same_logits_any_row": bool(
                np.any(tree_all < native_same_logits_all - 1e-6)
            ) if len(rows) else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result["summary"], indent=2))
    for row in rows:
        print(
            "match", row["match_index"],
            "native(tree logits)", f"{row['native_linear_expected_on_tree_logits']:.6f}",
            "tree", f"{row['tree_spine_expected_on_tree_logits']:.6f}",
            "committer_delta", f"{row['committer_delta_tree_minus_native_same_tree_logits']:.6g}",
            "verify_delta", f"{row['verify_delta_native_logits_minus_tree_logits']:.6f}",
        )


if __name__ == "__main__":
    main()
