#!/usr/bin/env python3
"""Compare FR12 layer-0 GDN subkernel captures on aligned spine rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


FALLBACK_TREE_SPINE_ROWS = [0, 1, 2, 4, 6]
STAGE_ORDER = ["pre_conv", "conv1d_out", "gdn_scan_out", "gate_out", "o_proj_out"]


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max().item())


def _mean_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().mean().item())


def _stage_tensor(capture: dict[str, Any], stage: str) -> torch.Tensor:
    try:
        return capture["stages"][stage]["tensor"]
    except KeyError as exc:
        raise KeyError(f"capture {capture.get('call_path')} missing stage {stage}") from exc


def _tensor_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, list):
        return [int(x) for x in value]
    return None


def _tree_path0_rows(capture: dict[str, Any], depth_count: int) -> list[int]:
    meta = capture.get("meta", {})
    detail = meta.get("tree_conv_detail", {})
    path0 = _tensor_list(detail.get("path0_nodes"))
    if path0:
        return path0[:depth_count]
    return FALLBACK_TREE_SPINE_ROWS[:depth_count]


def _draft_tokens_by_target_row(
    logits: dict[str, Any], target_rows: list[int], req: int
) -> list[int]:
    counts = [int(x) for x in logits["num_draft_tokens"]]
    start = sum(counts[:req])
    end = start + counts[req]
    target_logits_indices = logits["target_logits_indices"][start:end]
    draft_token_ids = logits["draft_token_ids"][start:end]
    out: list[int] = []
    for target in target_rows:
        matches = (target_logits_indices == int(target)).nonzero(as_tuple=True)[0]
        if matches.numel() == 0:
            raise RuntimeError(
                f"no draft token found for target row {target}; "
                f"available={target_logits_indices.detach().cpu().tolist()}"
            )
        out.append(int(draft_token_ids[int(matches[0])].item()))
    return out


def _detail_checks(
    tree_cap: dict[str, Any],
    native_cap: dict[str, Any],
    tree_rows: list[int],
    native_rows: list[int],
) -> dict[str, Any]:
    tree_detail = tree_cap.get("meta", {}).get("tree_conv_detail", {})
    native_detail = native_cap.get("meta", {}).get("native_conv_detail", {})
    out: dict[str, Any] = {}
    tree_window = tree_detail.get("window_path0")
    native_window = native_detail.get("window")
    if torch.is_tensor(tree_window) and torch.is_tensor(native_window):
        depth = min(len(tree_rows), len(native_rows), tree_window.shape[0], native_window.shape[0])
        out["conv_window"] = {
            "max_abs": _max_abs(tree_window[:depth], native_window[:depth]),
            "mean_abs": _mean_abs(tree_window[:depth], native_window[:depth]),
        }
    tree_prior = tree_detail.get("prior_window")
    native_prior = native_detail.get("prior_window")
    if torch.is_tensor(tree_prior) and torch.is_tensor(native_prior):
        out["conv_prior_window"] = {
            "max_abs": _max_abs(tree_prior, native_prior),
            "mean_abs": _mean_abs(tree_prior, native_prior),
            "native_source": native_detail.get("prior_window_source"),
            "tree_prior_cols": _tensor_list(tree_detail.get("prior_cols")),
            "native_prior_cols": _tensor_list(native_detail.get("prior_cols")),
        }
    for key in ["tap_products_fp32", "tap_products_bf16"]:
        tree_key = key + "_path0"
        tree_taps = tree_detail.get(tree_key)
        native_taps = native_detail.get(key)
        if torch.is_tensor(tree_taps) and torch.is_tensor(native_taps):
            depth = min(len(tree_rows), len(native_rows), tree_taps.shape[0], native_taps.shape[0])
            out[key] = {
                "max_abs": _max_abs(tree_taps[:depth], native_taps[:depth]),
                "mean_abs": _mean_abs(tree_taps[:depth], native_taps[:depth]),
            }
    return out


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

    native_counts = [int(x) for x in native_logits["num_draft_tokens"]]
    depth_count = min(5, native_counts[args.native_req])
    native_rows = list(range(depth_count))
    tree_rows = _tree_path0_rows(tree_cap, depth_count)
    tree_tokens = _draft_tokens_by_target_row(tree_logits, tree_rows, args.tree_req)
    native_tokens = _draft_tokens_by_target_row(
        native_logits, native_rows, args.native_req
    )

    rows = []
    for stage in STAGE_ORDER:
        if stage not in tree_cap.get("stages", {}) or stage not in native_cap.get("stages", {}):
            continue
        tree_t = _stage_tensor(tree_cap, stage)
        native_t = _stage_tensor(native_cap, stage)
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
        "tree_stage_rows": tree_rows,
        "native_stage_rows": native_rows,
        "tree_spine_tokens": tree_tokens,
        "native_spine_tokens": native_tokens,
        "spine_tokens_match": tree_tokens == native_tokens,
        "detail_checks": _detail_checks(tree_cap, native_cap, tree_rows, native_rows),
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
