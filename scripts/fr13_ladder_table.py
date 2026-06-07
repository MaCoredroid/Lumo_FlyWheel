#!/usr/bin/env python3
"""Build exact top-down ladder tables from FR10/FR13 capture tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def _csv_ints(value: str | None, default: list[int]) -> list[int]:
    if value is None or value.strip() == "":
        return list(default)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _row_positions(capture_rows: list[int], rows: list[int]) -> list[int]:
    return [capture_rows.index(row) for row in rows]


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    d = (a.to(torch.float32) - b.to(torch.float32)).abs()
    return {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "torch_equal": bool(torch.equal(a, b)),
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
        "nonzero": int((d != 0).sum().item()) if d.numel() else 0,
    }


def _select_rows(tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
    idx = torch.tensor(positions, dtype=torch.long)
    return tensor.index_select(0, idx)


def _logit_tensor(payload: dict[str, Any], key: str) -> torch.Tensor:
    tensor = payload[key]
    if not torch.is_tensor(tensor):
        raise TypeError(f"{key} is not a tensor")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _hidden_table(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    *,
    lhs_rows: list[int],
    rhs_rows: list[int],
    threshold: float,
) -> dict[str, Any]:
    lhs_capture_rows = [int(x) for x in lhs["rows"]]
    rhs_capture_rows = [int(x) for x in rhs["rows"]]
    lhs_pos = _row_positions(lhs_capture_rows, lhs_rows)
    rhs_pos = _row_positions(rhs_capture_rows, rhs_rows)

    stages = []
    input_stats = _metrics(
        _select_rows(lhs["input_hidden"], lhs_pos),
        _select_rows(rhs["input_hidden"], rhs_pos),
    )
    stages.append({"stage": "input_hidden", **input_stats})

    layer_count = min(len(lhs["layers"]), len(rhs["layers"]))
    layer_rows = []
    for layer_idx in range(layer_count):
        left_layer = lhs["layers"][layer_idx]
        right_layer = rhs["layers"][layer_idx]
        hidden = _metrics(
            _select_rows(left_layer["hidden"], lhs_pos),
            _select_rows(right_layer["hidden"], rhs_pos),
        )
        item = {
            "layer_idx": int(layer_idx),
            "lhs_layer_type": left_layer.get("layer_type"),
            "rhs_layer_type": right_layer.get("layer_type"),
            "hidden": hidden,
        }
        if "residual" in left_layer and "residual" in right_layer:
            item["residual"] = _metrics(
                _select_rows(left_layer["residual"], lhs_pos),
                _select_rows(right_layer["residual"], rhs_pos),
            )
        layer_rows.append(item)

    final_stats = _metrics(
        _select_rows(lhs["final_norm_hidden"], lhs_pos),
        _select_rows(rhs["final_norm_hidden"], rhs_pos),
    )

    first = None
    if input_stats["max_abs"] > threshold:
        first = {"stage": "input_hidden", "max_abs": input_stats["max_abs"]}
    else:
        for row in layer_rows:
            if row["hidden"]["max_abs"] > threshold:
                first = {
                    "stage": "layer_hidden",
                    "layer_idx": row["layer_idx"],
                    "max_abs": row["hidden"]["max_abs"],
                }
                break
            residual = row.get("residual")
            if residual is not None and residual["max_abs"] > threshold:
                first = {
                    "stage": "layer_residual",
                    "layer_idx": row["layer_idx"],
                    "max_abs": residual["max_abs"],
                }
                break
        if first is None and final_stats["max_abs"] > threshold:
            first = {"stage": "final_norm_hidden", "max_abs": final_stats["max_abs"]}
    return {
        "lhs_rows": lhs_rows,
        "rhs_rows": rhs_rows,
        "lhs_positions": lhs_pos,
        "rhs_positions": rhs_pos,
        "input": input_stats,
        "layers": layer_rows,
        "final_norm_hidden": final_stats,
        "first_nonzero": first,
    }


def _logit_table(
    lhs: dict[str, Any],
    rhs: dict[str, Any],
    *,
    lhs_key: str,
    rhs_key: str,
    lhs_rows: list[int],
    rhs_rows: list[int],
    threshold: float,
) -> dict[str, Any]:
    left = _logit_tensor(lhs, lhs_key)
    right = _logit_tensor(rhs, rhs_key)
    left_sel = _select_rows(left, lhs_rows)
    right_sel = _select_rows(right, rhs_rows)
    stats = _metrics(left_sel, right_sel)
    return {
        "lhs_key": lhs_key,
        "rhs_key": rhs_key,
        "lhs_rows": lhs_rows,
        "rhs_rows": rhs_rows,
        "stats": stats,
        "first_nonzero": (
            {"stage": "logits", "max_abs": stats["max_abs"]}
            if stats["max_abs"] > threshold
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lhs-hidden", type=Path, required=True)
    parser.add_argument("--rhs-hidden", type=Path, required=True)
    parser.add_argument("--lhs-hidden-rows", default="0")
    parser.add_argument("--rhs-hidden-rows", default="0")
    parser.add_argument("--lhs-logits", type=Path)
    parser.add_argument("--rhs-logits", type=Path)
    parser.add_argument("--lhs-logit-key", default="root_logits")
    parser.add_argument("--rhs-logit-key", default="root_logits")
    parser.add_argument("--lhs-logit-rows", default="0")
    parser.add_argument("--rhs-logit-rows", default="0")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    lhs_h = _load(args.lhs_hidden)
    rhs_h = _load(args.rhs_hidden)
    hidden = _hidden_table(
        lhs_h,
        rhs_h,
        lhs_rows=_csv_ints(args.lhs_hidden_rows, [0]),
        rhs_rows=_csv_ints(args.rhs_hidden_rows, [0]),
        threshold=float(args.threshold),
    )
    logits = None
    if args.lhs_logits and args.rhs_logits:
        logits = _logit_table(
            _load(args.lhs_logits),
            _load(args.rhs_logits),
            lhs_key=args.lhs_logit_key,
            rhs_key=args.rhs_logit_key,
            lhs_rows=_csv_ints(args.lhs_logit_rows, [0]),
            rhs_rows=_csv_ints(args.rhs_logit_rows, [0]),
            threshold=float(args.threshold),
        )

    passed = hidden["first_nonzero"] is None and (
        logits is None or logits["first_nonzero"] is None
    )
    payload = {
        "schema": "fr13.ladder_table.v1",
        "label": args.label,
        "threshold": float(args.threshold),
        "lhs_hidden": str(args.lhs_hidden),
        "rhs_hidden": str(args.rhs_hidden),
        "lhs_logits": None if args.lhs_logits is None else str(args.lhs_logits),
        "rhs_logits": None if args.rhs_logits is None else str(args.rhs_logits),
        "passed": bool(passed),
        "hidden": hidden,
        "logits": logits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "label": args.label,
                "passed": bool(passed),
                "hidden_first_nonzero": hidden["first_nonzero"],
                "input_max_abs": hidden["input"]["max_abs"],
                "final_norm_max_abs": hidden["final_norm_hidden"]["max_abs"],
                "logits_max_abs": None
                if logits is None
                else logits["stats"]["max_abs"],
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
