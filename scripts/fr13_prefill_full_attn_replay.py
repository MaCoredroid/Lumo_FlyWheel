#!/usr/bin/env python3
"""Direct row-wise prefill full-attention diff for TREE_ATTN vs FLASH_ATTN."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


STAGES = [
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


def _load(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False) | {"_path": str(path)}


def _layer_idx(payload: dict[str, Any]) -> int:
    name = str(payload.get("layer_prefix", ""))
    match = re.search(r"layers\.(\d+)\.self_attn", name)
    if not match:
        raise ValueError(f"cannot parse layer from {name!r}")
    return int(match.group(1))


def _by_layer(paths: list[Path]) -> dict[int, dict[str, Any]]:
    out = {}
    for path in paths:
        payload = _load(path)
        out[_layer_idx(payload)] = payload
    return out


def _tensor(payload: dict[str, Any], stage: str) -> torch.Tensor | None:
    item = payload.get("stages", {}).get(stage)
    if item is None:
        return None
    return item["tensor"]


def _metrics(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"missing": True}
    if a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
        a = a.reshape(-1)
        b = b.reshape(-1)
    if tuple(a.shape) != tuple(b.shape):
        return {
            "shape_mismatch": [list(a.shape), list(b.shape)],
            "dtype_a": str(a.dtype),
            "dtype_b": str(b.dtype),
        }
    d = (a.float() - b.float()).abs()
    out: dict[str, Any] = {
        "shape": list(a.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "torch_equal": bool(torch.equal(a, b)),
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
        "nonzero": int((d != 0).sum().item()) if d.numel() else 0,
    }
    if out["nonzero"]:
        flat = int(torch.argmax(d).item())
        idx = [int(x.item()) for x in torch.unravel_index(torch.tensor(flat), d.shape)]
        out["first_mismatch"] = {
            "index": idx,
            "lhs": float(a[tuple(idx)].float().item()),
            "rhs": float(b[tuple(idx)].float().item()),
            "abs": float(d[tuple(idx)].item()),
        }
    else:
        out["first_mismatch"] = None
    return out


def _compare_layer(tree: dict[str, Any], native: dict[str, Any]) -> dict[str, Any]:
    stages = {stage: _metrics(_tensor(tree, stage), _tensor(native, stage)) for stage in STAGES}
    first = None
    for stage in STAGES:
        m = stages[stage]
        if m.get("shape_mismatch") or (m.get("max_abs") not in (None, 0.0)):
            first = stage
            break
    return {
        "layer": _layer_idx(tree),
        "tree_capture": tree["_path"],
        "native_capture": native["_path"],
        "num_tokens": int(tree.get("num_tokens", -1)),
        "first_diverging_stage": first,
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", nargs="+", type=Path, required=True)
    parser.add_argument("--native", nargs="+", type=Path, required=True)
    parser.add_argument("--layers", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tree_layers = _by_layer(args.tree)
    native_layers = _by_layer(args.native)
    layers = sorted(set(tree_layers) & set(native_layers))
    if args.layers.strip():
        wanted = {int(x.strip()) for x in args.layers.split(",") if x.strip()}
        layers = [layer for layer in layers if layer in wanted]
    rows = [_compare_layer(tree_layers[layer], native_layers[layer]) for layer in layers]
    first_layer = next((row for row in rows if row["first_diverging_stage"] is not None), None)
    payload = {
        "schema": "fr13.prefill_full_attn_replay.v1",
        "tree_captures": [str(p) for p in args.tree],
        "native_captures": [str(p) for p in args.native],
        "layers": [row["layer"] for row in rows],
        "first_diverging_layer": None if first_layer is None else first_layer["layer"],
        "first_diverging_stage": None if first_layer is None else first_layer["first_diverging_stage"],
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "layers": payload["layers"],
                "first_diverging_layer": payload["first_diverging_layer"],
                "first_diverging_stage": payload["first_diverging_stage"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    for row in rows:
        print(
            json.dumps(
                {
                    "layer": row["layer"],
                    "first": row["first_diverging_stage"],
                    "stage_max": {
                        stage: row["stages"][stage].get("max_abs")
                        for stage in STAGES
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0 if first_layer is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
