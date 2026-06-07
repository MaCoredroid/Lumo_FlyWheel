#!/usr/bin/env python3
"""Offline per-stage GDN sub-op diff: tree vs native, to pin the first diverging op.

No GPU/server. Loads two `fr12.gdn_l0_subkernel_capture.v1` captures (tree + native)
and diffs each stage tensor per row, in pipeline order, reporting the FIRST stage that
diverges above a bf16-ULP threshold. The first diverging stage is the culprit op:
  input_hidden -> pre_conv -> conv1d_out -> [h0_state_in input] -> gdn_scan_out
  -> gate_z -> gate_out -> o_proj_out

Usage:
  python3 scripts/fr13_gdn_subop_diff.py --tree <tree/gdn_subop.callN.pt> \
      --native <native/gdn_subop.callN.pt> [--rows 0,1,2,3,4,5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# pipeline order; h0_state_in is an INPUT (state read), placed where it's consumed
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


def _tensor(stage: dict) -> torch.Tensor:
    return stage["tensor"].to(torch.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True, type=Path)
    ap.add_argument("--native", required=True, type=Path)
    ap.add_argument("--rows", default="0,1,2,3,4,5")
    ap.add_argument("--threshold", type=float, default=0.00390625)  # 2-ULP-ish
    ap.add_argument("--out-json", type=Path)
    args = ap.parse_args()

    rows = [int(x) for x in args.rows.split(",") if x.strip()]
    T = torch.load(args.tree, map_location="cpu", weights_only=False)
    N = torch.load(args.native, map_location="cpu", weights_only=False)
    ts, ns = T["stages"], N["stages"]
    print(f"tree  layer_prefix: {T.get('layer_prefix')}")
    print(f"native layer_prefix: {N.get('layer_prefix')}")

    report = {"tree": str(args.tree), "native": str(args.native), "stages": []}
    first_div = None
    for name in STAGE_ORDER:
        if name not in ts or name not in ns:
            print(f"  {name:<14} MISSING in one capture")
            continue
        a, b = _tensor(ts[name]), _tensor(ns[name])
        entry = {"stage": name, "tree_extra": ts[name].get("extra"),
                 "native_extra": ns[name].get("extra")}
        if a.shape != b.shape:
            print(f"  {name:<14} SHAPE MISMATCH tree={tuple(a.shape)} native={tuple(b.shape)}")
            entry["shape_mismatch"] = [list(a.shape), list(b.shape)]
            report["stages"].append(entry)
            if first_div is None:
                first_div = name
            continue
        d = (a - b).abs()
        if name == "h0_state_in":
            mx = float(d.max())
            entry["max_abs"] = mx
            flag = "  <== STATE READ DIFFERS" if mx > args.threshold else ""
            extra_t = ts[name].get("extra", {})
            extra_n = ns[name].get("extra", {})
            print(f"  {name:<14} max_abs={mx:<12} tree_h0_rows={extra_t.get('h0_rows')} "
                  f"native_h0_rows={extra_n.get('h0_rows')}{flag}")
            if mx > args.threshold and first_div is None:
                first_div = name
        else:
            # per-row max
            flat = d.reshape(d.shape[0], -1) if d.dim() > 1 else d.reshape(-1, 1)
            per_row = {r: round(float(flat[r].max()), 6) for r in rows if r < flat.shape[0]}
            mx = max(per_row.values()) if per_row else 0.0
            entry["max_abs"] = mx
            entry["per_row"] = per_row
            flag = "  <== FIRST DIVERGING STAGE" if (mx > args.threshold and first_div is None) else ""
            print(f"  {name:<14} max_abs={mx:<12} per_row={per_row}{flag}")
            if mx > args.threshold and first_div is None:
                first_div = name
        report["stages"].append(entry)

    report["first_diverging_stage"] = first_div
    print(f"\nFIRST DIVERGING STAGE: {first_div}")
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
