#!/usr/bin/env python3
"""FR13 APC multi-turn carrier localizer (offline diff, no GPU).

Compares the cache-ON write/restore SSM recurrent state against the no-cache
prefill ground-truth final_state, per GDN layer, to localize the multi-turn
cache-ON corruption carrier among FOUR candidates:

  1. STALE-WRITE     : stock_row (what stock get_temporal_copy_spec would have
                       copied into the cache block) diverges from no-cache
                       final_state  -> the snapshot SOURCE is stale.
  2. LEAF-UNFAITHFUL : leaf_row (the FR13_APC_SNAP_FIX redirect target) diverges
                       from no-cache final_state  -> SNAP_FIX's "faithful"
                       committed-leaf row is itself wrong at high occupancy
                       (the residual the SNAP_FIX bake did NOT cover).
  3. RESTORE-READ    : restored initial_state (ON arm, cache-hit prefill) diverges
                       from no-cache final_state  -> the WRITE may be fine but the
                       READ-BACK on the next cache-hit prefill reads the wrong row.
  4. WRAPAROUND      : #27264-class -- stock_row_idx / dest_row_idx >= state_rows,
                       or a block row index reused across turns (collision).

Inputs:
  --nocache DIR : dir of FR13_PREFILL_GDN_CAPTURE *.pt (CFG arm, NO cache).
                  Each payload has layer_prefix, final_state, initial_state, conv_out.
  --cacherow DIR: dir of FR13_APC_CACHEROW_DUMP cacherow_*.pt (ON arm).
                  Each has layer, stock_row, leaf_row, *_row_idx, num_block_ids,
                  max_block_id, state_rows.
  --on-capture DIR (optional): dir of FR13_PREFILL_GDN_CAPTURE *.pt from the ON
                  arm; its initial_state on a cache-hit prefill == the RESTORED
                  per-layer recurrent state (candidate 3). If omitted, the
                  restore-read axis is skipped.

Output: per-GDN-layer JSONL of max_abs for each axis + a summary naming the
FIRST divergent layer and WHICH axis carries it. Reuses the metric shape of
scripts/fr13_prefill_gdn_state_replay.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


def _layer_idx(s: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", str(s))
    return int(m.group(1)) if m else None


def _payload_layer(pay: dict[str, Any]) -> int | None:
    """Real GDN layer index from a prefill-capture payload.

    BUG-1 fix: prefer the explicit "layer_name" field (= str(self.prefix), the
    ACTUAL layer name now added to the payload), falling back to "layer_prefix"
    (also the real layer name in current captures, NOT the env filter '*') for
    backward-compat with older dumps.
    """
    for key in ("layer_name", "layer_prefix"):
        li = _layer_idx(pay.get(key, ""))
        if li is not None:
            return li
    return None


def _max_abs(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"missing": True}
    a = a.float()
    b = b.float()
    if a.shape != b.shape:
        # squeeze a leading batch dim if present (final_state is [B, vh, hv, hk];
        # a cache row is [vh, hv, hk])
        if a.ndim == b.ndim + 1 and a.shape[0] == 1:
            a = a[0]
        elif b.ndim == a.ndim + 1 and b.shape[0] == 1:
            b = b[0]
        if a.shape != b.shape:
            return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    d = (a - b).abs()
    return {
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
        "nonzero": int((d != 0).sum().item()) if d.numel() else 0,
        "shape": list(a.shape),
    }


def _load_nocache(d: Path) -> dict[int, dict[str, Any]]:
    """LATEST prefill capture per GDN layer (BUG-2 fix).

    The no-cache ground truth must be the HIGH-occupancy turn (seq49) so it is
    apples-to-apples with the latest-snap_fired cacherow. We pick the payload
    with the highest "capture_saved_index" per layer (last call seen), NOT the
    first (turn-0 low-occupancy). Drive this by sending ONLY the seq49 request
    to the CFG no-cache boot (see driver) so there is exactly ONE high-occ
    full-prefill per layer; latest-per-layer is then just that turn.
    """
    out: dict[int, dict[str, Any]] = {}
    for p in sorted(d.glob("*.pt")):
        try:
            pay = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        li = _payload_layer(pay)
        if li is None:
            continue
        prev = out.get(li)
        if prev is None or int(pay.get("capture_saved_index", 0)) >= int(
            prev.get("capture_saved_index", 0)
        ):
            out[li] = pay  # latest call per layer == high-occupancy turn
    return out


def _load_cacherows(d: Path) -> dict[int, dict[str, Any]]:
    """Latest (highest snap_fired) cacherow dump per GDN layer -> the high-occ snapshot."""
    out: dict[int, dict[str, Any]] = {}
    for p in sorted(d.glob("cacherow_*.pt")):
        try:
            pay = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        li = _layer_idx(pay.get("layer", ""))
        if li is None:
            continue
        prev = out.get(li)
        if prev is None or int(pay.get("snap_fired", 0)) >= int(prev.get("snap_fired", 0)):
            out[li] = pay
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nocache", required=True, type=Path)
    ap.add_argument("--cacherow", required=True, type=Path)
    ap.add_argument("--on-capture", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("output/fr13_apc_cacherow_diff.jsonl"))
    ap.add_argument("--thresh", type=float, default=1e-2,
                    help="max_abs above this flags a divergent axis")
    args = ap.parse_args()

    nocache = _load_nocache(args.nocache)
    cacherows = _load_cacherows(args.cacherow)
    on_cap = _load_nocache(args.on_capture) if args.on_capture else {}

    layers = sorted(set(nocache) | set(cacherows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    first_div: dict[str, int | None] = {
        "stale_write": None, "leaf_unfaithful": None, "restore_read": None,
        "wraparound": None,
    }
    records = []
    with args.out.open("w") as fh:
        for li in layers:
            nc = nocache.get(li)
            cr = cacherows.get(li)
            oc = on_cap.get(li)
            gt = nc.get("final_state") if nc else None  # ground-truth recurrent state

            rec: dict[str, Any] = {"layer": li}
            if cr is not None:
                rec["stale_write"] = _max_abs(cr.get("stock_row"), gt)
                rec["leaf_unfaithful"] = _max_abs(cr.get("leaf_row"), gt)
                sr = int(cr.get("state_rows", 0))
                rec["wraparound"] = {
                    "stock_row_idx": cr.get("stock_row_idx"),
                    "leaf_row_idx": cr.get("leaf_row_idx"),
                    "dest_row_idx": cr.get("dest_row_idx"),
                    "num_block_ids": cr.get("num_block_ids"),
                    "max_block_id": cr.get("max_block_id"),
                    "state_rows": sr,
                    "oob": bool(
                        sr and (
                            int(cr.get("stock_row_idx", 0)) >= sr
                            or int(cr.get("dest_row_idx", 0)) >= sr
                            or int(cr.get("max_block_id", 0)) >= sr
                        )
                    ),
                }
                if rec["wraparound"]["oob"] and first_div["wraparound"] is None:
                    first_div["wraparound"] = li
            if oc is not None:
                rec["restore_read"] = _max_abs(oc.get("initial_state"), gt)

            for axis in ("stale_write", "leaf_unfaithful", "restore_read"):
                m = rec.get(axis)
                if (
                    isinstance(m, dict)
                    and m.get("max_abs", 0.0) > args.thresh
                    and first_div[axis] is None
                ):
                    first_div[axis] = li

            fh.write(json.dumps(rec) + "\n")
            records.append(rec)

    print("=== FR13 APC cacherow carrier localization ===")
    print(f"layers compared: {len(layers)}  (nocache={len(nocache)} cacherow={len(cacherows)} on_cap={len(on_cap)})")
    for axis, li in first_div.items():
        print(f"  first-divergent[{axis:16s}] = {li}")
    carrier = min(
        (v for v in first_div.values() if v is not None),
        default=None,
    )
    if carrier is None:
        print("  VERDICT: no axis diverges above thresh -- carrier NOT in the SSM "
              "snapshot/restore (look elsewhere: conv prior-window, full-attn KV, "
              "or graph-read).")
    else:
        winners = [a for a, v in first_div.items() if v == carrier]
        print(f"  VERDICT: earliest divergence at GDN layer {carrier} via {winners}")
    print(f"  per-layer jsonl -> {args.out}")


if __name__ == "__main__":
    main()
