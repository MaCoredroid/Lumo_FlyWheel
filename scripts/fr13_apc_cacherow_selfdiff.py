#!/usr/bin/env python3
"""FR13 APC cacherow STOCK-vs-LEAF self-diff (offline, no GPU, no reference boot).

The cheapest discriminator for "does the conv/ssm redirect actually CHANGE the
committed cache value?" Each FR13_APC_CACHEROW_DUMP cacherow_*.pt holds BOTH:

  stock_row : state[block_ids[src_block_idx + accept_token_bias]]
              == what the STOCK get_conv_copy_spec / get_temporal_copy_spec
                 would have committed (no redirect).
  leaf_row  : state[_FR13_APC_*_LEAF_BY_REQ[req]]
              == what the FR13 committed-accepted-leaf redirect commits.

If stock_row == leaf_row BIT-EXACT for every fired event, the redirect is INERT
(stock copy was already correct) -> that state is NOT the Axis-A carrier; the
redirect firing (CONV_REDIRECT_FIRED>0) is engagement-only and the deviation
must live elsewhere. If they DIFFER, the redirect does real work and a
ground-truth reference run (fr13_apc_cacherow_diff.py --nocache ...) is needed to
decide which one is correct.

This needs NO second boot: both rows are in the same dump. Reports per-kind
(conv vs ssm) the count identical vs differing and the max |stock-leaf|.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def _layer_idx(s: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", str(s))
    return int(m.group(1)) if m else None


def _kind(p: Path) -> str:
    n = p.name
    if n.startswith("cacherow_conv_"):
        return "conv"
    if n.startswith("cacherow_ssm_"):
        return "ssm"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cacherow", required=True, type=Path)
    args = ap.parse_args()

    files = sorted(args.cacherow.glob("cacherow_*.pt"))
    if not files:
        print(f"NO cacherow dumps in {args.cacherow} -> CACHEROW_DUMP vacuous (no event fired/dumped)")
        return

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in files:
        try:
            pay = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"  !! load fail {p.name}: {e}")
            continue
        stock = pay.get("stock_row")
        leaf = pay.get("leaf_row")
        if stock is None or leaf is None:
            continue
        a = stock.float()
        b = leaf.float()
        if a.shape != b.shape:
            by_kind[_kind(p)].append({"file": p.name, "shape_mismatch": [list(a.shape), list(b.shape)]})
            continue
        d = (a - b).abs()
        by_kind[_kind(p)].append({
            "file": p.name,
            "layer": _layer_idx(pay.get("layer", "")),
            "snap_fired": int(pay.get("snap_fired", 0)),
            "stock_idx": pay.get("stock_row_idx"),
            "leaf_idx": pay.get("leaf_row_idx"),
            "max_abs": float(d.max().item()) if d.numel() else 0.0,
            "nonzero": int((d != 0).sum().item()) if d.numel() else 0,
            "numel": int(d.numel()),
            "same_idx": int(pay.get("stock_row_idx", -1)) == int(pay.get("leaf_row_idx", -2)),
        })

    print("=== FR13 APC cacherow STOCK-vs-LEAF self-diff ===")
    print(f"dumps: {len(files)}  dir={args.cacherow}")
    overall_changed = False
    for kind in ("conv", "ssm", "other"):
        recs = by_kind.get(kind)
        if not recs:
            continue
        diffs = [r for r in recs if r.get("max_abs", 0.0) > 0.0]
        ident = [r for r in recs if "max_abs" in r and r["max_abs"] == 0.0]
        mism = [r for r in recs if "shape_mismatch" in r]
        maxd = max((r.get("max_abs", 0.0) for r in recs), default=0.0)
        # how many of the differing events ALSO had a different physical row idx
        diff_diffidx = [r for r in diffs if not r.get("same_idx", False)]
        print(f"\n[{kind}]  events={len(recs)}  identical(stock==leaf)={len(ident)}  "
              f"DIFFERING={len(diffs)}  shape_mismatch={len(mism)}  max|stock-leaf|={maxd:.6g}")
        if diffs:
            overall_changed = True
            print(f"   -> redirect CHANGES the committed value on {len(diffs)} event(s) "
                  f"({len(diff_diffidx)} of which point at a different row idx).")
            for r in sorted(diffs, key=lambda x: -x.get("max_abs", 0.0))[:6]:
                print(f"      L{r['layer']} n={r['snap_fired']} stock_idx={r['stock_idx']} "
                      f"leaf_idx={r['leaf_idx']} max_abs={r['max_abs']:.6g} nonzero={r['nonzero']}/{r['numel']}")
        else:
            print(f"   -> stock==leaf on ALL {len(ident)} event(s): redirect is INERT for {kind} "
                  f"(stock copy already correct -> {kind} is NOT the Axis-A carrier).")
        if mism:
            for r in mism[:3]:
                print(f"      shape_mismatch {r['file']}: {r['shape_mismatch']}")

    print("\nVERDICT:", "redirect DOES change committed value (needs ground-truth ref to judge correctness)"
          if overall_changed else
          "redirect INERT everywhere (stock already correct) -> carrier is NOT this snapshot row")


if __name__ == "__main__":
    main()
