#!/usr/bin/env python3
"""FR13_GDN_SUBOP_MAB reduce: localize the L0-GDN co-residency carrier sub-op.

Reads the JSONL written by the in-process FR13_GDN_SUBOP_MAB hook (one record per
captured tree-verify event) and prints the localization verdict:

  * The CLEAN co-residency axis = M10-vs-M5 (full tree vs spine-slice; branch rows
    present vs absent, deep node's context otherwise identical).  The first sub-op
    (pre_conv -> conv1d_out -> scan_out) whose deep-row M10-vs-M5 RAW != 0 is where
    co-residency (row-occupancy) first perturbs the committed-spine row.
  * The receptive-field axis = M5-vs-M1 (spine ancestry vs deep-row-alone): a large
    value here with M10-vs-M5 ~0 localizes the divergence to the STATEFUL receptive
    field (conv window / recurrent ancestry), NOT a batch-tile occupancy effect.
  * conv1d_out.served_fused_vs_native_decode_max_abs = the kernel-identity
    realization seam (our bf16-tap MAC + ex2.silu vs native causal_conv1d_update),
    the ~9.77e-4 carrier FR13_CONV_FIX_DESIGN names (NOT a row-occupancy effect).

Usage:
  python3 scripts/fr13_gdn_subop_mab_reduce.py \
      --jsonl output/fr13_gdn_subop_mab/fr13_gdn_subop_mab.jsonl \
      [--out output/fr13_gdn_subop_mab/reduce.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUBOPS = ["pre_conv", "conv1d_out", "scan_out"]


def _load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def _agg_max(records: list[dict], subop: str, key: str) -> float:
    vals = []
    for r in records:
        v = r.get("subops", {}).get(subop, {}).get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return max(vals) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.0)
    args = ap.parse_args()

    records = _load(args.jsonl)
    if not records:
        print("FR13_GDN_SUBOP_MAB reduce: NO records (hook did not fire / vacuous)")
        return 1

    # Aggregate the worst (max) divergence per sub-op across all captured events.
    by_subop = {}
    for s in SUBOPS:
        by_subop[s] = {
            "m10_vs_m5_max_abs": _agg_max(records, s, "m10_deep_vs_m5_deep_max_abs"),
            "m5_vs_m1_max_abs": _agg_max(records, s, "m5_deep_vs_m1_max_abs"),
            "m10_vs_m1_max_abs": _agg_max(records, s, "m10_deep_vs_m1_max_abs"),
        }
    by_subop["conv1d_out"]["served_fused_vs_native_decode_max_abs"] = _agg_max(
        records, "conv1d_out", "served_fused_vs_native_decode_max_abs"
    )

    first_coresid = next(
        (s for s in SUBOPS if by_subop[s]["m10_vs_m5_max_abs"] > args.threshold), None
    )
    first_rf = next(
        (s for s in SUBOPS if by_subop[s]["m5_vs_m1_max_abs"] > args.threshold), None
    )
    pre_conv_m_invariant = all(
        bool(r.get("pre_conv_m_invariant", False)) for r in records
    )

    verdict = {
        "schema": "fr13.gdn_subop_mab_reduce.v1",
        "jsonl": str(args.jsonl),
        "num_events": len(records),
        "tree_n": sorted({int(r.get("tree_n", -1)) for r in records}),
        "deep_rows": sorted({int(r.get("deep_row", -1)) for r in records}),
        "by_subop": by_subop,
        "first_coresidency_subop_m10_vs_m5": first_coresid,
        "first_receptive_field_subop_m5_vs_m1": first_rf,
        "pre_conv_m_invariant_all_events": pre_conv_m_invariant,
        "threshold": args.threshold,
        "interpretation": (
            "first_coresidency_subop = where ROW-OCCUPANCY co-residency first "
            "perturbs the deep-spine row (None => row-occupancy M-invariant, as "
            "FR13_CONV_FIX_DESIGN predicts for the conv emulation). A non-None "
            "first_receptive_field_subop with row-occupancy invariant localizes "
            "the carrier to the STATEFUL receptive field; "
            "conv1d_out.served_fused_vs_native_decode_max_abs > 0 is the "
            "our-kernel-vs-native realization seam (the ~9.77e-4 carrier)."
        ),
    }
    blob = json.dumps(verdict, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob + "\n")
    print(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
