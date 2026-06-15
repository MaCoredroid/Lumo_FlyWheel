#!/usr/bin/env python3
"""FR13_PLUS2 de-cascade: collapse clear-margin flip CLUSTERS into independent
divergence events, per the canonical rule the K1 bind re-derived (12/18/3).

RULE (gap-from-immediately-PRECEDING flip position, per prompt):
  Walk the sorted clear-margin flip positions.  A flip starts a NEW divergence
  event iff its gap to the IMMEDIATELY-PRECEDING clear-margin flip position is
  > 2 (i.e. gap <= 2 means it is the SAME cascade as the preceding flip and is
  collapsed).  The first flip in each prompt always starts an event.  The
  de-cascaded count = number of events = number of "gap>2" boundaries + 1 per
  non-empty prompt.

  NB the naive "gap-from-last-KEPT-anchor" rule wrongly gives 14 for K1; the
  preceding-position rule is canonical (FR13_K1_STORE_BOUNDARY_BIND verify).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def decascade_positions(positions: list[int]) -> int:
    """Number of independent events among sorted flip `positions` (gap>2 rule)."""
    if not positions:
        return 0
    ps = sorted(int(p) for p in positions)
    events = 1
    for i in range(1, len(ps)):
        if ps[i] - ps[i - 1] > 2:
            events += 1
    return events


def decascade_recur_json(path: str) -> dict:
    d = json.loads(Path(path).read_text())
    per_prompt = d["per_prompt"]
    raw_total = 0
    decasc_total = 0
    per_prompt_raw = []
    per_prompt_decasc = []
    for p in per_prompt:
        clear = [f["pos"] for f in p.get("flips", []) if f.get("clear_margin")]
        per_prompt_raw.append(len(clear))
        ev = decascade_positions(clear)
        per_prompt_decasc.append(ev)
        raw_total += len(clear)
        decasc_total += ev
    return {
        "raw_clear_total": raw_total,
        "raw_clear_per_prompt": per_prompt_raw,
        "decascaded_total": decasc_total,
        "decascaded_per_prompt": per_prompt_decasc,
    }


if __name__ == "__main__":
    out = decascade_recur_json(sys.argv[1])
    print(json.dumps(out, indent=2))
