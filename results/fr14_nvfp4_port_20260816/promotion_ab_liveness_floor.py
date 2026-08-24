#!/usr/bin/env python3
"""LIVENESS FLOOR — a task that did not generate is VACUOUS, never 'failed'.

MTP-5 replicate A returned swerc=0 with astropy-13236 'failed' in 2.157s and 0 bytes.
Banked as-is that reads 'no degeneration', which is false in every part that matters: the
agent never generated. A vacuous pass is worse than a refusal because a refusal announces
itself.

FLOOR, from the banked distribution rather than taste: every real 13236 run served >5 min
(20.6 min for the degeneration, 5436s for H27n); even 13453's fast clean run was ~10 min.
60s plus nonzero generated tokens is conservative by two orders of magnitude and catches
exactly the 2-second case.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

WALL_FLOOR_S = 60.0

def classify(instance_id, elapsed_s, patch_bytes, generated_tokens=None):
    reasons = []
    if elapsed_s is not None and elapsed_s < WALL_FLOOR_S:
        reasons.append(f"elapsed {elapsed_s}s < {WALL_FLOOR_S}s floor")
    if not patch_bytes:
        reasons.append("empty patch")
    if generated_tokens == 0:
        reasons.append("zero generated tokens")
    vacuous = bool(reasons) and (elapsed_s is not None and elapsed_s < WALL_FLOOR_S)
    return {
        "instance_id": instance_id, "elapsed_s": elapsed_s, "patch_bytes": patch_bytes,
        "generated_tokens": generated_tokens,
        "verdict": "VACUOUS_NOT_RUN" if vacuous else "admissible",
        "reasons": reasons,
        "counts_in_tally": not vacuous,
        "note": ("the agent did not generate; this is NOT a task outcome and must not be "
                 "recorded as failed/resolved" if vacuous else ""),
    }

def main(argv):
    if len(argv) < 2:
        print("usage: promotion_ab_liveness_floor.py <runlog-or-summary.json>", file=sys.stderr); return 2
    import re
    txt = Path(argv[1]).read_text(errors="replace")
    m = re.search(r'\{\s*"swe_orchestrator_rc".*?\n\}', txt, re.S)
    if not m:
        print(json.dumps({"error": "no orchestrator summary found"}, indent=1)); return 2
    d = json.loads(m.group(0))
    out = [classify(t.get("instance_id"), t.get("codex_elapsed_s"), t.get("patch_bytes")) for t in d.get("tasks", [])]
    doc = {"schema": "fr14.liveness_floor.v1", "wall_floor_s": WALL_FLOOR_S,
           "tasks": out,
           "admissible": [t["instance_id"] for t in out if t["counts_in_tally"]],
           "vacuous": [t["instance_id"] for t in out if not t["counts_in_tally"]]}
    print(json.dumps(doc, indent=1))
    return 1 if doc["vacuous"] else 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
