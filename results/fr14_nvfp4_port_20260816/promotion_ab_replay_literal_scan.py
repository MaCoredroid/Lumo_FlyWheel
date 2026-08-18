#!/usr/bin/env python3
"""FR14: the scan dimension the shape-literal sweep does not cover — REPLAY counts.

WHY THIS EXISTS. `fr14_paired_contract_sweep.shape_literal_scan()` is correct and
covers all 65 injected blobs, but its candidate set is

    magnitudes = {15, 6, 10, 5, 4, 12, 16, 8, 14, 18}

which is the **column** dimension of the gate (15 native / 6 tail / 10 rescue /
31 pack). The split graph moves three independent dimensions, not one:

  columns  15/6/10 …   -> covered by the scan
  passes   5->3, post-root 4->2   -> partly covered (4 and 5 are in the set)
  REPLAYS  1->2                   -> NOT COVERED: neither 1 nor 2 is a magnitude

The 15th site found by the round-4 boot is exactly that:
`evidence.get("matching_replays") != 1` inside the `fixed_flush` blob's
`_fr13_f32_flush_reconcile`. It was scanned and it was not flagged, because the
stale literal is the number 1.

This scan closes that dimension. It reports every literal 1 or 2 sitting in a
REPLAY-counting position, in every injected blob, and marks the ones that are
provably fine (the forward-graph replay really is once per step) so the reviewer
adjudicates a short list instead of the whole file.

Output is a candidate list for a human, NOT a verdict: a `1` next to "replay" can
be perfectly correct, which is the whole reason the value cannot simply be banned.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path("/home/mark/shared/lumoFlyWheel-nvfp4-port-20260816")
sys.path.insert(0, str(REPO / "scripts"))
import fr14_paired_contract_sweep as sweep  # noqa: E402

# A replay-counting position: the identifier names a replay/capture count and the
# line compares it against, or assigns it, a small literal.
REPLAY_NAMES = (
    "replay", "replays", "graph_captures", "capture_count",
    "prior_replays", "matching_replays", "measured_replays",
    "unmeasured_replays",
)
LITERAL = re.compile(
    r"(?:[=!<>]=|[-+*]|\breturn\b|:)\s*(?:\w+\s*[-+*]\s*)?\b([12])\b"
)

# Adjudications are keyed on POSITION, never on the literal's text.
#
# This is not a stylistic choice -- the first draft of this scan keyed on text
# and promptly reproduced the exact blind spot it was written to expose. The
# fragment 'evidence.get("matching_replays") != 1' occurs TWICE in the
# `fixed_flush` blob: at blob line 379 it guards the FORWARD graph, where one
# replay per step is correct and must not change; at blob line 409 it guards the
# DRAFTER, where an armed ungated step replays twice and the literal is the 15th
# site. Identical text, opposite verdicts. A text-keyed allowlist marks both OK
# and hides the real one -- which is, to the character, the failure being
# reported. Adjudicate positions.
KNOWN = {
    (39286, 379): (
        "OK",
        "FORWARD-graph evidence: the forward CUDA graph really is replayed once "
        "per step regardless of the drafter's pass split. Correct; do not touch.",
    ),
    (39286, 409): (
        "STALE",
        "THE 15TH SITE, found by the round-4 boot. Drafter evidence: an armed "
        "UNGATED step replays lo then hi (2 replays, section 11.1), so this "
        "flush-time attestation refuses a legal step. Textual twin of line 379, "
        "opposite verdict.",
    ),
}


def main() -> int:
    rows = []
    for lineno, text, tree in sweep.all_injected_blobs():
        for offset, raw in enumerate(text.split("\n")):
            low = raw.lower()
            if not any(n in low for n in REPLAY_NAMES):
                continue
            if raw.lstrip().startswith("#"):
                continue
            m = LITERAL.search(raw)
            if m is None:
                continue
            verdict, why = KNOWN.get((lineno, offset + 1), (None, None))
            rows.append(
                {
                    "blob_lineno": lineno,
                    "line_in_blob": offset + 1,
                    "value": int(m.group(1)),
                    "source": raw.strip()[:200],
                    "parses": tree is not None,
                    "verdict": verdict or "REVIEW",
                    "why": why,
                }
            )

    out = {
        "schema": "fr14.promotion_ab.replay_literal_scan.v1",
        "rationale": (
            "shape_literal_scan()'s magnitudes {15,6,10,5,4,12,16,8,14,18} cover "
            "the COLUMN dimension of the gate. The split graph also moves the "
            "REPLAY dimension 1->2, and neither 1 nor 2 is in that set, so the "
            "15th site (matching_replays != 1 in the fixed_flush blob) was "
            "scanned without being flagged."
        ),
        "blobs_scanned": len(sweep.all_injected_blobs()),
        "candidates": len(rows),
        "for_review": sum(1 for r in rows if r["verdict"] == "REVIEW"),
        "known_stale": [r for r in rows if r["verdict"] == "STALE"],
        "rows": rows,
    }
    Path(sys.argv[1] if len(sys.argv) > 1
         else "promotion_ab_replay_literal_scan.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(f"blobs scanned: {out['blobs_scanned']}")
    print(f"replay-position 1/2 literals: {out['candidates']} "
          f"({out['for_review']} for review)")
    for r in rows:
        mark = {"OK": "OK    ", "STALE": "STALE!", None: "REVIEW"}.get(
            r["verdict"], "REVIEW")
        print(f"  {mark} blob@{r['blob_lineno']}+{r['line_in_blob']:<5} "
              f"val={r['value']}  {r['source'][:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
