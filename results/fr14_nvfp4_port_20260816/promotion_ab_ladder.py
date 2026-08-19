#!/usr/bin/env python3
"""FR14 round 21 — the acceptance ladder, and the five conditions it must clear.

Prep written BEFORE the counter landed, so the verification is fixed in advance of
seeing any number. The ladder is headline 1; it is also a brand-new instrument, and
a brand-new instrument reporting a zero is exactly the round-6 failure mode (a
missing metric read as a measured zero). So this refuses to report a ladder that
cannot prove itself.

Discovers the counter wherever the drafter lane put it, rather than pinning a path
I would be guessing at.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

LADDER_HINTS = (
    "accepted_per_pos", "accept_per_pos", "ladder", "per_pos",
    "accepted_tokens_per_pos", "pos_hist", "accept_ladder",
)


def _walk(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        yield path, obj


def find_ladder(*docs: tuple[str, Any]) -> tuple[str, list[int]] | None:
    """A ladder is an int list of plausible slot count whose key looks like one."""
    best = None
    for label, doc in docs:
        for path, val in _walk(doc):
            leaf = path.rsplit(".", 1)[-1].lower()
            if not any(h in leaf for h in LADDER_HINTS):
                continue
            if not val or not all(isinstance(x, (int, float)) for x in val):
                continue
            if not 8 <= len(val) <= 64:
                continue
            cand = (f"{label}:{path}", [int(x) for x in val])
            # prefer the longest plausible vector
            if best is None or len(cand[1]) > len(best[1]):
                best = cand
    return best


def selfproof(ladder: list[int], accepted_total_delta: float) -> dict:
    """CONDITION: ladder-sum == aggregate accept delta. Non-negotiable."""
    s = int(sum(ladder))
    exp = int(round(accepted_total_delta))
    return {
        "ladder_sum": s,
        "aggregate_accepted_tokens_delta": exp,
        "difference": s - exp,
        "PASS": s == exp,
        "note": (
            "the 16-slot counter must account for every accepted token the "
            "aggregate counted; any difference means the counter misses a path"
        ),
    }


def zero_case(ladder: list[int], drafts: float, accepted_total_delta: float) -> dict:
    """CONDITION: a zero must mean acceptance was zero, never that the metric absent.

    An all-zero ladder is only admissible if the aggregate ALSO says zero. If the
    aggregate saw accepted tokens and the ladder is flat zero, the instrument is
    absent/not wired -- report that, never 'the ladder is dead past position 10'.
    """
    all_zero = all(x == 0 for x in ladder)
    agg_zero = int(round(accepted_total_delta)) == 0
    if all_zero and not agg_zero:
        verdict = "INSTRUMENT ABSENT OR UNWIRED -- NOT a measured zero. Do not report a ladder."
        ok = False
    elif all_zero and agg_zero:
        verdict = "genuine zero: no acceptance occurred in the window"
        ok = True
    else:
        verdict = "ladder is populated; zero-case not triggered"
        ok = True
    return {
        "all_zero": all_zero,
        "aggregate_says_zero": agg_zero,
        "drafts": int(round(drafts)),
        "ADMISSIBLE": ok,
        "verdict": verdict,
    }


def report(ladder: list[int]) -> dict:
    nz = [i for i, v in enumerate(ladder) if v > 0]
    total = sum(ladder) or 1
    return {
        "ladder": ladder,
        "slots": len(ladder),
        "highest_nonzero_position": max(nz) if nz else None,
        "nonzero_positions": nz,
        # headline 1: does the tail10 ladder actually run past position 10?
        "past_position_10": {
            f"pos_{i}": ladder[i] for i in range(10, min(len(ladder), 15))
        },
        "nonzero_through_14": all(
            i < len(ladder) and ladder[i] > 0 for i in range(10, 15)
        ),
        "share_past_10": round(sum(ladder[10:]) / total, 6),
        "mean_accept_from_ladder": round(
            sum((i + 1) * v for i, v in enumerate(ladder)) / total, 6
        ),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: promotion_ab_ladder.py <arm.json> [deploy_speed.json]\n"
            "  prints the ladder with its self-proof; exits 1 if inadmissible",
            file=sys.stderr,
        )
        return 2
    docs = []
    for p in argv[1:]:
        path = Path(p)
        if path.is_file():
            docs.append((path.name, json.loads(path.read_text())))
    if not docs:
        print("no readable inputs", file=sys.stderr)
        return 2

    found = find_ladder(*docs)
    agg = {}
    for _, d in docs:
        for path, val in _walk(d):
            pass
    # locate the aggregate counters wherever they sit
    def dig(doc, key):
        if isinstance(doc, dict):
            for k, v in doc.items():
                if k == key:
                    return v
                r = dig(v, key)
                if r is not None:
                    return r
        return None

    accepted = None
    drafts = None
    for _, d in docs:
        accepted = accepted or dig(d, "vllm:spec_decode_num_accepted_tokens_total")
        drafts = drafts or dig(d, "vllm:spec_decode_num_drafts_total")

    out: dict[str, Any] = {
        "schema": "fr14.promotion_ab.ladder.v1",
        "ladder_found_at": found[0] if found else None,
        "aggregate_accepted_tokens_total": accepted,
        "aggregate_drafts_total": drafts,
    }
    if not found:
        out["VERDICT"] = (
            "NO LADDER IN ARTIFACTS -- headline 1 is instrument-blocked. Report it "
            "as blocked; do NOT substitute the aggregate accept for a distribution."
        )
        print(json.dumps(out, indent=1))
        return 1

    ladder = found[1]
    out["self_proof"] = selfproof(ladder, accepted or 0.0)
    out["zero_case"] = zero_case(ladder, drafts or 0.0, accepted or 0.0)
    out["report"] = report(ladder)
    admissible = out["self_proof"]["PASS"] and out["zero_case"]["ADMISSIBLE"]
    out["ADMISSIBLE"] = admissible
    out["VERDICT"] = (
        "ladder admissible; headline 1 reportable"
        if admissible
        else "LADDER INADMISSIBLE -- report the failure, not the numbers"
    )
    print(json.dumps(out, indent=1))
    return 0 if admissible else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
