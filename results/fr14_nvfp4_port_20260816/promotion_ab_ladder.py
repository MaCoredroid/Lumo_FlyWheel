#!/usr/bin/env python3
"""FR14 round 21 — the acceptance ladder, and the conditions it must clear.

Sealed before the counter landed; then CORRECTED against the landed payload's real
semantics (schema fr13.fixed32.accept_ladder.v1) before any run produced a number.
The first draft asserted `sum(ladder) == accepted-token delta`, which is wrong for
this schema and would have fail-closed a perfectly good ladder: the slot INDEX is the
accepted LENGTH, so sum(ladder) is ROWS, not tokens. Recording that here because
catching it after a multi-hour boot would have cost the run.

PAYLOAD SEMANTICS (src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py):
    slots      = 16, index i = "this step accepted i tokens", i in 0..15
    ladder[i]  = rows whose accepted length was i; lengths >= 15 CLAMPED into slot 15
    rows       = sum(ladder)
    accepted_tokens = sum(i * ladder[i]) + overflow_tokens   (exact; excess tracked)
    drain returns None when disabled -> absence is distinguishable from a zero ladder

THREE SELF-PROOFS, all must hold:
    1. rows          == delta spec_decode_num_drafts_total
    2. tokens        == delta spec_decode_num_accepted_tokens_total
    3. payload's own accepted_tokens == recomputed tokens   (internal consistency)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fr13.fixed32.accept_ladder.v1"
CLAMP_SLOT = 15


def _walk(obj: Any, path: str = ""):
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")


def find_payload(*docs: tuple[str, Any]) -> tuple[str, dict] | None:
    """Find the drain payload by its SCHEMA, not by key-name guessing."""
    for label, doc in docs:
        for path, node in _walk(doc):
            if isinstance(node, dict) and node.get("schema") == SCHEMA:
                return f"{label}:{path or '<root>'}", node
    return None


def dig(doc: Any, key: str):
    for _, node in _walk(doc):
        if isinstance(node, dict) and key in node:
            return node[key]
    return None


def selfproof(p: dict, drafts_delta: float | None, accepted_delta: float | None) -> dict:
    counts = [int(x) for x in p.get("ladder", [])]
    overflow_tokens = int(p.get("overflow_tokens", 0))
    rows = sum(counts)
    tokens = sum(i * c for i, c in enumerate(counts)) + overflow_tokens

    checks: dict[str, Any] = {
        "recomputed_rows": rows,
        "recomputed_tokens": tokens,
        "payload_rows": p.get("rows"),
        "payload_accepted_tokens": p.get("accepted_tokens"),
        "overflow_rows": p.get("overflow_rows"),
        "overflow_tokens": overflow_tokens,
    }
    # 3. internal consistency -- always checkable, needs no aggregate
    checks["internal_tokens_match"] = int(p.get("accepted_tokens", -1)) == tokens
    checks["internal_rows_match"] = int(p.get("rows", -1)) == rows

    # 1 & 2 -- against the counters we already trust
    if drafts_delta is not None:
        checks["aggregate_drafts_delta"] = int(round(drafts_delta))
        checks["rows_match_drafts"] = rows == int(round(drafts_delta))
    if accepted_delta is not None:
        checks["aggregate_accepted_delta"] = int(round(accepted_delta))
        checks["tokens_match_accepted"] = tokens == int(round(accepted_delta))

    gates = [v for k, v in checks.items() if isinstance(v, bool)]
    checks["PASS"] = bool(gates) and all(gates)
    checks["note"] = (
        "a ladder that cannot fail this check is not evidence; dropped or "
        "double-counted accumulation under graph replay diverges these sums"
    )
    return checks


def admissibility(p: dict, sp: dict, drafts_delta: float | None) -> dict:
    """A zero must mean acceptance was zero, never that the metric was absent."""
    counts = [int(x) for x in p.get("ladder", [])]
    rows = sum(counts)
    enabled = bool(p.get("enabled", False))
    saw_drafts = bool(drafts_delta and round(drafts_delta) > 0)

    if not enabled:
        return {"ADMISSIBLE": False,
                "verdict": "payload present but enabled=False -- instrument off, not a measured zero"}
    if rows == 0 and saw_drafts:
        return {"ADMISSIBLE": False,
                "verdict": "ZERO ROWS while the aggregate counted drafts -- "
                           "INSTRUMENT UNWIRED, NOT a measured zero. Do not report a ladder."}
    if rows == 0:
        return {"ADMISSIBLE": False,
                "verdict": "zero rows and no drafts -- nothing was served; no ladder to report"}
    if not sp["PASS"]:
        return {"ADMISSIBLE": False,
                "verdict": "self-proof FAILED -- the counter does not account for what the "
                           "aggregate counted; report the failure, not the numbers"}
    return {"ADMISSIBLE": True, "verdict": "ladder admissible; headline 1 reportable"}


def report(p: dict) -> dict:
    counts = [int(x) for x in p.get("ladder", [])]
    rows = sum(counts) or 1
    tokens = sum(i * c for i, c in enumerate(counts)) + int(p.get("overflow_tokens", 0))
    nz = [i for i, v in enumerate(counts) if v > 0]
    # slot index == accepted length. "past position 10" = accepted 10..14 tokens.
    # slot 15 is the CLAMP bucket (>=15) and is reported separately, not as "14".
    return {
        "ladder": counts,
        "slot_semantics": "index i = accepted i tokens; slot 15 is the >=15 clamp bucket",
        "rows": sum(counts),
        "accepted_tokens": tokens,
        "mean_accept_from_ladder": round(tokens / rows, 6),
        "highest_nonzero_slot": max(nz) if nz else None,
        "ladder_past_position_10": {f"accept_{i}": counts[i] for i in range(10, 15)
                                    if i < len(counts)},
        "nonzero_through_14": all(i < len(counts) and counts[i] > 0
                                  for i in range(10, 15)),
        "clamp_bucket_slot15": counts[CLAMP_SLOT] if len(counts) > CLAMP_SLOT else None,
        "overflow_rows": p.get("overflow_rows"),
        "share_of_rows_past_10": round(sum(counts[10:15]) / rows, 6),
        "flag": p.get("flag"),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: promotion_ab_ladder.py <artifact.json> [more.json ...]",
              file=sys.stderr)
        return 2
    docs = [(Path(a).name, json.loads(Path(a).read_text()))
            for a in argv[1:] if Path(a).is_file()]
    if not docs:
        print("no readable inputs", file=sys.stderr)
        return 2

    found = find_payload(*docs)
    out: dict[str, Any] = {"schema": "fr14.promotion_ab.ladder_check.v2",
                           "payload_found_at": found[0] if found else None}
    if not found:
        out["ADMISSIBLE"] = False
        out["VERDICT"] = (
            f"NO {SCHEMA} PAYLOAD IN ARTIFACTS -- headline 1 is instrument-blocked. "
            "Report it blocked; do NOT substitute the aggregate accept for a distribution."
        )
        print(json.dumps(out, indent=1))
        return 1

    p = found[1]
    # THE UNWIRED/DISABLED CASE emits enabled=False with ladder=None -- never zeros.
    # Handle it before any arithmetic: a crash here would be fail-closed by accident
    # rather than by design, and would not say WHY.
    if p.get("ladder") is None or not p.get("enabled", False):
        out["ADMISSIBLE"] = False
        out["payload"] = {k: p.get(k) for k in ("schema", "enabled", "flag", "slots")}
        out["VERDICT"] = (
            "INSTRUMENT ABSENT: the drain reported enabled=%s with ladder=%s. This is "
            "NOT a measured zero -- headline 1 is instrument-blocked for this run. Do "
            "NOT substitute the aggregate accept for a distribution."
            % (p.get("enabled"), p.get("ladder"))
        )
        print(json.dumps(out, indent=1))
        return 1

    drafts = accepted = None
    for _, d in docs:
        drafts = drafts if drafts is not None else dig(d, "vllm:spec_decode_num_drafts_total")
        accepted = accepted if accepted is not None else dig(
            d, "vllm:spec_decode_num_accepted_tokens_total")

    sp = selfproof(p, drafts, accepted)
    adm = admissibility(p, sp, drafts)
    out["self_proof"] = sp
    out["admissibility"] = adm
    out["report"] = report(p)
    out["ADMISSIBLE"] = adm["ADMISSIBLE"]
    out["VERDICT"] = adm["verdict"]
    print(json.dumps(out, indent=1))
    return 0 if adm["ADMISSIBLE"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
