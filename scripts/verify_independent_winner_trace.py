#!/usr/bin/env python3
"""Verify independent-row winner trace integrity."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _iter_rows(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_no, {"_parse_error": str(exc)}


def summarize(path: Path) -> dict:
    rows = 0
    parse_errors = 0
    superset_violations = 0
    copy_missing_sum = 0
    recovered_token_total = 0
    winner_nonzero_spine_events = 0
    winner_acc_total = 0
    spine0_acc_total = 0
    winner_spines: Counter[int] = Counter()
    malformed_rows = 0
    examples = []

    for line_no, row in _iter_rows(path):
        rows += 1
        if "_parse_error" in row:
            parse_errors += 1
            if len(examples) < 5:
                examples.append({"line": line_no, "error": row["_parse_error"]})
            continue

        try:
            counts = {str(k): int(v) for k, v in (row.get("counts") or {}).items()}
            winner_acc = int(row.get("winner_acc"))
            winner_spine = int(row.get("winner_spine", 0))
        except Exception as exc:
            malformed_rows += 1
            if len(examples) < 5:
                examples.append({"line": line_no, "error": f"malformed row: {exc}"})
            continue

        max_count = max(counts.values()) if counts else -1
        if winner_acc < max_count:
            superset_violations += 1
            if len(examples) < 5:
                examples.append({
                    "line": line_no,
                    "winner_acc": winner_acc,
                    "counts": counts,
                    "reason": "winner below max spine count",
                })

        copy = row.get("copy") or {}
        copy_missing_sum += int(copy.get("missing") or 0)
        spine0_acc = int(counts.get("0", 0))
        recovered_token_total += max(0, winner_acc - spine0_acc)
        winner_acc_total += winner_acc
        spine0_acc_total += spine0_acc
        winner_spines[winner_spine] += 1
        if winner_spine != 0:
            winner_nonzero_spine_events += 1

    return {
        "path": str(path),
        "rows": rows,
        "parse_errors": parse_errors,
        "malformed_rows": malformed_rows,
        "superset_violations": superset_violations,
        "copy_missing_sum": copy_missing_sum,
        "winner_nonzero_spine_events": winner_nonzero_spine_events,
        "recovered_token_total": recovered_token_total,
        "avg_winner_acc": (winner_acc_total / rows) if rows else None,
        "avg_spine0_acc": (spine0_acc_total / rows) if rows else None,
        "winner_spines": dict(sorted(winner_spines.items())),
        "examples": examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--require-recovery", action="store_true")
    args = ap.parse_args()

    if not args.trace.exists():
        sys.stderr.write(f"winner trace missing: {args.trace}\n")
        return 2

    summary = summarize(args.trace)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.json:
        print(text)
    else:
        print(
            f"rows={summary['rows']} "
            f"viol={summary['superset_violations']} "
            f"copy_missing_sum={summary['copy_missing_sum']} "
            f"recovered={summary['recovered_token_total']}"
        )

    failed = (
        summary["rows"] <= 0
        or summary["parse_errors"] > 0
        or summary["malformed_rows"] > 0
        or summary["superset_violations"] > 0
        or summary["copy_missing_sum"] > 0
    )
    if args.require_recovery and (
        summary["winner_nonzero_spine_events"] <= 0
        or summary["recovered_token_total"] <= 0
    ):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
