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
    lossless_suppressed_superset_events = 0
    copy_missing_sum = 0
    recovered_token_total = 0
    hidden_recovery_opportunity_total = 0
    winner_nonzero_spine_events = 0
    hidden_winner_suppressed_events = 0
    lossless_public_stream_events = 0
    non_lossless_public_stream_events = 0
    selector_enabled_events = 0
    winner_acc_total = 0
    spine0_acc_total = 0
    winner_spines: Counter[int] = Counter()
    policies: Counter[str] = Counter()
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
        hidden_suppressed = bool(row.get("hidden_winner_suppressed_reason"))
        lossless_public_stream = bool(row.get("lossless_public_stream"))
        policy = str(row.get("policy") or "unlabeled")
        policies[policy] += 1
        if row.get("selector_enabled"):
            selector_enabled_events += 1
        if lossless_public_stream:
            lossless_public_stream_events += 1
        else:
            non_lossless_public_stream_events += 1
        if winner_acc < max_count:
            if (
                policy == "lossless"
                and lossless_public_stream
                and winner_spine == 0
                and row.get("candidate_winner_spine") not in (None, 0)
                and row.get("hidden_winner_suppressed_reason")
                == "no_lossless_selector"
            ):
                lossless_suppressed_superset_events += 1
            else:
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
        if hidden_suppressed:
            hidden_winner_suppressed_events += 1
            if len(examples) < 5:
                examples.append({
                    "line": line_no,
                    "reason": "hidden winner suppressed",
                    "hidden_winner_suppressed_reason": row.get(
                        "hidden_winner_suppressed_reason"),
                    "candidate_winner_spine": row.get("candidate_winner_spine"),
                    "candidate_winner_acc": row.get("candidate_winner_acc"),
                    "winner_spine": winner_spine,
                    "winner_acc": winner_acc,
                })
        spine0_acc = int(counts.get("0", 0))
        candidate_winner_acc = row.get("candidate_winner_acc")
        try:
            hidden_recovery_opportunity_total += max(
                0,
                int(candidate_winner_acc) - spine0_acc,
            )
        except Exception:
            pass
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
        "lossless_suppressed_superset_events": lossless_suppressed_superset_events,
        "copy_missing_sum": copy_missing_sum,
        "winner_nonzero_spine_events": winner_nonzero_spine_events,
        "hidden_winner_suppressed_events": hidden_winner_suppressed_events,
        "recovered_token_total": recovered_token_total,
        "hidden_recovery_opportunity_total": hidden_recovery_opportunity_total,
        "lossless_public_stream_events": lossless_public_stream_events,
        "non_lossless_public_stream_events": non_lossless_public_stream_events,
        "selector_enabled_events": selector_enabled_events,
        "policies": dict(sorted(policies.items())),
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
            f"suppressed={summary['hidden_winner_suppressed_events']} "
            f"lossless_stream={summary['lossless_public_stream_events']} "
            f"recovered={summary['recovered_token_total']}"
        )

    failed = (
        summary["rows"] <= 0
        or summary["parse_errors"] > 0
        or summary["malformed_rows"] > 0
        or summary["superset_violations"] > 0
        or summary["copy_missing_sum"] > 0
        or summary["non_lossless_public_stream_events"] > 0
    )
    if args.require_recovery and (
        summary["hidden_recovery_opportunity_total"] <= 0
        and summary["recovered_token_total"] <= 0
    ):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
