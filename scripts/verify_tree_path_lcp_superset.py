#!/usr/bin/env python3
"""Verify FR7 tree path-LCP superset logs.

The runtime verifier writes one JSON object per request/event to
tree_path_lcp_max.jsonl. This checker makes the proof mechanical: every row must
report winner accepted length >= path0 LCP, and the reported winner must equal
the max over all logged root-to-leaf paths.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if row.get("event") == "tree_path_lcp_max" or "path_scores" in row:
            if int(row.get("node_count", 1) or 0) <= 0:
                continue
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def verify_rows(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for idx, row in enumerate(rows):
        loc = f"line {row.get('_line_no', idx + 1)}"
        accepted_len = int(row.get("accepted_len", -1))
        path0_lcp = int(row.get("path0_lcp", 0))
        if row.get("superset_violation") or accepted_len < path0_lcp:
            errors.append(
                f"{loc}: superset violation accepted_len={accepted_len} "
                f"path0_lcp={path0_lcp}"
            )
        path_scores = row.get("path_scores") or []
        if not isinstance(path_scores, list) or not path_scores:
            errors.append(f"{loc}: missing path_scores")
            continue
        max_lcp = max(int(item.get("lcp", -1)) for item in path_scores)
        if accepted_len != max_lcp:
            errors.append(
                f"{loc}: accepted_len={accepted_len} does not equal "
                f"max path lcp={max_lcp}"
            )
        first_lcp = int(path_scores[0].get("lcp", 0))
        if path0_lcp != first_lcp:
            errors.append(
                f"{loc}: path0_lcp={path0_lcp} does not match first "
                f"path score={first_lcp}"
            )
    return not errors, errors


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [int(row.get("accepted_len", 0)) for row in rows]
    path0 = [int(row.get("path0_lcp", 0)) for row in rows]
    deltas = [acc - p0 for acc, p0 in zip(accepted, path0, strict=True)]
    recovery_events = [delta for delta in deltas if delta > 0]
    path_counts = [
        len(row.get("path_scores") or [])
        for row in rows
    ]
    winner_indices: dict[str, int] = {}
    for row in rows:
        winner_leaf = row.get("winner_leaf")
        path_scores = row.get("path_scores") or []
        winner_idx = None
        for idx, score in enumerate(path_scores):
            if winner_leaf is not None and int(score.get("leaf", -1)) == int(winner_leaf):
                winner_idx = idx
                break
        key = str(winner_idx if winner_idx is not None else "unknown")
        winner_indices[key] = winner_indices.get(key, 0) + 1

    row_count = len(rows)
    accepted_total = sum(accepted)
    path0_total = sum(path0)
    return {
        "rows": row_count,
        "accepted_total": accepted_total,
        "path0_total": path0_total,
        "superset_delta_total": sum(deltas),
        "recovered_token_total": sum(recovery_events),
        "recovery_event_count": len(recovery_events),
        "recovery_event_rate": (len(recovery_events) / row_count) if row_count else 0.0,
        "avg_accepted_len": (accepted_total / row_count) if row_count else 0.0,
        "avg_path0_lcp": (path0_total / row_count) if row_count else 0.0,
        "avg_superset_delta": (sum(deltas) / row_count) if row_count else 0.0,
        "max_accept": max(accepted) if accepted else 0,
        "max_path_count": max(path_counts) if path_counts else 0,
        "min_path_count": min(path_counts) if path_counts else 0,
        "winner_index_counts": winner_indices,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--min-path0-avg", type=float, default=None,
                        help="fail if average path0 LCP is below this value")
    parser.add_argument("--min-winner-avg", type=float, default=None,
                        help="fail if average winner accepted length is below this value")
    parser.add_argument("--json", action="store_true",
                        help="print the summary as JSON")
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"missing trace: {args.path}")
    rows = _load_rows(args.path)
    if len(rows) < args.min_rows:
        raise SystemExit(
            f"insufficient rows: found {len(rows)}, expected >= {args.min_rows}")
    ok, errors = verify_rows(rows)
    if not ok:
        for error in errors[:50]:
            print(error, file=sys.stderr)
        if len(errors) > 50:
            print(f"... {len(errors) - 50} more errors", file=sys.stderr)
        return 1
    summary = summarize_rows(rows)
    threshold_errors = []
    if (args.min_path0_avg is not None
            and summary["avg_path0_lcp"] < args.min_path0_avg):
        threshold_errors.append(
            f"avg_path0_lcp={summary['avg_path0_lcp']:.3f} below "
            f"--min-path0-avg={args.min_path0_avg:.3f}"
        )
    if (args.min_winner_avg is not None
            and summary["avg_accepted_len"] < args.min_winner_avg):
        threshold_errors.append(
            f"avg_accepted_len={summary['avg_accepted_len']:.3f} below "
            f"--min-winner-avg={args.min_winner_avg:.3f}"
        )
    if threshold_errors:
        for error in threshold_errors:
            print(error, file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, sort_keys=True))
        return 0
    print(
        f"OK rows={summary['rows']} max_accept={summary['max_accept']} "
        f"avg_path0={summary['avg_path0_lcp']:.3f} "
        f"avg_winner={summary['avg_accepted_len']:.3f} "
        f"recovery_events={summary['recovery_event_count']} "
        f"recovered_tokens={summary['recovered_token_total']} violations=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
