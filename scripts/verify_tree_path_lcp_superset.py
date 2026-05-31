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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--min-rows", type=int, default=1)
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
    print(
        f"OK rows={len(rows)} max_accept="
        f"{max(int(row.get('accepted_len', 0)) for row in rows)} "
        f"violations=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
