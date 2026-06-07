#!/usr/bin/env python3
"""Compare FR13 SWE-4 deliverable probe artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _tv(left: Counter[Any], right: Counter[Any]) -> float:
    left_n = sum(left.values())
    right_n = sum(right.values())
    keys = set(left) | set(right)
    if left_n == 0 and right_n == 0:
        return 0.0
    return 0.5 * sum(
        abs((left.get(k, 0) / left_n if left_n else 0.0) - (right.get(k, 0) / right_n if right_n else 0.0))
        for k in keys
    )


def _records(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(artifact.get("records") or [])
    rows.sort(key=lambda row: (int(row["sample_index"]), int(row["prompt_id"])))
    return rows


def _first_diff(left: list[int], right: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = _records(left)
    right_rows = _records(right)
    left_by_key = {
        (int(row["sample_index"]), int(row["prompt_id"])): [int(x) for x in row.get("token_ids", [])]
        for row in left_rows
    }
    right_by_key = {
        (int(row["sample_index"]), int(row["prompt_id"])): [int(x) for x in row.get("token_ids", [])]
        for row in right_rows
    }
    keys = sorted(set(left_by_key) | set(right_by_key))
    diffs = []
    exact = 0
    for key in keys:
        ltok = left_by_key.get(key, [])
        rtok = right_by_key.get(key, [])
        diff = _first_diff(ltok, rtok)
        if diff is None:
            exact += 1
        else:
            diffs.append(
                {
                    "sample_index": key[0],
                    "prompt_id": key[1],
                    "first_diff": diff,
                    "left_token": ltok[diff] if diff < len(ltok) else None,
                    "right_token": rtok[diff] if diff < len(rtok) else None,
                    "left_len": len(ltok),
                    "right_len": len(rtok),
                }
            )

    left_lengths = Counter(len(v) for v in left_by_key.values())
    right_lengths = Counter(len(v) for v in right_by_key.values())
    left_first = Counter(v[0] for v in left_by_key.values() if v)
    right_first = Counter(v[0] for v in right_by_key.values() if v)
    left_bag = Counter(token for value in left_by_key.values() for token in value)
    right_bag = Counter(token for value in right_by_key.values() for token in value)

    ls = left.get("summary", {})
    rs = right.get("summary", {})
    left_accept = float(ls.get("accepted_per_draft_event") or 0.0)
    right_accept = float(rs.get("accepted_per_draft_event") or 0.0)
    left_tps = float(ls.get("warm_decode_tps") or 0.0)
    right_tps = float(rs.get("warm_decode_tps") or 0.0)
    return {
        "left_label": left.get("label"),
        "right_label": right.get("label"),
        "records_compared": len(keys),
        "exact_token_sequence_matches": exact,
        "exact_token_sequence_match_rate": exact / len(keys) if keys else None,
        "first_mismatch": diffs[0] if diffs else None,
        "token_count_tv": _tv(left_lengths, right_lengths),
        "first_token_tv": _tv(left_first, right_first),
        "emitted_token_bag_tv": _tv(left_bag, right_bag),
        "accepted_per_draft_event_left": left_accept,
        "accepted_per_draft_event_right": right_accept,
        "accepted_per_draft_event_delta": left_accept - right_accept,
        "warm_decode_tps_left": left_tps,
        "warm_decode_tps_right": right_tps,
        "warm_decode_tps_delta": left_tps - right_tps,
        "left_summary": ls,
        "right_summary": rs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    payload = {
        "schema": "fr13.deliverable_compare.v1",
        "left": str(args.left),
        "right": str(args.right),
        "comparison": _compare(_load(args.left), _load(args.right)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
