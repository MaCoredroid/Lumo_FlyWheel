#!/usr/bin/env python3
"""Compute FR10 same-regime temp sampling distribution distance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_equivalence_gate import (  # noqa: E402
    load_sampling_artifact,
    sampling_distribution_distance,
    summarize_sampling_distance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--out", required=True)
    parser.add_argument("--positions", type=int, default=16)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    left = load_sampling_artifact(args.left)
    right = load_sampling_artifact(args.right)
    rows = sampling_distribution_distance(
        left,
        right,
        positions=args.positions,
        per_prompt=True,
        top=args.top,
    )
    summary = summarize_sampling_distance(rows)
    payload = {
        "left": args.left,
        "right": args.right,
        "left_samples": len(left),
        "right_samples": len(right),
        "positions": args.positions,
        "top": args.top,
        "summary": summary,
        "rows": [row.__dict__ for row in rows],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
