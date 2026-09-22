#!/usr/bin/env python3
"""Render all planned E1 cells and supplied contrasts without changing analysis.

Use only after independent review of the aggregate. This file is presentation
code, not an eligibility checker or a new statistical analysis.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("aggregate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.aggregate.read_text())
    cells = sorted(data["cells"], key=lambda row: row["index"])
    assert len(cells) == 18
    assert [row["index"] for row in cells] == list(range(1, 19))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in cells:
        support = row.get("support") or {}
        if row["rate"] is not None:
            assert row["status"] == "VALID" and row.get("terminal_seal_utc")
            tokens = support["sum_emitted_tokens_api_bound_pure_support"]
            wall = support["sum_wall_s_unique_physical_steps"]
            assert wall > 0 and abs(tokens / wall - row["rate"]) < 1e-10
        records.append({
            "cell": row["index"], "block": row["block"],
            "arm": row["arm"], "batch": row["batch"],
            "status": row["status"], "attempt": row.get("attempt"),
            "retained_intervals": support.get("n_usable"),
            "api_bound_tokens": support.get("sum_emitted_tokens_api_bound_pure_support"),
            "retained_wall_s": support.get("sum_wall_s_unique_physical_steps"),
            "tokens_per_retained_wall_s": row["rate"],
            "seal_utc": row.get("terminal_seal_utc"),
        })
    with (args.output_dir / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    contrasts = []
    for batch, result in data["by_batch"].items():
        for name, row in result["contrasts"].items():
            interval = row.get("bootstrap_95pct_interval", [None, None])
            contrasts.append({
                "batch": batch, "contrast": name, "status": row["status"],
                "mean_difference": row.get("mean_difference"),
                "lower_95pct": interval[0], "upper_95pct": interval[1],
                "precision_half_width_over_baseline": row.get("precision_half_width_over_baseline"),
                "target": row.get("target"),
            })
    with (args.output_dir / "contrasts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    proof = {
        "source": str(args.aggregate.resolve()),
        "source_sha256": hashlib.sha256(args.aggregate.read_bytes()).hexdigest(),
        "scope": "All 18 planned cells; missing numerical support stays empty, not zero. Contrasts and intervals are copied from the frozen aggregate without recomputation.",
        "files": {name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
                  for name in ("cells.csv", "contrasts.csv")},
    }
    (args.output_dir / "tables-source.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
