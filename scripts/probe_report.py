#!/usr/bin/env python3
"""
CNB-55 Track 10 probe report.

Reads report/probe_runs.jsonl produced by probe_family.sh and prints a
calibration report:

    - per-variant mean / stdev / min / max / n
    - family mean (mean of per-variant means)
    - monotonicity check (V1 >= V2 >= V3 >= V4 >= V5 within tolerance)
    - pass/fail against the calibration targets in benchmark_run.md:
        family_mean in [15, 25]
        max_variant_score <= 40
        >= 1 variant mean <= 10
        monotonicity tolerance +/- 3

Usage:
    probe_report.py report/probe_runs.jsonl
    probe_report.py report/probe_runs.jsonl --probe-run-id 20260419T220000Z
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

VARIANT_ORDER = [
    "v1-clean-baseline",
    "v2-noisy-distractor",
    "v3-dirty-state",
    "v4-multi-corpus-objective",
    "v5-recovery-in-thread",
]

FAMILY_MEAN_WINDOW = (15.0, 25.0)
MAX_VARIANT_SCORE = 40.0
MIN_HARD_VARIANT_SCORE = 10.0
MONOTONICITY_TOLERANCE = 3.0


def load(path: Path, probe_run_id: str | None):
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if probe_run_id and rec.get("probe_run_id") != probe_run_id:
            continue
        rows.append(rec)
    return rows


def summarize(rows):
    by_variant = {v: [] for v in VARIANT_ORDER}
    for r in rows:
        if r["variant"] in by_variant:
            by_variant[r["variant"]].append(r)
    summary = {}
    for v, rs in by_variant.items():
        scores = [r["score"] for r in rs]
        if not scores:
            summary[v] = None
            continue
        summary[v] = {
            "n": len(scores),
            "mean": statistics.mean(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "scores": scores,
            "ceiling_hits": {},
            "shortcut_runs": sum(1 for r in rs if r.get("shortcut_detected")),
        }
        for r in rs:
            for c in r.get("ceilings_applied", []):
                summary[v]["ceiling_hits"][c] = summary[v]["ceiling_hits"].get(c, 0) + 1
    return summary


def assess(summary):
    variant_means = [
        summary[v]["mean"] for v in VARIANT_ORDER if summary.get(v) is not None
    ]
    family_mean = statistics.mean(variant_means) if variant_means else float("nan")
    max_variant_mean = max(variant_means) if variant_means else float("nan")
    min_variant_mean = min(variant_means) if variant_means else float("nan")

    # Monotonicity: each consecutive pair must satisfy earlier >= later - tolerance
    monotonic_ok = True
    monotonic_breaks = []
    for i in range(len(VARIANT_ORDER) - 1):
        a, b = VARIANT_ORDER[i], VARIANT_ORDER[i + 1]
        sa, sb = summary.get(a), summary.get(b)
        if sa is None or sb is None:
            continue
        if sa["mean"] + MONOTONICITY_TOLERANCE < sb["mean"]:
            monotonic_ok = False
            monotonic_breaks.append(
                f"{a} ({sa['mean']:.1f}) < {b} ({sb['mean']:.1f}) beyond +/-{MONOTONICITY_TOLERANCE}"
            )

    family_ok = FAMILY_MEAN_WINDOW[0] <= family_mean <= FAMILY_MEAN_WINDOW[1]
    max_variant_ok = max_variant_mean <= MAX_VARIANT_SCORE
    hard_variant_ok = min_variant_mean <= MIN_HARD_VARIANT_SCORE

    return {
        "family_mean": family_mean,
        "max_variant_mean": max_variant_mean,
        "min_variant_mean": min_variant_mean,
        "family_mean_ok": family_ok,
        "max_variant_ok": max_variant_ok,
        "hard_variant_ok": hard_variant_ok,
        "monotonic_ok": monotonic_ok,
        "monotonic_breaks": monotonic_breaks,
        "all_pass": family_ok and max_variant_ok and hard_variant_ok and monotonic_ok,
    }


def fmt_check(name, ok, detail):
    mark = "PASS" if ok else "FAIL"
    return f"  [{mark}] {name}: {detail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--probe-run-id", default=None)
    ap.add_argument("--emit-json", action="store_true",
                    help="Also print a JSON summary block")
    args = ap.parse_args()

    rows = load(args.jsonl, args.probe_run_id)
    if not rows:
        print("no matching rows", file=sys.stderr)
        sys.exit(2)

    summary = summarize(rows)
    assessment = assess(summary)

    scope = f" probe_run_id={args.probe_run_id}" if args.probe_run_id else ""
    print(f"CNB-55 Track 10 probe report — {len(rows)} runs{scope}")
    print("")
    header = f"{'variant':<32} {'n':>3} {'mean':>7} {'stdev':>7} {'min':>5} {'max':>5}  scores  ceilings"
    print(header)
    print("-" * len(header))
    for v in VARIANT_ORDER:
        s = summary.get(v)
        if s is None:
            print(f"{v:<32}   -       -       -     -     -   (no runs)")
            continue
        ceil = ",".join(f"{k}x{n}" for k, n in sorted(s["ceiling_hits"].items())) or "-"
        scores_str = ",".join(str(x) for x in s["scores"])
        print(
            f"{v:<32} {s['n']:>3} {s['mean']:>7.2f} {s['stdev']:>7.2f} "
            f"{s['min']:>5} {s['max']:>5}  [{scores_str}]  {ceil}"
        )

    print("")
    print(f"family_mean = {assessment['family_mean']:.2f}   "
          f"(window {FAMILY_MEAN_WINDOW[0]}-{FAMILY_MEAN_WINDOW[1]})")
    print(f"max_variant_mean = {assessment['max_variant_mean']:.2f}   "
          f"(cap {MAX_VARIANT_SCORE})")
    print(f"min_variant_mean = {assessment['min_variant_mean']:.2f}   "
          f"(must have at least one <= {MIN_HARD_VARIANT_SCORE})")
    print("")
    print("Acceptance checks:")
    print(fmt_check("family_mean in window", assessment["family_mean_ok"],
                   f"{assessment['family_mean']:.2f}"))
    print(fmt_check("max variant <= cap", assessment["max_variant_ok"],
                   f"{assessment['max_variant_mean']:.2f}"))
    print(fmt_check("at least one variant <= hard floor",
                   assessment["hard_variant_ok"],
                   f"min variant mean = {assessment['min_variant_mean']:.2f}"))
    print(fmt_check("monotonic V1>=V2>=V3>=V4>=V5 +/-tolerance",
                   assessment["monotonic_ok"],
                   "; ".join(assessment["monotonic_breaks"]) or "ok"))
    print("")
    overall = "ALL PASS" if assessment["all_pass"] else "HARDEN NEEDED"
    print(f"overall: {overall}")

    if args.emit_json:
        print("")
        print("JSON_SUMMARY:")
        print(json.dumps({
            "summary": {v: s for v, s in summary.items()},
            "assessment": assessment,
        }, indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
