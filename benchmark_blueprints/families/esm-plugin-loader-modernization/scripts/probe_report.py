#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def load_rows(path: Path, probe_run_id: str | None) -> list[dict]:
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


def summarize(rows: list[dict]) -> dict[str, dict | None]:
    by_variant = {variant: [] for variant in VARIANT_ORDER}
    for row in rows:
        if row["variant"] in by_variant:
            by_variant[row["variant"]].append(row)

    summary: dict[str, dict | None] = {}
    for variant, variant_rows in by_variant.items():
        if not variant_rows:
            summary[variant] = None
            continue
        scores = [row["score"] for row in variant_rows]
        ceiling_hits: dict[str, int] = {}
        for row in variant_rows:
            for ceiling in row.get("ceilings_applied", []):
                ceiling_hits[ceiling] = ceiling_hits.get(ceiling, 0) + 1
        summary[variant] = {
            "n": len(variant_rows),
            "mean": statistics.mean(scores),
            "stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "scores": scores,
            "ceiling_hits": ceiling_hits,
            "m_training_mean": statistics.mean(row.get("M_training", 0.0) for row in variant_rows),
            "raw_mean": statistics.mean(row.get("raw_score_pre_ceiling", 0) for row in variant_rows),
        }
    return summary


def assess(summary: dict[str, dict | None]) -> dict[str, object]:
    variant_means = [
        summary[variant]["mean"] for variant in VARIANT_ORDER if summary[variant] is not None
    ]
    family_mean = statistics.mean(variant_means) if variant_means else float("nan")
    max_variant_mean = max(variant_means) if variant_means else float("nan")
    min_variant_mean = min(variant_means) if variant_means else float("nan")

    monotonic_ok = True
    monotonic_breaks = []
    for index in range(len(VARIANT_ORDER) - 1):
        left = VARIANT_ORDER[index]
        right = VARIANT_ORDER[index + 1]
        left_summary = summary.get(left)
        right_summary = summary.get(right)
        if left_summary is None or right_summary is None:
            continue
        if left_summary["mean"] + MONOTONICITY_TOLERANCE < right_summary["mean"]:
            monotonic_ok = False
            monotonic_breaks.append(
                f"{left} ({left_summary['mean']:.2f}) < {right} ({right_summary['mean']:.2f}) beyond +/-{MONOTONICITY_TOLERANCE}"
            )

    return {
        "family_mean": family_mean,
        "max_variant_mean": max_variant_mean,
        "min_variant_mean": min_variant_mean,
        "family_mean_ok": FAMILY_MEAN_WINDOW[0] <= family_mean <= FAMILY_MEAN_WINDOW[1],
        "max_variant_ok": max_variant_mean <= MAX_VARIANT_SCORE,
        "hard_variant_ok": min_variant_mean <= MIN_HARD_VARIANT_SCORE,
        "monotonic_ok": monotonic_ok,
        "monotonic_breaks": monotonic_breaks,
        "all_pass": (
            FAMILY_MEAN_WINDOW[0] <= family_mean <= FAMILY_MEAN_WINDOW[1]
            and max_variant_mean <= MAX_VARIANT_SCORE
            and min_variant_mean <= MIN_HARD_VARIANT_SCORE
            and monotonic_ok
        ),
    }


def fmt_check(name: str, ok: bool, detail: str) -> str:
    return f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--probe-run-id", default=None)
    parser.add_argument("--emit-json", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.jsonl, args.probe_run_id)
    if not rows:
        print("no matching rows", file=sys.stderr)
        return 2

    summary = summarize(rows)
    assessment = assess(summary)

    scope = f" probe_run_id={args.probe_run_id}" if args.probe_run_id else ""
    print(f"ESM plugin-loader probe report — {len(rows)} runs{scope}")
    print("")
    header = f"{'variant':<32} {'n':>3} {'mean':>7} {'stdev':>7} {'min':>5} {'max':>5} {'M_mean':>8} {'raw_mean':>9}  scores  ceilings"
    print(header)
    print("-" * len(header))
    for variant in VARIANT_ORDER:
        variant_summary = summary.get(variant)
        if variant_summary is None:
            print(f"{variant:<32}   -       -       -     -     -        -         -  (no runs)")
            continue
        ceilings = ",".join(f"{name}x{count}" for name, count in sorted(variant_summary["ceiling_hits"].items())) or "-"
        scores = ",".join(str(score) for score in variant_summary["scores"])
        print(
            f"{variant:<32} {variant_summary['n']:>3} {variant_summary['mean']:>7.2f} {variant_summary['stdev']:>7.2f} "
            f"{variant_summary['min']:>5} {variant_summary['max']:>5} {variant_summary['m_training_mean']:>8.4f} "
            f"{variant_summary['raw_mean']:>9.2f}  [{scores}]  {ceilings}"
        )

    print("")
    print(
        f"family_mean = {assessment['family_mean']:.2f} "
        f"(window {FAMILY_MEAN_WINDOW[0]}-{FAMILY_MEAN_WINDOW[1]})"
    )
    print(f"max_variant_mean = {assessment['max_variant_mean']:.2f} (cap {MAX_VARIANT_SCORE})")
    print(
        f"min_variant_mean = {assessment['min_variant_mean']:.2f} "
        f"(must have at least one <= {MIN_HARD_VARIANT_SCORE})"
    )
    print("")
    print("Acceptance checks:")
    print(fmt_check("family_mean in window", assessment["family_mean_ok"], f"{assessment['family_mean']:.2f}"))
    print(fmt_check("max variant <= cap", assessment["max_variant_ok"], f"{assessment['max_variant_mean']:.2f}"))
    print(
        fmt_check(
            "at least one variant <= hard floor",
            assessment["hard_variant_ok"],
            f"min variant mean = {assessment['min_variant_mean']:.2f}",
        )
    )
    print(
        fmt_check(
            "monotonic V1>=V2>=V3>=V4>=V5 +/-tolerance",
            assessment["monotonic_ok"],
            "; ".join(assessment["monotonic_breaks"]) or "ok",
        )
    )
    print("")
    print(f"overall: {'ALL PASS' if assessment['all_pass'] else 'HARDEN NEEDED'}")

    if args.emit_json:
        print("")
        print("JSON_SUMMARY:")
        print(json.dumps({"summary": summary, "assessment": assessment}, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
