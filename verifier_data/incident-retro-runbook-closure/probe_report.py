#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
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
        if not line.strip():
            continue
        row = json.loads(line)
        if probe_run_id and row.get("probe_run_id") != probe_run_id:
            continue
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    summary = {v: [] for v in VARIANT_ORDER}
    for row in rows:
        summary.setdefault(row["variant"], []).append(row)
    out = {}
    for variant in VARIANT_ORDER:
        items = summary.get(variant, [])
        if not items:
            out[variant] = None
            continue
        p_scores = [item.get("P_benchmark", item["score"]) for item in items]
        m_scores = [item.get("M_training", item["score"] / 100.0) for item in items]
        hits: dict[str, int] = {}
        integrity_hits = 0
        for item in items:
            for ceiling in item.get("ceilings_applied", []):
                hits[ceiling] = hits.get(ceiling, 0) + 1
            if item.get("integrity_flag", 0):
                integrity_hits += 1
        out[variant] = {
            "n": len(p_scores),
            "mean_P": statistics.mean(p_scores),
            "stdev_P": statistics.stdev(p_scores) if len(p_scores) > 1 else 0.0,
            "min_P": min(p_scores),
            "max_P": max(p_scores),
            "mean_M": statistics.mean(m_scores),
            "stdev_M": statistics.stdev(m_scores) if len(m_scores) > 1 else 0.0,
            "scores_P": p_scores,
            "scores_M": m_scores,
            "ceiling_hits": hits,
            "integrity_hits": integrity_hits,
        }
    return out


def assess(summary: dict) -> dict:
    p_means = [summary[v]["mean_P"] for v in VARIANT_ORDER if summary.get(v)]
    m_means = [summary[v]["mean_M"] for v in VARIANT_ORDER if summary.get(v)]
    family_mean_P = statistics.mean(p_means) if p_means else float("nan")
    family_mean_M = statistics.mean(m_means) if m_means else float("nan")
    observed_stdev_M = statistics.stdev(
        [item for v in VARIANT_ORDER if summary.get(v) for item in summary[v]["scores_M"]]
    ) if sum(len(summary[v]["scores_M"]) for v in VARIANT_ORDER if summary.get(v)) > 1 else 0.0
    max_variant_mean = max(p_means) if p_means else float("nan")
    min_variant_mean = min(p_means) if p_means else float("nan")
    monotonic_ok = True
    monotonic_breaks = []
    for left, right in zip(VARIANT_ORDER, VARIANT_ORDER[1:]):
        if not summary.get(left) or not summary.get(right):
            continue
        if summary[left]["mean_P"] + MONOTONICITY_TOLERANCE < summary[right]["mean_P"]:
            monotonic_ok = False
            monotonic_breaks.append(
                f"{left} ({summary[left]['mean_P']:.2f}) < {right} ({summary[right]['mean_P']:.2f}) beyond +/-{MONOTONICITY_TOLERANCE}"
            )
    return {
        "family_mean_P": family_mean_P,
        "family_mean_M": family_mean_M,
        "observed_stdev_M": observed_stdev_M,
        "max_variant_mean": max_variant_mean,
        "min_variant_mean": min_variant_mean,
        "family_mean_ok": FAMILY_MEAN_WINDOW[0] <= family_mean_P <= FAMILY_MEAN_WINDOW[1],
        "max_variant_ok": max_variant_mean <= MAX_VARIANT_SCORE,
        "hard_variant_ok": min_variant_mean <= MIN_HARD_VARIANT_SCORE,
        "monotonic_ok": monotonic_ok,
        "monotonic_breaks": monotonic_breaks,
    }


def render(summary: dict, assessment: dict, probe_run_id: str) -> str:
    lines = [f"CNB-55 incident-retro-runbook-closure probe report — probe_run_id={probe_run_id}", ""]
    lines.append(
        f"{'variant':<32} {'n':>3} {'mean_P':>7} {'stdev_P':>7} {'mean_M':>7} {'stdev_M':>7} {'min_P':>5} {'max_P':>5}  scores_P  ceilings"
    )
    lines.append("-" * 124)
    for variant in VARIANT_ORDER:
        data = summary.get(variant)
        if not data:
            lines.append(f"{variant:<32}   -       -       -     -     -   (no runs)")
            continue
        ceilings = ",".join(f"{k}x{v}" for k, v in sorted(data["ceiling_hits"].items())) or "-"
        scores = ",".join(str(x) for x in data["scores_P"])
        lines.append(
            f"{variant:<32} {data['n']:>3} {data['mean_P']:>7.2f} {data['stdev_P']:>7.2f} {data['mean_M']:>7.2f} {data['stdev_M']:>7.2f} {data['min_P']:>5} {data['max_P']:>5}  [{scores}]  {ceilings}"
        )
    lines.append("")
    lines.append(f"family_mean_P = {assessment['family_mean_P']:.2f}   (window {FAMILY_MEAN_WINDOW[0]}-{FAMILY_MEAN_WINDOW[1]})")
    lines.append(f"family_mean_M = {assessment['family_mean_M']:.4f}")
    lines.append(f"observed_stdev_M = {assessment['observed_stdev_M']:.4f}")
    lines.append(f"max_variant_mean = {assessment['max_variant_mean']:.2f}   (cap {MAX_VARIANT_SCORE})")
    lines.append(f"min_variant_mean = {assessment['min_variant_mean']:.2f}   (must have at least one <= {MIN_HARD_VARIANT_SCORE})")
    lines.append("")
    lines.append("Acceptance checks:")
    lines.append(f"  [{'PASS' if assessment['family_mean_ok'] else 'FAIL'}] family_mean_P in window: {assessment['family_mean_P']:.2f}")
    lines.append(f"  [{'PASS' if assessment['max_variant_ok'] else 'FAIL'}] max variant <= cap: {assessment['max_variant_mean']:.2f}")
    lines.append(f"  [{'PASS' if assessment['hard_variant_ok'] else 'FAIL'}] at least one variant <= hard floor: {assessment['min_variant_mean']:.2f}")
    mono_detail = '; '.join(assessment['monotonic_breaks']) or 'ok'
    lines.append(f"  [{'PASS' if assessment['monotonic_ok'] else 'FAIL'}] monotonic V1>=V2>=V3>=V4>=V5 +/-tolerance: {mono_detail}")
    lines.append("")
    all_pass = all(
        (
            assessment["family_mean_ok"],
            assessment["max_variant_ok"],
            assessment["hard_variant_ok"],
            assessment["monotonic_ok"],
        )
    )
    lines.append(f"overall: {'ALL PASS' if all_pass else 'HARDEN NEEDED'}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--probe-run-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = load_rows(args.jsonl, args.probe_run_id)
    if not rows:
        raise SystemExit("no matching rows")
    summary = summarize(rows)
    assessment = assess(summary)
    text = render(summary, assessment, args.probe_run_id)
    args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
