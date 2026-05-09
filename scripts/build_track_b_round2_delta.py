#!/usr/bin/env python3
"""Round 2 applicability-delta report — baseline vs patched runtime.

Reads two applicability JSON files produced by
``build_track_b_round2_applicability.py`` (one before vLLM is patched,
one after) and emits a structured comparison: corpus totals delta,
per-regime deltas, and per-technique delta against its theoretical
ceiling. Lets the operator answer the Round 2 acceptance question
("did the T1+T3 patches actually move decode time?") with one
command instead of staring at two JSONs.

Inputs:
- ``--baseline``: path to Round 0 (or whichever pre-patch baseline)
  applicability JSON.
- ``--patched``: path to the post-patch sweep's applicability JSON.

Output JSON schema ``lumo.track_b.round2_delta.v1``:
- ``totals_delta``: absolute and percent change in turns,
  prefill_sum_s, decode_sum_s, wallclock_s.
- ``regimes_delta``: per-regime absolute and percent change in
  prefill_sum_s, decode_sum_s.
- ``techniques_delta``: per-technique change in
  ``decode_sum_s_covered`` plus a ``measured_vs_ceiling_ratio``
  field for techniques whose ceilings are independent of patched
  state (T1 / T3 / T5).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA = "lumo.track_b.round2_delta.v1"


def _safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den is None:
        return None
    if not math.isfinite(num) or not math.isfinite(den) or den == 0:
        return None
    return num / den


def _percent_change(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    if not math.isfinite(before) or not math.isfinite(after) or before == 0:
        return None
    return (after - before) / before


def _delta_block(before: dict[str, Any], after: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        b = before.get(key)
        a = after.get(key)
        out[f"{key}_before"] = b
        out[f"{key}_after"] = a
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            out[f"{key}_delta"] = a - b
            out[f"{key}_pct_change"] = _percent_change(b, a)
    return out


def build_delta(baseline: dict[str, Any], patched: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "producer": "build_track_b_round2_delta",
        "baseline_files_read_count": baseline.get("files_read_count"),
        "patched_files_read_count": patched.get("files_read_count"),
    }

    out["totals_delta"] = _delta_block(
        baseline.get("totals", {}),
        patched.get("totals", {}),
        [
            "turns",
            "prompt_tokens_total",
            "completion_tokens_total",
            "prefill_sum_s_total",
            "decode_sum_s_total",
            "wallclock_s_total",
        ],
    )

    regimes_before = baseline.get("regimes", {}) or {}
    regimes_after = patched.get("regimes", {}) or {}
    all_regimes = sorted(set(regimes_before) | set(regimes_after))
    out["regimes_delta"] = {
        regime: _delta_block(
            regimes_before.get(regime, {}) or {},
            regimes_after.get(regime, {}) or {},
            ["turns", "prefill_sum_s", "decode_sum_s"],
        )
        for regime in all_regimes
    }

    techniques_before = baseline.get("techniques", {}) or {}
    techniques_after = patched.get("techniques", {}) or {}
    all_techniques = sorted(set(techniques_before) | set(techniques_after))
    techniques_delta: dict[str, Any] = {}
    for technique in all_techniques:
        b = techniques_before.get(technique, {}) or {}
        a = techniques_after.get(technique, {}) or {}
        block = _delta_block(
            b, a,
            [
                "turns_covered",
                "decode_sum_s_covered",
                "decode_sum_s_fraction_of_corpus",
                "decode_reduction_ceiling_s",
            ],
        )
        # Measured-vs-ceiling: how close did the patched run get to
        # the theoretical decode-reduction ceiling the technique
        # advertised on the baseline corpus? Useful only when the
        # baseline ceiling is positive.
        ceiling = b.get("decode_reduction_ceiling_s")
        decode_before = b.get("decode_sum_s_covered")
        decode_after = a.get("decode_sum_s_covered")
        if (
            isinstance(ceiling, (int, float))
            and ceiling > 0
            and isinstance(decode_before, (int, float))
            and isinstance(decode_after, (int, float))
        ):
            measured_reduction = decode_before - decode_after
            block["measured_decode_reduction_s"] = measured_reduction
            block["measured_vs_ceiling_ratio"] = _safe_div(measured_reduction, ceiling)
        techniques_delta[technique] = block
    out["techniques_delta"] = techniques_delta

    # Headline -- the single number an operator can paste into a slack
    # update. Sum of decode reduction across all techniques whose
    # ceiling is non-zero. Capped at the corpus decode total so a
    # measurement noise spike doesn't claim more than 100%.
    decode_total_before = baseline.get("totals", {}).get("decode_sum_s_total")
    decode_total_after = patched.get("totals", {}).get("decode_sum_s_total")
    if (
        isinstance(decode_total_before, (int, float))
        and isinstance(decode_total_after, (int, float))
    ):
        delta = decode_total_before - decode_total_after
        out["headline"] = {
            "corpus_decode_reduction_s": delta,
            "corpus_decode_reduction_pct": _safe_div(delta, decode_total_before),
        }
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Round 2 applicability-delta report -- pairs a baseline "
            "JSON (pre-patch) with a patched JSON (post-relaunch) and "
            "emits per-technique decode-reduction measurements."
        )
    )
    parser.add_argument("--baseline", required=True, help="Baseline applicability JSON path")
    parser.add_argument("--patched", required=True, help="Post-patch applicability JSON path")
    parser.add_argument("--output", required=True, help="Where to write the delta JSON")
    parser.add_argument("--print", action="store_true", help="Print headline + per-technique table to stdout")
    return parser.parse_args(argv)


def _print_table(report: dict[str, Any]) -> None:
    headline = report.get("headline") or {}
    print(
        "Round 2 corpus decode delta: "
        f"{headline.get('corpus_decode_reduction_s', 0):.1f} s "
        f"({(headline.get('corpus_decode_reduction_pct') or 0) * 100:+.1f}%)"
    )
    print()
    print("Per-technique:")
    print(f"  {'technique':<32} {'before_s':>10} {'after_s':>10} {'reduce_s':>10} {'vs_ceil':>10}")
    for name, block in sorted(report.get("techniques_delta", {}).items()):
        before = block.get("decode_sum_s_covered_before") or 0
        after = block.get("decode_sum_s_covered_after") or 0
        red = block.get("measured_decode_reduction_s")
        ratio = block.get("measured_vs_ceiling_ratio")
        red_str = f"{red:.1f}" if isinstance(red, (int, float)) else "—"
        ratio_str = f"{ratio*100:+.1f}%" if isinstance(ratio, (int, float)) else "—"
        print(f"  {name:<32} {before:>10.1f} {after:>10.1f} {red_str:>10} {ratio_str:>10}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read baseline: {exc}", file=sys.stderr)
        return 2
    try:
        patched = json.loads(Path(args.patched).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read patched: {exc}", file=sys.stderr)
        return 2
    report = build_delta(baseline, patched)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.print:
        _print_table(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
