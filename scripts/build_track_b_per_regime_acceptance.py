#!/usr/bin/env python3
"""Aggregate per-regime acceptance from inference-proxy capture rows.

Reads JSONL rows produced by ``lumo_flywheel_serving.inference_proxy``'s
Track B per-request capture (schema ``lumo.track_b.vllm_request_metrics.v1``)
and groups by ``regime`` (``tool-call`` / ``reasoning`` / ``summary`` /
``unknown``). Emits per-regime summaries: row count, p50/p90 of
``accepted_per_draft_token`` (= ``spec_decode_num_accepted_tokens /
spec_decode_num_draft_tokens``), p50/p90 of ``decode_tps``
(= ``completion_tokens / decode_sum_s``), aggregate accepted/draft.

Reuses the regime classifier already in proxy capture rows; no separate
inference pass needed. Diagnostic for the §6.5 taxonomy in the parent
agentic-saturation plan: which regime has the most acceptance headroom
and therefore the most uplift potential from harness-coupled techniques.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(p) - 1]


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None:
        return None
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return None
    return num / den


def _row_acceptance(row: dict[str, Any]) -> float | None:
    return _safe_ratio(row.get("spec_decode_num_accepted_tokens"), row.get("spec_decode_num_draft_tokens"))


def _row_decode_tps(row: dict[str, Any]) -> float | None:
    return _safe_ratio(row.get("completion_tokens"), row.get("decode_sum_s"))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        regime = row.get("regime") or "unknown"
        if not isinstance(regime, str):
            regime = "unknown"
        by_regime[regime].append(row)

    summaries: dict[str, Any] = {}
    aggregate_accepted = 0.0
    aggregate_draft = 0.0
    for regime, regime_rows in by_regime.items():
        acceptance_values = [v for v in (_row_acceptance(r) for r in regime_rows) if v is not None]
        decode_tps_values = [v for v in (_row_decode_tps(r) for r in regime_rows) if v is not None]
        regime_accepted = sum(
            (r.get("spec_decode_num_accepted_tokens") or 0)
            for r in regime_rows
            if isinstance(r.get("spec_decode_num_accepted_tokens"), (int, float))
        )
        regime_draft = sum(
            (r.get("spec_decode_num_draft_tokens") or 0)
            for r in regime_rows
            if isinstance(r.get("spec_decode_num_draft_tokens"), (int, float))
        )
        aggregate_accepted += regime_accepted
        aggregate_draft += regime_draft
        completion_tokens = sum(
            (r.get("completion_tokens") or 0)
            for r in regime_rows
            if isinstance(r.get("completion_tokens"), (int, float))
        )
        prompt_tokens = sum(
            (r.get("prompt_tokens") or 0)
            for r in regime_rows
            if isinstance(r.get("prompt_tokens"), (int, float))
        )
        decode_sum_s = sum(
            (r.get("decode_sum_s") or 0)
            for r in regime_rows
            if isinstance(r.get("decode_sum_s"), (int, float))
        )
        summaries[regime] = {
            "row_count": len(regime_rows),
            "accepted_per_draft_token_p50": _percentile(acceptance_values, 50),
            "accepted_per_draft_token_p90": _percentile(acceptance_values, 90),
            "accepted_per_draft_token_aggregate": _safe_ratio(regime_accepted, regime_draft),
            "decode_tps_p50": _percentile(decode_tps_values, 50),
            "decode_tps_p90": _percentile(decode_tps_values, 90),
            "decode_tps_aggregate": _safe_ratio(completion_tokens, decode_sum_s),
            "completion_tokens_total": int(completion_tokens),
            "prompt_tokens_total": int(prompt_tokens),
            "spec_decode_num_accepted_tokens_total": int(regime_accepted),
            "spec_decode_num_draft_tokens_total": int(regime_draft),
            "decode_sum_s_total": decode_sum_s,
        }
    return {
        "schema": "lumo.track_b.per_regime_acceptance.v1",
        "row_count": len(rows),
        "regimes": summaries,
        "aggregate_accepted_per_draft_token": _safe_ratio(aggregate_accepted, aggregate_draft),
        "aggregate_spec_decode_num_accepted_tokens": int(aggregate_accepted),
        "aggregate_spec_decode_num_draft_tokens": int(aggregate_draft),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed proxy capture JSONL at {path}:{line_number}: {exc.msg}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _fmt(value: float | None, spec: str, width: int) -> str:
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{format(value, spec):>{width}}"


def format_table(summary: dict[str, Any]) -> str:
    header = f"{'regime':<14} {'rows':>5} {'agg accept':>11} {'p50 accept':>11} {'p50 tps':>9} {'tokens out':>11}"
    lines = [header, "-" * len(header)]
    for regime, stats in sorted(summary["regimes"].items()):
        agg = stats.get("accepted_per_draft_token_aggregate")
        p50 = stats.get("accepted_per_draft_token_p50")
        tps = stats.get("decode_tps_p50")
        lines.append(
            f"{regime:<14} {stats['row_count']:>5} "
            f"{_fmt(agg, '.3f', 11)} "
            f"{_fmt(p50, '.3f', 11)} "
            f"{_fmt(tps, '.2f', 9)} "
            f"{stats['completion_tokens_total']:>11}"
        )
    lines.append("-" * len(header))
    agg = summary.get("aggregate_accepted_per_draft_token")
    lines.append(
        f"{'AGGREGATE':<14} {summary['row_count']:>5} "
        f"{_fmt(agg, '.3f', 11)}"
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-regime acceptance from Track B proxy capture rows.")
    parser.add_argument("source", type=Path, help="Path to vllm_request_metrics.jsonl from inference proxy capture.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path; default stdout.")
    parser.add_argument("--text", action="store_true", help="Print a human-readable table to stderr alongside JSON.")
    args = parser.parse_args()

    if not args.source.is_file():
        print(f"build_track_b_per_regime_acceptance: source not found: {args.source}", file=sys.stderr)
        return 2
    rows = load_rows(args.source)
    summary = aggregate(rows)

    if args.text:
        print(format_table(summary), file=sys.stderr)

    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
