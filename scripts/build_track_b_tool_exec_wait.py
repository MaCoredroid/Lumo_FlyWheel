#!/usr/bin/env python3
"""Compute tool-exec-wait gaps from Track B proxy capture rows.

Tool-exec-wait is the wallclock spent between Codex finishing a turn
(ts_completed of the /v1/responses call that emitted the tool call)
and Codex starting the next turn (ts_request_received of the next
call to /v1/responses). On the proxy side this is the only thing
that can drive that gap: Codex receives the tool call, the tool
runs locally on the host (apply_patch, exec_command, write_file,
read_file), Codex parses the result, and Codex sends the next
request with the tool output appended to the conversation.

The v2 spec recalibration flagged tool-exec-wait as the largest
open lever for absolute-wallclock leverage, since the measured
89% tool-call regime share means most turns transition through a
tool-execution phase. This script answers: how much aggregate
wallclock is in tool-exec-wait vs decode vs prefill?

Schema emitted: lumo.track_b.tool_exec_wait.v1.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA = "lumo.track_b.tool_exec_wait.v1"


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("schema") == "lumo.track_b.vllm_request_metrics.v1":
                rows.append(payload)
    return rows


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    points = statistics.quantiles(values, n=100, method="inclusive")
    idx = max(0, min(len(points) - 1, int(round(q * 100)) - 1))
    return points[idx]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute tool-exec-wait gaps by inferring the boundaries between
    consecutive Codex turns from request timestamps. Adjacent rows
    (sorted by ts_request_received) belong to the same agent task if
    their gap is < 60s -- 60s is well above any reasonable tool-exec
    wait but well below the inter-task gap during sweeps."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        ts_recv = _parse_ts(row.get("ts_request_received"))
        ts_done = _parse_ts(row.get("ts_completed"))
        if ts_recv is None or ts_done is None:
            continue
        wallclock_s = row.get("wallclock_s")
        if not isinstance(wallclock_s, (int, float)):
            wallclock_s = (ts_done - ts_recv).total_seconds()
        enriched.append({
            "ts_recv": ts_recv,
            "ts_done": ts_done,
            "wallclock_s": float(wallclock_s),
            "regime": row.get("regime"),
            "decode_sum_s": row.get("decode_sum_s"),
            "prefill_sum_s": row.get("prefill_sum_s"),
            "completion_tokens": row.get("completion_tokens"),
        })
    enriched.sort(key=lambda r: r["ts_recv"])

    INTRA_TASK_GAP_S = 60.0
    gaps: list[float] = []
    by_regime: dict[str, list[float]] = {}
    for prev, curr in zip(enriched, enriched[1:]):
        gap_s = (curr["ts_recv"] - prev["ts_done"]).total_seconds()
        if gap_s < 0 or gap_s > INTRA_TASK_GAP_S:
            continue
        gaps.append(gap_s)
        # Attribute the wait to the regime of the PRIOR turn (the turn
        # that emitted the tool call).
        regime = prev.get("regime") or "unknown"
        by_regime.setdefault(regime, []).append(gap_s)

    decode_sum_total = sum(
        r["decode_sum_s"] for r in enriched if isinstance(r.get("decode_sum_s"), (int, float))
    )
    prefill_sum_total = sum(
        r["prefill_sum_s"] for r in enriched if isinstance(r.get("prefill_sum_s"), (int, float))
    )
    wallclock_sum_total = sum(r["wallclock_s"] for r in enriched)
    tool_wait_sum = sum(gaps)
    served_sum_total = decode_sum_total + prefill_sum_total

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "row_count": len(enriched),
        "intra_task_gap_threshold_s": INTRA_TASK_GAP_S,
        "tool_exec_wait_gap_count": len(gaps),
        "tool_exec_wait_sum_s": tool_wait_sum,
        "tool_exec_wait_p50_s": _quantile(sorted(gaps), 0.50),
        "tool_exec_wait_p90_s": _quantile(sorted(gaps), 0.90),
        "tool_exec_wait_p99_s": _quantile(sorted(gaps), 0.99),
        "tool_exec_wait_max_s": max(gaps) if gaps else None,
        "tool_exec_wait_mean_s": statistics.mean(gaps) if gaps else None,
        "decode_sum_total_s": decode_sum_total,
        "prefill_sum_total_s": prefill_sum_total,
        "wallclock_sum_total_s": wallclock_sum_total,
        "served_sum_total_s": served_sum_total,
    }
    if tool_wait_sum + served_sum_total > 0:
        denom = tool_wait_sum + served_sum_total
        summary["tool_exec_wait_share_of_total"] = tool_wait_sum / denom
        summary["served_share_of_total"] = served_sum_total / denom

    by_regime_summary: dict[str, dict[str, Any]] = {}
    for regime, regime_gaps in sorted(by_regime.items()):
        regime_gaps_sorted = sorted(regime_gaps)
        by_regime_summary[regime] = {
            "gap_count": len(regime_gaps_sorted),
            "sum_s": sum(regime_gaps_sorted),
            "p50_s": _quantile(regime_gaps_sorted, 0.50),
            "p90_s": _quantile(regime_gaps_sorted, 0.90),
            "max_s": regime_gaps_sorted[-1] if regime_gaps_sorted else None,
            "mean_s": statistics.mean(regime_gaps_sorted) if regime_gaps_sorted else None,
        }
    summary["by_prior_turn_regime"] = by_regime_summary
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute tool-exec-wait gaps from proxy capture rows.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--text", action="store_true", help="Emit a short human summary on stdout")
    args = parser.parse_args()

    rows = _load_rows(args.source)
    summary = aggregate(rows)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.text or not args.out:
        print(json.dumps(summary, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
