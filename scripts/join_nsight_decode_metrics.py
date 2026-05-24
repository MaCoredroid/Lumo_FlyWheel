#!/usr/bin/env python3
"""Offline join of Nsight GPU-metrics (point-in-time) with per-completion decode
speed (continuous) — the correlation the Round 5 spec §13.7 asks for, done as a
wall-clock join rather than wiring Nsight into the live metrics path.

Why offline: Nsight adds ~10% profiling overhead and is post-hoc, so it must NOT
sit in the per-completion measurement path (that would bias every decode-tps and
acceptance number — the §0 hygiene we just fixed). Instead, keep three
timestamped producers and join by wall clock for the windows Nsight actually ran:

  1. proxy request_metrics.jsonl  -> per-completion decode tps + acceptance
  2. Nsight gb20y .sqlite         -> 10Hz GPU saturation (Tensor/SM Issue/SMs)
  3. swe_dgx_steptrace .jsonl     -> 10Hz vLLM batch + power_w (optional)

For each completion whose [ts_request_received, ts_completed] overlaps the Nsight
capture, we attribute the GPU-metric samples in that interval and report decode
tps alongside median Tensor Active / SM Issue / SMs Active.

Nsight time base: GPU_METRICS.timestamp is ns from report start;
TARGET_INFO_SESSION_START_TIME.utcEpochNs is the wall clock at report start, so
  wall_ns(sample) = utcEpochNs + GPU_METRICS.timestamp
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path


def _iso_to_epoch_ns(value: str) -> int | None:
    if not value:
        return None
    v = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1e9)


def load_nsight(sqlite_path: Path) -> tuple[list[tuple[int, dict[str, float]]], dict[int, str]]:
    """Return (samples, metric_names). samples = [(wall_ns, {metricId: value})]."""
    con = sqlite3.connect(str(sqlite_path))
    names = dict(con.execute("SELECT metricId, metricName FROM TARGET_INFO_GPU_METRICS"))
    (utc_epoch_ns,) = con.execute(
        "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"
    ).fetchone()
    by_ts: dict[int, dict[str, float]] = {}
    for metric_id, value, ts in con.execute(
        "SELECT metricId, value, timestamp FROM GPU_METRICS"
    ):
        by_ts.setdefault(ts, {})[metric_id] = float(value)
    con.close()
    samples = sorted((int(utc_epoch_ns) + int(ts), vals) for ts, vals in by_ts.items())
    return samples, names


def _metric_id(names: dict[int, str], needle: str) -> int | None:
    for mid, name in names.items():
        if name == needle:
            return mid
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsight-sqlite", required=True, type=Path)
    ap.add_argument("--request-metrics", required=True, type=Path,
                    help="proxy request_metrics.jsonl (full file ok; filtered to window)")
    ap.add_argument("--steptrace", type=Path, default=None,
                    help="optional swe_dgx_steptrace jsonl (epoch 'ts' + power_w)")
    ap.add_argument("--out", type=Path, required=True, help="output joined CSV")
    args = ap.parse_args()

    samples, names = load_nsight(args.nsight_sqlite)
    if not samples:
        print("ERROR: no GPU_METRICS samples in nsight sqlite", file=sys.stderr)
        return 2
    win_lo, win_hi = samples[0][0], samples[-1][0]
    sample_ns = [s[0] for s in samples]
    print(f"Nsight window: {datetime.fromtimestamp(win_lo/1e9, timezone.utc).isoformat()} "
          f"-> {datetime.fromtimestamp(win_hi/1e9, timezone.utc).isoformat()} "
          f"({len(samples)} samples)")

    mid_tensor = _metric_id(names, "Tensor Active [Throughput %]")
    mid_issue = _metric_id(names, "SM Issue [Throughput %]")
    mid_sms = _metric_id(names, "SMs Active [Throughput %]")

    import bisect

    def samples_in(lo: int, hi: int) -> list[dict[str, float]]:
        i = bisect.bisect_left(sample_ns, lo)
        j = bisect.bisect_right(sample_ns, hi)
        return [samples[k][1] for k in range(i, j)]

    def med(vals: list[dict[str, float]], mid: int | None) -> float | None:
        if mid is None:
            return None
        xs = [v[mid] for v in vals if mid in v]
        return round(st.median(xs), 1) if xs else None

    # optional steptrace: power_w by epoch second
    step_power: dict[int, float] = {}
    if args.steptrace and args.steptrace.exists():
        for line in args.steptrace.open():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = d.get("ts")
            pw = d.get("power_w")
            if ts is not None and pw is not None:
                step_power[int(float(ts))] = float(pw)

    rows_out = []
    n_scanned = n_joined = 0
    with args.request_metrics.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recv = _iso_to_epoch_ns(r.get("ts_request_received", ""))
            done = _iso_to_epoch_ns(r.get("ts_completed", ""))
            if recv is None or done is None:
                continue
            n_scanned += 1
            if done < win_lo or recv > win_hi:
                continue  # no overlap with Nsight window
            gpu = samples_in(recv, done)
            if not gpu:
                continue
            n_joined += 1
            ctok = r.get("completion_tokens")
            dsum = r.get("decode_sum_s")
            decode_tps = (ctok / dsum) if (ctok and dsum and dsum > 0) else None
            acc = r.get("spec_decode_num_accepted_tokens")
            drf = r.get("spec_decode_num_draft_tokens")
            accept = (acc / drf) if (acc is not None and drf) else None
            sec = int(done / 1e9)
            rows_out.append({
                "ts_completed": r.get("ts_completed"),
                "oracle_session_id": r.get("oracle_session_id"),
                "oracle_run_anchor": r.get("oracle_run_anchor"),
                "completion_tokens": ctok,
                "decode_tps": round(decode_tps, 2) if decode_tps else None,
                "accept_ratio": round(accept, 3) if accept is not None else None,
                "running_before": r.get("num_requests_running_before"),
                "running_after": r.get("num_requests_running_after"),
                "gpu_samples": len(gpu),
                "tensor_active_med": med(gpu, mid_tensor),
                "sm_issue_med": med(gpu, mid_issue),
                "sms_active_med": med(gpu, mid_sms),
                "power_w": step_power.get(sec) or step_power.get(sec - 1),
            })

    rows_out.sort(key=lambda x: x["ts_completed"] or "")
    fields = ["ts_completed", "oracle_session_id", "oracle_run_anchor",
              "completion_tokens", "decode_tps", "accept_ratio",
              "running_before", "running_after", "gpu_samples",
              "tensor_active_med", "sm_issue_med", "sms_active_med", "power_w"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)

    print(f"scanned {n_scanned} completions, joined {n_joined} overlapping the window "
          f"-> {args.out}")
    if rows_out:
        def agg(key):
            xs = [r[key] for r in rows_out if r[key] is not None]
            return round(st.median(xs), 2) if xs else None
        print("\nWindow medians across joined completions:")
        print(f"  decode_tps         = {agg('decode_tps')}")
        print(f"  accept_ratio       = {agg('accept_ratio')}")
        print(f"  Tensor Active %    = {agg('tensor_active_med')}")
        print(f"  SM Issue %         = {agg('sm_issue_med')}")
        print(f"  SMs Active %       = {agg('sms_active_med')}")
        print(f"  power_w            = {agg('power_w')}")
        clean = sum(1 for r in rows_out
                    if (r["running_before"] in (0, 0.0, 1, 1.0))
                    and (r["running_after"] in (0, 0.0, 1, 1.0)))
        print(f"  B=1-clean rows     = {clean}/{len(rows_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
