#!/usr/bin/env python3
"""FR13 host-residual rung: offline host-timeline attribution of a fixed32 capture.

Analysis-only, read-only, no GPU. Reduces the nsys sqlite export of a fixed32
B1 capture along the HOST timeline, which the existing GPU-projection reducer
(``scripts/fr13_fixed32_nsys_reduce.py``) deliberately does not do.

Why a second view. ``fr13.fixed32.*`` NVTX ranges are host push/pop pairs, so
they measure what the *host* was doing as well as what they project onto the
GPU. Reading them that way answers the only question this rung cares about --
during which host activity is the GPU idle -- and it exposes two things the
GPU projection cannot:

  * the post-DFWD "tail" MEAN is carried by a handful of steps whose window
    contains a chunked-prefill forward (the fixed32 step markers wrap only the
    spec-decode path), so the mean is 3.3x the median;
  * ``Command buffer full`` driver records show exactly where the host is
    blocked on backpressure rather than being the critical path.

``analysis_only=true``, ``acceptance_valid=false``. This carries no TPS,
floor or acceptance claim, and reads nothing but a banked capture.

Usage:
    python3 scripts/fr13_host_tail_attribution.py \
        --sqlite <runroot>/<arm>/logs/fr13_fixed32_b1_real_swe.sqlite \
        --out results/fr13_host_residual_20260811/tail_attribution.json
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PHASES = ("step", "sfwd", "postprocess", "cfwd", "dfwd")
# Steps whose host wall falls outside this band are not decode-cadence steps:
# their window contains a prefill forward, a request boundary, or the capture
# edge. They are reported separately, never averaged in silently.
DECODE_WALL_MS = (150.0, 400.0)
SEGMENTS = (
    "A sfwd(host)",
    "B gap sfwd->postprocess",
    "C postprocess(host)",
    "D gap postprocess->cfwd",
    "E cfwd(host)",
    "F gap cfwd->dfwd",
    "G dfwd(host)",
    "H TAIL dfwd->next sfwd",
)


def _connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"capture sqlite not found: {path}")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _nvtx_thread(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "select globalTid, count(*) from NVTX_EVENTS group by globalTid "
        "order by 2 desc limit 1"
    ).fetchall()
    if not rows:
        raise SystemExit("capture has no NVTX events")
    return int(rows[0][0])


def _ranges(conn: sqlite3.Connection, tid: int, name: str):
    query = (
        "select e.start, e.end from NVTX_EVENTS e "
        "left join StringIds s on s.id = e.textId "
        "where coalesce(s.value, e.text) = ? and e.globalTid = ? "
        "and e.end is not null and e.end > e.start order by e.start"
    )
    out = [
        (a, b)
        for a, b in conn.execute(query, (f"fr13.fixed32.{name}", tid))
        # A range longer than five seconds is a capture-boundary artefact, not
        # a step; the reducer applies the same kind of boundary allowance.
        if (b - a) < 5_000_000_000
    ]
    if not out:
        raise SystemExit(f"capture has no fr13.fixed32.{name} ranges")
    return out


def _busy_union(conn: sqlite3.Connection):
    ops = []
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        ops.extend(conn.execute(f"select start, end from {table} where end > start"))
    if not ops:
        raise SystemExit("capture has no GPU activity")
    ops.sort()
    merged = []
    cur_s, cur_e = ops[0]
    for start, end in ops[1:]:
        if start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = start, end
    merged.append((cur_s, cur_e))
    return merged


def _overlap_ns(intervals, starts, lo, hi) -> int:
    if hi <= lo:
        return 0
    total = 0
    index = max(0, bisect.bisect_right(starts, lo) - 1)
    while index < len(intervals) and intervals[index][0] < hi:
        s, e = intervals[index]
        total += max(0, min(hi, e) - max(lo, s))
        index += 1
    return total


def _stats(values):
    ordered = sorted(values)
    if not ordered:
        return {"n": 0}
    return {
        "n": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": ordered[len(ordered) // 2],
        "p05_ms": ordered[int(len(ordered) * 0.05)],
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max_ms": ordered[-1],
    }


def reduce_capture(sqlite_path: Path) -> dict:
    conn = _connect(sqlite_path)
    tid = _nvtx_thread(conn)
    ranges = {name: _ranges(conn, tid, name) for name in PHASES}
    busy = _busy_union(conn)
    busy_starts = [b[0] for b in busy]

    api = list(
        conn.execute(
            "select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r "
            "join StringIds s on s.id = r.nameId where r.globalTid = ? "
            "order by r.start",
            (tid,),
        )
    )
    api_starts = [a[0] for a in api]
    overhead = sorted(
        conn.execute(
            "select o.start, o.end from CUPTI_ACTIVITY_KIND_OVERHEAD o "
            "where o.globalTid = ?",
            (tid,),
        )
    )
    overhead_starts = [o[0] for o in overhead]

    count = min(len(ranges[name]) for name in ("sfwd", "postprocess", "cfwd", "dfwd"))
    wall = defaultdict(list)
    idle = defaultdict(list)
    in_api = defaultdict(float)
    outside_api = defaultdict(float)
    idle_by_host = defaultdict(lambda: defaultdict(float))
    cbf = defaultdict(float)
    tails_all = []
    kept = 0

    for i in range(count - 1):
        sfwd, post = ranges["sfwd"], ranges["postprocess"]
        cfwd, dfwd = ranges["cfwd"], ranges["dfwd"]
        total_ms = (sfwd[i + 1][0] - sfwd[i][0]) / 1e6
        tails_all.append((sfwd[i + 1][0] - dfwd[i][1]) / 1e6)
        if not DECODE_WALL_MS[0] < total_ms < DECODE_WALL_MS[1]:
            continue
        kept += 1
        bounds = (
            sfwd[i],
            (sfwd[i][1], post[i][0]),
            post[i],
            (post[i][1], cfwd[i][0]),
            cfwd[i],
            (cfwd[i][1], dfwd[i][0]),
            dfwd[i],
            (dfwd[i][1], sfwd[i + 1][0]),
        )
        for label, (lo, hi) in zip(SEGMENTS, bounds):
            wall[label].append((hi - lo) / 1e6)
            gpu_busy = _overlap_ns(busy, busy_starts, lo, hi)
            idle[label].append(((hi - lo) - gpu_busy) / 1e6)
            cbf[label] += _overlap_ns(overhead, overhead_starts, lo, hi)
            # API coverage inside the window, and the complement (host CPU
            # inside no CUDA call at all).
            index = max(0, bisect.bisect_right(api_starts, lo) - 1)
            covered = []
            while index < len(api) and api[index][0] < hi:
                s, e, name = api[index]
                if min(hi, e) > max(lo, s):
                    covered.append((max(lo, s), min(hi, e), name))
                index += 1
            prev = lo
            for s, e, _name in covered:
                if s > prev:
                    outside_api[label] += s - prev
                prev = max(prev, e)
            outside_api[label] += max(0, hi - prev)
            in_api[label] += sum(e - s for s, e, _ in covered)
            # And which host activity was running while the GPU was idle.
            index = max(0, bisect.bisect_right(busy_starts, lo) - 1)
            gaps = []
            prev_end = lo
            while index < len(busy) and busy[index][0] < hi:
                s, e = busy[index]
                if s > prev_end:
                    gaps.append((prev_end, min(s, hi)))
                prev_end = max(prev_end, e)
                index += 1
            if prev_end < hi:
                gaps.append((prev_end, hi))
            for g0, g1 in gaps:
                j = max(0, bisect.bisect_right(api_starts, g0) - 1)
                seen = []
                while j < len(api) and api[j][0] < g1:
                    s, e, name = api[j]
                    if min(g1, e) > max(g0, s):
                        key = name.split("_v")[0]
                        idle_by_host[label][key] += min(g1, e) - max(g0, s)
                        seen.append((max(g0, s), min(g1, e)))
                    j += 1
                prev = g0
                python_ns = 0
                for s, e in seen:
                    if s > prev:
                        python_ns += s - prev
                    prev = max(prev, e)
                python_ns += max(0, g1 - prev)
                idle_by_host[label]["<python: no CUDA call>"] += python_ns

    if not kept:
        raise SystemExit("no decode-cadence steps in this capture")

    segments = {}
    for label in SEGMENTS:
        segments[label] = {
            "wall_ms_per_step": statistics.fmean(wall[label]),
            "gpu_idle_ms_per_step": statistics.fmean(idle[label]),
            "gpu_busy_ms_per_step": statistics.fmean(wall[label])
            - statistics.fmean(idle[label]),
            "in_cuda_api_ms_per_step": in_api[label] / 1e6 / kept,
            "outside_cuda_api_ms_per_step": outside_api[label] / 1e6 / kept,
            "command_buffer_full_ms_per_step": cbf[label] / 1e6 / kept,
            "gpu_idle_by_host_activity_ms_per_step": {
                k: v / 1e6 / kept
                for k, v in sorted(
                    idle_by_host[label].items(), key=lambda kv: -kv[1]
                )
                if v / 1e6 / kept >= 0.001
            },
        }

    outliers = [t for t in tails_all if t > 10.0]
    inliers = [t for t in tails_all if t <= 10.0]
    return {
        "schema": "fr13.host_tail_attribution.v1",
        "analysis_only": True,
        "acceptance_valid": False,
        "attribution_only": True,
        "curated_publishable": False,
        "stamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "capture": str(sqlite_path),
        "steps_total": len(tails_all),
        "steps_decode_cadence": kept,
        "decode_wall_band_ms": list(DECODE_WALL_MS),
        "host_step_wall_ms_per_step": sum(
            segments[label]["wall_ms_per_step"] for label in SEGMENTS
        ),
        "host_attributable_gpu_idle_ms_per_step": sum(
            segments[label]["gpu_idle_ms_per_step"] for label in SEGMENTS
        ),
        "segments": segments,
        "tail_distribution_all_steps": _stats(tails_all),
        "tail_outliers": {
            "threshold_ms": 10.0,
            "count": len(outliers),
            "share_of_steps": len(outliers) / len(tails_all),
            "contribution_to_mean_ms": sum(outliers) / len(tails_all),
            "remainder_mean_ms": statistics.fmean(inliers) if inliers else None,
        },
    }


def _render(payload: dict) -> str:
    lines = [
        "FR13 host-residual: HOST-TIMELINE attribution",
        f"  capture               {payload['capture']}",
        f"  steps                 {payload['steps_decode_cadence']} decode-cadence "
        f"of {payload['steps_total']}",
        f"  host step wall        {payload['host_step_wall_ms_per_step']:9.3f} ms/step",
        f"  host-attributable GPU idle "
        f"{payload['host_attributable_gpu_idle_ms_per_step']:9.3f} ms/step",
        "",
        f"  {'segment':28s} {'wall':>9s} {'idle':>9s} {'busy':>9s} "
        f"{'in_api':>9s} {'py':>9s} {'cbfull':>9s}",
    ]
    for label, row in payload["segments"].items():
        lines.append(
            f"  {label:28s} {row['wall_ms_per_step']:9.3f} "
            f"{row['gpu_idle_ms_per_step']:9.3f} {row['gpu_busy_ms_per_step']:9.3f} "
            f"{row['in_cuda_api_ms_per_step']:9.3f} "
            f"{row['outside_cuda_api_ms_per_step']:9.3f} "
            f"{row['command_buffer_full_ms_per_step']:9.3f}"
        )
    dist = payload["tail_distribution_all_steps"]
    out = payload["tail_outliers"]
    lines += [
        "",
        "  HOST-window TAIL over ALL steps (dfwd_end -> next sfwd_start).",
        "  The ladder quotes the GPU-PROJECTED tail (dfwd_gpu_end ->",
        "  step_gpu_end), whose mean is 11.977 ms and whose median is the same",
        "  3.588 ms -- both are inflated by the same prefill-carrying steps;",
        "  the host window is longer only because it spans the whole prefill",
        "  wall rather than being bounded by the step range's last GPU op.",
        f"    mean {dist['mean_ms']:.3f}  p50 {dist['p50_ms']:.3f}  "
        f"p95 {dist['p95_ms']:.3f}  max {dist['max_ms']:.3f} ms",
        f"    steps > {out['threshold_ms']:.0f} ms: {out['count']} "
        f"({100 * out['share_of_steps']:.1f}%), carrying "
        f"{out['contribution_to_mean_ms']:.3f} ms/step of that mean;",
        f"    the remaining steps average {out['remainder_mean_ms']:.3f} ms.",
        "",
        "  GPU idle by concurrent host activity:",
    ]
    for label, row in payload["segments"].items():
        detail = row["gpu_idle_by_host_activity_ms_per_step"]
        if not detail:
            continue
        lines.append(f"    {label}")
        for name, value in detail.items():
            lines.append(f"      {value:8.3f} ms/step  {name}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-out")
    args = parser.parse_args()

    payload = reduce_capture(Path(args.sqlite))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rendered = _render(payload)
    if args.text_out:
        Path(args.text_out).write_text(rendered)
    print(rendered)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
