#!/usr/bin/env python3
"""Decompose the out-of-decode wall of an fr13 fixed32 Nsight capture.

DIAGNOSTIC. NOT CITABLE. The arm this reads was served with CUPTI attached for
its whole lifetime; every absolute number is profiler-perturbed. What is claimed
here is a decomposition of SHARES of the captured window, and the shares are
what the lever ladder is priced against.

THE OBJECT BEING DECOMPOSED
---------------------------
`fr13_b4_width4_nsys_reduce.py` established that 40.7% of the captured window
wall lies outside the `fr13.fixed32.step` NVTX ranges, in 33 discrete gaps. That
range is pushed ONLY on a pure-decode forward pass:

    _fr13_fixed32_step_nvtx_begin(_fr13_sfwd_ev is not None)

and `_fr13_sfwd_begin` returns None whenever `_fr13_sfwd_is_pure_decode` is
false. So "outside a decode step" does not mean "idle" and does not mean
"not a forward pass": it means the forward pass was NOT uniform-decode, i.e. it
was a chunked-prefill/mixed batch. The whole question is what fraction of that
mass is (a) prefill compute, (b) GPU idle awaiting demand, (c) host/scheduler,
(d) decode work that the pure-decode counter simply does not count.

THREE MEASUREMENTS, KEPT APART
------------------------------
  gap wall        span between consecutive pure-decode NVTX step ranges.
  gap GPU busy    UNION of kernel [start,end) inside the gap. Never a plain sum:
                  a plain sum double-counts concurrent streams.
  gap GPU idle    gap wall - gap GPU busy. This is the ONLY quantity that can be
                  demand starvation, and it bounds class (b) from above.

CLASSIFYING THE BUSY MASS WITHOUT GUESSING
------------------------------------------
Kernels are classified by their own observed distribution, not by a name
whitelist an author picked:

  prefill-exclusive   a (shortName, gridX-bucket) class with ZERO instances
                      inside any pure-decode step range. It cannot be decode
                      work, because decode never launches it.
  co-batched decode   for a class that DOES appear in decode steps, the
                      out-of-step time up to that class's own in-step rate per
                      pass. This is decode progress made inside a mixed batch.
  prefill excess      the remainder of a shared class's out-of-step time. A
                      mixed batch's GEMM covers prefill tokens AND decode rows
                      in one call; the excess over the decode rate is the
                      prefill share of it.

FAIL-CLOSED
-----------
The pass identity (`sample_readback` instances == pure-decode step ranges +
mixed passes, and every mixed pass inside a gap over threshold) is checked
before any class total is emitted. If it fails the tool raises; it does not
emit a partial decomposition.
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fr13.b4_prefill_gaps.v1"

STEP_RANGE = "fr13.fixed32.step"
PASS_RANGE = "fr13.fixed32.sample_readback"

# A capture boundary can leave one NVTX range with a bogus (unclosed, rebased)
# timestamp. The B1/width-4 reducers pin the same +/-2 allowance.
MAX_BOGUS_STEP_RANGES = 2
# No real fixed32 decode step is anywhere near a minute.
MAX_PLAUSIBLE_STEP_NS = 60_000_000_000


def _q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()


def _nvtx(conn: sqlite3.Connection, name: str) -> list[tuple[int, int]]:
    rows = _q(conn, """
        SELECT n.start, n.end FROM NVTX_EVENTS n
        LEFT JOIN StringIds si ON si.id = n.textId
        WHERE COALESCE(si.value, n.text) = ? AND n.end IS NOT NULL
        ORDER BY n.start
    """, (name,))
    return [(int(a), int(b)) for a, b in rows]


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    """Union length of [start,end) intervals. Never a plain sum."""
    if not intervals:
        return 0
    total = 0
    cur_s, cur_e = intervals[0]
    for a, b in intervals[1:]:
        if a <= cur_e:
            if b > cur_e:
                cur_e = b
        else:
            total += cur_e - cur_s
            cur_s, cur_e = a, b
    return total + cur_e - cur_s


def _grid_bucket(grid_x: int) -> str:
    """gridX buckets separate a decode-shaped launch from a prefill-shaped one.

    A pure-decode fixed32 batch is at most 4 requests x 32 tree rows = 128 rows,
    so its M dimension never needs many tiles. A chunked-prefill batch carries
    hundreds to thousands of tokens and its M tile count is an order of
    magnitude larger. The bucket is a shape fact read off the trace, not a
    token count inferred from a tile size the trace does not record.
    """
    if grid_x <= 1:
        return "gx1"
    if grid_x == 2:
        return "gx2"
    if grid_x <= 8:
        return "gx3_8"
    return "gx9plus"


def _family(short_name: str, demangled: str) -> str:
    """Group kernels into the families the lever ladder is priced against.

    The only subtle one is FA2: the tree attention and the prefill attention are
    the SAME `flash_fwd_splitkv_kernel`, differing only in the first bool
    template argument (Is_causal). The gqa_pair lever targets the non-causal
    tree instantiation; whether it transfers to the causal one is the single
    largest open upside on the board, so the two are never merged here.
    """
    if short_name == "flash_fwd_splitkv_kernel":
        causal = ", (bool)1, (bool)0, (bool)0, (bool)0, (bool)1" in demangled
        return ("FA2 splitkv causal (prefill + MTP)" if causal
                else "FA2 splitkv tree (gqa_pair target)")
    if short_name == "device_kernel" and "cutlass_3x_gemm_fp8_blockwise" in demangled:
        return "GEMM fp8 blockwise"
    if short_name == "_tree_gdn_path_kernel":
        return "GDN tree scan (single_launch target)"
    if short_name in {
        "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
        "chunk_fwd_kernel_o",
        "recompute_w_u_fwd_kernel",
        "chunk_scaled_dot_kkt_fwd_kernel",
        "chunk_local_cumsum_scalar_kernel",
        "merge_16x16_to_64x64_inverse_kernel",
    }:
        return "GDN chunked (prefill-only)"
    if short_name == "fused_sigmoid_gating_delta_rule_update_kernel":
        return "GDN delta-rule update"
    if short_name == "kernel_unified_attention_2d":
        return "unified attention (full-attn layers)"
    if short_name.startswith("nvjet") or short_name == "Kernel2":
        return "bf16 GEMM (LM head / misc)"
    return "other"


class _StepIndex:
    """Membership test for 'this timestamp is inside a pure-decode step'."""

    def __init__(self, steps: list[tuple[int, int]]) -> None:
        self._steps = steps
        self._starts = [s for s, _ in steps]

    def inside(self, t: int) -> bool:
        i = bisect.bisect_right(self._starts, t) - 1
        return i >= 0 and t < self._steps[i][1]


def _load_steps(conn: sqlite3.Connection) -> tuple[list[tuple[int, int]], int]:
    raw = _nvtx(conn, STEP_RANGE)
    if not raw:
        raise SystemExit(f"no {STEP_RANGE} NVTX ranges in trace")
    floor = min(a for a, _ in raw)
    good = [
        (a, b) for a, b in raw
        if (b - a) < MAX_PLAUSIBLE_STEP_NS and (a - floor) < MAX_PLAUSIBLE_STEP_NS * 60
    ]
    dropped = len(raw) - len(good)
    if dropped > MAX_BOGUS_STEP_RANGES:
        raise SystemExit(
            f"{dropped} {STEP_RANGE} ranges outside the plausible window; "
            f"allowance is {MAX_BOGUS_STEP_RANGES}. Refusing to decompose."
        )
    return good, dropped


def _gaps(steps: list[tuple[int, int]], threshold_ns: int) -> list[tuple[int, int]]:
    out = []
    for i in range(1, len(steps)):
        a, b = steps[i - 1][1], steps[i][0]
        if b - a >= threshold_ns:
            out.append((a, b))
    return out


def _kernels(conn: sqlite3.Connection) -> list[tuple[int, int, int, str, str]]:
    has_demangled = bool(_q(
        conn, "SELECT 1 FROM pragma_table_info('CUPTI_ACTIVITY_KIND_KERNEL') "
              "WHERE name='demangledName'"))
    if has_demangled:
        sql = """
            SELECT k.start, k.end, k.gridX, sn.value,
                   COALESCE(sd.value, sn.value)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds sn ON sn.id = k.shortName
            LEFT JOIN StringIds sd ON sd.id = k.demangledName
            ORDER BY k.start
        """
    else:
        sql = """
            SELECT k.start, k.end, k.gridX, sn.value, sn.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds sn ON sn.id = k.shortName
            ORDER BY k.start
        """
    return [
        (int(s), int(e), int(gx or 0), nm, dn)
        for s, e, gx, nm, dn in _q(conn, sql)
    ]


def reduce_capture(sqlite_path: str, gap_threshold_ms: float = 1000.0) -> dict[str, Any]:
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        steps, dropped = _load_steps(conn)
        idx = _StepIndex(steps)
        threshold_ns = int(gap_threshold_ms * 1e6)
        gaps = _gaps(steps, threshold_ns)

        window_start, window_end = steps[0][0], steps[-1][1]
        window_ns = window_end - window_start
        inside_ns = sum(b - a for a, b in steps)
        all_gap_ns = sum(
            steps[i][0] - steps[i - 1][1] for i in range(1, len(steps))
            if steps[i][0] > steps[i - 1][1]
        )
        big_gap_ns = sum(b - a for a, b in gaps)

        # --- pass identity -------------------------------------------------
        passes = _nvtx(conn, PASS_RANGE)
        mixed = [(a, b) for a, b in passes if not idx.inside(a)]
        pure = len(passes) - len(mixed)
        if pure != len(steps):
            raise SystemExit(
                f"pass identity broken: {pure} {PASS_RANGE} instances inside a "
                f"step range vs {len(steps)} step ranges"
            )
        gap_starts = [a for a, _ in gaps]
        stray = 0
        for a, _ in mixed:
            i = bisect.bisect_right(gap_starts, a) - 1
            if not (i >= 0 and a < gaps[i][1]):
                stray += 1
        if stray:
            raise SystemExit(
                f"{stray} mixed forward passes fall outside every gap over "
                f"{gap_threshold_ms} ms; the gap set does not cover them"
            )

        # --- kernel classification ----------------------------------------
        kernels = _kernels(conn)
        classes: dict[tuple[str, str], list[int]] = {}
        families: dict[str, list[int]] = {}
        for s, e, gx, nm, dn in kernels:
            key = (nm, _grid_bucket(gx))
            row = classes.setdefault(key, [0, 0, 0, 0])
            fam = families.setdefault(_family(nm, dn), [0, 0])
            if idx.inside(s):
                row[0] += 1
                row[1] += e - s
                fam[0] += e - s
            else:
                row[2] += 1
                row[3] += e - s
                fam[1] += e - s

        n_pure, n_mixed = len(steps), len(mixed)
        prefill_exclusive_ns = 0
        prefill_excess_ns = 0
        codecode_ns = 0
        prefill_classes: set[tuple[str, str]] = set()
        for key, (i_n, i_ns, o_n, o_ns) in classes.items():
            if o_ns <= 0:
                continue
            if i_n == 0:
                prefill_exclusive_ns += o_ns
                prefill_classes.add(key)
                continue
            per_mixed = o_ns / n_mixed
            per_pure = i_ns / n_pure
            decode_share = min(per_mixed, per_pure) * n_mixed
            codecode_ns += decode_share
            prefill_excess_ns += o_ns - decode_share

        # --- idle -----------------------------------------------------------
        gap_busy_ns = 0
        per_gap = []
        gap_starts_all = [a for a, _ in gaps]
        gap_kern: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(gaps))}
        gap_class: dict[int, dict[str, float]] = {
            i: {"prefill": 0.0, "codecode": 0.0} for i in range(len(gaps))
        }
        for s, e, gx, nm, _dn in kernels:
            i = bisect.bisect_right(gap_starts_all, s) - 1
            if not (i >= 0 and s < gaps[i][1]):
                continue
            a, b = gaps[i]
            gap_kern[i].append((max(s, a), min(e, b)))
            key = (nm, _grid_bucket(gx))
            i_n, i_ns, o_n, o_ns = classes[key]
            if i_n == 0:
                gap_class[i]["prefill"] += e - s
            else:
                frac_decode = min(1.0, (i_ns / n_pure) / (o_ns / n_mixed)) if o_ns else 0.0
                gap_class[i]["codecode"] += (e - s) * frac_decode
                gap_class[i]["prefill"] += (e - s) * (1.0 - frac_decode)

        for i, (a, b) in enumerate(gaps):
            busy = _union_ns(sorted(gap_kern[i]))
            gap_busy_ns += busy
            n_p = sum(1 for ps, _ in mixed if a <= ps < b)
            plain = gap_class[i]["prefill"] + gap_class[i]["codecode"]
            scale = (busy / plain) if plain > 0 else 0.0
            per_gap.append({
                "index": i,
                "t_start_s": (a - window_start) / 1e9,
                "wall_s": (b - a) / 1e9,
                "mixed_passes": n_p,
                "gpu_busy_s": busy / 1e9,
                "gpu_busy_pct": 100.0 * busy / (b - a),
                "gpu_idle_s": (b - a - busy) / 1e9,
                "class_a_prefill_s": gap_class[i]["prefill"] * scale / 1e9,
                "class_d_codecode_s": gap_class[i]["codecode"] * scale / 1e9,
                "class_bc_idle_s": (b - a - busy) / 1e9,
                "kernels": len(gap_kern[i]),
            })

        # Whole-window idle, so class (b) can be bounded against the WHOLE
        # window and not only against the gaps.
        merged_idle: list[tuple[int, int]] = []
        cur_s, cur_e = kernels[0][0], kernels[0][1]
        for s, e, _gx, _nm, _dn in kernels[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                merged_idle.append((cur_e, s - cur_e))
                cur_s, cur_e = s, e
        window_idle_ns = sum(d for _, d in merged_idle)
        idle_in_step = sum(d for t, d in merged_idle if idx.inside(t))
        longest_idle_ns = max((d for _, d in merged_idle), default=0)
        longest_gap_idle_ns = max(
            (d for t, d in merged_idle if not idx.inside(t)), default=0
        )

        plain_out = prefill_exclusive_ns + prefill_excess_ns + codecode_ns
        scale = (gap_busy_ns / plain_out) if plain_out > 0 else 0.0
        class_a = (prefill_exclusive_ns + prefill_excess_ns) * scale
        class_d = codecode_ns * scale
        class_bc = big_gap_ns - gap_busy_ns

        return {
            "schema": SCHEMA,
            "citable": False,
            "acceptance_valid": False,
            "diagnostic_only": True,
            "source_sqlite": str(sqlite_path),
            "gap_threshold_ms": gap_threshold_ms,
            "window": {
                "wall_s": window_ns / 1e9,
                "pure_decode_step_ranges": n_pure,
                "bogus_step_ranges_dropped": dropped,
                "mixed_forward_passes": n_mixed,
                "forward_passes_total": len(passes),
                "inside_decode_s": inside_ns / 1e9,
                "inside_decode_pct": 100.0 * inside_ns / window_ns,
                "outside_decode_s": all_gap_ns / 1e9,
                "outside_decode_pct": 100.0 * all_gap_ns / window_ns,
                "gaps_over_threshold": len(gaps),
                "gaps_over_threshold_s": big_gap_ns / 1e9,
                "longest_gap_s": max((b - a for a, b in gaps), default=0) / 1e9,
                "sub_threshold_gap_s": (all_gap_ns - big_gap_ns) / 1e9,
                "mean_pure_decode_step_ms": inside_ns / n_pure / 1e6,
                "mean_mixed_pass_ms": (big_gap_ns / n_mixed / 1e6) if n_mixed else None,
            },
            "classes": {
                "a_chunked_prefill_compute_s": class_a / 1e9,
                "a_chunked_prefill_compute_pct_window": 100.0 * class_a / window_ns,
                "b_gpu_idle_awaiting_demand_s": class_bc / 1e9,
                "b_gpu_idle_awaiting_demand_pct_window": 100.0 * class_bc / window_ns,
                "d_cobatched_decode_compute_s": class_d / 1e9,
                "d_cobatched_decode_compute_pct_window": 100.0 * class_d / window_ns,
                "note": (
                    "class (c) scheduler/host overhead is not separable from "
                    "class (b): both can only appear as gap GPU idle, and the "
                    "gap idle is bounded above by "
                    f"{class_bc / 1e9:.3f} s in total with a longest single "
                    f"in-gap idle interval of {longest_gap_idle_ns / 1e6:.1f} ms."
                ),
            },
            "idle": {
                "window_gpu_idle_s": window_idle_ns / 1e9,
                "window_gpu_idle_pct": 100.0 * window_idle_ns / window_ns,
                "idle_inside_decode_steps_s": idle_in_step / 1e9,
                "idle_outside_decode_steps_s": (window_idle_ns - idle_in_step) / 1e9,
                "longest_idle_interval_ms": longest_idle_ns / 1e6,
                "longest_out_of_step_idle_interval_ms": longest_gap_idle_ns / 1e6,
            },
            "kernel_evidence": {
                "prefill_exclusive_plain_sum_s": prefill_exclusive_ns / 1e9,
                "prefill_excess_on_shared_plain_sum_s": prefill_excess_ns / 1e9,
                "cobatched_decode_plain_sum_s": codecode_ns / 1e9,
                "plain_sum_vs_union_ratio": (plain_out / gap_busy_ns) if gap_busy_ns else None,
                "prefill_exclusive_classes": sorted(
                    (
                        {
                            "kernel": nm,
                            "grid_bucket": gb,
                            "out_of_step_instances": classes[(nm, gb)][2],
                            "out_of_step_s": classes[(nm, gb)][3] / 1e9,
                        }
                        for nm, gb in prefill_classes
                    ),
                    key=lambda r: -r["out_of_step_s"],
                )[:24],
                "shared_classes_top": sorted(
                    (
                        {
                            "kernel": nm,
                            "grid_bucket": gb,
                            "in_step_s": v[1] / 1e9,
                            "out_of_step_s": v[3] / 1e9,
                            "per_pass_ratio_out_over_in": (
                                (v[3] / n_mixed) / (v[1] / n_pure) if v[1] else None
                            ),
                        }
                        for (nm, gb), v in classes.items()
                        if v[0] > 0 and v[3] > 0
                    ),
                    key=lambda r: -r["out_of_step_s"],
                )[:20],
            },
            "lever_transfer": {
                "note": (
                    "base_dilution = inside_decode / window: the factor the "
                    "width-4 attribution applied to EVERY decode lever. It is "
                    "correct only for a kernel that runs nowhere but inside a "
                    "pure-decode step. transfer = window/in-step time for the "
                    "family; D = transfer x base_dilution is what a 1% "
                    "step-wall saving on that family is worth as a % of window "
                    "wall. The GPU is ~95% busy and effectively serial here "
                    "(plain kernel sum / window = "
                    f"{sum(v[0] + v[1] for v in families.values()) / window_ns:.3f}), "
                    "so kernel time removed is window wall removed."
                ),
                "base_dilution": inside_ns / window_ns,
                "families": sorted(
                    (
                        {
                            "family": name,
                            "in_step_s": v[0] / 1e9,
                            "out_of_step_s": v[1] / 1e9,
                            "window_s": (v[0] + v[1]) / 1e9,
                            "pct_of_window": 100.0 * (v[0] + v[1]) / window_ns,
                            "transfer": ((v[0] + v[1]) / v[0]) if v[0] else None,
                            "D": (
                                ((v[0] + v[1]) / v[0]) * inside_ns / window_ns
                                if v[0] else None
                            ),
                        }
                        for name, v in families.items()
                    ),
                    key=lambda r: -r["window_s"],
                ),
            },
            "gaps": per_gap,
        }
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sqlite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap-threshold-ms", type=float, default=1000.0)
    args = ap.parse_args(argv)
    doc = reduce_capture(args.sqlite, args.gap_threshold_ms)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    w, cl = doc["window"], doc["classes"]
    print(f"window {w['wall_s']:.2f} s  "
          f"inside decode {w['inside_decode_pct']:.1f}%  "
          f"outside {w['outside_decode_pct']:.1f}%")
    print(f"  {w['pure_decode_step_ranges']} pure-decode steps + "
          f"{w['mixed_forward_passes']} mixed passes in "
          f"{w['gaps_over_threshold']} gaps")
    print(f"  (a) chunked prefill compute   "
          f"{cl['a_chunked_prefill_compute_s']:8.2f} s  "
          f"{cl['a_chunked_prefill_compute_pct_window']:5.1f}% of window")
    print(f"  (d) co-batched decode compute "
          f"{cl['d_cobatched_decode_compute_s']:8.2f} s  "
          f"{cl['d_cobatched_decode_compute_pct_window']:5.1f}% of window")
    print(f"  (b/c) GPU idle in gaps        "
          f"{cl['b_gpu_idle_awaiting_demand_s']:8.2f} s  "
          f"{cl['b_gpu_idle_awaiting_demand_pct_window']:5.1f}% of window")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
