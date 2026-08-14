#!/usr/bin/env python3
"""Bin the banked width-4 Nsight capture's GDN tree-scan launches by batch width.

Prices the parked `_tree_gdn_kernel_fixed32_single_launch` lever at the B4
width-4 operating point from measured evidence only:

  * the deployed two-launch schedule (`_tree_gdn_path_kernel`) is issued once
    per (layer, level, request) -- 48 layers x {L0,L1} x width = 96*width
    launches per decode step.  The launch count is therefore SELF-IDENTIFYING:
    a step's width falls out of its launch count with no census join.
  * the B1 kernel probe (results/fr13_gdn_scan_20260811/cost_probe.json)
    supplies the single-vs-two ratio and the handoff decomposition.

Nothing here is a step-envelope claim.  The capture is CUPTI-attached for its
whole lifetime; every absolute time it carries is profiler-perturbed and is
reported, never subtracted (the same rule the width-4 attribution used).

Fail-closed: every cross-check against the two prior independent reductions
(attribution_final.json's gdn_scan ms/step, gaps.json's in-step seconds) is an
assertion, not a print.  A drifted capture cannot produce a scope artifact.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import sqlite3
import sys
from pathlib import Path

SCHEMA = "fr13.gdn_single_launch_b4_scope.v1"
KERNEL = "_tree_gdn_path_kernel"
STEP_RANGE = "fr13.fixed32.step"
GDN_LAYERS = 48
LAUNCHES_PER_LAYER = 2  # level 0 export, level 1 parent-read
LAUNCHES_PER_EVENT = GDN_LAYERS * LAUNCHES_PER_LAYER  # 96

# Prior independent reductions this binning must reproduce.
XCHECK_ATTRIBUTION_GDN_MS_PER_STEP = 40.978
XCHECK_GAPS_IN_STEP_S = 22.149822117
XCHECK_GAPS_OUT_OF_STEP_S = 4.27909517
XCHECK_REL_TOL = 2e-3

# Sealed, UNPROFILED width-4 operating point (Hydra27) and its n=4 MDE.
SEALED_WIDTH4_STEP_WALL_MS = 411.05
WIDTH4_STEP_WALL_MDE_MS = 4.20


class ScopeError(RuntimeError):
    """The evidence cannot support a scope artifact."""


def _q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    return list(conn.execute(sql, params))


def bin_by_width(sqlite_path: Path) -> dict:
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        steps = _q(conn, """
            SELECT n.start, n.end
            FROM NVTX_EVENTS n LEFT JOIN StringIds si ON si.id = n.textId
            WHERE COALESCE(si.value, n.text) = ?
              AND n.end IS NOT NULL
            ORDER BY n.start
        """, (STEP_RANGE,))
        if not steps:
            raise ScopeError(f"capture carries no {STEP_RANGE} NVTX range")

        # A GDN kernel is bound to the decode step whose NVTX host range
        # contains its LAUNCHING runtime call.  Graph-replayed nodes correlate
        # to their cudaGraphLaunch, which is the intended relation (same shape
        # the width-4 attribution reducer uses).
        rows = _q(conn, f"""
            SELECT r.start, k.start, k.end, k.gridZ,
                   k.gridX, k.gridY, k.blockX, k.registersPerThread,
                   k.sharedMemoryExecuted
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON s.id = k.shortName
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
              ON r.correlationId = k.correlationId
            WHERE s.value = '{KERNEL}'
        """)
        if not rows:
            raise ScopeError(f"capture carries no {KERNEL} rows")
    finally:
        conn.close()

    # The deployed schedule must present exactly two geometries: the level-0
    # launch (gridZ == 1) and the level-1 launch (gridZ == 11 parallel chains).
    geoms = collections.Counter(
        (gx, gy, gz, bx) for _, _, _, gz, gx, gy, bx, _, _ in rows)
    if len(geoms) != 2:
        raise ScopeError(f"expected 2 GDN launch geometries, got {dict(geoms)}")
    levels = sorted({gz for _, _, _, gz, *_ in rows})
    if levels != [1, 11]:
        raise ScopeError(f"unexpected GDN gridZ set {levels}")

    starts = [s for s, _ in steps]
    in_step: dict[int, list[tuple[int, int, int]]] = collections.defaultdict(list)
    out_of_step: list[tuple[int, int, int]] = []
    for host, ks, ke, gz, *_ in rows:
        i = bisect.bisect_right(starts, host) - 1
        if i < 0 or host >= steps[i][1]:
            out_of_step.append((ks, ke, gz))
        else:
            in_step[i].append((ks, ke, gz))

    by_width: dict[int, dict] = collections.defaultdict(
        lambda: {"steps": 0, "L0_n": 0, "L1_n": 0, "L0_ns": 0, "L1_ns": 0,
                 "step_wall_ns": 0, "sum_ns": 0, "union_ns": 0})
    for i, launches in in_step.items():
        n = len(launches)
        if n % LAUNCHES_PER_EVENT:
            raise ScopeError(
                f"step {i} carries {n} GDN launches, not a multiple of "
                f"{LAUNCHES_PER_EVENT}: the per-request launch identity that "
                "makes width self-identifying does not hold")
        width = n // LAUNCHES_PER_EVENT
        rec = by_width[width]
        rec["steps"] += 1
        rec["step_wall_ns"] += steps[i][1] - steps[i][0]
        for ks, ke, gz in launches:
            key = "L0" if gz == 1 else "L1"
            rec[key + "_n"] += 1
            rec[key + "_ns"] += ke - ks
        # Union vs plain sum: if the launches overlapped, kernel time removed
        # would NOT equal wall removed and the whole pricing would be void.
        ordered = sorted((ks, ke) for ks, ke, _ in launches)
        rec["sum_ns"] += sum(e - s for s, e in ordered)
        union = 0
        cs, ce = ordered[0]
        for s, e in ordered[1:]:
            if s > ce:
                union += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        rec["union_ns"] += union + (ce - cs)

    # Out-of-step bursts: separated by >2 ms of no GDN activity.  Their SIZE
    # says which batch widths the mixed passes actually run at, which decides
    # whether a batch==4-only route can ever reach that time.
    out_of_step.sort()
    bursts: list[int] = []
    if out_of_step:
        run = 1
        for j in range(1, len(out_of_step)):
            if out_of_step[j][0] - out_of_step[j - 1][1] > 2_000_000:
                bursts.append(run)
                run = 1
            else:
                run += 1
        bursts.append(run)
    burst_hist = collections.Counter(bursts)
    # Each (layer, request) contributes 2 launches, so burst size / 2 is the
    # number of co-resident requests in that mixed pass.
    burst_widths = {str(size // LAUNCHES_PER_LAYER): count
                    for size, count in sorted(burst_hist.items())}
    if any(size % LAUNCHES_PER_LAYER for size in burst_hist):
        raise ScopeError(f"out-of-step burst sizes not request-paired: "
                         f"{dict(burst_hist)}")

    widths = {}
    tot_steps = tot_ns = 0
    for w in sorted(by_width):
        r = by_width[w]
        s = r["steps"]
        widths[str(w)] = {
            "steps": s,
            "launches_per_step": w * LAUNCHES_PER_EVENT,
            "gdn_ms_per_step": (r["L0_ns"] + r["L1_ns"]) / s / 1e6,
            "l0_us_per_launch": r["L0_ns"] / r["L0_n"] / 1e3,
            "l1_us_per_launch": r["L1_ns"] / r["L1_n"] / 1e3,
            "pair_us": (r["L0_ns"] / r["L0_n"] + r["L1_ns"] / r["L1_n"]) / 1e3,
            "step_wall_ms_profiled": r["step_wall_ns"] / s / 1e6,
            "overlap_fraction": 1.0 - r["union_ns"] / r["sum_ns"],
        }
        tot_steps += s
        tot_ns += r["L0_ns"] + r["L1_ns"]

    return {
        "kernel": KERNEL,
        "geometries": [{"gridX": gx, "gridY": gy, "gridZ": gz, "blockX": bx,
                        "launches": n}
                       for (gx, gy, gz, bx), n in sorted(geoms.items(),
                                                         key=lambda kv: kv[0][2])],
        "nvtx_step_instances": len(steps),
        "steps_with_gdn": tot_steps,
        "in_step_gdn_s": tot_ns / 1e9,
        "in_step_blended_ms_per_step": tot_ns / tot_steps / 1e6,
        "out_of_step_gdn_s": sum(e - s for s, e, _ in out_of_step) / 1e9,
        "out_of_step_launches": len(out_of_step),
        "out_of_step_burst_width_histogram": burst_widths,
        "by_width": widths,
    }


def price(bins: dict, probe: dict) -> dict:
    d = probe["derived"]
    b = probe["bench_us"]
    two = float(d["two_launch_us"])
    single = float(d["single_launch_us"])
    ratio = single / two
    saving_us_per_pair = two - single

    # Handoff decomposition, and the denominator honesty note.  The probe
    # published handoff_fraction against two_launch_total, but its numerator is
    # built from ISOLATED ablations whose parents sum to L0+L1, not to the
    # back-to-back pair.  Both are reported; the consistent one is used.
    l0 = float(b["L0_deployed"]["us_p50"])
    l1 = float(b["L1_deployed"]["us_p50"])
    export_write = float(d["l0_export_write_us"])
    parent_read_step = float(d["l1_parent_read_plus_one_step_us"])
    handoff = float(d["handoff_us_per_layer"])
    alpha = export_write / l0          # handoff share inside an L0 launch
    beta = parent_read_step / l1       # handoff share inside an L1 launch

    per_width = {}
    for w, rec in bins["by_width"].items():
        s0 = rec["l0_us_per_launch"] / rec["pair_us"]
        s1 = rec["l1_us_per_launch"] / rec["pair_us"]
        pairs = int(w) * GDN_LAYERS
        floor_ms = pairs * saving_us_per_pair / 1e3
        ratio_ms = pairs * (1.0 - ratio) * rec["pair_us"] / 1e3
        upper_ms = pairs * max(0.0, rec["pair_us"] - single) / 1e3
        per_width[w] = {
            "pairs_per_step": pairs,
            "l0_time_share_of_pair": s0,
            "l1_time_share_of_pair": s1,
            "handoff_fraction_in_situ": alpha * s0 + beta * s1,
            "saving_ms_per_step_floor_absolute_transfer": floor_ms,
            "saving_ms_per_step_ratio_transfer": ratio_ms,
            "saving_ms_per_step_upper_if_single_launch_insensitive": upper_ms,
            "pct_of_profiled_step_wall_ratio_transfer":
                100.0 * ratio_ms / rec["step_wall_ms_profiled"],
        }
    return {
        "b1_probe": {
            "two_launch_us_per_layer": two,
            "single_launch_us_per_layer": single,
            "single_over_two_ratio": ratio,
            "saving_us_per_layer_request_pair": saving_us_per_pair,
            "saving_ms_per_step_at_b1_48_layers":
                -float(d["single_minus_two_ms_step"]),
            "handoff_us_per_layer": handoff,
            "handoff_fraction_as_published": float(
                d["handoff_fraction_of_two_launch"]),
            "handoff_fraction_consistent_denominator": handoff / (l0 + l1),
            "handoff_share_inside_l0_launch": alpha,
            "handoff_share_inside_l1_launch": beta,
        },
        "per_width": per_width,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True, type=Path)
    ap.add_argument("--probe", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    bins = bin_by_width(args.sqlite)

    # Fail closed against both prior independent reductions of this capture.
    xchecks = {
        "attribution_gdn_ms_per_step": (
            bins["in_step_blended_ms_per_step"],
            XCHECK_ATTRIBUTION_GDN_MS_PER_STEP),
        "gaps_in_step_s": (bins["in_step_gdn_s"], XCHECK_GAPS_IN_STEP_S),
        "gaps_out_of_step_s": (bins["out_of_step_gdn_s"],
                               XCHECK_GAPS_OUT_OF_STEP_S),
    }
    reproduced = {}
    for name, (got, want) in xchecks.items():
        if not math.isclose(got, want, rel_tol=XCHECK_REL_TOL):
            raise ScopeError(
                f"{name}: this binning gives {got!r}, prior reduction {want!r}")
        reproduced[name] = {"this_binning": got, "prior_reduction": want}

    for w, rec in bins["by_width"].items():
        if rec["overlap_fraction"] > 0.01:
            raise ScopeError(
                f"width {w} GDN launches overlap {rec['overlap_fraction']:.3f} "
                "-- kernel time removed is not wall removed")

    probe = json.loads(args.probe.read_text())
    pricing = price(bins, probe)

    out = {
        "schema": SCHEMA,
        "acceptance_valid": False,
        "citable": False,
        "analysis_only": True,
        "step_envelope_claim": False,
        "gpu_used": False,
        "docker_used": False,
        "source_sqlite": str(args.sqlite),
        "source_probe": str(args.probe),
        "cross_checks_reproduced": reproduced,
        "bins": bins,
        "pricing": pricing,
        "reference_points": {
            "sealed_width4_step_wall_ms_unprofiled": SEALED_WIDTH4_STEP_WALL_MS,
            "width4_step_wall_mde_ms_n4_hydra27": WIDTH4_STEP_WALL_MDE_MS,
        },
    }
    w4 = pricing["per_width"].get("4")
    if w4:
        out["reference_points"]["width4_saving_over_mde_floor"] = (
            w4["saving_ms_per_step_floor_absolute_transfer"]
            / WIDTH4_STEP_WALL_MDE_MS)
        out["reference_points"]["width4_saving_over_mde_ratio"] = (
            w4["saving_ms_per_step_ratio_transfer"]
            / WIDTH4_STEP_WALL_MDE_MS)
        out["reference_points"]["width4_pct_of_sealed_unprofiled_wall_ratio"] = (
            100.0 * w4["saving_ms_per_step_ratio_transfer"]
            / SEALED_WIDTH4_STEP_WALL_MS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ScopeError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(2)
