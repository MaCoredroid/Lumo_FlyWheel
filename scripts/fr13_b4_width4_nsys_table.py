#!/usr/bin/env python3
"""Render the B4 width-4 attribution JSON as the tables the analysis quotes.

DIAGNOSTIC. NOT CITABLE. Every table this prints carries profiler-perturbed
absolute numbers; they are attribution weights, not acceptance readings.

The renderer refuses to print kernel tables when the capture-validity section
did not pass, mirroring the reducer's own ordering: a profile that cannot prove
which steps it profiled may not rename a kernel.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Detection thresholds for the width-4 operating point, recomputed from the
# sealed artifact rather than hardcoded prose: MDE = CV * t/sqrt(n) with the
# repo's pinned one-sided critical at df=3.
T95_ONE_SIDED_DF3 = 2.3534


def _bar(width: int = 78) -> str:
    return "-" * width


def _fmt(value: Any, spec: str = "8.2f") -> str:
    if value is None:
        return "n/a".rjust(int(spec.split(".")[0]))
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def render(doc: dict[str, Any]) -> str:
    out: list[str] = []
    w = out.append

    w("=" * 78)
    w("FR13 B4 WIDTH-4 NSYS ATTRIBUTION -- DIAGNOSTIC, NOT CITABLE")
    w("=" * 78)
    w("")
    w("acceptance_valid = false. CUPTI was attached for the whole arm lifetime,")
    w("so every absolute number below is profiler-perturbed and must NOT be")
    w("compared as a regression against an unprofiled wall point.")
    w("")

    # ---- capture validity, FIRST ---------------------------------------
    w(_bar())
    w("1. CAPTURE VALIDITY  (must pass before any kernel is named)")
    w(_bar())
    cap = doc.get("capture", {})
    w(f"  session                  {cap.get('session')}")
    w(f"  inner bracket steps      [{cap.get('inner_first_step')}, "
      f"{cap.get('inner_last_step_exclusive')}) = {cap.get('inner_steps')} steps")
    w(f"  outer bracket steps      [{cap.get('outer_first_step')}, "
      f"{cap.get('outer_last_step_exclusive')}) = {cap.get('outer_steps')} steps")
    w(f"  edge ambiguity           open {cap.get('open_edge_ambiguity_steps')} / "
      f"close {cap.get('close_edge_ambiguity_steps')} steps "
      f"({_fmt(100 * (cap.get('ambiguity_fraction_of_inner') or 0), '.2f')}% of inner)")
    arm = cap.get("arm_condition", {})
    w(f"  armed at step            {arm.get('armed_at_steps')} "
      f"(trailing events/step {_fmt(arm.get('trailing_events_per_step'), '.3f')} "
      f"over {arm.get('trailing_step_span')} steps)")
    w("")

    cc = doc.get("census_cross_check", {})
    if cc.get("status") == "census_absent":
        w("  census cross-check       ABSENT -- capture cannot be width-verified")
    else:
        ok_s = cc.get("records_equal_counter_steps")
        ok_e = cc.get("events_equal_counter_events")
        w(f"  census records in range  {cc.get('census_records_in_range')} vs "
          f"counter steps {cc.get('counter_forward_steps')}   "
          f"{'IDENTITY HOLDS' if ok_s else '*** MISMATCH ***'}")
        w(f"  census events in range   {cc.get('census_events_in_range')} vs "
          f"counter events {cc.get('counter_events')}   "
          f"{'IDENTITY HOLDS' if ok_e else '*** MISMATCH ***'}")
        w(f"  batch-width histogram    {cc.get('batch_width_histogram')}")
        w(f"  width-4 fraction         "
          f"{_fmt(100 * (cc.get('width4_fraction') or 0), '.1f')}%   "
          f"census events/step {_fmt(cc.get('census_events_per_step'), '.4f')}")
    nvc = doc.get("nvtx_vs_counter_steps")
    if nvc:
        w(f"  NVTX step instances      {nvc.get('nvtx_step_instances')} "
          f"(counter inner {nvc.get('counter_inner_steps')}, "
          f"outer {nvc.get('counter_outer_steps')})   "
          f"{'CONSISTENT' if nvc.get('nvtx_within_outer_bracket') else '*** OUT OF BRACKET ***'}")
    w("")

    # ---- split reconciliation ------------------------------------------
    w(_bar())
    w("2. PHASE SPLIT RECONCILIATION vs THE SEALED (UNPROFILED) WIDTH-4 POINT")
    w(_bar())
    ref = doc.get("sealed_reference", {})
    w(f"  sealed source: {ref.get('source')}")
    w(f"  topology {ref.get('topology')}, n={ref.get('arms')} arms")
    w("")
    w(f"  {'component':<26}{'capture':>11}{'sealed':>11}{'delta':>10}{'pct':>9}")
    recon = doc.get("reconciliation", {})
    order = ["sfwd_gpu_ms_per_step", "dfwd_gpu_ms_per_step", "cfwd_gpu_ms_per_step",
             "gpu_component_ms_per_step", "other_wall_ms_per_step",
             "step_wall_ms", "events_per_step"]
    for k in order:
        if k not in recon:
            continue
        r = recon[k]
        w(f"  {k:<26}{_fmt(r['capture'], '11.3f')}{_fmt(r['sealed'], '11.3f')}"
          f"{_fmt(r['delta'], '10.3f')}{_fmt(r['pct'], '8.1f')}%")
    w("")
    w("  A capture ABOVE the sealed point is expected: CUPTI inflates the arm.")
    w("  The reconciliation is read for SHAPE (relative phase shares), not for")
    w("  absolute agreement.")
    w("")

    inner = doc.get("counter_split_inner", {})
    wall = inner.get("step_wall_ms")
    if wall:
        w(f"  {'capture phase shares':<26}{'ms/step':>11}{'% of wall':>11}")
        for k, lab in (("sfwd_gpu_ms_per_step", "sfwd"),
                       ("cfwd_gpu_ms_per_step", "cfwd"),
                       ("dfwd_gpu_ms_per_step", "dfwd"),
                       ("other_wall_ms_per_step", "other (wall residual)")):
            v = inner.get(k)
            if v is not None:
                w(f"  {lab:<26}{_fmt(v, '11.3f')}{_fmt(100 * v / wall, '10.1f')}%")
        w(f"  {'TOTAL step wall':<26}{_fmt(wall, '11.3f')}{'100.0':>10}%")
    w("")

    valid = (
        cc.get("records_equal_counter_steps") is not False
        and cc.get("events_equal_counter_events") is not False
    )
    if not valid:
        w("*** CENSUS IDENTITY FAILED -- kernel tables withheld. ***")
        return "\n".join(out)

    # ---- GPU phase projection ------------------------------------------
    proj = doc.get("phase_projection", {})
    if proj:
        w(_bar())
        w("3. NVTX->GPU PROJECTION  (span / busy / idle, ms per captured step)")
        w(_bar())
        w(f"  {'range':<16}{'span':>10}{'busy':>10}{'idle':>10}{'ops':>12}"
          f"{'instances':>11}")
        for name in ("step", "sfwd", "cfwd", "dfwd", "postprocess"):
            if name not in proj:
                continue
            p = proj[name]
            w(f"  {name:<16}{_fmt(p.get('span_ms_per_step'), '10.3f')}"
              f"{_fmt(p.get('ms_per_step'), '10.3f')}"
              f"{_fmt(p.get('idle_ms_per_step'), '10.3f')}"
              f"{p.get('ops', 0):>12,}{p.get('instances', 0):>11,}")
        w("")
        u = proj.get("step", {}).get("union_equals_plain_sum")
        w(f"  busy computed as interval UNION; equals plain sum: {u}")
        pr = doc.get("projection_residual")
        if pr:
            w(f"  GPU-PROJECTION residual (step envelope - disjoint phases): "
              f"{_fmt(pr.get('residual_ms_per_step'), '.3f')} ms/step")
            w("    NOTE: this is NOT other_wall_ms_per_step and NOT gpu idle.")
        w("")

    ht = doc.get("host_tail_projection")
    if ht:
        w(f"  {'host-tail sub-range':<22}{'busy':>10}{'idle':>10}{'ops':>10}")
        for k, p in sorted(ht.items(), key=lambda kv: -(kv[1].get("ms_per_step") or 0)):
            w(f"  {k:<22}{_fmt(p.get('ms_per_step'), '10.3f')}"
              f"{_fmt(p.get('idle_ms_per_step'), '10.3f')}{p.get('ops', 0):>10,}")
        w("")

    # ---- within-phase kernel groups -------------------------------------
    kern = doc.get("phase_kernels", {})
    for phase in ("sfwd", "cfwd", "dfwd", "postprocess"):
        if phase not in kern:
            continue
        w(_bar())
        w(f"4.{phase.upper()}  kernel groups (ms/step, GPU rows only)")
        w(_bar())
        groups = kern[phase]["groups_ms_per_step"]
        ptot = proj.get(phase, {}).get("ms_per_step") or sum(
            g["ms_per_step"] for g in groups.values())
        w(f"  {'group':<24}{'ms/step':>10}{'% phase':>10}{'% wall':>10}{'inst':>12}")
        for g, v in groups.items():
            w(f"  {g:<24}{_fmt(v['ms_per_step'], '10.3f')}"
              f"{_fmt(100 * v['ms_per_step'] / ptot if ptot else 0, '9.1f')}%"
              f"{_fmt(100 * v['ms_per_step'] / wall if wall else 0, '9.1f')}%"
              f"{v['instances']:>12,}")
        w(f"  {'PHASE TOTAL (projected)':<24}{_fmt(ptot, '10.3f')}")
        w("")
        w("  top kernels:")
        for r in kern[phase]["top_kernels"][:10]:
            nm = r["name"]
            nm = nm if len(nm) <= 62 else nm[:59] + "..."
            w(f"    {_fmt(r['ms_per_step'], '8.3f')} ms/step  "
              f"{_fmt(r['instances_per_step'], '7.1f')}/step  [{r['group']}]")
            w(f"      {nm}")
        w("")

    # ---- memcpy / F-window ----------------------------------------------
    mc = doc.get("memcpy_census")
    if mc:
        w(_bar())
        w("5. MEMCPY CENSUS  (the F-window 4-byte D2H lives here)")
        w(_bar())
        w(f"  {'copy_kind':<12}{'count':>10}{'per_step':>10}{'mean_B':>12}"
          f"{'gpu_ms/step':>13}")
        for m in mc:
            w(f"  {str(m['copy_kind']):<12}{m['count']:>10,}"
              f"{_fmt(m['per_step'], '10.2f')}{_fmt(m['mean_bytes'], '12.1f')}"
              f"{_fmt(m['gpu_ms_per_step'], '13.4f')}")
        sm = doc.get("small_memcpy_le_8_bytes")
        if sm:
            w(f"  <=8-byte copies: {sm['count']:,} total, "
              f"{_fmt(sm['per_step'], '.2f')}/step, "
              f"{_fmt(sm['gpu_ms_per_step'], '.4f')} gpu ms/step")
        w("")

    # ---- CUPTI, located not subtracted ----------------------------------
    po = doc.get("profiler_overhead_by_thread")
    if po:
        w(_bar())
        w("6. PROFILER OVERHEAD -- LOCATED, NEVER SUBTRACTED")
        w(_bar())
        for t in po:
            w(f"  tid {t['global_tid']:<20} {t['records']:>7,} records  "
              f"mean {_fmt(t['mean_us'], '9.1f')} us  "
              f"{_fmt(t['ms_per_step'], '8.3f')} ms/step")
        w("")
        w("  No host row enters any GPU total in this artifact. Overhead that")
        w("  sits on a flush thread is not on the critical path; overhead on")
        w("  the main CUDA thread would be.")
        w("")

    gi = doc.get("graph_inventory")
    if gi:
        w(_bar())
        w("7. CUDA GRAPH INVENTORY")
        w(_bar())
        w(f"  {'graph':<10}{'kernels':>12}{'replays':>10}{'nodes/replay':>14}"
          f"{'gpu ms/step':>13}")
        for g in gi[:8]:
            w(f"  {str(g['graph_id']):<10}{g['kernels']:>12,}{g['replays']:>10,}"
              f"{str(g['nodes_per_replay']):>14}"
              f"{_fmt(g['gpu_ms_per_step'], '13.3f')}")
        w("")

    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("attribution_json")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    doc = json.loads(Path(args.attribution_json).read_text())
    text = render(doc)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
