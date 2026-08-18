#!/usr/bin/env python3
"""FR14 promotion A/B: the PAIRED comparison of two drained arms.

Pairing is the whole point. seam_move_economics.md 9.4 banked the SAME arm at
accept 3.81 / 4.04 / 4.28 across three runs (+/-10%), so an unpaired accept
delta below ~10% measures the task mix, not the lever. This reducer therefore
reports:

  * the arm-level delta on each instrument, AND
  * the per-task (per-bracket) delta on the four canonical instances, which is
    the only basis on which a sub-10% accept move can be read at all, AND
  * the +/-10% variance band drawn explicitly around every accept number, so a
    delta inside it is reported as INSIDE VARIANCE rather than as a result.

Verdict instruments (pass 24): step_wall_ms and s_per_fwd_gpu. TPS is reported
and is never the verdict for a per-step lever.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VARIANCE = 0.10  # seam_move_economics.md 9.4


def pct(new: float, old: float) -> float:
    return (new - old) / old * 100.0 if old else float("nan")


def arm_numbers(rec: dict) -> dict:
    ds = [d for d in rec.get("deploy_speed", []) if not d.get("MISSING")]
    if not ds:
        return {}
    d = ds[0]
    return {
        "step_wall_ms": d.get("step_wall_ms"),
        "s_per_fwd_gpu": d.get("s_per_fwd_gpu"),
        "s_per_fwd": d.get("s_per_fwd"),
        "accept_per_event": d.get("accept_per_event"),
        "committed_per_event": d.get("committed_per_event"),
        "drafter_gpu_ms_per_step": d.get("drafter_gpu_ms_per_step"),
        "committer_gpu_ms_per_step": d.get("committer_gpu_ms_per_step"),
        "overhead_other_ms_per_event": d.get("overhead_other_ms_per_event"),
        "per_request_decode_tps": d.get("per_request_decode_tps"),
        "floor_ratio": d.get("floor_ratio"),
        "wall_steps_measured": d.get("wall_steps_measured"),
        "prefill_frac": d.get("prefill_frac"),
        "per_task": {t["instance_id"]: t for t in d.get("per_task", [])},
        "n_tasks": d.get("n_tasks"),
        "task_instance_ids": d.get("task_instance_ids"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="ARM C reduction json")
    ap.add_argument("--treated", required=True, help="ARM G reduction json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c_rec = json.loads(Path(a.control).read_text())
    g_rec = json.loads(Path(a.treated).read_text())
    c, g = arm_numbers(c_rec), arm_numbers(g_rec)

    out = {
        "schema": "fr14.promotion_ab.pair.v1",
        "control": {"label": c_rec.get("label"), "arm": c_rec.get("arm")},
        "treated": {"label": g_rec.get("label"), "arm": g_rec.get("arm")},
        "variance_doctrine": (
            "seam_move_economics.md 9.4: the same arm banked accept 3.81/4.04/"
            "4.28 across three runs. An accept delta inside +/-10% is INSIDE "
            "VARIANCE on the arm level; only the paired per-task brackets can "
            "resolve smaller moves, and even those carry trajectory drift."
        ),
        "arm_level": {},
        "per_task": {},
    }

    for k in (
        "step_wall_ms", "s_per_fwd_gpu", "s_per_fwd", "accept_per_event",
        "committed_per_event", "drafter_gpu_ms_per_step",
        "committer_gpu_ms_per_step", "overhead_other_ms_per_event",
        "per_request_decode_tps", "floor_ratio", "prefill_frac",
        "wall_steps_measured",
    ):
        cv, gv = c.get(k), g.get(k)
        rec = {"control": cv, "treated": gv}
        if isinstance(cv, (int, float)) and isinstance(gv, (int, float)):
            rec["delta"] = gv - cv
            rec["delta_pct"] = pct(gv, cv)
            if k in ("accept_per_event", "committed_per_event",
                     "per_request_decode_tps"):
                rec["inside_10pct_variance"] = abs(rec["delta_pct"]) <= 100 * VARIANCE
        out["arm_level"][k] = rec

    shared = sorted(set(c.get("per_task", {})) & set(g.get("per_task", {})))
    out["paired_instances"] = shared
    out["control_only_instances"] = sorted(
        set(c.get("per_task", {})) - set(g.get("per_task", {}))
    )
    out["treated_only_instances"] = sorted(
        set(g.get("per_task", {})) - set(c.get("per_task", {}))
    )
    for iid in shared:
        ct, gt = c["per_task"][iid], g["per_task"][iid]
        row = {}
        for k in ("s_per_fwd_gpu", "s_per_fwd", "accept_per_event",
                  "per_request_decode_tps", "drafts", "wall_steps"):
            cv, gv = ct.get(k), gt.get(k)
            r = {"control": cv, "treated": gv}
            if isinstance(cv, (int, float)) and isinstance(gv, (int, float)):
                r["delta"] = gv - cv
                r["delta_pct"] = pct(gv, cv)
            row[k] = r
        for label, t in (("control", ct), ("treated", gt)):
            if t.get("wall_steps"):
                row.setdefault("step_wall_ms", {})[label] = (
                    t["wall_seconds"] / t["wall_steps"] * 1000.0
                )
            if t.get("drafter_gpu_spans"):
                row.setdefault("drafter_gpu_ms_per_step", {})[label] = (
                    t["drafter_gpu_seconds"] / t["drafter_gpu_spans"] * 1000.0
                )
            if t.get("committer_gpu_spans"):
                row.setdefault("committer_gpu_ms_per_step", {})[label] = (
                    t["committer_gpu_seconds"] / t["committer_gpu_spans"] * 1000.0
                )
        for k in ("step_wall_ms", "drafter_gpu_ms_per_step",
                  "committer_gpu_ms_per_step"):
            r = row.get(k, {})
            if "control" in r and "treated" in r:
                r["delta"] = r["treated"] - r["control"]
                r["delta_pct"] = pct(r["treated"], r["control"])
        out["per_task"][iid] = row

    # Census-side pairing: the pass-gate arm's warm rate and shape checks.
    for label, rec in (("control", c_rec), ("treated", g_rec)):
        cen = rec.get("census", {})
        out.setdefault("census", {})[label] = {
            "step_events": cen.get("step_events"),
            "active_nodes": cen.get("active_nodes"),
            "verify_rows": cen.get("verify_rows"),
            "mtp_forward_calls": cen.get("mtp_forward_calls"),
            "graph_replays": cen.get("graph_replays"),
            "warm_step_rate": cen.get("warm_step_rate"),
            "graph_signatures": cen.get("graph_signatures"),
            "section_11_7_checks": cen.get("section_11_7_checks"),
            "section_11_7_all_pass": cen.get("section_11_7_all_pass"),
            "drafter_graph_registry": cen.get("drafter_graph_registry"),
        }

    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

    print(f"{'instrument':32s} {'control':>14s} {'treated':>14s} {'delta':>12s} {'delta%':>9s}")
    for k, r in out["arm_level"].items():
        cv, gv = r.get("control"), r.get("treated")
        if not isinstance(cv, (int, float)) or not isinstance(gv, (int, float)):
            continue
        flag = ""
        if r.get("inside_10pct_variance") is True:
            flag = "  <- INSIDE +/-10% VARIANCE"
        print(f"{k:32s} {cv:14.5f} {gv:14.5f} {r['delta']:12.5f} "
              f"{r['delta_pct']:8.2f}%{flag}")
    print()
    print("paired per-task (same instance, both arms):")
    for iid, row in out["per_task"].items():
        sw = row.get("step_wall_ms", {})
        sg = row.get("s_per_fwd_gpu", {})
        ac = row.get("accept_per_event", {})
        print(
            f"  {iid:26s} step_wall {sw.get('control', float('nan')):8.2f} ->"
            f" {sw.get('treated', float('nan')):8.2f} ({sw.get('delta_pct', float('nan')):6.2f}%)"
            f"   s/fwd_gpu {sg.get('control', float('nan')):.5f} ->"
            f" {sg.get('treated', float('nan')):.5f} ({sg.get('delta_pct', float('nan')):6.2f}%)"
            f"   accept {ac.get('control', float('nan')):.3f} ->"
            f" {ac.get('treated', float('nan')):.3f} ({ac.get('delta_pct', float('nan')):6.2f}%)"
        )
    if out["control_only_instances"] or out["treated_only_instances"]:
        print()
        print(f"  UNPAIRED control-only: {out['control_only_instances']}")
        print(f"  UNPAIRED treated-only: {out['treated_only_instances']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
