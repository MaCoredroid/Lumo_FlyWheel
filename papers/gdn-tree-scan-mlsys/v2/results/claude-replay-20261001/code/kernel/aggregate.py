#!/usr/bin/env python3
"""Combine <run>/methods/*.json and <run>/sweep/sweep_*.json into <run>/summary.json and <run>/summary.md (CPU only)."""
from __future__ import annotations

import glob
import json
import os
import sys

MiB = 1 << 20


def _f(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _e(x):
    return "-" if x is None else f"{x:.2e}"


def scratch_bytes(method, acc):
    """The method's verification/commit scratch beyond the durable state (allocated bytes; see the JSON for logical bytes)."""
    if not isinstance(acc, dict):
        return None
    if method.startswith("lumo"):
        return (acc.get("cut_state_buffer") or {}).get("allocated_bytes", 0) + (acc.get("replay_rings") or {}).get("bytes", 0) + (acc.get("committer_fixed16_staging") or {}).get("bytes", 0)
    if method.startswith("weaver"):
        return (acc.get("stashes") or {}).get("bytes", 0) + (acc.get("verifier_workspace_cache") or {}).get("bytes", 0) + (acc.get("topology") or {}).get("bytes", 0)
    if method.startswith("treewy"):
        return (acc.get("stashes_allocated") or {}).get("bytes", 0) + (acc.get("dfs_staging_and_remap") or {}).get("bytes", 0)
    if method.startswith("naive"):
        return (acc.get("node_state_bank") or {}).get("allocated_bytes", 0) + (acc.get("commit_staging") or {}).get("bytes", 0)
    return None


def method_rows(recs):
    rows = []
    for r in recs:
        m = r.get("method")
        if r.get("status") not in ("complete", "smoke_complete"):
            rows.append({"method": m, "status": r.get("status"), "error": (r.get("error") or {}).get("message")})
            continue
        num = r.get("numerics") or {}
        vo = num.get("verify_output") or {}
        cs = num.get("committed_state") or {}
        acc = (r.get("memory") or {}).get("accounting") or {}
        for mode, exps in (r.get("timing") or {}).items():
            st = exps.get("step")
            if not st or "total" not in st:
                continue
            reg = st["regions"]
            first = next(iter(reg.values()))
            row = {"method": m, "mode": mode, "status": r["status"],
                   "step_median_ms": st["total"]["median_ms"], "step_p10_ms": st["total"]["p10_ms"], "step_p90_ms": st["total"]["p90_ms"],
                   "regions": {k: {"median_ms": v["median_ms"], "p10_ms": v["p10_ms"], "p90_ms": v["p90_ms"]} for k, v in reg.items()},
                   "first_region_per_layer_ms": first["median_ms"] / r["geometry"]["gdn_layers"],
                   "transient_peak_allocated_bytes": st["memory"]["transient_peak_allocated_bytes"], "peak_allocated_bytes": st["memory"]["peak_allocated_bytes"],
                   "peak_reserved_bytes": st["memory"]["peak_reserved_bytes"], "scratch_allocated_bytes": scratch_bytes(m, acc),
                   "verify_out_max_abs": vo.get("max_abs_max"), "verify_out_rms_max": vo.get("rms_max"),
                   "state_max_abs": {p: (v or {}).get("max_abs_max") for p, v in cs.items()},
                   "derived": (r.get("derived") or {}).get(mode), "capture": r.get("graph_capture") if mode == "graph" else None}
            rows.append(row)
    return rows


def markdown(rows, sweeps, lumo_acc):
    L = ["# Kernel benchmark summary", "",
         "Times are CUDA-event medians (p10-p90) in ms over the timed repeats; one step = 48 GDN layers, B1, 32-row fixed32 tree.",
         "Errors are max |cand - C2(float64)| over 48 layers x 28 active nodes x 48 heads (outputs) or over 48 layers x 48 heads (committed state); not a gate.", "",
         "| method | mode | regions (median ms) | step median (p10-p90) | per-layer first region | transient peak | scratch alloc | out max-abs err | state max-abs err (n31) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "mode" not in r:
            L.append(f"| {r['method']} | - | {r.get('status')}: {r.get('error')} | | | | | | |")
            continue
        regs = ", ".join(f"{k}={_f(v['median_ms'])}" for k, v in r["regions"].items())
        L.append(f"| {r['method']} | {r['mode']} | {regs} | {_f(r['step_median_ms'])} ({_f(r['step_p10_ms'])}-{_f(r['step_p90_ms'])}) | "
                 f"{_f(r['first_region_per_layer_ms'], 4)} | {_f((r['transient_peak_allocated_bytes'] or 0) / MiB, 1)} MiB | "
                 f"{_f((r['scratch_allocated_bytes'] or 0) / MiB, 1)} MiB | {_e(r['verify_out_max_abs'])} | {_e((r['state_max_abs'] or {}).get('n31'))} |")
    for r in rows:
        d = r.get("derived") or {}
        if r.get("method", "").startswith("treewy") and "commit_estimate_median_ms" in d:
            L.append(f"\nTreeWY ({r['mode']}): verify-only (sentinel) {_f(d.get('verify_only_median_ms'))} ms; derived commit estimate "
                     f"{_f(d.get('commit_estimate_median_ms'))} ms; final flush {_f(d.get('final_flush_median_ms'))} ms.")
    if lumo_acc:
        cut, pf = lumo_acc.get("cut_state_buffer") or {}, lumo_acc.get("paper_formula") or {}
        L += ["", "## LumoTree cut-state memory", "",
              f"N_c = {cut.get('N_c')} cut nodes {cut.get('logical_cut_nodes')}; allocated export buffer {_f((cut.get('allocated_bytes') or 0) / MiB, 1)} MiB (shared by all layers); "
              f"logical bytes written per layer {_f((cut.get('logical_bytes_written_per_layer') or 0) / MiB, 2)} MiB; "
              f"paper formula M_cut = 4*N_c*B_v*d_k = {pf.get('bytes_per_head_value_tile')} B per head/value-tile x {pf.get('head_value_tiles_per_layer')} tiles = "
              f"{_f((pf.get('bytes_per_layer') or 0) / MiB, 2)} MiB per layer."]
    for s in sweeps:
        L += ["", f"## Sweep: {s.get('family')} ({s.get('status')})", ""]
        res = s.get("result") or {}
        groups = {k: v for k, v in res.items() if isinstance(v, dict) and k not in ("env",)}
        if s.get("family") == "lumo_generic":
            groups = {"lumo_generic": res.get("shapes", {})}
        for g, cells in groups.items():
            if not isinstance(cells, dict) or not all(isinstance(c, dict) for c in cells.values()):
                continue
            L.append(f"| {g} | n | graph verify median ms (per layer) | eager verify median ms | layer-0 max-abs err |")
            L.append("|---|---|---|---|---|")
            for name, c in cells.items():
                if "error" in c:
                    L.append(f"| {name} | | error: {c['error']} | | |")
                    continue
                t = c.get("timing") or {}
                gr, eg = t.get("graph") or {}, t.get("eager") or {}
                gm = (gr.get("total") or {}).get("median_ms")
                em = (eg.get("total") or {}).get("median_ms")
                L.append(f"| {name} | {(c.get('shape') or {}).get('n')} | {_f(gm)} ({_f(gr.get('per_layer_median_ms'), 4)}) {gr.get('capture_failed', '')} | {_f(em)} | "
                         f"{_e((c.get('layer0_error_vs_c2') or {}).get('max_abs_max'))} |")
            L.append("")
    return "\n".join(L) + "\n"


def main(run_dir):
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "methods", "*.json")))]
    sweeps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(run_dir, "sweep", "sweep_*.json")))]
    rows = method_rows(recs)
    lumo_acc = next(((r.get("memory") or {}).get("accounting") for r in recs if r.get("method") == "lumo_fixed32" and r.get("status") in ("complete", "smoke_complete")), None)
    summary = {"schema": "lumo.v2exp.kernel-bench.summary.v1", "run_dir": run_dir, "methods": rows, "lumo_cut_state": lumo_acc and {k: lumo_acc.get(k) for k in ("cut_state_buffer", "paper_formula")},
               "statuses": {r.get("method"): r.get("status") for r in recs}, "sweeps": {s.get("family"): s.get("status") for s in sweeps}}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    with open(os.path.join(run_dir, "summary.md"), "w") as f:
        f.write(markdown(rows, sweeps, lumo_acc))
    print(open(os.path.join(run_dir, "summary.md")).read())
    return 0 if recs and all(r.get("status") in ("complete", "smoke_complete") for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/out"))
