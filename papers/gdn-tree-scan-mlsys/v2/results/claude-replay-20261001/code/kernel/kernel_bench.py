#!/usr/bin/env python3
"""Per-method GDN tree-verification timing + memory benchmark (ONE method per process; run inside the pinned vLLM image).

  python3 kernel_bench.py --method lumo_fixed32 --out /out/methods [--repeats 100 --warmup 20 --modes graph,eager]

Writes <out>/<method>.json: identities, geometry checks, numerical error vs the float64 C2 reference (M1 comparator), CUDA-event
timing (median/p10/p90 over >= 50 repeats) per region and per step, per-layer derived costs, and memory (peak/reserved/transient
plus per-method scratch and cut-state buffer accounting).  Exit 0 ok, 2 geometry/identity refusal, 3 other failure (a JSON with
the traceback is still written)."""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kbench_common as C  # noqa: E402

METHODS = ("lumo_fixed32", "weaver_author_default", "weaver_aligned_local", "treewy_author_default", "treewy_dotbf16_false",
           "naive_torch_node", "naive_native_paths")


def make_method(name, torch, device, layers, paths, args):
    if name == "lumo_fixed32":
        from kbench_lumo import LumoMethod
        return LumoMethod(torch, device, layers, paths, args)
    if name in ("weaver_author_default", "weaver_aligned_local"):
        from kbench_author import WeaverMethod
        return WeaverMethod(torch, device, layers, paths, args, variant=name)
    if name in ("treewy_author_default", "treewy_dotbf16_false"):
        from kbench_author import TreeWYMethod
        return TreeWYMethod(torch, device, layers, paths, args, dot_bf16=(name == "treewy_author_default"))
    if name == "naive_torch_node":
        from kbench_naive import NaiveTorchNode
        return NaiveTorchNode(torch, device, layers, paths, args)
    if name == "naive_native_paths":
        from kbench_naive import NaiveNativePaths
        return NaiveNativePaths(torch, device, layers, paths, args)
    raise C.GeometryError(f"unknown method {name!r}; choose from {METHODS}")


def derived(method_name, timing, layers):
    """Per-layer and per-step headline numbers from the 'step' experiments (medians; per-layer = per-step / layers)."""
    out = {}
    for mode, exps in timing.items():
        st = exps.get("step")
        if not st or "total" not in st:
            continue
        d = {"step_total_median_ms": st["total"]["median_ms"], "step_total_p10_ms": st["total"]["p10_ms"], "step_total_p90_ms": st["total"]["p90_ms"]}
        for lab, s in st["regions"].items():
            d[f"{lab}_median_ms"] = s["median_ms"]
            d[f"{lab}_per_layer_median_ms"] = s["median_ms"] / layers
        if method_name.startswith("treewy"):
            vo = exps.get("verify_only_sentinel")
            if vo and "total" in vo:
                d["verify_only_median_ms"] = vo["total"]["median_ms"]
                d["commit_estimate_median_ms"] = st["total"]["median_ms"] - vo["total"]["median_ms"]
                d["commit_estimate_note"] = "derived: steady fused step minus sentinel (commit-skipped) step; not separately measurable"
            fl = exps.get("final_flush")
            if fl and "total" in fl:
                d["final_flush_median_ms"] = fl["total"]["median_ms"]
        for k in ("verify_layer0", "layer0_step"):
            if exps.get(k) and "total" in exps[k]:
                d[f"{k}_median_ms"] = exps[k]["total"]["median_ms"]
        if exps.get("topology_rebuild") and "total" in exps["topology_rebuild"]:
            d["topology_rebuild_median_ms"] = exps["topology_rebuild"]["total"]["median_ms"]
        out[mode] = d
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--modes", default="graph,eager")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--stratum", default="ordinary-random", choices=("ordinary-random", "mixed-stress"))
    ap.add_argument("--layers", type=int, default=C.GDN_LAYERS)
    ap.add_argument("--l2-flush-mib", type=int, default=256)
    ap.add_argument("--no-sync-each", action="store_true", help="do not synchronize before each timed iteration")
    ap.add_argument("--capture-warmup", type=int, default=3)
    ap.add_argument("--skip-numerics", action="store_true")
    ap.add_argument("--no-raw", action="store_true", help="omit per-iteration raw milliseconds from the JSON")
    ap.add_argument("--smoke", action="store_true", help="few repeats; results labelled SMOKE (not for the paper)")
    ap.add_argument("--allow-missing-model-config", action="store_true")
    ap.add_argument("--allow-native-sha-mismatch", action="store_true")
    ap.add_argument("--allow-layer-subset", action="store_true", help="debug only: fewer than 48 layers (results labelled)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.method}.json")
    rec = {"schema": "lumo.v2exp.kernel-bench.method.v1", "method": a.method, "args": vars(a), "started_utc": C.utc(), "status": "running"}
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    try:
        C.require(all(m in ("graph", "eager") for m in modes), f"modes {modes}")
        if not a.smoke:
            C.require(a.repeats >= 50, f"--repeats {a.repeats} < 50 (use --smoke for a labelled quick run)")
            C.require(a.layers == C.GDN_LAYERS, f"--layers {a.layers} != {C.GDN_LAYERS} outside --smoke")
        rec["smoke"] = bool(a.smoke)
        C.setup_paths()
        import torch
        C.require(torch.cuda.is_available(), "CUDA not available")
        device = torch.device("cuda", torch.cuda.current_device())    # canonical (the Lumo subtree cache key is str(device))
        rec["identity"] = C.runtime_identity(torch)
        rec["source_hashes"] = C.source_hashes()
        rec["pinned_sources"] = C.check_pinned_sources(rec["source_hashes"], strict=True)
        rec["model_config"] = C.check_model_config(allow_missing=a.allow_missing_model_config)
        rec["topology"] = C.check_topology()
        paths = C.accepted_paths()
        rec["geometry"] = {"gdn_layers": a.layers, "H_k": C.H_K, "H_v": C.H_V, "d_k": C.D_K, "d_v": C.D_V, "physical_rows": C.N_PHYS,
                           "active_drafts": C.N_ACTIVE_DRAFTS, "inactive_physical_rows": list(C.INACTIVE_PHYSICAL), "max_depth": C.MAX_DEPTH,
                           "batch": 1, "scale": C.SCALE, "operand_dtypes": "q,k,v,a,b bf16; A_log, dt_bias, S0 fp32; state fp32",
                           "accepted_paths_cycled": sorted(paths, key=lambda p: (len(paths[p]), p))}
        torch.cuda.reset_peak_memory_stats(device)
        mem = {"process_start": C.mem_snapshot(torch, device)}
        t0 = time.time()
        ops = C.build_operands(a.seed, a.layers, a.stratum)
        rec["operands"] = {"generator": "m1_cycle_driver.build_shared_operands -> q1_2b_fixtures.make_instance (one instance per layer)",
                           "seed": a.seed, "stratum": a.stratum, "layers": len(ops), "build_s": round(time.time() - t0, 3)}
        m = make_method(a.method, torch, device, a.layers, paths, a)
        t0 = time.time()
        m.setup(ops)
        rec["setup_s"] = round(time.time() - t0, 3)
        mem["after_setup"] = C.mem_snapshot(torch, device)
        if getattr(m, "loader", None):
            rec["author_loader"] = m.loader
        if not a.skip_numerics:
            t0 = time.time()
            outs, states = m.numerics(list(C.NUMERICS_PATHS))
            rec["numerics"] = C.numerics_report(torch, ops, outs, states, paths)
            rec["numerics"]["elapsed_s"] = round(time.time() - t0, 3)
            del outs, states
        mem["after_numerics"] = C.mem_snapshot(torch, device)
        flush = C.L2Flusher(torch, device, a.l2_flush_mib)
        timing, captures, mem_modes = {}, {}, {}
        for mode in modes:
            if mode == "graph":
                torch.cuda.reset_peak_memory_stats(device)
                before = C.mem_snapshot(torch, device)
                t0 = time.time()
                m.prepare_graphs()
                captures = {k: dict(v) for k, v in m.capture.items()}
                after = C.mem_snapshot(torch, device)
                mem_modes["graph_capture"] = {"before": before, "after": after, "capture_s": round(time.time() - t0, 3),
                                              "graph_pool_and_capture_bytes": after["allocated"] - before["allocated"]}
            timing[mode] = {}
            for ex in m.experiments(mode):
                m.reset()
                torch.cuda.reset_peak_memory_stats(device)
                before = C.mem_snapshot(torch, device)
                raw = C.time_iterations(torch, ex["fn"], warmup=a.warmup, repeats=a.repeats, pre=ex.get("pre"), flush=flush, sync_each=not a.no_sync_each)
                after = C.mem_snapshot(torch, device)
                r = C.timing_record(raw, keep_raw=not a.no_raw)
                r["note"] = ex.get("note")
                r["memory"] = {"before": before, "after": after, "transient_peak_allocated_bytes": after["max_allocated"] - before["allocated"],
                               "peak_allocated_bytes": after["max_allocated"], "peak_reserved_bytes": after["max_reserved"]}
                timing[mode][ex["name"]] = r
            if mode == "graph":
                missing = [k for k, v in captures.items() if not v.get("ok")]
                if missing:
                    timing[mode]["_capture_failures"] = {k: captures[k] for k in missing}
        rec["timing"] = timing
        rec["graph_capture"] = captures
        rec["derived"] = derived(a.method, timing, a.layers)
        mem["end"] = C.mem_snapshot(torch, device)
        rec["memory"] = {"snapshots": mem, "modes": mem_modes, "l2_flush_buffer_bytes": a.l2_flush_mib << 20,
                         "persistent_allocated_after_setup_bytes": mem["after_setup"]["allocated"] - mem["process_start"]["allocated"],
                         "accounting": m.memory_accounting(),
                         "process_peak_reserved_bytes": max(s["max_reserved"] for s in mem.values()),
                         "note": "torch caching-allocator statistics (GB10 unified memory); peaks reset per experiment; graph pools count as allocated"}
        rec["method_description"] = m.describe()
        rec["timing_protocol"] = {"clock": "torch.cuda.Event pairs on the current stream", "warmup": a.warmup, "repeats": a.repeats,
                                  "sync_before_each_iteration": not a.no_sync_each, "l2_flush_mib_between_iterations": a.l2_flush_mib,
                                  "state_reset": "durable state reset to S0 before each experiment (not between iterations)",
                                  "accepted_paths": "28 Hydra27 accepted paths cycled per iteration from a device table (D2D copy inside the commit region)"}
        rec["status"] = "smoke_complete" if a.smoke else "complete"
        rc = 0
    except C.GeometryError as e:
        rec["status"] = "refused_geometry"
        rec["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        rc = 2
    except Exception as e:  # noqa: BLE001
        rec["status"] = "failed"
        rec["error"] = {"type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()}
        rc = 3
    rec["ended_utc"] = C.utc()
    C.write_json(path, rec)
    print(f"[kernel_bench] {a.method}: {rec['status']} -> {path}", flush=True)
    if rc:
        print(rec["error"]["traceback"], file=sys.stderr, flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
