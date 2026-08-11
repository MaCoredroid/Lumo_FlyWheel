#!/usr/bin/env python3
"""FR13 GDN fused-scan rung: offline kernel-level cost probe.

Analysis-only. Drives the DEPLOYED two-launch tree-GDN scan
(``_tree_gdn_path_kernel``) and the parked one-launch candidate
(``_tree_gdn_kernel_fixed32_single_launch``) on synthetic tensors at the
deployed fixed32 geometry, and decomposes the two-launch cost into

  * fp32 state-handoff HBM traffic (level-0 export writes, level-1 parent
    reads), and
  * per-node-step serial recurrence latency.

That decomposition is what prices any fused-scan design. It needs no campaign
credential, no served model, no offload host: synthetic tensors, one container,
CPU-launched.

``performance_measurement`` here is a KERNEL microbenchmark on synthetic data.
It is NOT a step-envelope measurement and NOT acceptance-valid. It carries no
TPS, floor, or acceptance claim.

Usage (inside the pinned image, GPU visible):
    python3 scripts/fr13_gdn_scan_fusion_cost_probe.py --out results/....json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lumo_flywheel_serving.fr10_gdn_tree_kernel as G  # noqa: E402

# --- deployed fixed32 geometry (pinned; see design.md §2) -------------------
N_ACTUAL = 32
NUM_VH = 48
NUM_KH = 16
DIM_K = 128
DIM_V = 128
BLOCK_V = 8          # deployed via FR13_TREE_GDN_GEOM_OVERRIDE=BV=8
NUM_WARPS = 8        # G._DEPLOYED_NUM_WARPS
OUTPUT_SCALE = 1.0
USE_QK_L2NORM = True
RAW_GATING = True
SCAN_ALIGN = False

STATE_TILE_BYTES = NUM_VH * DIM_V * DIM_K * 4  # one node's fp32 state export


def _dev() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("fr13 gdn cost probe requires a visible CUDA device")
    return torch.device("cuda")


def build_inputs(device, seed: int = 20260811):
    g = torch.Generator(device="cpu").manual_seed(seed)

    def r(*shape, dtype=torch.bfloat16, scale=1.0):
        t = torch.randn(*shape, generator=g, dtype=torch.float32) * scale
        return t.to(device=device, dtype=dtype)

    return {
        "q": r(N_ACTUAL, NUM_KH, DIM_K),
        "k": r(N_ACTUAL, NUM_KH, DIM_K),
        "v": r(N_ACTUAL, NUM_VH, DIM_V),
        # gating scalars: keep decay in a sane range so the scan does not
        # under/overflow across a 12-deep chain.
        "g": r(N_ACTUAL, NUM_VH, dtype=torch.float32, scale=0.1),
        "beta": r(N_ACTUAL, NUM_VH, dtype=torch.float32, scale=0.1),
        "raw_a": r(N_ACTUAL, NUM_VH, dtype=torch.float32, scale=0.5),
        "raw_b": r(N_ACTUAL, NUM_VH, dtype=torch.float32, scale=0.5),
        "A_log": r(NUM_VH, dtype=torch.float32, scale=0.5),
        "dt_bias": r(NUM_VH, dtype=torch.float32, scale=0.5),
        "h0": r(NUM_VH, DIM_V, DIM_K, dtype=torch.float32, scale=0.05),
    }


def build_outputs(device):
    return {
        "out": torch.zeros(
            N_ACTUAL, NUM_VH, DIM_V, dtype=torch.bfloat16, device=device
        ),
        "ring_k": torch.zeros(
            N_ACTUAL, NUM_KH, DIM_K, dtype=torch.bfloat16, device=device
        ),
        "ring_v": torch.zeros(
            N_ACTUAL, NUM_VH, DIM_V, dtype=torch.bfloat16, device=device
        ),
        "ring_a": torch.zeros(
            N_ACTUAL, NUM_VH, dtype=torch.float32, device=device
        ),
        "ring_b": torch.zeros(
            N_ACTUAL, NUM_VH, dtype=torch.float32, device=device
        ),
        "flags": torch.zeros(4, dtype=torch.int32, device=device),
        "counter": torch.zeros(1, dtype=torch.int32, device=device),
        "dummy_idx": torch.zeros(1, 1, dtype=torch.int32, device=device),
        "dummy_acc": torch.zeros(1, 1, dtype=torch.int32, device=device),
    }


def preseed(device):
    """Build the real level descriptors + single-launch descriptors."""
    G._FR13_FIXED32_GDN_PATH_BV_CANDIDATE = "single_launch"
    G._FR13_SUBTREE_CACHE.clear()
    G.subtree_preseed(
        G._FR13_FIXED32_PARENT, N_ACTUAL, NUM_VH, DIM_V, DIM_K, device
    )
    key = G._subtree_cache_key(N_ACTUAL, NUM_VH, DIM_V, DIM_K, device)
    return G._FR13_SUBTREE_CACHE[key]


def launch_level(st, io, out_t, level_idx, *, lengths=None, export_mode=None,
                 state_source=None, ring_export=False):
    nodes, pars, mlen, npaths, lens = st["levels"][level_idx]
    if lengths is not None:
        lens = lengths
    if state_source is None:
        state_source = 1 if level_idx == 0 else 2
    if export_mode is None:
        export_mode = 1 if level_idx == 0 else 2
    G._tree_gdn_path_kernel[(NUM_VH, triton.cdiv(DIM_V, BLOCK_V), npaths)](
        io["q"], io["k"], io["v"], io["g"], io["beta"],
        io["raw_a"], io["raw_b"], io["A_log"], io["dt_bias"],
        io["h0"], io["dummy_idx"], io["dummy_acc"], io["counter"],
        nodes, pars, lens,
        st["export"], st["emask"], out_t,
        io["ring_k"], io["ring_v"], io["ring_a"], io["ring_b"], io["flags"],
        N_ACTUAL=N_ACTUAL, NUM_KH=NUM_KH, NUM_VH=NUM_VH,
        DIM_K=DIM_K, DIM_V=DIM_V, BLOCK_V=BLOCK_V,
        OUTPUT_SCALE=OUTPUT_SCALE,
        USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM,
        H0_IS_BANK=False, H0_INDEX_ROW=0, H0_BATCH_INDEX=0,
        H0_BANK_STRIDE=0, H0_USE_ACCEPTED_COLUMN=False,
        RAW_GATING=RAW_GATING, COUNT_INVOCATION=False,
        SCAN_ALIGN=SCAN_ALIGN, MAX_PATH_LEN=mlen,
        STATE_SOURCE=state_source, EXPORT_MODE=export_mode,
        RING_EXPORT=ring_export, FLAGS_EXPORT=False, FLAGS_ROWS=0,
        num_warps=NUM_WARPS,
    )


def launch_two_launch(st, io, out_t, ring_export=False):
    launch_level(st, io, out_t, 0, ring_export=ring_export)
    launch_level(st, io, out_t, 1, ring_export=ring_export)


def launch_single(st, io, out_t, ring_export=False):
    sl = st["fixed32_single_launch"]
    contract = sl["contract"]
    root_nodes = st["levels"][0][0]
    branch_nodes, _bp, branch_max_len, _npaths, branch_lengths = (
        st["levels"][1]
    )
    lengths_arg, indices_arg, prescaled = (
        G._fr13_fixed32_gdn_single_launch_path_args(
            sl, branch_lengths, branch_max_len
        )
    )
    G._tree_gdn_kernel_fixed32_single_launch[
        (NUM_VH, triton.cdiv(DIM_V, BLOCK_V), 1)
    ](
        io["q"], io["k"], io["v"], io["g"], io["beta"],
        io["raw_a"], io["raw_b"], io["A_log"], io["dt_bias"],
        io["h0"], io["dummy_idx"], io["dummy_acc"], io["counter"],
        root_nodes, branch_nodes, lengths_arg, indices_arg, sl["path_counts"],
        out_t,
        io["ring_k"], io["ring_v"], io["ring_a"], io["ring_b"], io["flags"],
        io["ring_a"], io["ring_b"],
        N_ACTUAL=N_ACTUAL, NUM_KH=NUM_KH, NUM_VH=NUM_VH,
        DIM_K=DIM_K, DIM_V=DIM_V, BLOCK_V=BLOCK_V,
        OUTPUT_SCALE=OUTPUT_SCALE,
        USE_QK_L2NORM_IN_KERNEL=USE_QK_L2NORM,
        H0_IS_BANK=False, H0_INDEX_ROW=0, H0_INDEX_BATCH_STRIDE=0,
        H0_BATCH_INDEX=0, H0_ACCEPTED_BATCH_STRIDE=0,
        H0_BANK_STRIDE=0, H0_USE_ACCEPTED_COLUMN=False,
        RAW_GATING=RAW_GATING, COUNT_INVOCATION=False,
        SCAN_ALIGN=SCAN_ALIGN,
        ROOT_STEPS=int(contract["groups"]),
        MAX_PATH_LEN=branch_max_len,
        MAX_GROUP_PATHS=int(contract["max_group_paths"]),
        NUM_GROUPS=int(contract["groups"]),
        PRESCALED_PATH_BASE=prescaled,
        RING_EXPORT=ring_export, K_NORM_EXPORT=False, GATE_EXPORT=False,
        DECAY_EXPORT=False, FLAGS_EXPORT=False, FLAGS_ROWS=0,
        num_warps=NUM_WARPS,
    )


def timeit(fn, warmup=20, reps=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)  # us
    samples.sort()
    n = len(samples)
    return {
        "us_min": samples[0],
        "us_p05": samples[max(0, int(0.05 * n) - 1)],
        "us_p50": samples[n // 2],
        "us_mean": sum(samples) / n,
        "reps": n,
    }


def byte_equal(a, b) -> bool:
    return torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8),
        b.contiguous().reshape(-1).view(torch.uint8),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--byte-seeds", type=int, default=4)
    args = ap.parse_args()

    device = _dev()
    props = torch.cuda.get_device_properties(0)
    st = preseed(device)
    io = build_inputs(device)
    io.update(build_outputs(device))

    out_two = torch.zeros_like(io["out"])
    out_one = torch.zeros_like(io["out"])

    # ---- correctness: does the parked one-launch kernel reproduce bytes? ----
    launch_two_launch(st, io, out_two)
    torch.cuda.synchronize()
    export_after_two = st["export"].clone()
    launch_single(st, io, out_one)
    torch.cuda.synchronize()

    out_equal = byte_equal(out_two, out_one)
    diff = (out_two.float() - out_one.float()).abs()
    max_abs = float(diff.max().item())
    n_diff = int((diff != 0).sum().item())

    # ---- byte A/B sweep: seeds x input regimes ----------------------------
    sweep = []
    regimes = {
        "nominal": {},
        "zero_state": {"h0_zero": True},
        "large_gating": {"gate_scale": 4.0},
        "tiny_gating": {"gate_scale": 1e-3},
        "zero_inputs": {"zero_all": True},
    }
    for seed in range(args.byte_seeds):
        for rname, opts in regimes.items():
            probe_io = build_inputs(device, seed=20260811 + 1000 * seed)
            probe_io.update(build_outputs(device))
            if opts.get("h0_zero"):
                probe_io["h0"].zero_()
            if opts.get("zero_all"):
                for key in ("q", "k", "v", "g", "beta", "raw_a", "raw_b",
                            "h0", "A_log", "dt_bias"):
                    probe_io[key].zero_()
            if "gate_scale" in opts:
                probe_io["raw_a"] *= opts["gate_scale"]
                probe_io["raw_b"] *= opts["gate_scale"]
            a = torch.zeros_like(probe_io["out"])
            b = torch.zeros_like(probe_io["out"])
            st["export"].zero_()
            launch_two_launch(st, probe_io, a, ring_export=True)
            torch.cuda.synchronize()
            ring_a_ref = [probe_io[r].clone() for r in
                          ("ring_k", "ring_v", "ring_a", "ring_b")]
            for r in ("ring_k", "ring_v", "ring_a", "ring_b"):
                probe_io[r].zero_()
            launch_single(st, probe_io, b, ring_export=True)
            torch.cuda.synchronize()
            rings_equal = all(
                byte_equal(ref, probe_io[r])
                for ref, r in zip(
                    ring_a_ref, ("ring_k", "ring_v", "ring_a", "ring_b"))
            )
            eq = byte_equal(a, b)
            sweep.append({
                "seed": 20260811 + 1000 * seed,
                "regime": rname,
                "out_byte_equal": bool(eq),
                "rings_byte_equal": bool(rings_equal),
                "max_abs": float(
                    (a.float() - b.float()).abs().max().item()),
                "out_all_finite": bool(torch.isfinite(a.float()).all().item()),
            })
    sweep_pass = all(
        s["out_byte_equal"] and s["rings_byte_equal"] for s in sweep)

    # ---- cost decomposition on the DEPLOYED two-launch route ---------------
    lens0 = st["levels"][0][4]
    lens1 = st["levels"][1][4]
    ones1 = torch.ones_like(lens1)
    full1 = torch.full_like(lens1, int(st["levels"][1][2]))
    scratch = torch.zeros_like(io["out"])

    bench = {}
    bench["L0_deployed"] = timeit(
        lambda: launch_level(st, io, scratch, 0), reps=args.reps)
    bench["L0_no_export"] = timeit(
        lambda: launch_level(st, io, scratch, 0, export_mode=2),
        reps=args.reps)
    bench["L0_len1"] = timeit(
        lambda: launch_level(
            st, io, scratch, 0, lengths=torch.ones_like(lens0)),
        reps=args.reps)
    bench["L0_len1_no_export"] = timeit(
        lambda: launch_level(
            st, io, scratch, 0, lengths=torch.ones_like(lens0),
            export_mode=2),
        reps=args.reps)
    bench["L1_deployed"] = timeit(
        lambda: launch_level(st, io, scratch, 1), reps=args.reps)
    bench["L1_len1"] = timeit(
        lambda: launch_level(st, io, scratch, 1, lengths=ones1),
        reps=args.reps)
    bench["L1_len_full_padded"] = timeit(
        lambda: launch_level(st, io, scratch, 1, lengths=full1),
        reps=args.reps)
    bench["L1_from_h0"] = timeit(
        lambda: launch_level(st, io, scratch, 1, state_source=1),
        reps=args.reps)
    bench["two_launch_total"] = timeit(
        lambda: launch_two_launch(st, io, scratch), reps=args.reps)
    bench["single_launch"] = timeit(
        lambda: launch_single(st, io, scratch), reps=args.reps)

    # ---- derived model ----------------------------------------------------
    L0 = bench["L0_deployed"]["us_p50"]
    L1 = bench["L1_deployed"]["us_p50"]
    two = bench["two_launch_total"]["us_p50"]
    one = bench["single_launch"]["us_p50"]

    # marginal cost of one serial node-step, from the level-1 length sweep:
    # deployed lengths are (7,5,7,1x8); all-ones removes 4 critical node-steps
    # from path A/C and 4 from B.
    node_step_us = (
        (bench["L1_deployed"]["us_p50"] - bench["L1_len1"]["us_p50"]) / 6.0
    )
    export_write_us = (
        bench["L0_deployed"]["us_p50"] - bench["L0_no_export"]["us_p50"]
    )
    parent_read_us = bench["L1_len1"]["us_p50"]

    layers = args.layers
    result = {
        "schema": "fr13.gdn_scan_fusion_cost_probe.v1",
        "analysis_only": True,
        "acceptance_valid": False,
        "performance_measurement": "kernel_microbenchmark_synthetic",
        "step_envelope_claim": False,
        "stamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "device": {
            "name": props.name,
            "sms": props.multi_processor_count,
            "l2_bytes": getattr(props, "L2_cache_size", None),
        },
        "geometry": {
            "n_actual": N_ACTUAL, "num_vh": NUM_VH, "num_kh": NUM_KH,
            "dim_k": DIM_K, "dim_v": DIM_V, "block_v": BLOCK_V,
            "num_warps": NUM_WARPS,
            "grid_l0": [NUM_VH, DIM_V // BLOCK_V, 1],
            "grid_l1": [NUM_VH, DIM_V // BLOCK_V, 11],
            "state_tile_bytes": STATE_TILE_BYTES,
        },
        "topology": {
            "levels": [
                [[list(p), int(par)] for p, par in lvl]
                for lvl in G._FR13_FIXED32_SUBTREE_LEVELS
            ],
            "tree_critical_path_nodes": 12,
            "two_launch_waves": 12,
            "single_launch_critical_node_steps": 32,
        },
        "byte_ab_single_vs_two": {
            "out_byte_equal": bool(out_equal),
            "out_max_abs_diff": max_abs,
            "out_elements_differing": n_diff,
            "note": (
                "synthetic-tensor kernel A/B; not the campaign byte gate"
            ),
        },
        "byte_ab_sweep": {
            "all_pass": bool(sweep_pass),
            "cases": sweep,
            "surfaces": ["out", "ring_k", "ring_v", "ring_a", "ring_b"],
        },
        "bench_us": bench,
        "derived": {
            "l0_us": L0,
            "l1_us": L1,
            "two_launch_us": two,
            "single_launch_us": one,
            "per_node_step_us": node_step_us,
            "l0_export_write_us": export_write_us,
            "l1_parent_read_plus_one_step_us": parent_read_us,
            "handoff_us_per_layer": export_write_us + parent_read_us,
            "handoff_fraction_of_two_launch": (
                (export_write_us + parent_read_us) / two if two else None
            ),
            "two_launch_ms_step_at_%d_layers" % layers: two * layers / 1000.0,
            "single_launch_ms_step_at_%d_layers" % layers: (
                one * layers / 1000.0
            ),
            "single_minus_two_ms_step": (one - two) * layers / 1000.0,
        },
        "export_untouched_by_single_launch": bool(
            byte_equal(export_after_two, st["export"])
        ),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["derived"], indent=2))
    print("byte_ab:", json.dumps(result["byte_ab_single_vs_two"]))
    print("byte_ab_sweep_all_pass:", sweep_pass, "cases:", len(sweep))
    for s in sweep:
        if not (s["out_byte_equal"] and s["rings_byte_equal"]):
            print("  MISMATCH:", json.dumps(s))
    return 0 if sweep_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
