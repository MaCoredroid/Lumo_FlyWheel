#!/usr/bin/env python3
"""FR13 host-residual rung: offline host-cost probe.

Analysis-only. This probe measures the *host* (CPU) cost of the three
mechanisms that the banked post-Qrow capture shows are the whole of the
step-boundary GPU-idle bill on decode-cadence steps:

  1. ``cudaGraphLaunch`` host cost as a function of graph node count.
     The capture measures 1.945 ms/step of GPU idle spent *inside*
     ``cudaGraphLaunch`` for the 1894-node SFWD graph. If that cost is
     proportional to node count, then every graph-node reduction anywhere in
     SFWD carries a previously unpriced host-side saving, and the number
     below is its price list.
  2. Eager ATen dispatch cost for the op mix the post-DFWD tail actually
     runs (small index/copy/fill/arange/scatter/compare kernels). The tail
     issues ~154 CUDA ops that produce 0.209 ms of GPU work but cost
     3.458 ms of host critical path; this converts "remove K ops" into ms.
  3. Small H2D copy cost, pageable vs pinned. The tail issues 37 H2D
     copies/step of 1-768 B. A pageable small copy is a synchronous staged
     copy on the runtime's own buffer; a pinned one is not.

It needs no campaign credential, no served model, no offload host: synthetic
tensors, one container, CPU-launched.

``performance_measurement`` here is a HOST microbenchmark on synthetic work.
It is NOT a step-envelope measurement and NOT acceptance-valid. It carries no
TPS, floor, or acceptance claim. Nothing it measures changes any served byte:
every quantity is a launch/dispatch cost, not an arithmetic result.

Usage:
    python3 scripts/fr13_host_tail_cost_probe.py \
        --out results/fr13_host_residual_20260811/host_cost_probe.json \
        --reps 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# Node counts to sweep. 1894 is the measured SFWD graph node count
# (results/fr13_attack_ladder_analysis_20260808 SS0); the rest bracket it so
# the relationship can be fitted rather than asserted.
NODE_COUNTS = (16, 64, 256, 512, 1024, 1894, 3072)

# The post-DFWD tail's measured eager op mix, from the banked capture
# (see results/fr13_host_residual_20260811/design.md SS3). Counts are per step.
TAIL_OP_MIX = (
    ("index_select", 12),
    ("copy_", 10),
    ("fill_", 6),
    ("add_out", 6),
    ("arange", 6),
    ("add_", 4),
    ("scatter_", 4),
    ("ge_scalar", 3),
    ("clamp_", 3),
    ("nonzero_free_where", 3),
)


def _dev() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "fr13_host_tail_cost_probe requires a CUDA device: it measures "
            "host launch cost against a real driver. There is no CPU stand-in."
        )
    return torch.device("cuda")


def _host_us(fn, warmup: int, reps: int) -> dict:
    """Host-side wall cost of ``fn``, with the GPU quiesced before each rep.

    The GPU is synchronised BEFORE the timed region and never inside it, so
    what is measured is submission cost on an empty queue -- which is exactly
    the regime the step boundary is in (the capture measures zero
    ``Command buffer full`` stalls in the tail and in the sfwd enqueue).
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e3)
    torch.cuda.synchronize()
    samples.sort()
    return {
        "us_min": samples[0],
        "us_p05": samples[max(0, int(len(samples) * 0.05))],
        "us_p50": samples[len(samples) // 2],
        "us_mean": statistics.fmean(samples),
        "us_p95": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
        "reps": len(samples),
    }


def _gpu_ms(fn, warmup: int, reps: int) -> float:
    """GPU execution time of ``fn`` in ms, CUDA-event timed."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


def build_graph(device: torch.device, nodes: int):
    """A CUDA graph with exactly ``nodes`` trivial kernel nodes.

    Each node is an in-place add on a 256-element tensor: the smallest
    launch-dominated kernel available, so the replay's GPU time stays far
    below its host submission cost and the two do not confound.
    """
    x = torch.ones(256, device=device, dtype=torch.float32)
    # Warm the allocator and the kernel on the capture stream's ancestor.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            x.add_(1.0)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            x.add_(1.0)
    return graph, x


def probe_graph_launch(device: torch.device, reps: int) -> dict:
    """Host cost of graph replay vs node count, plus the eviction question."""
    rows = []
    graphs = {}
    for nodes in NODE_COUNTS:
        graph, _x = build_graph(device, nodes)
        graphs[nodes] = graph
        host = _host_us(graph.replay, warmup=20, reps=reps)
        gpu = _gpu_ms(graph.replay, warmup=5, reps=max(20, reps // 4))
        rows.append(
            {
                "nodes": nodes,
                "host_launch": host,
                "gpu_ms_per_replay": gpu,
                "host_ns_per_node": host["us_p50"] * 1e3 / nodes,
            }
        )
    # Interleave question: does replaying a DIFFERENT graph in between change
    # the host cost of the big one? If it does, the driver is re-uploading and
    # a hoisted upload is a real lever; if it does not, the cost is inherent
    # per-node submission and only node-count reduction touches it.
    big = graphs[max(NODE_COUNTS)]
    small = graphs[min(NODE_COUNTS)]

    def _interleaved():
        small.replay()
        big.replay()

    solo = _host_us(big.replay, warmup=20, reps=reps)
    inter = _host_us(_interleaved, warmup=20, reps=reps)
    small_solo = _host_us(small.replay, warmup=20, reps=reps)
    return {
        "by_node_count": rows,
        "eviction_probe": {
            "big_nodes": max(NODE_COUNTS),
            "small_nodes": min(NODE_COUNTS),
            "big_solo": solo,
            "small_solo": small_solo,
            "small_then_big": inter,
            "big_cost_delta_us_p50": (
                inter["us_p50"] - small_solo["us_p50"] - solo["us_p50"]
            ),
        },
    }


def _tail_ops(device: torch.device):
    """Callables mirroring the tail's measured eager op mix.

    Shapes are the tail's: block tables, slot mappings and request-indexed
    bookkeeping over a handful of rows. Nothing here is a served computation;
    it exists only to be dispatched.
    """
    n = 64
    src = torch.arange(n, device=device, dtype=torch.int32)
    dst = torch.zeros(n, device=device, dtype=torch.int32)
    idx = torch.arange(n, device=device, dtype=torch.int64)
    big = torch.zeros(1024, device=device, dtype=torch.int32)
    return {
        "index_select": lambda: torch.index_select(src, 0, idx),
        "copy_": lambda: dst.copy_(src),
        "fill_": lambda: dst.fill_(0),
        "add_out": lambda: torch.add(src, 1, out=dst),
        "arange": lambda: torch.arange(n, device=device, dtype=torch.int32),
        "add_": lambda: dst.add_(1),
        "scatter_": lambda: big.scatter_(0, idx, src),
        "ge_scalar": lambda: src.ge(0),
        "clamp_": lambda: dst.clamp_(0, 7),
        "nonzero_free_where": lambda: torch.where(src.ge(0), src, dst),
    }


def probe_eager_dispatch(device: torch.device, reps: int) -> dict:
    ops = _tail_ops(device)
    per_op = {}
    for name, fn in ops.items():
        per_op[name] = _host_us(fn, warmup=50, reps=reps)

    counts = dict(TAIL_OP_MIX)

    def _mix():
        for name, k in TAIL_OP_MIX:
            fn = ops[name]
            for _ in range(k):
                fn()

    total_ops = sum(counts.values())
    mix = _host_us(_mix, warmup=20, reps=max(20, reps // 4))
    return {
        "per_op": per_op,
        "mix_ops": total_ops,
        "mix_host": mix,
        "mix_us_per_op": mix["us_p50"] / total_ops,
    }


def probe_small_h2d(device: torch.device, reps: int) -> dict:
    out = {}
    for nbytes in (4, 8, 256, 636, 768):
        n = max(1, nbytes // 4)
        dev = torch.zeros(n, device=device, dtype=torch.int32)
        pageable = torch.zeros(n, dtype=torch.int32)
        pinned = torch.zeros(n, dtype=torch.int32).pin_memory()
        out[str(nbytes)] = {
            "pageable_blocking": _host_us(
                lambda: dev.copy_(pageable), warmup=50, reps=reps
            ),
            "pinned_non_blocking": _host_us(
                lambda: dev.copy_(pinned, non_blocking=True),
                warmup=50,
                reps=reps,
            ),
        }
    return out


def probe_python_bookkeeping(reps: int) -> dict:
    """Cost of the three uncached per-step derivations the tail actually runs.

    All three are pure functions of an immutable input, recomputed every step:
      * ``ast.literal_eval`` + ``sorted`` + two ``np.array`` builds over the
        speculative token tree (``# FR10_TREE_DEPTH_POSITIONS``);
      * ``json.loads`` of every stored FULL-graph capture manifest (the
        undeferred census open ``_fr13_fixed32_observed_begin``);
      * a stringified set over the Arctic suffix cache's cached-request ids.

    Sizes are swept rather than assumed: the deployed tree path count and the
    deployed cache occupancy are configuration, and this probe does not get to
    see them. What it produces is a price per unit, not a per-step total.
    """
    import ast
    import json

    import numpy as np

    def _tree_src(paths: int) -> str:
        out = []
        depth = 1
        while len(out) < paths:
            out.append(tuple([0] * depth))
            if len(out) < paths:
                out.append(tuple([0] * (depth - 1) + [1]))
            depth += 1
        return repr(out[:paths])

    def _tree_work(src):
        def run():
            choices = sorted(ast.literal_eval(src), key=lambda p: (len(p), p))
            np.array([0] + [len(c) for c in choices], dtype=np.int64)
            spine = [c for c in choices if all(int(p) == 0 for p in c)]
            leaf = [c for c in choices if not all(int(p) == 0 for p in c)]
            np.array(
                [0] + [len(c) for c in spine] + [len(c) for c in leaf],
                dtype=np.int64,
            )

        return run

    def _cpu_us(fn):
        for _ in range(20):
            fn()
        samples = []
        for _ in range(reps):
            t0 = time.perf_counter_ns()
            fn()
            samples.append((time.perf_counter_ns() - t0) / 1e3)
        samples.sort()
        return {
            "us_p50": samples[len(samples) // 2],
            "us_mean": statistics.fmean(samples),
            "reps": len(samples),
        }

    tree = {
        str(p): _cpu_us(_tree_work(_tree_src(p))) for p in (9, 15, 31, 63)
    }

    manifests = {}
    for kb in (2, 5, 10):
        blob = json.dumps(
            {
                "mode": "tail6_fixed32",
                "descriptor": {"runtime_mode": "FULL"},
                "physical_rows_per_request": 32,
                "batch_size": 1,
                "gdn_layers": [f"model.layers.{i}.mixer" for i in range(48)],
                "pad": "x" * max(0, kb * 1024 - 2048),
            }
        )
        for count in (1, 4):
            blobs = [blob] * count
            manifests[f"{kb}KB_x{count}"] = _cpu_us(
                lambda b=blobs: [json.loads(x) for x in b]
            )

    cached = {}
    for n in (10, 100, 1000, 10000):
        ids = [f"req-{i:08d}" for i in range(n)]
        cached[str(n)] = _cpu_us(lambda i=ids: {str(x) for x in i})

    return {
        "tree_depth_offsets_by_path_count": tree,
        "capture_manifest_json_loads": manifests,
        "arctic_cached_id_set_by_size": cached,
    }


def _fit_ns_per_node(rows) -> dict:
    """OLS of host p50 launch us on node count. Reported with its residual."""
    xs = [float(r["nodes"]) for r in rows]
    ys = [float(r["host_launch"]["us_p50"]) for r in rows]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return {
        "ns_per_node": slope * 1e3,
        "fixed_us": intercept,
        "max_abs_residual_us": max(abs(r) for r in resid),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    device = _dev()
    props = torch.cuda.get_device_properties(0)

    graph = probe_graph_launch(device, args.reps)
    eager = probe_eager_dispatch(device, args.reps)
    h2d = probe_small_h2d(device, args.reps)
    bookkeeping = probe_python_bookkeeping(args.reps)
    fit = _fit_ns_per_node(graph["by_node_count"])

    # Derived pricing. Both are *host* costs; neither is a step-envelope claim.
    sfwd_nodes = 1894
    tail_ops_per_step = 154
    derived = {
        "sfwd_graph_nodes": sfwd_nodes,
        "modelled_sfwd_graph_launch_ms": fit["ns_per_node"] * sfwd_nodes / 1e6
        + fit["fixed_us"] / 1e3,
        "measured_sfwd_graph_launch_idle_ms_from_capture": 1.945,
        "host_us_per_graph_node": fit["ns_per_node"] / 1e3,
        "host_us_per_eager_op": eager["mix_us_per_op"],
        "tail_eager_ops_per_step_from_capture": tail_ops_per_step,
        "modelled_tail_dispatch_ms": eager["mix_us_per_op"]
        * tail_ops_per_step
        / 1e3,
        "measured_tail_idle_ms_from_capture": 3.458,
    }

    payload = {
        "schema": "fr13.host_tail_cost_probe.v1",
        "analysis_only": True,
        "acceptance_valid": False,
        "performance_measurement": "host_microbenchmark_synthetic",
        "step_envelope_claim": False,
        "byte_claim": False,
        "stamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "device": {
            "name": props.name,
            "multi_processor_count": props.multi_processor_count,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "graph_launch": graph,
        "graph_launch_fit": fit,
        "eager_dispatch": eager,
        "small_h2d": h2d,
        "python_bookkeeping": bookkeeping,
        "derived": derived,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(json.dumps(derived, indent=2, sort_keys=True))
    print("\nhost graph-replay cost by node count (p50 us):")
    for r in graph["by_node_count"]:
        print(
            f"  nodes {r['nodes']:5d}  host {r['host_launch']['us_p50']:9.2f} us"
            f"  ({r['host_ns_per_node']:6.1f} ns/node)"
            f"  gpu {r['gpu_ms_per_replay'] * 1e3:9.2f} us"
        )
    ev = graph["eviction_probe"]
    print(
        "\neviction probe: big solo p50 %.2f us, small solo %.2f us, "
        "small+big %.2f us, delta on big %.2f us"
        % (
            ev["big_solo"]["us_p50"],
            ev["small_solo"]["us_p50"],
            ev["small_then_big"]["us_p50"],
            ev["big_cost_delta_us_p50"],
        )
    )
    print("\nsmall H2D copy host cost (p50 us): pageable vs pinned")
    for k, v in h2d.items():
        print(
            f"  {k:>4s} B  pageable {v['pageable_blocking']['us_p50']:7.2f}"
            f"   pinned {v['pinned_non_blocking']['us_p50']:7.2f}"
        )
    print("\nper-step python bookkeeping (p50 us), uncached today:")
    bk = bookkeeping
    for k, v in bk["tree_depth_offsets_by_path_count"].items():
        print(f"  tree depth-offsets, {k:>3s} paths      {v['us_p50']:8.2f}")
    for k, v in bk["capture_manifest_json_loads"].items():
        print(f"  capture manifest json.loads {k:<10s} {v['us_p50']:8.2f}")
    for k, v in bk["arctic_cached_id_set_by_size"].items():
        print(f"  arctic cached-id set, n={k:<6s}      {v['us_p50']:8.2f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
