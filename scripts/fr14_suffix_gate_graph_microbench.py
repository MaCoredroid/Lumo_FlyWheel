#!/usr/bin/env python3
"""FR14 lever 2 -- CUDA-graph mechanics microbench for the gated drafter.

WHAT THIS PRICES, AND WHAT IT DOES NOT
--------------------------------------
The gate needs the drafter to be able to run 2 post-root MTP passes on some
steps and 4 on others.  Today all 4 live in ONE captured graph
(`drafter_runtime.graph_replays = 1`, `graph_captures = 0`, on 20 579/20 579
consecutive decode steps).  Two shapes are possible:

  TWIN   two independent graphs per batch size: a 4-pass and a 2-pass.
  SPLIT  one 2-pass graph (`lo`) + one 2-pass graph (`hi`) sharing static
         buffers; ungated = replay(lo)+replay(hi), gated = replay(lo).

SPLIT is the one worth having: the ungated path still executes exactly 4 MTP
forwards, so every per-step invariant that counts 4 forwards is untouched, and
only one extra graph launch is added.  TWIN doubles the captured kernel memory
and needs a second manifest for the same batch.

This microbench measures the things that are actually uncertain:
  1. does a split capture REPRODUCE the single-graph result bit-exactly?
  2. what does the second capture cost in wall time?
  3. what does it cost in device memory (pool growth)?
  4. what does the extra graph launch cost at replay?
  5. what does replaying 2 passes instead of 4 save?

It does NOT re-measure the MTP pass cost.  That is already kernel-confirmed at
10.3 ms/pass by nsys attribution on the banked sqlite (seam_move_economics.md
§5).  The synthetic pass here is byte-sized to the pinned floor ledger
(`fr13_hardware_floor_ledger.py`: MTP block 849 398 784 B/pass + draft head
715 161 608 B/pass) so the graph-mechanics numbers land at a realistic scale --
it is a stand-in for the block, not a model of it.

Run inside the CUDA container; see fr14_suffix_gate_graph_microbench.sh.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

MTP_BLOCK_BYTES_PER_PASS = 849_398_784
DRAFT_HEAD_BYTES_PER_PASS = 715_161_608
TARGET_BYTES_PER_PASS = MTP_BLOCK_BYTES_PER_PASS + DRAFT_HEAD_BYTES_PER_PASS
HIDDEN = 4096
BANDWIDTH_BYTES_PER_S = 273_000_000_000


def build_pass_weights(device, dtype=torch.bfloat16):
    """A byte-sized stand-in for one MTP pass: a chain of square GEMMs."""
    per_layer = HIDDEN * HIDDEN * torch.finfo(dtype).bits // 8
    n_layers = max(1, round(TARGET_BYTES_PER_PASS / per_layer))
    weights = [
        torch.randn(HIDDEN, HIDDEN, device=device, dtype=dtype) / (HIDDEN ** 0.5)
        for _ in range(n_layers)
    ]
    return weights, n_layers * per_layer


def run_pass(x, weights, out_slot):
    for w in weights:
        x = torch.nn.functional.silu(x @ w)
    out_slot.copy_(x[0])
    return x


class Runner:
    """Holds static buffers shared by every graph variant."""

    def __init__(self, device, n_passes=4):
        self.device = device
        self.n_passes = n_passes
        self.weights, self.bytes_per_pass = build_pass_weights(device)
        self.x = torch.zeros(1, HIDDEN, device=device, dtype=torch.bfloat16)
        self.seed = torch.randn(1, HIDDEN, device=device, dtype=torch.bfloat16)
        self.out = torch.zeros(
            n_passes, HIDDEN, device=device, dtype=torch.bfloat16
        )

    def reset(self):
        self.x.copy_(self.seed)
        self.out.zero_()

    def passes(self, lo, hi):
        """Execute passes [lo, hi) against the static buffers."""
        cur = self.x
        for i in range(lo, hi):
            cur = run_pass(cur, self.weights, self.out[i])
        self.x.copy_(cur)


def _capture(runner, lo, hi, pool):
    """Capture passes [lo,hi) into a graph on a non-default stream."""
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    prev = torch.cuda.current_stream()
    torch.cuda.set_stream(stream)
    t0 = time.perf_counter()
    graph = torch.cuda.CUDAGraph()
    graph.capture_begin(pool=pool)
    runner.passes(lo, hi)
    graph.capture_end()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    torch.cuda.set_stream(prev)
    return graph, dt


def _time_replays(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--out", default="/logs/fr14_gate_graph_microbench.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    device = torch.device("cuda")
    torch.cuda.init()

    result = {
        "schema": "fr14.gate_graph_microbench.v1",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hidden": HIDDEN,
        "target_bytes_per_pass": TARGET_BYTES_PER_PASS,
    }

    runner = Runner(device)
    result["actual_bytes_per_pass"] = runner.bytes_per_pass
    result["pass_byte_floor_ms"] = (
        runner.bytes_per_pass / BANDWIDTH_BYTES_PER_S * 1000.0
    )

    base_mem = torch.cuda.memory_reserved()

    # ---- eager reference (also the correctness oracle) --------------------
    runner.reset()
    torch.cuda.synchronize()
    runner.passes(0, 4)
    torch.cuda.synchronize()
    ref = runner.out.clone()
    result["eager_4pass_ms"] = _time_replays(
        lambda: (runner.reset(), runner.passes(0, 4)), max(4, args.iters // 4)
    )

    # ---- TODAY: one graph, 4 passes --------------------------------------
    pool = torch.cuda.graph_pool_handle()
    runner.reset()
    g4, cap_ms_4 = _capture(runner, 0, 4, pool)
    mem_after_g4 = torch.cuda.memory_reserved()

    def replay4():
        runner.x.copy_(runner.seed)
        g4.replay()

    single_out = None
    replay4()
    torch.cuda.synchronize()
    single_out = runner.out.clone()
    result["single_graph"] = {
        "capture_ms": cap_ms_4 * 1000.0,
        "pool_growth_bytes": mem_after_g4 - base_mem,
        "replay_ms": _time_replays(replay4, args.iters),
        "matches_eager": bool(torch.equal(single_out, ref)),
    }

    # ---- SPLIT: two 2-pass graphs sharing the pool and the static buffers -
    runner.reset()
    g_lo, cap_ms_lo = _capture(runner, 0, 2, pool)
    mem_after_lo = torch.cuda.memory_reserved()
    g_hi, cap_ms_hi = _capture(runner, 2, 4, pool)
    mem_after_hi = torch.cuda.memory_reserved()

    def replay_ungated():
        runner.x.copy_(runner.seed)
        g_lo.replay()
        g_hi.replay()

    def replay_gated():
        runner.x.copy_(runner.seed)
        g_lo.replay()

    replay_ungated()
    torch.cuda.synchronize()
    split_out = runner.out.clone()

    runner.out.zero_()
    replay_gated()
    torch.cuda.synchronize()
    gated_out = runner.out.clone()

    result["split_graph"] = {
        "capture_ms_lo": cap_ms_lo * 1000.0,
        "capture_ms_hi": cap_ms_hi * 1000.0,
        "capture_ms_total": (cap_ms_lo + cap_ms_hi) * 1000.0,
        "extra_capture_ms_vs_single": (
            (cap_ms_lo + cap_ms_hi) - cap_ms_4
        ) * 1000.0,
        "pool_growth_bytes_lo": mem_after_lo - mem_after_g4,
        "pool_growth_bytes_hi": mem_after_hi - mem_after_lo,
        "pool_growth_bytes_total": mem_after_hi - mem_after_g4,
        "replay_ms_ungated_lo_plus_hi": _time_replays(replay_ungated, args.iters),
        "replay_ms_gated_lo_only": _time_replays(replay_gated, args.iters),
        # THE correctness claim: splitting the capture must not change a bit.
        "ungated_matches_single_graph": bool(torch.equal(split_out, single_out)),
        "gated_fills_only_first_two_passes": bool(
            torch.equal(gated_out[:2], single_out[:2])
            and not gated_out[2:].any()
        ),
    }

    sg = result["single_graph"]
    sp = result["split_graph"]
    sp["launch_overhead_ms"] = (
        sp["replay_ms_ungated_lo_plus_hi"] - sg["replay_ms"]
    )
    sp["saving_ms_on_gated_steps"] = (
        sp["replay_ms_ungated_lo_plus_hi"] - sp["replay_ms_gated_lo_only"]
    )
    sp["measured_per_pass_ms"] = sp["saving_ms_on_gated_steps"] / 2.0

    print(json.dumps(result, indent=1))
    try:
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)
        print(f"wrote {args.out}")
    except OSError as exc:
        print(f"could not write {args.out}: {exc}")


if __name__ == "__main__":
    main()
