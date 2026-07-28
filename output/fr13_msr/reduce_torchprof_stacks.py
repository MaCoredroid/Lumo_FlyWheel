#!/usr/bin/env python3
"""Reduce a torch-profiler trace (with_stack) to NAMED residual sites.

DIAGNOSTIC ONLY (observer-effect standing rule): in-window numbers carry
profiler overhead — use for site naming + relative ranking, never as clean
speed, never merged into the fr13_measure ledger.

Input: a chrome-trace json (.pt.trace.json or .json.gz) written by vLLM's
TorchProfilerWrapper (tensorboard_trace_handler output dir).
Output: <trace>.sites.json with
  - top cpu_op self-time (aten/host glue ranking),
  - top python_function frames filtered to OUR seams (gdn_linear_attn,
    rejection_sampler, gpu_model_runner, fr10_gdn_tree_kernel,
    fr13_tree_conv_fused) by total time — the host-gap/index-soup/norms/
    sampler residual sites the next build round attacks,
  - cuda kernel top list (cross-check vs the nsys reducer buckets).
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

OUR_SEAMS = (
    "gdn_linear_attn", "rejection_sampler", "gpu_model_runner",
    "fr10_gdn_tree_kernel", "fr13_tree_conv_fused", "gdn_attn",
    "mamba", "eagle",
)


def load(path: Path):
    raw = gzip.open(path, "rt") if path.suffix == ".gz" else open(path)
    with raw as fh:
        return json.load(fh)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    data = load(path)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    cpu_self = defaultdict(float)
    py_frames = defaultdict(float)
    kernels = defaultdict(float)
    n = {"cpu_op": 0, "python_function": 0, "kernel": 0}
    for ev in events:
        if not isinstance(ev, dict) or ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        dur = float(ev.get("dur", 0.0))  # us
        name = ev.get("name", "")
        if cat == "cpu_op":
            cpu_self[name] += dur
            n["cpu_op"] += 1
        elif cat == "python_function":
            if any(s in name for s in OUR_SEAMS):
                py_frames[name] += dur
                n["python_function"] += 1
        elif cat in ("kernel", "gpu_op", "cuda_runtime"):
            if cat == "kernel":
                kernels[name] += dur
                n["kernel"] += 1

    def top(d, k=40):
        return [
            {"name": nm[:200], "total_ms": round(v / 1000.0, 3)}
            for nm, v in sorted(d.items(), key=lambda kv: -kv[1])[:k]
        ]

    out = {
        "label": "DIAGNOSTIC-ONLY torchprof window (profiler overhead included)",
        "trace": str(path),
        "event_counts": n,
        "top_cpu_ops_self_ms": top(cpu_self),
        "top_our_python_frames_ms": top(py_frames, 60),
        "top_cuda_kernels_ms": top(kernels),
    }
    outp = path.with_suffix(path.suffix + ".sites.json")
    outp.write_text(json.dumps(out, indent=1))
    print(f"wrote {outp}")
    for row in out["top_our_python_frames_ms"][:15]:
        print(f"  {row['total_ms']:>10.2f}ms  {row['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
