#!/usr/bin/env python3
"""Shared helpers for the LumoTree GDN kernel timing + memory benchmark (scripts/v2exp/kernel).

Nothing here launches work at import time.  All geometry is asserted (fail loudly) against
  * the served model config (Qwen3.8-27B NVFP4: 48 GDN layers, Hk=16, Hv=48, d_k=d_v=128),
  * the fixed32 topology authority scripts/fr13_fixed32_topology.py (32 physical rows, Hydra27, depth 11),
  * the frozen M1/Q1 constants (m1_adapters: H, HV, K, V, N).
Container layout (run_kernel_bench.sh): /workspace = v2exp worktree (ro), /review = M1 review repo (ro),
/out = run directory (rw), /model_config.json = served model config (ro).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import platform
import socket
import sys

# ----------------------------------------------------------------------------------------------- geometry (paper/deployment)
GDN_LAYERS = 48
ATTN_LAYERS = 16
H_K = 16            # GDN key heads
H_V = 48            # GDN value heads
D_K = 128
D_V = 128
N_PHYS = 32         # 27 Hydra27 drafts + root, padded to 32 physical rows
N_ACTIVE_DRAFTS = 27
INACTIVE_PHYSICAL = (18, 23, 25, 27)
MAX_DEPTH = 11
LUMO_BV = 8         # FR13_TREE_GDN_GEOM_OVERRIDE=BV=8 (deployed)
SCALE = D_K ** -0.5
STATE_BYTES = H_V * D_V * D_K * 4          # one fp32 [HV,V,K] state = 3,145,728 B

# ----------------------------------------------------------------------------------------------- pinned identities (M1 freeze)
EXPECTED_IMAGE_ID = "sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc"
EXPECTED_IMAGE_REPO_DIGEST = "vllm/vllm-openai@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776"
EXPECTED_KERNEL_SHA = "d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8"     # src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
EXPECTED_TOPOLOGY_SHA = "c02703b4bc777e5edfa01623b446a289cba8f1023e31d8efd7fc8126666b38bc"   # scripts/fr13_fixed32_topology.py
EXPECTED_NATIVE_SHA = "000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f"     # vllm fla ops fused_sigmoid_gating.py
EXPECTED_RUNNER_SHA = "ab27fc9cf680bdd2b7c3ad994fc9f7bffe0f9ff1b0f35dc2cc36018f364dea29"     # tools/q1_component_runner_v2_2.py
EXPECTED_TORCH, EXPECTED_TRITON = "2.11.0+cu130", "3.6.0"

# Deployed fixed32 GDN route (identical to M1/Q1.2b REQUIRED_ENV and to the v2exp tree arm container_env.txt).
DEPLOYED_ENV = {"FR13_FIXED32_MODE": "hydra27_fixed32", "FR13_SUBTREE_PARALLEL": "1", "FR13_SCAN_ALIGN": "0", "FR13_TREE_GDN_GEOM_OVERRIDE": "BV=8",
                "FR13_RING_EXPORT": "1", "FR13_TREE_RUNROW_INIT": "1", "FR13_FLAGS_INKERNEL": "1", "FR13_FIXED32_COMMIT_DEVICE_FILL": "1",
                "FR13_FIXED32_KV_REMAP16": "1", "FR13_FIXED32_COMMITTER_LAYER_BATCH": "0", "FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION": "0",
                "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION": "0", "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE": "0", "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL": "0"}
# Generic (non-fixed32) Lumo scan route used ONLY by the tree-size sweep (fixed32 geometry cannot change size).
GENERIC_ENV_SET = {"FR13_SUBTREE_PARALLEL": "1", "FR13_SCAN_ALIGN": "0", "FR13_TREE_GDN_GEOM_OVERRIDE": "BV=8", "FR13_RING_EXPORT": "1", "FR13_FLAGS_INKERNEL": "1"}
GENERIC_ENV_UNSET = ("FR13_FIXED32_MODE", "FR13_FIXED32_COMMIT_DEVICE_FILL", "FR13_FIXED32_KV_REMAP16")

# ----------------------------------------------------------------------------------------------- paths
WORKSPACE = os.environ.get("KBENCH_WORKSPACE", "/workspace")
REVIEW_ROOT = os.environ.get("KBENCH_REVIEW_ROOT", "/review")
CAMPAIGN = os.path.join(REVIEW_ROOT, "papers", "gdn-tree-scan-mlsys", "v2", "experiments", "review-response-20260927")
M1_TOOLS = os.path.join(CAMPAIGN, "tools", "m1")
Q1_TOOLS = os.path.join(CAMPAIGN, "tools")
MODEL_CONFIG = os.environ.get("KBENCH_MODEL_CONFIG", "/model_config.json")

NUMERICS_PATHS = ("root-only", "n14", "n15", "n31")        # spine depth 4, off-spine sibling depth 4, deepest spine depth 11


class GeometryError(RuntimeError):
    """Shape/geometry/identity mismatch: the benchmark refuses to produce numbers."""


def require(cond, msg):
    if not cond:
        raise GeometryError(msg)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def setup_paths():
    """Workspace src first (the Lumo kernel under test), then the frozen M1/Q1 tools (read-only reuse)."""
    for p in (M1_TOOLS, Q1_TOOLS, os.path.join(WORKSPACE, "scripts"), os.path.join(WORKSPACE, "src")):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def check_model_config(path=MODEL_CONFIG, allow_missing=False):
    """Assert the served model's GDN geometry (d_k, d_v, heads, 48 GDN + 16 attention layers)."""
    if not os.path.exists(path):
        require(allow_missing, f"model config {path} not mounted; pass --allow-missing-model-config to skip")
        return {"path": path, "checked": False}
    c = json.load(open(path))
    t = c.get("text_config", c)
    lt = list(t.get("layer_types") or [])
    got = {"linear_key_head_dim": t.get("linear_key_head_dim"), "linear_value_head_dim": t.get("linear_value_head_dim"),
           "linear_num_key_heads": t.get("linear_num_key_heads"), "linear_num_value_heads": t.get("linear_num_value_heads"),
           "gdn_layers": sum(1 for x in lt if x == "linear_attention"), "attention_layers": sum(1 for x in lt if x == "full_attention"),
           "num_hidden_layers": t.get("num_hidden_layers")}
    want = {"linear_key_head_dim": D_K, "linear_value_head_dim": D_V, "linear_num_key_heads": H_K, "linear_num_value_heads": H_V,
            "gdn_layers": GDN_LAYERS, "attention_layers": ATTN_LAYERS, "num_hidden_layers": GDN_LAYERS + ATTN_LAYERS}
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    require(not bad, f"model config geometry mismatch (got, want): {bad}")
    return {"path": path, "checked": True, "sha256": sha256_file(path), **got}


def check_topology():
    """Assert the fixed32 topology authority and M1 constants agree with the benchmark geometry."""
    import fr13_fixed32_topology as T
    import m1_adapters as A1
    import m1_topology_mapping as M
    require(len(T.PHYSICAL_PARENT) == N_PHYS, f"physical rows {len(T.PHYSICAL_PARENT)} != {N_PHYS}")
    require(int(T.HYDRA27_ACTIVE_DRAFTS) == N_ACTIVE_DRAFTS, "Hydra27 active drafts drifted")
    require(int(T.MAX_PHYSICAL_DEPTH) == MAX_DEPTH, "max physical depth drifted")
    require(tuple(M.INACTIVE_PHYSICAL) == INACTIVE_PHYSICAL, f"inactive physical rows {M.INACTIVE_PHYSICAL}")
    require((A1.H, A1.HV, A1.K, A1.V, A1.N) == (H_K, H_V, D_K, D_V, N_PHYS), f"M1 constants {(A1.H, A1.HV, A1.K, A1.V, A1.N)}")
    require(abs(A1.SCALE - SCALE) < 1e-12, "M1 scale drifted")
    require(hex(T.HYDRA27_VALID_MASK) == "0x7abdffff", "Hydra27 valid mask drifted")
    return {"physical_parent": [int(p) for p in T.PHYSICAL_PARENT], "hydra27_valid_mask": hex(T.HYDRA27_VALID_MASK),
            "subtree_levels": [[list(p), int(par)] for lvl in T.SUBTREE_LEVELS for p, par in lvl] if hasattr(T, "SUBTREE_LEVELS") else None,
            "inactive_physical": list(INACTIVE_PHYSICAL), "max_depth": MAX_DEPTH}


def accepted_paths():
    """{path_id: root-inclusive physical node list} for the 28 Hydra27 accepted paths (topology authority via M1 fixtures)."""
    import q1_2b_fixtures as FX
    out = {p["path_id"]: list(p["nodes"]) for p in FX.accepted_paths()}
    require(len(out) == 28, f"expected 28 accepted paths, got {len(out)}")
    require(max(len(v) for v in out.values()) == MAX_DEPTH + 1, "deepest accepted path must have 12 nodes")
    for n in NUMERICS_PATHS:
        require(n in out, f"numerics path {n} missing")
    return out


def build_operands(seed, layers, stratum):
    """M1 shared logical operands (m1_cycle_driver.build_shared_operands = q1_2b_fixtures.make_instance per layer)."""
    import m1_cycle_driver as D1
    ops = D1.build_shared_operands(seed=seed, L=layers, stratum=stratum)
    import m1_adapters as A1
    for l, inp in enumerate(ops):
        pr = A1.validate_raw_operands(inp)
        require(not pr, f"layer {l} raw operands: {pr}")
    return ops


def sync(torch, device):
    """torch.cuda.synchronize only for CUDA devices (keeps the naive baselines CPU-testable)."""
    if getattr(device, "type", str(device)) == "cuda" or str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def stage_operands(ops, device):
    return [{k: (v.to(device).contiguous() if hasattr(v, "to") else v) for k, v in inp.items()} for inp in ops]


# ----------------------------------------------------------------------------------------------- statistics
def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


def summarize(vals):
    v = sorted(float(x) for x in vals)
    if not v:
        return {"n": 0}
    mean = sum(v) / len(v)
    sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return {"n": len(v), "median_ms": percentile(v, 50), "p10_ms": percentile(v, 10), "p90_ms": percentile(v, 90),
            "mean_ms": mean, "std_ms": sd, "min_ms": v[0], "max_ms": v[-1]}


# ----------------------------------------------------------------------------------------------- timing
class L2Flusher:
    """Writes a scratch buffer between timed iterations (outside the timed region) so each iteration starts with a cold L2,
    as in serving where attention/MLP weights stream between GDN layers."""

    def __init__(self, torch, device, mib):
        self.buf = torch.empty(int(mib) * (1 << 20), dtype=torch.uint8, device=device) if mib and mib > 0 else None
        self.mib = int(mib or 0)

    def __call__(self):
        if self.buf is not None:
            self.buf.fill_(1)


def time_iterations(torch, step_fn, *, warmup, repeats, pre=None, flush=None, sync_each=True):
    """Run step_fn(i, mark) warmup+repeats times.  step_fn calls mark(label) at the END of each region; region k's duration is the
    CUDA-event interval between the previous mark (or the iteration start) and mark k.  pre(i) is untimed per-iteration setup.
    Returns {"regions": {label: [ms...]}, "total": [ms...], "labels": [...]}."""
    for i in range(warmup):
        if pre:
            pre(i)
        step_fn(i, lambda _label: None)
    torch.cuda.synchronize()
    recs = []
    for r in range(repeats):
        i = warmup + r
        if pre:
            pre(i)
        if flush:
            flush()
        if sync_each:
            torch.cuda.synchronize()
        marks = []

        def mark(label, _marks=marks):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            _marks.append((label, ev))
        mark("__start__")
        step_fn(i, mark)
        recs.append(marks)
    torch.cuda.synchronize()
    labels = [m[0] for m in recs[0][1:]] if recs else []
    regions = {lab: [] for lab in labels}
    total = []
    for marks in recs:
        require([m[0] for m in marks[1:]] == labels, "inconsistent region labels across iterations")
        for (_, a), (lab, b) in zip(marks[:-1], marks[1:]):
            regions[lab].append(a.elapsed_time(b))
        total.append(marks[0][1].elapsed_time(marks[-1][1]))
    return {"labels": labels, "regions": regions, "total": total}


def timing_record(raw, keep_raw=True):
    rec = {"regions": {k: summarize(v) for k, v in raw["regions"].items()}, "total": summarize(raw["total"]), "labels": raw["labels"]}
    if keep_raw:
        rec["raw_ms"] = {"regions": raw["regions"], "total": raw["total"]}
    return rec


def capture_graph(torch, fn, warmup_iters=3, pool=None):
    """Warm fn on a side stream, then capture it once into a CUDA graph.  Returns (graph, None) or (None, error-string)."""
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup_iters):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            fn()
        torch.cuda.synchronize()
        return g, None
    except Exception as e:  # noqa: BLE001  (capture failure is a recorded result, not a crash)
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        return None, f"{type(e).__name__}: {e}"


# ----------------------------------------------------------------------------------------------- memory
def mem_snapshot(torch, device):
    torch.cuda.synchronize(device)
    free_b, total_b = torch.cuda.mem_get_info(device)
    return {"allocated": int(torch.cuda.memory_allocated(device)), "reserved": int(torch.cuda.memory_reserved(device)),
            "max_allocated": int(torch.cuda.max_memory_allocated(device)), "max_reserved": int(torch.cuda.max_memory_reserved(device)),
            "device_free": int(free_b), "device_total": int(total_b)}


def tensor_bytes(tensors):
    """Unique-storage bytes of an iterable of tensors (views of one storage counted once)."""
    seen, total = set(), 0
    for t in tensors:
        if t is None or not hasattr(t, "untyped_storage"):
            continue
        st = t.untyped_storage()
        key = st.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        total += int(st.nbytes())
    return total


def m_cut_formula(n_c, b_v=LUMO_BV, d_k=D_K, h_v=H_V, d_v=D_V):
    """Paper's logical cut-state formula: M_cut = 4 * N_c * B_v * d_k bytes per (head, value tile); also scaled to one layer."""
    per_tile = 4 * n_c * b_v * d_k
    tiles = h_v * (d_v // b_v)
    return {"N_c": int(n_c), "B_v": int(b_v), "d_k": int(d_k), "bytes_per_head_value_tile": per_tile, "head_value_tiles_per_layer": tiles,
            "bytes_per_layer": per_tile * tiles, "bytes_all_layers_if_resident": per_tile * tiles * GDN_LAYERS,
            "formula": "M_cut = 4 * N_c * B_v * d_k (bytes, fp32) per head/value-tile"}


# ----------------------------------------------------------------------------------------------- numerics (M1 comparator, float64 C2 reference)
ERR_NOTE = ("M1 comparator: q1_2b_fixtures.c2_node_refs (float64 C2, one cast of the recorded bf16/fp32 operands) + q1_oracle.per_head_errors; "
            "reported next to the time; NOT a pass/fail gate")


def _agg_cells(cells):
    """cells: list of per_head_errors dicts -> scalar summary (max over heads/cells; median of per-cell max RMS)."""
    mx, rms, rel, nf, invalid = [], [], [], 0, 0
    for e in cells:
        nf += int(e.get("nonfinite_count", 0))
        if not e.get("metrics_valid"):
            invalid += 1
            continue
        m = max(v for v in e["max_abs"] if v is not None)
        r = max(v for v in e["rms"] if v is not None)
        refm = max((v for v in e["ref_max_abs"] if v is not None), default=0.0)
        mx.append(m)
        rms.append(r)
        rel.append(m / refm if refm > 0 else float("inf"))
    s = sorted(rms)
    return {"cells": len(cells), "invalid_cells": invalid, "nonfinite_values": nf, "max_abs_max": max(mx) if mx else None,
            "rms_max": max(rms) if rms else None, "rms_median": percentile(s, 50) if s else None, "rel_max_abs_max": max(rel) if rel else None}


def numerics_report(torch, operands, outputs_phys, path_states, paths, nodes=None):
    """outputs_phys[l]: [32,HV,V] (any dtype/device) verification output in PHYSICAL row order after ONE verification from S0.
    path_states[path_id][l]: [HV,V,K] durable state after committing that path from S0.  Reference per layer is recomputed and
    dropped immediately (float64 node states are 200 MB per layer)."""
    import q1_2b_fixtures as FX
    import q1_oracle as O
    import m1_topology_mapping as M
    nodes = list(M.ACTIVE_PHYSICAL) if nodes is None else list(nodes)
    out_cells, out_by_layer = [], []
    state_cells = {pid: [] for pid in path_states}
    state_by_layer = {pid: [] for pid in path_states}
    for l, inp in enumerate(operands):
        states, outs = FX.c2_node_refs(inp, SCALE)
        if outputs_phys is not None and outputs_phys[l] is not None:
            o = outputs_phys[l]
            require(tuple(o.shape) == (N_PHYS, H_V, D_V), f"layer {l}: output shape {tuple(o.shape)} != {(N_PHYS, H_V, D_V)}")
            lc = [O.per_head_errors(o[n], outs[n]) for n in nodes]
            out_cells += lc
            out_by_layer.append(_agg_cells(lc)["max_abs_max"])
        for pid, per_layer in path_states.items():
            st = per_layer[l]
            require(tuple(st.shape) == (H_V, D_V, D_K), f"{pid} layer {l}: state shape {tuple(st.shape)}")
            leaf = paths[pid][-1]
            e = O.per_head_errors(st, states[leaf])
            state_cells[pid].append(e)
            state_by_layer[pid].append(_agg_cells([e])["max_abs_max"])
        del states, outs
    rep = {"note": ERR_NOTE, "nodes_compared": nodes,
           "verify_output": ({**_agg_cells(out_cells), "max_abs_by_layer": out_by_layer} if out_cells else None),
           "committed_state": {pid: {**_agg_cells(c), "accepted_nodes": paths[pid], "max_abs_by_layer": state_by_layer[pid]} for pid, c in state_cells.items()}}
    return rep


# ----------------------------------------------------------------------------------------------- run metadata
def runtime_identity(torch):
    rec = {"utc": utc(), "hostname": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__,
           "cuda": str(torch.version.cuda), "image_id_env": os.environ.get("KBENCH_IMAGE_ID"),
           "env_fr13": {k: v for k, v in sorted(os.environ.items()) if k.startswith("FR13_")}}
    try:
        import triton
        rec["triton"] = str(triton.__version__)
    except Exception as e:  # noqa: BLE001
        rec["triton"] = f"unavailable: {e}"
    if torch.cuda.is_available():
        d = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(d)
        rec.update({"gpu_name": torch.cuda.get_device_name(d), "gpu_sm_count": int(p.multi_processor_count),
                    "gpu_total_memory": int(p.total_memory), "capability": list(torch.cuda.get_device_capability(d))})
    rec["version_expectations"] = {"torch": EXPECTED_TORCH, "triton": EXPECTED_TRITON,
                                   "torch_matches": rec["torch"] == EXPECTED_TORCH, "triton_matches": rec.get("triton") == EXPECTED_TRITON}
    return rec


def source_hashes():
    files = {"workspace:src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": os.path.join(WORKSPACE, "src", "lumo_flywheel_serving", "fr10_gdn_tree_kernel.py"),
             "workspace:scripts/fr13_fixed32_topology.py": os.path.join(WORKSPACE, "scripts", "fr13_fixed32_topology.py"),
             "review:src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": os.path.join(REVIEW_ROOT, "src", "lumo_flywheel_serving", "fr10_gdn_tree_kernel.py"),
             "review:scripts/fr13_fixed32_topology.py": os.path.join(REVIEW_ROOT, "scripts", "fr13_fixed32_topology.py"),
             "m1:m1_adapters_v3.py": os.path.join(M1_TOOLS, "m1_adapters_v3.py"), "m1:m1_adapters_v2.py": os.path.join(M1_TOOLS, "m1_adapters_v2.py"),
             "m1:m1_adapters.py": os.path.join(M1_TOOLS, "m1_adapters.py"), "m1:m1_loaders.py": os.path.join(M1_TOOLS, "m1_loaders.py"),
             "m1:m1_cycle_driver_v2.py": os.path.join(M1_TOOLS, "m1_cycle_driver_v2.py"), "q1:q1_component_runner_v2_2.py": os.path.join(Q1_TOOLS, "q1_component_runner_v2_2.py"),
             "q1:q1_oracle.py": os.path.join(Q1_TOOLS, "q1_oracle.py"), "q1:q1_2b_fixtures.py": os.path.join(Q1_TOOLS, "q1_2b_fixtures.py")}
    here = os.path.dirname(os.path.abspath(__file__))
    for f in sorted(os.listdir(here)):
        if f.endswith((".py", ".sh")):
            files[f"kbench:{f}"] = os.path.join(here, f)
    return {k: (sha256_file(p) if os.path.exists(p) else None) for k, p in files.items()}


def check_pinned_sources(hashes, strict=True):
    """Lumo kernel/topology bytes must be the M1-frozen bytes in BOTH trees (the Q1 runner puts review/src on sys.path)."""
    want = {"workspace:src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": EXPECTED_KERNEL_SHA, "review:src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py": EXPECTED_KERNEL_SHA,
            "workspace:scripts/fr13_fixed32_topology.py": EXPECTED_TOPOLOGY_SHA, "review:scripts/fr13_fixed32_topology.py": EXPECTED_TOPOLOGY_SHA,
            "q1:q1_component_runner_v2_2.py": EXPECTED_RUNNER_SHA}
    bad = {k: (hashes.get(k), v) for k, v in want.items() if hashes.get(k) != v}
    if strict:
        require(not bad, f"pinned source mismatch (got, want): {bad}")
    return {"pinned_ok": not bad, "mismatches": bad}


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str, allow_nan=True)
    os.replace(tmp, path)
