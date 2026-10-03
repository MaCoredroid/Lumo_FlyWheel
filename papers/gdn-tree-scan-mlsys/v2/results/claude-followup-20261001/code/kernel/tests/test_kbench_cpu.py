#!/usr/bin/env python3
"""CPU-only tests for the kernel benchmark (never touch CUDA, Triton, Lumo kernels or author kernels).

Run with the review repo (or a mirror with the same layout) and a worktree providing scripts/fr13_fixed32_topology.py:
  KBENCH_REVIEW_ROOT=/home/mark/lumotree-review-20260927 KBENCH_WORKSPACE=/home/mark/shared/lumotree-v2exp-20260930 \
  CUDA_VISIBLE_DEVICES= python3 -m pytest -q scripts/v2exp/kernel/tests/test_kbench_cpu.py
The naive baselines run on CPU at L=2 and are checked against the float64 C2 reference with the M1 comparator; the native
vLLM kernel is replaced by a torch emulator transcribed from the pinned source (fla ops fused_sigmoid_gating.py, sha 000ab899)
so the path-cover slot layout of naive_native_paths and of the sweep's depth-level schedule is exercised."""
from __future__ import annotations

import json
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.dirname(HERE)
sys.path.insert(0, KDIR)
import kbench_common as C  # noqa: E402
import kbench_trees as TR  # noqa: E402

C.setup_paths()
torch.set_num_threads(min(8, os.cpu_count() or 1))
CPU = torch.device("cpu")


def args(**kw):
    d = dict(allow_layer_subset=True, allow_native_sha_mismatch=True, capture_warmup=1)
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture(scope="module")
def ops2():
    return C.build_operands(20260928, 2, "ordinary-random")


@pytest.fixture(scope="module")
def paths():
    return C.accepted_paths()


# ------------------------------------------------------------------------------------------------ geometry / topology
def test_topology_and_paths(paths):
    t = C.check_topology()
    assert t["physical_parent"][:4] == [-1, 0, 0, 0] and t["inactive_physical"] == [18, 23, 25, 27]
    assert len(paths) == 28 and paths["n31"] == [0, 1, 4, 9, 14, 19, 24, 26, 28, 29, 30, 31] and paths["root-only"] == [0]
    assert list(TR.PHYS_PARENT) == t["physical_parent"]


def test_model_config(tmp_path):
    good = {"text_config": {"linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_key_heads": 16, "linear_num_value_heads": 48,
                            "num_hidden_layers": 64, "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 16}}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(good))
    assert C.check_model_config(str(p))["gdn_layers"] == 48
    bad = json.loads(json.dumps(good))
    bad["text_config"]["linear_key_head_dim"] = 64
    p.write_text(json.dumps(bad))
    with pytest.raises(C.GeometryError):
        C.check_model_config(str(p))
    with pytest.raises(C.GeometryError):
        C.check_model_config(str(tmp_path / "missing.json"))


def test_cut_formula():
    f = C.m_cut_formula(5)
    assert f["bytes_per_head_value_tile"] == 4 * 5 * 8 * 128 == 20480
    assert f["head_value_tiles_per_layer"] == 48 * 16
    assert f["bytes_per_layer"] == 5 * C.STATE_BYTES == 15_728_640


def test_stats():
    s = C.summarize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert s["median_ms"] == 5.5 and abs(s["p10_ms"] - 1.9) < 1e-9 and abs(s["p90_ms"] - 9.1) < 1e-9 and s["n"] == 10


def test_trees():
    want = {"chain12": (12, 11), "tree8": (9, 2), "tree16": (17, 4), "tree27": (28, 11)}
    for name, (n, depth) in want.items():
        sh = TR.make_shape(name)
        assert sh["n"] == n and sh["depth"] == depth, (name, sh)
        assert not set(sh["rows"]) & set(TR.INACTIVE)
        order = TR.dfs_preorder(sh["parent"])
        assert sorted(order) == list(range(n)) and order[0] == 0
        pd = TR.relabel(sh["parent"], order)
        for i in range(n):          # ancestor sets are preserved by the relabelling
            assert sorted(order[a] for a in TR.ancestors(pd, i)) == sorted(TR.ancestors(sh["parent"], order[i]))
        s, v = TR.ancestor_masks(sh["parent"], TR.next_pow2(n))
        assert all(v[i][i] == 1 and s[i][i] == 0 for i in range(n))
    with pytest.raises(ValueError):
        TR.make_shape("tree99")


class _Ev:
    def __init__(self, enable_timing=True):
        self.t = None

    def record(self):
        import time
        self.t = time.perf_counter()

    def elapsed_time(self, other):
        return (other.t - self.t) * 1e3


def test_time_iterations_regions():
    fake = SimpleNamespace(cuda=SimpleNamespace(Event=_Ev, synchronize=lambda *a: None))
    calls = []

    def step(i, mark):
        calls.append(i)
        mark("verify")
        mark("commit")
    raw = C.time_iterations(fake, step, warmup=2, repeats=5, pre=lambda i: None, flush=None)
    assert raw["labels"] == ["verify", "commit"] and len(raw["total"]) == 5 and calls == list(range(7))
    rec = C.timing_record(raw)
    assert rec["total"]["n"] == 5 and set(rec["regions"]) == {"verify", "commit"}


# ------------------------------------------------------------------------------------------------ naive baselines on CPU
def test_naive_torch_node_matches_c2(ops2, paths):
    from kbench_naive import NaiveTorchNode
    m = NaiveTorchNode(torch, CPU, 2, paths, args())
    m.setup(ops2)
    outs, states = m.numerics(["root-only", "n14", "n31"])
    rep = C.numerics_report(torch, ops2, outs, states, paths)
    assert rep["verify_output"]["cells"] == 2 * 28 and rep["verify_output"]["invalid_cells"] == 0
    assert rep["verify_output"]["rel_max_abs_max"] < 1e-2          # bf16 output rounding
    for pid in ("root-only", "n14", "n31"):
        assert rep["committed_state"][pid]["max_abs_max"] < 1e-3, (pid, rep["committed_state"][pid])
    acc = m.memory_accounting()
    assert acc["node_state_bank"]["logical_node_state_bytes_per_layer"] == 32 * C.STATE_BYTES


def emulate_native(A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0, scale=None, initial_state=None, inplace_final_state=True,
                   cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None, use_qk_l2norm_in_kernel=False, is_kda=False):
    """Torch transcription of fused_sigmoid_gating_delta_rule_update(_kernel) (varlen + continuous batching + spec decoding)."""
    assert inplace_final_state and cu_seqlens is not None and ssm_state_indices is not None and not is_kda
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    scale = K ** -0.5 if scale is None else scale
    hm = torch.arange(HV) // (HV // H)
    o = torch.zeros((1,) + tuple(v.shape), dtype=q.dtype)
    cu = cu_seqlens.tolist()
    for i_n in range(len(cu) - 1):
        bos, eos = cu[i_n], cu[i_n + 1]
        i_t = int(num_accepted_tokens[i_n]) - 1 if num_accepted_tokens is not None else 0
        idx = int(ssm_state_indices[i_n, i_t])
        if idx <= 0:
            continue
        h = initial_state[idx].float().clone()
        for t in range(eos - bos):
            p = bos + t
            qt, kt, vt = q[0, p].float(), k[0, p].float(), v[0, p].float()
            x = a[0, p].float() + dt_bias.float()
            sp = torch.where(beta * x <= threshold, (1 / beta) * torch.log(1 + torch.exp(beta * x)), x)
            g = -torch.exp(A_log.float()) * sp
            bt = torch.sigmoid(b[0, p].float())
            if use_qk_l2norm_in_kernel:
                qt = qt * torch.rsqrt((qt * qt).sum(-1, keepdim=True) + 1e-6)
                kt = kt * torch.rsqrt((kt * kt).sum(-1, keepdim=True) + 1e-6)
            qt = qt * scale
            qh, kh = qt[hm], kt[hm]
            h = h * torch.exp(g)[:, None, None]
            vt = (vt - (h * kh[:, None, :]).sum(-1)) * bt[:, None]
            h = h + vt[:, :, None] * kh[:, None, :]
            o[0, 0, p] = (h * qh[:, None, :]).sum(-1).to(q.dtype)
            f = int(ssm_state_indices[i_n, t])
            if f > 0:
                initial_state[f] = h.to(initial_state.dtype)
    return o.squeeze(0), initial_state


@pytest.fixture()
def fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm.model_executor.layers.fla.ops.fused_sigmoid_gating")
    mod.fused_sigmoid_gating_delta_rule_update = emulate_native
    mod.__file__ = __file__
    ops_pkg = types.ModuleType("vllm.model_executor.layers.fla.ops")
    ops_pkg.fused_sigmoid_gating = mod
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.fla"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fla.ops", ops_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.fla.ops.fused_sigmoid_gating", mod)
    return mod


def test_naive_native_paths_slot_layout(ops2, paths, fake_vllm):
    from kbench_naive import NaiveNativePaths
    m = NaiveNativePaths(torch, CPU, 2, paths, args())
    m.setup(ops2)
    assert [lv["T"] for lv in m.lv] == [5, 27] and int(m.lv[1]["ssi"][:, 15].min()) >= 1
    outs, states = m.numerics(["n15", "n31"])
    rep = C.numerics_report(torch, ops2, outs, states, paths)
    assert rep["verify_output"]["rel_max_abs_max"] < 1e-2
    for pid in ("n15", "n31"):
        assert rep["committed_state"][pid]["max_abs_max"] < 1e-3, rep["committed_state"][pid]


def test_sweep_naive_cpu(ops2, fake_vllm):
    import kernel_sweep as SW
    a = SimpleNamespace(layers=2, modes=[], ops_cpu=ops2, warmup=0, repeats=0, capture_warmup=1)
    shapes = [TR.make_shape("chain12"), TR.make_shape("tree8")]
    res = SW.run_naive(torch, CPU, ops2, shapes, a, flush=None)
    for fam in ("naive_native_depth", "naive_torch_node"):
        for name in ("chain12", "tree8"):
            cell = res[fam][name]
            assert "error" not in cell, cell.get("traceback")
            assert cell["layer0_error_vs_c2"]["rel_max_abs_max"] < 1e-2, (fam, name, cell["layer0_error_vs_c2"])
    assert res["naive_native_depth"]["chain12"]["launches_per_layer"] == 12


# ------------------------------------------------------------------------------------------------ aggregation
def test_aggregate(tmp_path):
    import aggregate as AG
    (tmp_path / "methods").mkdir()
    (tmp_path / "sweep").mkdir()
    s = C.summarize([1.0, 2.0, 3.0])
    rec = {"method": "lumo_fixed32", "status": "complete", "geometry": {"gdn_layers": 48},
           "numerics": {"verify_output": {"max_abs_max": 1e-3, "rms_max": 1e-4}, "committed_state": {"n31": {"max_abs_max": 2e-5}}},
           "memory": {"accounting": {"cut_state_buffer": {"allocated_bytes": 100663296, "N_c": 5, "logical_cut_nodes": [0, 1, 4, 9, 14], "logical_bytes_written_per_layer": 15728640},
                                     "paper_formula": C.m_cut_formula(5), "replay_rings": {"bytes": 10}, "committer_fixed16_staging": {"bytes": 20}}},
           "timing": {"graph": {"step": {"regions": {"verify": s, "commit": s}, "total": s,
                                         "memory": {"transient_peak_allocated_bytes": 0, "peak_allocated_bytes": 1, "peak_reserved_bytes": 2}}}},
           "derived": {"graph": {}}, "graph_capture": {"verify": {"ok": True}}}
    (tmp_path / "methods" / "lumo_fixed32.json").write_text(json.dumps(rec))
    (tmp_path / "sweep" / "sweep_naive.json").write_text(json.dumps({"family": "naive", "status": "complete", "result": {
        "naive_torch_node": {"tree8": {"shape": {"n": 9}, "timing": {"graph": {"total": s, "per_layer_median_ms": 0.1}}, "layer0_error_vs_c2": {"max_abs_max": 1e-4}}}}}))
    assert AG.main(str(tmp_path)) == 0
    md = (tmp_path / "summary.md").read_text()
    assert "lumo_fixed32" in md and "N_c = 5" in md and "tree8" in md


# ------------------------------------------------------------------------------------------------ author-method plumbing (fake kernels)
@pytest.fixture()
def fake_authors(monkeypatch):
    """Shape-faithful stand-ins for the author kernels so the M1-plan execution path (resolve -> call -> bind) runs on CPU.
    They do NOT reproduce author numerics; only call/argument plumbing, stash shapes and the per-step flow are exercised."""
    import m1_adapters as A1
    import m1_adapters_v2 as A2
    import kbench_author as KA
    fla = "sglang.srt.layers.attention.fla"
    log = []

    def mk(name, **fns):
        m = types.ModuleType(name)
        for k, v in fns.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    def gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0, out_g=None, out_beta=None):
        g, be = A1.local_fp32_gating(A_log, a, b, dt_bias)
        out_g.copy_(g.view_as(out_g)); out_beta.copy_(be.view_as(out_beta)); log.append("gating")
        return out_g, out_beta

    def l2(x, eps=1e-6, out=None):
        y = A1.native_rsqrt_l2norm(x).to(x.dtype)
        if out is None:
            return y
        out.copy_(y.reshape(out.shape))
        return out

    def verify(A_log, a, dt_bias, sb, st, q, k, v, b, init, idx, tree, scale=None, **kw):
        assert q.shape == k.shape == (1, 32, 16, 128) and v.shape == (1, 32, 48, 128) and init.shape == (2, 48, 128, 128)
        assert kw["precision"] == "tf32" and kw["bf16_mode"] == "none" and kw["use_qk_l2norm_in_kernel"] is False
        log.append("verify")
        return torch.zeros_like(v)

    def replay(**kw):
        assert kw["ssm_states"].shape[2:] == (48, 128, 128) and kw["last_correct_steps"].shape == (1,)
        log.append(("replay", int(kw["last_correct_steps"][0])))

    def treewy(qs, ks, vs, a, b, A_log, dt_bias, anc_s, anc_i, vt, kk, gc, kt, leaf, slots, ssm, n, scale, **kw):
        assert qs.shape == (32, 16, 128) and vt.shape == (2, 48, 32, 128) and n == 32
        log.append(("treewy", int(leaf[0]), bool(gc[1, 0, 0] > 0)))
        return torch.zeros((1, 32, 48, 128), dtype=qs.dtype)
    mk(f"{fla}.gdn_tree_fused", alloc_tree_structure_buffers=lambda N, T, max_depth, device: A2.cpu_alloc_tree_structure_buffers(N, T, max_depth, device),
       build_tree_structure_into_fast=A2.cpu_build_tree_structure_into)
    mk(f"{fla}.chunk_tree_verify", build_tree_ancestor_masks=A2.cpu_build_tree_ancestor_masks, advance_ssm_states_along_accept_paths=replay)
    mk(f"{fla}.fused_gdn_gating", fused_gdn_gating=gating)
    mk(f"{fla}.l2norm", l2norm_fwd_strided=l2)
    mk(f"{fla}.gdn_tree_triton", tree_gdn_triton_verify=verify, _WORKSPACE={}, _TRI_CACHE={})
    mk("m1_treewy.tree_wy_triton", tree_wy_tree_commit_capture_triton=treewy)
    monkeypatch.setattr(KA, "_load_author_modules", lambda: {"fake": True})
    return log


FAKE_TORCH = SimpleNamespace(cuda=SimpleNamespace(Event=_Ev, synchronize=lambda *a: None))


@pytest.mark.parametrize("variant", ["weaver_author_default", "weaver_aligned_local"])
def test_weaver_plumbing(ops2, paths, fake_authors, variant):
    from kbench_author import WeaverMethod
    m = WeaverMethod(torch, CPU, 2, paths, args(), variant=variant)
    m.setup(ops2)
    outs, states = m.numerics(["n31"])
    assert len(outs) == 2 and tuple(outs[0].shape) == (32, 48, 128) and tuple(states["n31"][0].shape) == (48, 128, 128)
    fake_authors.clear()
    for ex in m.experiments("eager"):
        raw = C.time_iterations(FAKE_TORCH, ex["fn"], warmup=1, repeats=2, pre=ex.get("pre"))
        assert raw["labels"]
    replays = [x[1] for x in fake_authors if isinstance(x, tuple) and x[0] == "replay"]
    assert replays[:3] == [m.paths[m.path_ids[i]][-1] for i in range(3)]          # leaf cycles through the device table
    assert fake_authors.count("verify") >= 3 * 2
    assert m.memory_accounting()["stashes"]["bytes"] > 0


def test_treewy_plumbing(ops2, paths, fake_authors):
    import m1_adapters as A1
    from kbench_author import TreeWYMethod
    m = TreeWYMethod(torch, CPU, 2, paths, args())
    m.setup(ops2)
    outs, states = m.numerics(["n14"])
    assert tuple(outs[1].shape) == (32, 48, 128)
    fake_authors.clear()
    exps = {e["name"]: e for e in m.experiments("eager")}
    assert set(exps) == {"step", "verify_only_sentinel", "final_flush", "layer0_step"}
    C.time_iterations(FAKE_TORCH, exps["verify_only_sentinel"]["fn"], warmup=0, repeats=1, pre=exps["verify_only_sentinel"]["pre"])
    calls = [x for x in fake_authors if isinstance(x, tuple) and x[0] == "treewy"]
    assert len(calls) == 2 and all(c[2] for c in calls)                            # sentinel visible to every layer's call
    assert calls[0][1] == A1.treewy_leaf_author_id(m.paths[m.path_ids[0]])
