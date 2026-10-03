#!/usr/bin/env python3
"""Tree-size sensitivity of the VERIFY (scan) kernels: chain12 vs tree8 / tree16 / tree27 (see kbench_trees.py), 48 layers, B1.

The deployed fixed32 Lumo route has a FIXED 32-row geometry (Hydra27 padded to 32; masks do not shrink the work), so its cost
does not depend on the number of active nodes by construction: its reference point is the main benchmark (lumo_fixed32).
Sizes are therefore swept with:
  lumo_generic       : the same Lumo path-scan kernel family through the GENERIC (non-fixed32) subtree route
                       (FR13_FIXED32_MODE unset, FR13_SUBTREE_PARALLEL=1, BV=8, ring export, plain fp32 h0) -- NOT the deployed
                       configuration; its own process (the fixed32 mode is fixed at kernel import).
  treewy_verify_only : author tree_wy_tree_commit_capture_triton on the shape's DFS pre-order, positive gcum sentinel (commit skipped).
  weaver_verifier    : author tree_gdn_triton_verify (precision tf32, bf16_mode none) on pre-normalized q/k (preparation excluded).
  naive_native_depth : vLLM native fused_sigmoid_gating with per-node state stores, one launch per depth level per layer.
  naive_torch_node   : pure-torch sequential per-node update storing every node state.
Each (family, shape, mode) cell is independent: a failure is recorded in the JSON and the sweep continues.
Layer-0 outputs are compared with the float64 C2 reference (q1_oracle.c2_step along each node's ancestor chain)."""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kbench_common as C  # noqa: E402
import kbench_trees as TR  # noqa: E402

FAMILIES = ("lumo_generic", "authors", "naive")


def c2_layer_outputs(inp, rows, parent):
    """float64 C2 outputs [n,HV,V] for a compact tree whose node i uses physical operand row rows[i]."""
    import torch
    import q1_oracle as O
    f = {k: inp[k].to(torch.float64) for k in ("q", "k", "v", "a", "b")}
    A, d = inp["A_log"].to(torch.float64), inp["dt_bias"].to(torch.float64)
    states, outs = [], []
    for i, r in enumerate(rows):
        s_prev = inp["S0"].to(torch.float64) if parent[i] < 0 else states[parent[i]]
        s, o, _ = O.c2_step(s_prev, f["q"][r], f["k"][r], f["v"][r], f["a"][r], f["b"][r], A, d, scale=C.SCALE)
        states.append(s)
        outs.append(o)
    return outs


def err_layer0(cand_nodes, ref_nodes):
    import q1_oracle as O
    return C._agg_cells([O.per_head_errors(c, r) for c, r in zip(cand_nodes, ref_nodes)])


def time_modes(torch, verify_fn, args, flush, pre=None):
    res = {}
    for mode in args.modes:
        try:
            if mode == "graph":
                g, err = C.capture_graph(torch, verify_fn, warmup_iters=args.capture_warmup)
                if g is None:
                    res[mode] = {"capture_failed": err}
                    continue
                fn = lambda i, mark, _g=g: (_g.replay(), mark("verify"))  # noqa: E731
            else:
                fn = lambda i, mark: (verify_fn(), mark("verify"))  # noqa: E731
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            raw = C.time_iterations(torch, fn, warmup=args.warmup, repeats=args.repeats, pre=pre, flush=flush, sync_each=True)
            r = C.timing_record(raw, keep_raw=False)
            r["per_layer_median_ms"] = r["total"]["median_ms"] / args.layers
            r["transient_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() - before
            res[mode] = r
        except Exception as e:  # noqa: BLE001
            res[mode] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    return res


# ------------------------------------------------------------------------------------------------------------- lumo generic
def run_lumo_generic(torch, device, ops, shapes, args, flush):
    for k, v in C.GENERIC_ENV_SET.items():
        os.environ[k] = v
    for k in C.GENERIC_ENV_UNSET:
        os.environ.pop(k, None)
    import lumo_flywheel_serving.fr10_gdn_tree_kernel as kernel
    C.require(C.sha256_file(kernel.__file__) == C.EXPECTED_KERNEL_SHA, "kernel bytes are not the M1-pinned bytes")
    C.require(kernel._FR13_FIXED32_MODE is None, f"generic sweep needs FR13_FIXED32_MODE unset, got {kernel._FR13_FIXED32_MODE!r}")
    out = {"route": "generic subtree path scan (non-fixed32)", "env": {k: os.environ.get(k) for k in list(C.GENERIC_ENV_SET) + list(C.GENERIC_ENV_UNSET)}, "shapes": {}}
    for sh in shapes:
        cell = {"shape": sh}
        try:
            n, parent, rows = sh["n"], sh["parent"], sh["rows"]
            n_pad = TR.next_pow2(n)
            kernel.subtree_preseed(parent, n, C.H_V, C.D_V, C.D_K, device)
            st = kernel.subtree_get(n, C.H_V, C.D_V, C.D_K, device)
            ridx = torch.tensor(rows, dtype=torch.long, device=device)

            def pad(t, rows_n=n_pad):
                z = torch.zeros((rows_n,) + tuple(t.shape[1:]), dtype=t.dtype, device=device)
                z[:n].copy_(t.index_select(0, ridx))
                return z
            strict, vis = TR.ancestor_masks(parent, n_pad)
            sm = torch.tensor(strict, dtype=torch.int32, device=device)
            vm = torch.tensor(vis, dtype=torch.int32, device=device)
            L = args.layers
            outb = torch.zeros((L, n_pad, C.H_V, C.D_V), dtype=torch.bfloat16, device=device)
            rk = torch.zeros((L, n_pad, C.H_K, C.D_K), dtype=torch.bfloat16, device=device)
            rv = torch.zeros((L, n_pad, C.H_V, C.D_V), dtype=torch.bfloat16, device=device)
            ra = torch.zeros((L, n_pad, C.H_V), dtype=torch.bfloat16, device=device)
            rb = torch.zeros((L, n_pad, C.H_V), dtype=torch.bfloat16, device=device)
            flags = torch.zeros((2,), dtype=torch.int32, device=device)
            g0 = torch.zeros((n_pad, C.H_V), dtype=torch.float32, device=device)
            b0 = torch.zeros((n_pad, C.H_V), dtype=torch.float32, device=device)
            kws = []
            for l, inp in enumerate(ops):
                kws.append(dict(q=pad(inp["q"]), k=pad(inp["k"]), v=pad(inp["v"]), g=g0, beta=b0, raw_a=pad(inp["a"]), raw_b=pad(inp["b"]),
                                A_log=inp["A_log"], dt_bias=inp["dt_bias"], h0=inp["S0"], h0_is_bank=False, n_actual=n, n_pad=n_pad,
                                strict_mask=sm, visible_mask=vm, out=outb[l], state=None, output_scale=C.SCALE, use_qk_l2norm_in_kernel=True,
                                ring_k=rk[l], ring_v=rv[l], ring_a=ra[l], ring_b=rb[l], staging_flags=flags, staging_rows=1))

            def verify(_kws=kws):
                for kw in _kws:
                    kernel.launch_tree_gdn_prepared(**kw)
            verify()
            C.sync(torch, device)
            ref = c2_layer_outputs(args.ops_cpu[0], rows, parent)
            cell["layer0_error_vs_c2"] = err_layer0([outb[0, i].cpu() for i in range(n)], ref)
            levels = st["levels"]
            n_c = int(st["emask"].sum().item())
            cell["schedule"] = {"schedule": st.get("schedule"), "levels": len(levels), "paths_per_level": [int(x[3]) for x in levels],
                                "max_len_per_level": [int(x[2]) for x in levels], "critical_path": int(st["critical"]), "n_pad": n_pad,
                                "cut_states_N_c": n_c, "export_buffer_allocated_bytes": int(st["export"].numel() * 4),
                                "cut_bytes_written_per_layer": n_c * C.STATE_BYTES, "paper_formula": C.m_cut_formula(n_c)}
            cell["timing"] = time_modes(torch, verify, args, flush)
        except Exception as e:  # noqa: BLE001
            cell["error"] = f"{type(e).__name__}: {e}"
            cell["traceback"] = traceback.format_exc()
        out["shapes"][sh["name"]] = cell
    return out


# ------------------------------------------------------------------------------------------------------------- authors
def run_authors(torch, device, ops, shapes, args, flush):
    import m1_loaders as LD
    import m1_adapters as A1
    rec = LD.load(execute=True)
    tw = sys.modules["m1_treewy.tree_wy_triton"]
    fla = "sglang.srt.layers.attention.fla"
    wt, wf, l2 = sys.modules[f"{fla}.gdn_tree_triton"], sys.modules[f"{fla}.gdn_tree_fused"], sys.modules[f"{fla}.l2norm"]
    out = {"loader_created": rec.get("created"), "treewy": {}, "weaver": {}}
    L = args.layers
    for sh in shapes:
        n, parent, rows = sh["n"], sh["parent"], sh["rows"]
        ref = c2_layer_outputs(args.ops_cpu[0], rows, parent)
        # ---------------- TreeWY verify-only (sentinel) on the DFS pre-order
        cell = {"shape": sh}
        try:
            order = TR.dfs_preorder(parent)
            pdfs = TR.relabel(parent, order)
            rows_dfs = torch.tensor([rows[o] for o in order], dtype=torch.long, device=device)
            strict, vis = TR.ancestor_masks(pdfs)
            anc_i = torch.tensor(vis, dtype=torch.float32, device=device)
            anc_s = torch.tensor(strict, dtype=torch.float32, device=device)
            slots = torch.ones((1,), dtype=torch.int64, device=device)
            leaf = torch.zeros((1,), dtype=torch.int64, device=device)
            vt = torch.zeros((L, 2, C.H_V, n, C.D_V), dtype=torch.float32, device=device)
            kk = torch.zeros((L, 2, n, C.H_V, C.D_K), dtype=torch.float32, device=device)
            gc = torch.ones((L, 2, C.H_V, n), dtype=torch.float32, device=device)
            ssm = torch.zeros((L, 2, C.H_V, C.D_V, C.D_K), dtype=torch.float32, device=device)
            staged = []
            for l, inp in enumerate(ops):
                ssm[l, 1].copy_(inp["S0"])
                staged.append({x: inp[x].index_select(0, rows_dfs).contiguous() for x in ("q", "k", "v", "a", "b")})
            P = A1.TREEWY_PINS
            outs = {}

            def tw_verify():
                for l, s in enumerate(staged):
                    outs[l] = tw.tree_wy_tree_commit_capture_triton(s["q"], s["k"], s["v"], s["a"], s["b"], ops[l]["A_log"], ops[l]["dt_bias"], anc_s, anc_i,
                                                                    vt[l], kk[l], gc[l], anc_i, leaf, slots, ssm[l], n, C.SCALE,
                                                                    BV=P["BV"], num_warps=P["num_warps"], num_stages=P["num_stages"], dot_bf16=True)
            gc.fill_(1.0)
            tw_verify()
            C.sync(torch, device)
            o0 = outs[0][0].cpu()                                   # [n,HV,V] in DFS order
            cell["layer0_error_vs_c2"] = err_layer0([o0[order.index(i)] for i in range(n)], ref)
            cell["stash_bytes"] = C.tensor_bytes([vt, kk, gc])
            cell["timing"] = time_modes(torch, tw_verify, args, flush, pre=lambda i: gc.fill_(1.0))
        except Exception as e:  # noqa: BLE001
            cell["error"] = f"{type(e).__name__}: {e}"
            cell["traceback"] = traceback.format_exc()
        out["treewy"][sh["name"]] = cell
        # ---------------- Weaver verifier only (q/k pre-normalized by the author l2norm helper outside the timed region)
        cell = {"shape": sh}
        try:
            ridx = torch.tensor(rows, dtype=torch.long, device=device)
            tree = wf.alloc_tree_structure_buffers(1, n, sh["depth"], device)
            ptok = torch.tensor([parent], dtype=torch.int64, device=device)
            tree = wf.build_tree_structure_into_fast(ptok, tree)
            ssm_w = torch.zeros((L, 2, C.H_V, C.D_V, C.D_K), dtype=torch.float32, device=device)
            cache_idx = torch.ones((1,), dtype=torch.int64, device=device)
            staged = []
            for l, inp in enumerate(ops):
                ssm_w[l, 1].copy_(inp["S0"])
                q4 = inp["q"].index_select(0, ridx).view(1, n, C.H_K, C.D_K).contiguous()
                k4 = inp["k"].index_select(0, ridx).view(1, n, C.H_K, C.D_K).contiguous()
                staged.append({"q": l2.l2norm_fwd_strided(q4, eps=A1.NORM_EPS), "k": l2.l2norm_fwd_strided(k4, eps=A1.NORM_EPS),
                               "v": inp["v"].index_select(0, ridx).view(1, n, C.H_V, C.D_V).contiguous(),
                               "a": inp["a"].index_select(0, ridx).view(1, n, C.H_V).contiguous(), "b": inp["b"].index_select(0, ridx).view(1, n, C.H_V).contiguous()})
            W = A1.WEAVER_PINS
            outs = {}

            def w_verify():
                for l, s in enumerate(staged):
                    outs[l] = wt.tree_gdn_triton_verify(ops[l]["A_log"], s["a"], ops[l]["dt_bias"], A1.SOFTPLUS_BETA, A1.SOFTPLUS_THRESHOLD, s["q"], s["k"], s["v"], s["b"],
                                                        ssm_w[l], cache_idx, tree, scale=C.SCALE, use_qk_l2norm_in_kernel=False, return_lazy_state=False,
                                                        precision=W["precision"], bf16_mode=W["bf16_mode"])
            w_verify()
            C.sync(torch, device)
            o0 = outs[0][0].cpu()
            cell["layer0_error_vs_c2"] = err_layer0([o0[i] for i in range(n)], ref)
            cell["timing"] = time_modes(torch, w_verify, args, flush)
        except Exception as e:  # noqa: BLE001
            cell["error"] = f"{type(e).__name__}: {e}"
            cell["traceback"] = traceback.format_exc()
        out["weaver"][sh["name"]] = cell
    return out


# ------------------------------------------------------------------------------------------------------------- naive
def run_naive(torch, device, ops, shapes, args, flush):
    from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as mod
    fn = mod.fused_sigmoid_gating_delta_rule_update
    F = torch.nn.functional
    out = {"native_sha256": C.sha256_file(mod.__file__), "naive_native_depth": {}, "naive_torch_node": {}}
    L = args.layers
    head_map = torch.arange(C.H_V, device=device) // (C.H_V // C.H_K)
    for sh in shapes:
        n, parent, rows = sh["n"], sh["parent"], sh["rows"]
        ref = c2_layer_outputs(args.ops_cpu[0], rows, parent)
        dur = n + 1
        ridx = torch.tensor(rows, dtype=torch.long, device=device)
        staged = [{x: inp[x].index_select(0, ridx).contiguous() for x in ("q", "k", "v", "a", "b")} for inp in ops]
        bank = torch.zeros((L, n + 2, C.H_V, C.D_V, C.D_K), dtype=torch.float32, device=device)
        outb = torch.zeros((L, n, C.H_V, C.D_V), dtype=torch.bfloat16, device=device)

        def reset():
            bank.zero_()
            for l, inp in enumerate(ops):
                bank[l, dur].copy_(inp["S0"])
        # ---------------- native, one launch per depth level (1-token sequences reading the parent slot)
        cell = {"shape": sh}
        try:
            lv = []
            for nodes in TR.depth_levels(parent):
                rowsi = [[i + 1] + [0] * 14 + [dur if parent[i] < 0 else parent[i] + 1] for i in nodes]
                o = torch.tensor(nodes, dtype=torch.long, device=device)
                lv.append({"o": o, "T": len(nodes), "cu": torch.arange(len(nodes) + 1, dtype=torch.int32, device=device),
                           "ssi": torch.tensor(rowsi, dtype=torch.int32, device=device), "nacc": torch.full((len(nodes),), 16, dtype=torch.int32, device=device),
                           "per_layer": [{x: staged[l][x].index_select(0, o).unsqueeze(0).contiguous() for x in ("q", "k", "v", "a", "b")} for l in range(L)]})

            def nat_verify():
                for l in range(L):
                    for d in lv:
                        s = d["per_layer"][l]
                        y, _ = fn(A_log=ops[l]["A_log"], a=s["a"], b=s["b"], dt_bias=ops[l]["dt_bias"], q=s["q"], k=s["k"], v=s["v"], scale=C.SCALE,
                                  initial_state=bank[l], inplace_final_state=True, cu_seqlens=d["cu"], ssm_state_indices=d["ssi"],
                                  num_accepted_tokens=d["nacc"], use_qk_l2norm_in_kernel=True)
                        outb[l].index_copy_(0, d["o"], y.reshape(d["T"], C.H_V, C.D_V))
            reset()
            nat_verify()
            C.sync(torch, device)
            cell["layer0_error_vs_c2"] = err_layer0([outb[0, i].cpu() for i in range(n)], ref)
            cell["launches_per_layer"] = len(lv)
            cell["node_state_bytes_all_layers"] = n * C.STATE_BYTES * L
            reset()
            cell["timing"] = time_modes(torch, nat_verify, args, flush)
        except Exception as e:  # noqa: BLE001
            cell["error"] = f"{type(e).__name__}: {e}"
            cell["traceback"] = traceback.format_exc()
        out["naive_native_depth"][sh["name"]] = cell
        # ---------------- pure torch, sequential per node
        cell = {"shape": sh}
        try:
            def torch_verify():
                for l in range(L):
                    s, inp = staged[l], ops[l]
                    q, k = s["q"].float(), s["k"].float()
                    qn = (q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)).index_select(1, head_map) * C.SCALE
                    kn = (k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)).index_select(1, head_map)
                    vf = s["v"].float()
                    dec = torch.exp(-torch.exp(inp["A_log"]) * F.softplus(s["a"].float() + inp["dt_bias"], beta=1.0, threshold=20.0))
                    beta = torch.sigmoid(s["b"].float())
                    for i in range(n):
                        sp = bank[l, dur] if parent[i] < 0 else bank[l, parent[i] + 1]
                        st = bank[l, i + 1]
                        torch.mul(sp, dec[i].view(C.H_V, 1, 1), out=st)
                        r = (vf[i] - torch.bmm(st, kn[i].unsqueeze(-1)).squeeze(-1)) * beta[i].unsqueeze(-1)
                        st.baddbmm_(r.unsqueeze(-1), kn[i].unsqueeze(1))
                        outb[l, i].copy_(torch.bmm(st, qn[i].unsqueeze(-1)).squeeze(-1))
            reset()
            torch_verify()
            C.sync(torch, device)
            cell["layer0_error_vs_c2"] = err_layer0([outb[0, i].cpu() for i in range(n)], ref)
            reset()
            cell["timing"] = time_modes(torch, torch_verify, args, flush)
        except Exception as e:  # noqa: BLE001
            cell["error"] = f"{type(e).__name__}: {e}"
            cell["traceback"] = traceback.format_exc()
        out["naive_torch_node"][sh["name"]] = cell
        del bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default=",".join(TR.SHAPES))
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--modes", default="graph,eager")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--layers", type=int, default=C.GDN_LAYERS)
    ap.add_argument("--l2-flush-mib", type=int, default=256)
    ap.add_argument("--capture-warmup", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    a.modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"sweep_{a.family}.json")
    rec = {"schema": "lumo.v2exp.kernel-bench.sweep.v1", "family": a.family, "args": {k: v for k, v in vars(a).items()}, "started_utc": C.utc()}
    rc = 0
    try:
        if not a.smoke:
            C.require(a.repeats >= 50 and a.layers == C.GDN_LAYERS, "sweep needs >= 50 repeats and 48 layers outside --smoke")
        C.setup_paths()
        import torch
        C.require(torch.cuda.is_available(), "CUDA not available")
        device = torch.device("cuda", torch.cuda.current_device())
        rec["identity"] = C.runtime_identity(torch)
        rec["topology"] = C.check_topology()
        shapes = [TR.make_shape(s.strip()) for s in a.shapes.split(",") if s.strip()]
        ops_cpu = C.build_operands(a.seed, a.layers, "ordinary-random")
        a.ops_cpu = ops_cpu
        ops = C.stage_operands(ops_cpu, device)
        flush = C.L2Flusher(torch, device, a.l2_flush_mib)
        t0 = time.time()
        rec["result"] = {"lumo_generic": run_lumo_generic, "authors": run_authors, "naive": run_naive}[a.family](torch, device, ops, shapes, a, flush)
        rec["elapsed_s"] = round(time.time() - t0, 1)
        rec["status"] = "smoke_complete" if a.smoke else "complete"
    except C.GeometryError as e:
        rec.update(status="refused_geometry", error=str(e), traceback=traceback.format_exc())
        rc = 2
    except Exception as e:  # noqa: BLE001
        rec.update(status="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
        rc = 3
    rec["args"].pop("ops_cpu", None)
    rec["ended_utc"] = C.utc()
    C.write_json(path, rec)
    print(f"[kernel_sweep] {a.family}: {rec['status']} -> {path}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
