#!/usr/bin/env python3
"""E7a device harness. RUN INSIDE THE PINNED vLLM IMAGE (see run_in_image.sh).

Realizations exercised on identical rounded operands (bf16 q/k/v, bf16 raw gates a/b, fp32 A_log,
bf16 dt_bias, fp32 h0 pre-tree state, topology):

  oracle      float64 serial recurrence (CPU torch)                                 [reference: math]
  native_sg   pinned image `fused_sigmoid_gating_delta_rule_update` per root->node chain
              (the kernel vLLM dispatches for the speculative verify)               [reference: replacement]
  native_pk   pinned image `fused_recurrent_gated_delta_rule_packed_decode`, one token at a
              time along the chain (the default non-spec decode kernel)             [reference: continuation]
  A_prod      production scan `_tree_gdn_kernel` (out) + `_tree_gdn_replay_kernel` (commit) from THIS worktree
  A_legacy    June-2026 scan with per-node fp32 state export (vendored file; identity check vs payload bytes)
  B_legacyWY  June-2026 corrected WY kernel (vendored file, post state-write fix), two precision modes
  B_fs        new Triton forward-substitution solver + compact commit  (WY / TreeWY family, local impl)
  C_nm        new Triton finite-Neumann solver + compact commit          (Bole family, local impl)

Stage isolation: (1) verifier outputs + correction factors; (2) commit from IDENTICAL oracle factors;
(3) each method's own factors + own commit. Timing is CUDA-event based and reported separately for
verify and commit; it is labeled by the contention record the wrapper collects (shared-device
diagnostic unless an uncontended window is qualified).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import triton

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e7a_core as C  # noqa: E402
import e7a_kernels as K  # noqa: E402

PROD_PATH = Path("/work/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")
LEGACY_PATH = HERE / "legacy_wy_8a975837.py"
SPEC_COLS = 16
BANK_ROWS = SPEC_COLS + 4
PATH_COLS = 16


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return bool(torch.equal(a.contiguous(), b.contiguous()))


def exact_frac(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.contiguous().view(-1) == b.contiguous().view(-1).to(a.dtype)).double().mean().item()


# ----------------------------------------------------------------------------- native references
def native_sg_path(native_sg, pay: C.Payload, path: list[int], *, fp32_io: bool):
    dev = pay.q.device
    idx = torch.tensor(path, dtype=torch.long, device=dev)
    T = len(path)
    KH, DK = pay.q.shape[1], pay.q.shape[2]
    VH, DV = pay.v.shape[1], pay.v.shape[2]
    cast = (lambda t: t.float()) if fp32_io else (lambda t: t)
    q = cast(pay.q.index_select(0, idx)).reshape(1, T, KH, DK).contiguous()
    k = cast(pay.k.index_select(0, idx)).reshape(1, T, KH, DK).contiguous()
    v = cast(pay.v.index_select(0, idx)).reshape(1, T, VH, DV).contiguous()
    a = pay.a.index_select(0, idx).reshape(1, T, VH).contiguous()
    b = pay.b.index_select(0, idx).reshape(1, T, VH).contiguous()
    h0 = pay.h0.clone().reshape(1, VH, DV, DK).contiguous()
    o, st = native_sg(A_log=pay.A_log, a=a, b=b, dt_bias=pay.dt_bias, q=q, k=k, v=v, scale=pay.output_scale,
                      initial_state=h0, inplace_final_state=False, use_qk_l2norm_in_kernel=True)
    return o.reshape(T, VH, DV), st.reshape(T, VH, DV, DK)


def native_packed_path(native_pk, pay: C.Payload, path: list[int], *, fp32_io: bool):
    dev = pay.q.device
    KH, DK = pay.q.shape[1], pay.q.shape[2]
    VH, DV = pay.v.shape[1], pay.v.shape[2]
    bank = torch.zeros((2, VH, DV, DK), dtype=torch.float32, device=dev)
    bank[1] = pay.h0
    idx = torch.tensor([1], dtype=torch.int32, device=dev)
    outs = []
    states = []
    io_dtype = torch.float32 if fp32_io else pay.q.dtype
    for node in path:
        mixed = torch.cat([pay.q[node].reshape(-1), pay.k[node].reshape(-1), pay.v[node].reshape(-1)]).to(io_dtype).reshape(1, -1).contiguous()
        a = pay.a[node].reshape(1, VH).contiguous()
        b = pay.b[node].reshape(1, VH).contiguous()
        out = torch.empty((1, 1, VH, DV), dtype=io_dtype, device=dev)
        native_pk(mixed_qkv=mixed, a=a, b=b, A_log=pay.A_log, dt_bias=pay.dt_bias, scale=pay.output_scale,
                  initial_state=bank, out=out, ssm_state_indices=idx, use_qk_l2norm_in_kernel=True)
        outs.append(out.reshape(VH, DV).clone())
        states.append(bank[1].clone())
    return torch.stack(outs), torch.stack(states)


# ----------------------------------------------------------------------------- production A
def prod_bank(pay: C.Payload):
    dev = pay.q.device
    VH, DV, DK = pay.h0.shape
    bank = torch.zeros((BANK_ROWS, VH, DV, DK), dtype=torch.float32, device=dev)
    bank[0] = pay.h0
    spec_idx = torch.arange(SPEC_COLS, dtype=torch.int32, device=dev).view(1, -1)
    prev = torch.tensor([1], dtype=torch.int32, device=dev)  # h0 sits in column 0 (running row)
    return bank, spec_idx, prev


def prod_scan(prod, pay: C.Payload, n_pad: int, out_dtype):
    dev = pay.q.device
    n = pay.n
    tree = prod.Tree(tuple(pay.parents))
    strict, visible = tree.masks(dev, n_pad)
    bank, spec_idx, prev = prod_bank(pay)
    VH, DV = pay.v.shape[1], pay.v.shape[2]
    out = torch.zeros((n_pad, VH, DV), dtype=out_dtype, device=dev)
    g_ph = torch.zeros((n, VH), dtype=torch.float32, device=dev)
    beta_ph = torch.zeros((n, VH), dtype=torch.float32, device=dev)

    def launch():
        prod.launch_tree_gdn_prepared(
            q=pay.q, k=pay.k, v=pay.v, g=g_ph, beta=beta_ph, h0=bank, n_actual=n, n_pad=n_pad,
            strict_mask=strict, visible_mask=visible, out=out, output_scale=pay.output_scale,
            use_qk_l2norm_in_kernel=True, h0_indices=spec_idx, h0_num_accepted_tokens=prev, h0_is_bank=True,
            h0_index_row=0, h0_batch_index=0, h0_use_accepted_column=True,
            raw_a=pay.a, raw_b=pay.b, A_log=pay.A_log, dt_bias=pay.dt_bias,
        )
    launch()
    torch.cuda.synchronize()
    return out[:n].clone(), launch


def prod_replay(prod, pay: C.Payload, drafts: list[int], n_pad: int):
    """drafts = accepted path WITHOUT the root (the kernel always processes ROOT_NODE=0 first)."""
    dev = pay.q.device
    n = pay.n
    KH, DK = pay.k.shape[1], pay.k.shape[2]
    VH, DV = pay.v.shape[1], pay.v.shape[2]
    ring_k = torch.zeros((1, n_pad, KH, DK), device=dev, dtype=pay.k.dtype)
    ring_v = torch.zeros((1, n_pad, VH, DV), device=dev, dtype=pay.v.dtype)
    ring_a = torch.zeros((1, n_pad, VH), device=dev, dtype=pay.a.dtype)
    ring_b = torch.zeros((1, n_pad, VH), device=dev, dtype=pay.b.dtype)
    ring_k[0, :n].copy_(pay.k); ring_v[0, :n].copy_(pay.v); ring_a[0, :n].copy_(pay.a); ring_b[0, :n].copy_(pay.b)
    bank, spec_idx, prev = prod_bank(pay)
    paths_t = torch.zeros((1, PATH_COLS), device=dev, dtype=torch.int32)
    for t, node in enumerate(drafts):
        paths_t[0, t] = node
    lens_t = torch.tensor([len(drafts)], device=dev, dtype=torch.int32)

    def launch():
        prod.launch_tree_gdn_replay(
            state_bank=bank, spec_state_indices=spec_idx, prev_lens=prev, accepted_paths=paths_t,
            accepted_lens=lens_t, k_ring=ring_k, v_ring=ring_v, a_ring=ring_a, b_ring=ring_b,
            A_log=pay.A_log, dt_bias=pay.dt_bias, num_spec_decodes=1, output_scale=pay.output_scale,
            use_qk_l2norm_in_kernel=True,
        )
    launch()
    torch.cuda.synchronize()
    col = 0 if len(drafts) == 0 else len(drafts) - 1
    committed = bank[int(spec_idx[0, col].item())].clone()
    return committed, launch


# ----------------------------------------------------------------------------- legacy (June 2026) kernels
def legacy_scan(leg, pay: C.Payload, n_pad: int, *, use_wy: bool, bf16_boundaries: bool, out_dtype):
    dev = pay.q.device
    n = pay.n
    tree = leg.Tree(tuple(pay.parents))
    strict, visible = tree.masks(dev, n_pad)
    VH, DV, DK = pay.h0.shape
    out = torch.zeros((n_pad, VH, DV), dtype=out_dtype, device=dev)
    state = torch.zeros((n_pad, VH, DV, DK), dtype=torch.float32, device=dev)
    g_ph = torch.zeros((n, VH), dtype=torch.float32, device=dev)
    beta_ph = torch.zeros((n, VH), dtype=torch.float32, device=dev)

    def launch():
        leg.launch_tree_gdn_prepared(
            pay.q, pay.k, pay.v, g_ph, beta_ph, pay.h0, n_actual=n, n_pad=n_pad, strict_mask=strict,
            visible_mask=visible, out=out, state=state, output_scale=pay.output_scale,
            use_qk_l2norm_in_kernel=True, raw_a=pay.a, raw_b=pay.b, A_log=pay.A_log, dt_bias=pay.dt_bias,
            use_wy=use_wy, fla_bf16_boundaries=bf16_boundaries,
        )
    launch()
    torch.cuda.synchronize()
    return out[:n].clone(), state[:n].clone(), launch


# ----------------------------------------------------------------------------- new B/C kernels
class FactorBatch:
    """Padded batch operands for the factor kernels (all requests share topology + layer params)."""

    def __init__(self, pays: list[C.Payload], n_pad: int):
        p0 = pays[0]
        dev = p0.q.device
        B = len(pays)
        n = p0.n
        KH, DK = p0.q.shape[1], p0.q.shape[2]
        VH, DV = p0.v.shape[1], p0.v.shape[2]
        for p in pays:
            if p.parents != p0.parents:
                raise ValueError("factor batch requires one topology")
            if not (bits_equal(p.A_log, p0.A_log) and bits_equal(p.dt_bias, p0.dt_bias)):
                raise ValueError("factor batch requires shared layer parameters (A_log, dt_bias)")
        self.B, self.n, self.n_pad = B, n, n_pad
        self.q = torch.zeros((B, n_pad, KH, DK), dtype=p0.q.dtype, device=dev)
        self.k = torch.zeros((B, n_pad, KH, DK), dtype=p0.k.dtype, device=dev)
        self.v = torch.zeros((B, n_pad, VH, DV), dtype=p0.v.dtype, device=dev)
        self.a = torch.zeros((B, n_pad, VH), dtype=p0.a.dtype, device=dev)
        self.b = torch.zeros((B, n_pad, VH), dtype=p0.b.dtype, device=dev)
        self.h0 = torch.zeros((B, VH, DV, DK), dtype=torch.float32, device=dev)
        for i, p in enumerate(pays):
            self.q[i, :n] = p.q; self.k[i, :n] = p.k; self.v[i, :n] = p.v; self.a[i, :n] = p.a; self.b[i, :n] = p.b
            self.h0[i] = p.h0
        self.h0_rows = torch.arange(B, dtype=torch.int32, device=dev)
        strict = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=dev)
        visible = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=dev)
        sm = C.strict_mask(p0.parents)
        strict[:n, :n] = sm.to(torch.int32).to(dev)
        visible[:n, :n] = (sm | torch.eye(n, dtype=torch.bool)).to(torch.int32).to(dev)
        self.strict, self.visible = strict, visible
        self.A_log, self.dt_bias = p0.A_log, p0.dt_bias
        self.output_scale = p0.output_scale
        self.parents = list(p0.parents)
        self.depth = C.max_strict_ancestors(p0.parents)
        self.VH, self.DV, self.DK = VH, DV, DK

    def verify(self, solver: str, *, dot_prec: str, block_v: int, out_dtype, store_state: bool, d_terms: int | None = None):
        d = self.depth if d_terms is None else d_terms
        out, U, cumg, st = K.launch_factor_verify(
            q=self.q, k=self.k, v=self.v, a=self.a, b=self.b, A_log=self.A_log, dt_bias=self.dt_bias,
            h0_bank=self.h0, h0_rows=self.h0_rows, strict_mask=self.strict, visible_mask=self.visible,
            n_actual=self.n, n_pad=self.n_pad, output_scale=self.output_scale, solver=solver,
            d_terms=d, dot_prec=dot_prec, block_v=block_v, out_dtype=out_dtype, store_state=store_state)
        torch.cuda.synchronize()

        def launch():
            K.launch_factor_verify(
                q=self.q, k=self.k, v=self.v, a=self.a, b=self.b, A_log=self.A_log, dt_bias=self.dt_bias,
                h0_bank=self.h0, h0_rows=self.h0_rows, strict_mask=self.strict, visible_mask=self.visible,
                n_actual=self.n, n_pad=self.n_pad, output_scale=self.output_scale, solver=solver,
                d_terms=d, dot_prec=dot_prec, block_v=block_v, out_dtype=out_dtype, store_state=False,
                out=out, u_out=U, cumg_out=cumg)
        return out[:, :self.n], U, cumg, (st[:, :self.n] if st is not None else None), launch

    def commit(self, U: torch.Tensor, cumg: torch.Tensor, targets: list[int], *, dot_prec: str, block_v: int):
        dev = self.q.device
        dst = torch.zeros_like(self.h0)
        tgt = torch.tensor(targets, dtype=torch.int32, device=dev)
        rows = torch.arange(self.B, dtype=torch.int32, device=dev)

        def launch():
            K.launch_compact_commit(k=self.k, u=U, cumg=cumg, visible_mask=self.visible, h0_bank=self.h0,
                                    h0_rows=self.h0_rows, target_node=tgt, dst_bank=dst, dst_rows=rows,
                                    n_pad=self.n_pad, dot_prec=dot_prec, block_v=block_v)
        launch()
        torch.cuda.synchronize()
        return dst.clone(), launch


# ----------------------------------------------------------------------------- timing
def time_launch(fn, *, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    per = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); e.synchronize()
        per.append(s.elapsed_time(e) * 1000.0)
    per.sort()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record(); e.synchronize()
    return {"iters": iters, "warmup": warmup,
            "sync_median_us": per[len(per) // 2], "sync_p10_us": per[len(per) // 10], "sync_p90_us": per[(9 * len(per)) // 10],
            "sync_min_us": per[0], "pipelined_mean_us": s.elapsed_time(e) * 1000.0 / iters}


def matmul_probe(dev, n=4096, iters=20) -> float:
    a = torch.randn((n, n), device=dev, dtype=torch.float16)
    b = torch.randn((n, n), device=dev, dtype=torch.float16)
    for _ in range(3):
        a @ b
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        a @ b
    e.record(); e.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters


# ----------------------------------------------------------------------------- per-payload evaluation
def evaluate_payload(pay_cpu: C.Payload, mods: dict, args, dev) -> dict:
    prod, leg, native_sg, native_pk = mods["prod"], mods["legacy"], mods["native_sg"], mods["native_pk"]
    pay = pay_cpu.to(dev)
    n = pay.n
    n_pad = max(16, 1 << (n - 1).bit_length())
    parents = pay.parents
    paths = C.all_accept_paths(parents)
    depth = C.max_strict_ancestors(parents)
    res = {"label": pay.label, "synthetic": pay.synthetic, "provenance": C.to_jsonable(pay.provenance),
           "parents": parents, "n": n, "n_pad": n_pad, "depth": depth}
    t0 = time.time()

    # ---- oracle (CPU fp64) + fp32 torch mirrors of the factors
    ops64 = C.lift(pay_cpu, torch.float64)
    orc = C.sequential_tree(ops64, checkpoint_parent=False)
    sys64 = C.build_system(ops64)
    res["gates"] = C.gate_stats(ops64, sys64)
    o_out, o_state, o_u = orc.out.to(dev), orc.state.to(dev), orc.u.to(dev)
    Uref32 = o_u.float().permute(1, 0, 2).contiguous()  # (VH, N, DV) oracle factors rounded to fp32
    res["t_oracle_s"] = time.time() - t0

    # ---- native references per root->node chain
    sg_out16 = torch.empty_like(o_out, dtype=torch.bfloat16); sg_out32 = torch.empty_like(o_out, dtype=torch.float32)
    sg_state = torch.empty_like(o_state, dtype=torch.float32)
    pk_out16 = torch.empty_like(sg_out16); pk_out32 = torch.empty_like(sg_out32); pk_state = torch.empty_like(sg_state)
    for path in paths:
        a = path[-1]
        o16, st = native_sg_path(native_sg, pay, path, fp32_io=False)
        o32, st32 = native_sg_path(native_sg, pay, path, fp32_io=True)
        sg_out16[a], sg_out32[a], sg_state[a] = o16[-1], o32[-1], st[-1]
        if not bits_equal(st, st32):
            res.setdefault("notes", []).append(f"native_sg fp32-io state differs from bf16-io state on path to {a}")
        p16, pst = native_packed_path(native_pk, pay, path, fp32_io=False)
        p32, _ = native_packed_path(native_pk, pay, path, fp32_io=True)
        pk_out16[a], pk_out32[a], pk_state[a] = p16[-1], p32[-1], pst[-1]
    torch.cuda.synchronize()
    res["references"] = {
        "native_sg_out32_vs_oracle": C.compare(sg_out32, o_out, ulp="fp32"),
        "native_sg_out16_vs_oracle": C.compare(sg_out16, o_out, ulp="bf16"),
        "native_sg_state_vs_oracle": C.compare(sg_state, o_state, ulp="fp32"),
        "native_pk_out32_vs_oracle": C.compare(pk_out32, o_out, ulp="fp32"),
        "native_pk_state_vs_oracle": C.compare(pk_state, o_state, ulp="fp32"),
        "native_pk_vs_native_sg_state": C.compare(pk_state, sg_state, ulp="fp32"),
        "native_pk_vs_native_sg_out16_exact_frac": exact_frac(pk_out16, sg_out16),
        "native_sg_out16_equals_round(out32)": exact_frac(sg_out16, sg_out32.to(torch.bfloat16)),
    }

    # ---- A: production scan (out) + replay (commit)
    a_out16, launch_scan16 = prod_scan(prod, pay, n_pad, torch.bfloat16)
    a_out32, launch_scan32 = prod_scan(prod, pay, n_pad, torch.float32)
    a_commit = torch.empty_like(o_state)
    replay_launchers = {}
    for path in paths:
        a = path[-1]
        st, launch_rep = prod_replay(prod, pay, path[1:], n_pad)
        a_commit[a] = st
        replay_launchers[a] = launch_rep
    # ---- A_legacy scan with state export; B_legacyWY two modes
    al_out16, al_state, launch_leg_scan = legacy_scan(leg, pay, n_pad, use_wy=False, bf16_boundaries=False, out_dtype=torch.bfloat16)
    al_out32, _, _ = legacy_scan(leg, pay, n_pad, use_wy=False, bf16_boundaries=False, out_dtype=torch.float32)
    wy_out16, wy_state, launch_wy = legacy_scan(leg, pay, n_pad, use_wy=True, bf16_boundaries=False, out_dtype=torch.bfloat16)
    wy_out32, _, _ = legacy_scan(leg, pay, n_pad, use_wy=True, bf16_boundaries=False, out_dtype=torch.float32)
    wyb_out16, wyb_state, launch_wyb = legacy_scan(leg, pay, n_pad, use_wy=True, bf16_boundaries=True, out_dtype=torch.bfloat16)
    wyb_out32, _, _ = legacy_scan(leg, pay, n_pad, use_wy=True, bf16_boundaries=True, out_dtype=torch.float32)

    # ---- B_fs / C_nm (new) — ieee and tf32 dot precision; fp32 and bf16 out stores
    fb = FactorBatch([pay], n_pad)
    new = {}
    launchers_new = {}
    for solver, tag in (("fs", "B_fs"), ("neumann", "C_nm")):
        for prec in args.dot_precs:
            out32, U, cumg, st_diag, launch_v = fb.verify(solver, dot_prec=prec, block_v=args.block_v, out_dtype=torch.float32, store_state=True)
            out16, U16, _, _, _ = fb.verify(solver, dot_prec=prec, block_v=args.block_v, out_dtype=torch.bfloat16, store_state=False)
            commits_own = torch.empty_like(o_state); commits_ref = torch.empty_like(o_state)
            for path in paths:
                a = path[-1]
                dst, launch_c = fb.commit(U, cumg, [a], dot_prec=prec, block_v=args.block_v)
                commits_own[a] = dst[0]
                dst_r, _ = fb.commit(Uref32.permute(1, 0, 2).contiguous().unsqueeze(0), cumg, [a], dot_prec=prec, block_v=args.block_v)
                commits_ref[a] = dst_r[0]
                if a == paths[-1][-1]:
                    launchers_new[f"{tag}[{prec}]_commit_deepest"] = launch_c
            launchers_new[f"{tag}[{prec}]_verify"] = launch_v
            new[f"{tag}[{prec}]"] = {
                "out32": out32[0], "out16": out16[0], "U": U[0].permute(1, 0, 2).contiguous(), "cumg": cumg[0],
                "state_diag": st_diag[0], "commit_own": commits_own, "commit_ref": commits_ref,
                "U16_equals_U": bits_equal(U16, U),
            }
    # Neumann truncation sensitivity: d-1 terms must NOT be exact (checks the nilpotency argument is load-bearing)
    if depth >= 1:
        out_trunc, U_trunc, _, _, _ = fb.verify("neumann", dot_prec=args.dot_precs[0], block_v=args.block_v, out_dtype=torch.float32, store_state=False, d_terms=depth - 1)
        res["neumann_truncation_control"] = {"d_terms": depth - 1, "U_vs_oracle": C.compare(U_trunc[0].permute(1, 0, 2), o_u, ulp="fp32")["max_abs"],
                                             "out_vs_oracle": C.compare(out_trunc[0], o_out, ulp="fp32")["max_abs"]}
    # padding control (N_PAD 32) and sibling-reorder control for the new kernels
    fb32 = FactorBatch([pay], 32)
    ctrl = {}
    for solver, tag in (("fs", "B_fs"), ("neumann", "C_nm")):
        prec = args.dot_precs[0]
        o32b, Ub, cgb, _, _ = fb32.verify(solver, dot_prec=prec, block_v=args.block_v, out_dtype=torch.float32, store_state=False)
        ref = new[f"{tag}[{prec}]"]
        ctrl[f"{tag}_npad32_vs_npad16"] = {"out_exact_frac": exact_frac(o32b[0], ref["out32"]), "U_exact_frac": exact_frac(Ub[0].permute(1, 0, 2), ref["U"]),
                                          "out_max_abs": (o32b[0].double() - ref["out32"].double()).abs().max().item()}
    new_par, perm = C.sibling_reorder_permutation(parents, "reverse")
    pay_perm = pay.permuted(perm, new_par)
    inv = torch.empty(n, dtype=torch.long, device=dev); inv[torch.tensor(perm, device=dev)] = torch.arange(n, device=dev)
    fbp = FactorBatch([pay_perm], n_pad)
    for solver, tag in (("fs", "B_fs"), ("neumann", "C_nm")):
        prec = args.dot_precs[0]
        o32p, Up, _, _, _ = fbp.verify(solver, dot_prec=prec, block_v=args.block_v, out_dtype=torch.float32, store_state=False)
        back_o = o32p[0].index_select(0, inv); back_U = Up[0].permute(1, 0, 2).index_select(0, inv)
        ref = new[f"{tag}[{prec}]"]
        ctrl[f"{tag}_sibling_reverse"] = {"out_exact_frac": exact_frac(back_o, ref["out32"]), "U_exact_frac": exact_frac(back_U, ref["U"]),
                                         "out_max_abs": (back_o.double() - ref["out32"].double()).abs().max().item(),
                                         "U_vs_oracle_max_abs": C.compare(back_U, o_u)["max_abs"]}
    ap_out32, _ = prod_scan(prod, pay_perm, n_pad, torch.float32)
    ctrl["A_prod_sibling_reverse"] = {"out_exact_frac": exact_frac(ap_out32.index_select(0, inv), a_out32),
                                      "out_max_abs": (ap_out32.index_select(0, inv).double() - a_out32.double()).abs().max().item()}
    res["controls"] = ctrl

    # ---- stage 1: verifier outputs & factors
    s1 = {}
    s1["A_prod_out32_vs_oracle"] = C.compare(a_out32, o_out, ulp="fp32")
    s1["A_prod_out16_vs_oracle"] = C.compare(a_out16, o_out, ulp="bf16")
    s1["A_prod_out16_vs_native_sg16_exact_frac"] = exact_frac(a_out16, sg_out16)
    s1["A_prod_out32_vs_native_sg32"] = C.compare(a_out32, sg_out32, ulp="fp32")
    s1["A_legacy_out32_vs_oracle"] = C.compare(al_out32, o_out, ulp="fp32")
    s1["A_legacy_out16_vs_A_prod_out16_exact_frac"] = exact_frac(al_out16, a_out16)
    s1["A_legacy_state_vs_oracle"] = C.compare(al_state, o_state, ulp="fp32")
    s1["A_legacy_state_vs_native_sg_state"] = C.compare(al_state, sg_state, ulp="fp32")
    if pay.serving_state is not None:
        s1["A_legacy_state_bytes_equal_payload_serving_state"] = bits_equal(al_state, pay.serving_state[:n].float())
        s1["A_prod_out16_bytes_equal_payload_serving_out"] = bits_equal(a_out16, pay.serving_out[:n]) if pay.serving_out is not None else None
    s1["B_legacyWY_fp32closed_out32_vs_oracle"] = C.compare(wy_out32, o_out, ulp="fp32")
    s1["B_legacyWY_fp32closed_state_vs_oracle"] = C.compare(wy_state, o_state, ulp="fp32")
    s1["B_legacyWY_fp32closed_state_vs_native_sg"] = C.compare(wy_state, sg_state, ulp="fp32")
    s1["B_legacyWY_bf16bnd_out32_vs_oracle"] = C.compare(wyb_out32, o_out, ulp="fp32")
    s1["B_legacyWY_bf16bnd_state_vs_oracle"] = C.compare(wyb_state, o_state, ulp="fp32")
    for tag, d in new.items():
        s1[f"{tag}_U_vs_oracle_u"] = C.compare(d["U"], o_u, ulp="fp32")
        s1[f"{tag}_out32_vs_oracle"] = C.compare(d["out32"], o_out, ulp="fp32")
        s1[f"{tag}_out16_vs_oracle"] = C.compare(d["out16"], o_out, ulp="bf16")
        s1[f"{tag}_out16_vs_native_sg16_exact_frac"] = exact_frac(d["out16"], sg_out16)
        s1[f"{tag}_out32_vs_native_sg32"] = C.compare(d["out32"], sg_out32, ulp="fp32")
        s1[f"{tag}_statediag_vs_oracle"] = C.compare(d["state_diag"], o_state, ulp="fp32")
    res["stage1_verify"] = s1

    # ---- stage 2: commit from IDENTICAL oracle factors; stage 3: own factors + own commit
    s2, s3 = {}, {}
    s3["A_prod_replay_vs_oracle"] = C.compare(a_commit, o_state, ulp="fp32")
    s3["A_prod_replay_vs_native_sg_state"] = C.compare(a_commit, sg_state, ulp="fp32")
    s3["A_prod_replay_exact_frac_vs_native_sg"] = exact_frac(a_commit, sg_state)
    s3["A_prod_replay_vs_native_pk_state"] = C.compare(a_commit, pk_state, ulp="fp32")
    s3["A_prod_replay_vs_A_legacy_export_exact_frac"] = exact_frac(a_commit, al_state)
    s3["B_legacyWY_fp32closed_state_as_commit_vs_oracle"] = s1["B_legacyWY_fp32closed_state_vs_oracle"]
    for tag, d in new.items():
        s2[f"{tag}_compact_with_oracle_factors_vs_oracle"] = C.compare(d["commit_ref"], o_state, ulp="fp32")
        s2[f"{tag}_compact_with_oracle_factors_vs_native_sg"] = C.compare(d["commit_ref"], sg_state, ulp="fp32")
        s3[f"{tag}_compact_own_vs_oracle"] = C.compare(d["commit_own"], o_state, ulp="fp32")
        s3[f"{tag}_compact_own_vs_native_sg"] = C.compare(d["commit_own"], sg_state, ulp="fp32")
        s3[f"{tag}_compact_own_vs_native_pk"] = C.compare(d["commit_own"], pk_state, ulp="fp32")
        s3[f"{tag}_compact_own_equals_statediag"] = bits_equal(d["commit_own"], d["state_diag"])
    # per-depth breakdown of committed-state error (vs oracle) for A and the new routes
    per_depth = {}
    for a in range(n):
        dpt = C.depth_of(parents, a)
        row = per_depth.setdefault(dpt, {})
        row.setdefault("A_prod_replay", []).append(C.compare(a_commit[a], o_state[a])["max_abs"])
        for tag, d in new.items():
            row.setdefault(f"{tag}_own", []).append(C.compare(d["commit_own"][a], o_state[a])["max_abs"])
            row.setdefault(f"{tag}_oraclefactors", []).append(C.compare(d["commit_ref"][a], o_state[a])["max_abs"])
        row.setdefault("native_sg", []).append(C.compare(sg_state[a], o_state[a])["max_abs"])
        row.setdefault("native_pk", []).append(C.compare(pk_state[a], o_state[a])["max_abs"])
    res["stage2_commit_identical_factors"] = s2
    res["stage3_own_factors_own_commit"] = s3
    res["commit_maxabs_by_depth"] = {str(k): {kk: max(vv) for kk, vv in v.items()} for k, v in per_depth.items()}

    # ---- scratch/transient memory accounting (bytes; analytic from shapes)
    VH, DV, DK = pay.h0.shape
    KH = pay.q.shape[1]
    res["transient_memory_bytes"] = {
        "A_prod_scan_hbm_per_node_state_export": 0,
        "A_prod_scan_register_h_cache_per_program": n_pad * args.block_v_prod * DK * 4,
        "A_prod_replay_hbm_written_rows(depth+1)": (depth + 1) * VH * DV * DK * 4,
        "A_prod_activation_ring(k,v,a,b)": n_pad * (KH * DK * 2 + VH * DV * 2 + VH * 2 * 2),
        "A_legacy_state_export_all_nodes": n_pad * VH * DV * DK * 4,
        "B_legacyWY_state_export_all_nodes": n_pad * VH * DV * DK * 4,
        "BC_compact_factors_U+cumg": n_pad * VH * DV * 4 + n_pad * VH * 4,
        "BC_compact_commit_hbm_written_rows": VH * DV * DK * 4,
        "one_full_state_row": VH * DV * DK * 4,
    }

    # ---- timing (labeled by contention record; verify and commit separately)
    if args.timing:
        tm = {"probe_matmul4096_fp16_us_before": matmul_probe(dev)}
        it, wu = args.iters, args.warmup
        tm["A_prod_scan_verify_out16"] = time_launch(launch_scan16, iters=it, warmup=wu)
        deepest = paths[-1][-1]
        tm["A_prod_replay_commit_deepest"] = time_launch(replay_launchers[deepest], iters=it, warmup=wu)
        tm["A_prod_replay_commit_zero_accept"] = time_launch(replay_launchers[0], iters=it, warmup=wu)
        tm["A_legacy_scan_with_state_export"] = time_launch(launch_leg_scan, iters=it, warmup=wu)
        tm["B_legacyWY_fp32closed_with_state_export"] = time_launch(launch_wy, iters=it, warmup=wu)
        tm["B_legacyWY_bf16bnd_with_state_export"] = time_launch(launch_wyb, iters=it, warmup=wu)
        for k_, fn in launchers_new.items():
            tm[k_] = time_launch(fn, iters=it, warmup=wu)
        spine = paths[-1]
        tm["native_sg_chain_depth%d_verify(context)" % (len(spine) - 1)] = time_launch(lambda: native_sg_path(native_sg, pay, spine, fp32_io=False), iters=it, warmup=wu)
        tm["native_pk_one_token(context)"] = time_launch(lambda: native_packed_path(native_pk, pay, [0], fp32_io=False), iters=it, warmup=wu)
        tm["probe_matmul4096_fp16_us_after"] = matmul_probe(dev)
        res["timing_us"] = tm
    res["elapsed_s"] = time.time() - t0
    return res


def run_b4(args, mods, dev) -> dict:
    """B4 arm for the batched factor kernels: 4 SYNTHETIC requests sharing layer params + topology.
    (The production plain-route scan has no batched kernel outside fixed32 mode: B4 = 4 launches.)"""
    parents = list(C.CATERPILLAR_10)
    base = C.synth_payload(parents, args.seed, regime="historical-like")
    pays = [base] + [C.synth_payload(parents, args.seed + i, regime="historical-like") for i in range(1, 4)]
    for p in pays[1:]:
        p.A_log = base.A_log.clone(); p.dt_bias = base.dt_bias.clone()
    pays_dev = [p.to(dev) for p in pays]
    fb = FactorBatch(pays_dev, 16)
    res = {"label": "B4 synthetic caterpillar (shared layer params)", "synthetic": True, "n": fb.n, "depth": fb.depth}
    per = {}
    for solver, tag in (("fs", "B_fs"), ("neumann", "C_nm")):
        out32, U, cumg, _, launch_v = fb.verify(solver, dot_prec=args.dot_precs[0], block_v=args.block_v, out_dtype=torch.float32, store_state=False)
        targets = [9, 7, 5, 0]
        dst, launch_c = fb.commit(U, cumg, targets, dot_prec=args.dot_precs[0], block_v=args.block_v)
        errs = []
        for i, p in enumerate(pays):
            orc = C.sequential_tree(C.lift(p, torch.float64), checkpoint_parent=False)
            errs.append({"request": i, "out32_vs_oracle": C.compare(out32[i], orc.out.to(dev), ulp="fp32")["max_abs"],
                         "U_vs_oracle": C.compare(U[i].permute(1, 0, 2), orc.u.to(dev), ulp="fp32")["max_abs"],
                         "commit_target": targets[i], "commit_vs_oracle": C.compare(dst[i], orc.state[targets[i]].to(dev), ulp="fp32")["max_abs"]})
        # batched result must equal the B1 result for the same request bit-for-bit (batch invariance)
        fb1 = FactorBatch([pays_dev[2]], 16)
        o1, U1, _, _, _ = fb1.verify(solver, dot_prec=args.dot_precs[0], block_v=args.block_v, out_dtype=torch.float32, store_state=False)
        per[tag] = {"per_request": errs, "B4_vs_B1_bitwise_request2": {"out": bits_equal(o1[0], out32[2]), "U": bits_equal(U1[0], U[2])}}
        if args.timing:
            per[tag]["timing_us_B4_verify"] = time_launch(launch_v, iters=args.iters, warmup=args.warmup)
            per[tag]["timing_us_B4_commit"] = time_launch(launch_c, iters=args.iters, warmup=args.warmup)
    if args.timing:
        prod = mods["prod"]
        launches = [prod_scan(prod, p, 16, torch.bfloat16)[1] for p in pays_dev]

        def four():
            for fn in launches:
                fn()
        per["A_prod_scan_4x_launches_timing_us"] = time_launch(four, iters=args.iters, warmup=args.warmup)
    res["results"] = per
    return res


def manifest(args, mods_paths: dict, dev) -> dict:
    cache = Path(os.environ.get("TRITON_CACHE_DIR", ""))
    cubins = {}
    if cache.is_dir():
        for p in sorted(cache.rglob("*.cubin")):
            cubins[str(p.relative_to(cache))] = sha256_file(p)
    try:
        drv = Path("/proc/driver/nvidia/version").read_text().strip().splitlines()[0]
    except Exception:
        drv = None
    return {
        "schema": "e7a.device_manifest.v1",
        "start_utc": os.environ.get("E7A_START_UTC"),
        "image_digest": os.environ.get("E7A_IMAGE_DIGEST"),
        "torch": torch.__version__, "triton": triton.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0), "driver": drv,
        "source_sha256": {k: sha256_file(Path(v)) for k, v in mods_paths.items()},
        "flags": {"FIXED32_MODE": mods_paths.get("_fixed32_mode"), "scan_align_on": mods_paths.get("_scan_align"),
                  "npad_invariant_on": mods_paths.get("_npad_inv"), "parent_gather_on": mods_paths.get("_pg"),
                  "hc_internal_on": mods_paths.get("_hc"), "BV_prod": mods_paths.get("_bv"), "num_warps_prod": mods_paths.get("_warps"),
                  "env_FR13_FR10": {k: v for k, v in os.environ.items() if k.startswith(("FR13_", "FR10_", "FR12_"))}},
        "args": vars(args),
        "triton_cache_cubins_sha256": cubins,
        "timing_label": "SHARED-DEVICE DIAGNOSTIC unless telemetry/ shows an uncontended window (unrelated vLLM server resident on this GB10)",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payloads", nargs="*", default=[], help="historical payload .pt paths (mounted under /hist)")
    ap.add_argument("--synthetic", nargs="*", default=[], help="chain:<depth> | binary:<depth> | caterpillar  (labeled synthetic)")
    ap.add_argument("--regime", default="historical-like")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--dot-precs", nargs="*", default=["ieee", "tf32"])
    ap.add_argument("--block-v", type=int, default=16)
    ap.add_argument("--block-v-prod", type=int, default=16)
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--b4", action="store_true")
    ap.add_argument("--out-dir", default=os.environ.get("E7A_OUT_DIR", "."))
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    prod = load_module(PROD_PATH, "e7a_prod_kernel")
    leg = load_module(LEGACY_PATH, "e7a_legacy_wy")
    from vllm.model_executor.layers.fla.ops import (  # noqa: E402
        fused_recurrent_gated_delta_rule_packed_decode as native_pk,
        fused_sigmoid_gating_delta_rule_update as native_sg,
    )
    import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as _sgmod
    import vllm.model_executor.layers.fla.ops.fused_recurrent as _frmod
    mods = {"prod": prod, "legacy": leg, "native_sg": native_sg, "native_pk": native_pk}
    mods_paths = {"prod_kernel": str(PROD_PATH), "legacy_wy": str(LEGACY_PATH), "e7a_core": str(HERE / "e7a_core.py"),
                  "e7a_kernels": str(HERE / "e7a_kernels.py"), "e7a_device": str(HERE / "e7a_device.py"),
                  "native_fused_sigmoid_gating": _sgmod.__file__, "native_fused_recurrent": _frmod.__file__,
                  "_fixed32_mode": prod._FR13_FIXED32_MODE, "_scan_align": prod.scan_align_on(), "_npad_inv": prod.npad_invariant_on(),
                  "_pg": prod.parent_gather_on(), "_hc": prod.hc_internal_on(), "_bv": prod.BV, "_warps": prod._DEPLOYED_NUM_WARPS}
    src_paths = {k: v for k, v in mods_paths.items() if not k.startswith("_")}

    results = []
    for p in args.payloads:
        pay = C.load_payload(p)
        print(f"[e7a] payload {pay.label} ({p})", flush=True)
        r = evaluate_payload(pay, mods, args, dev)
        results.append(r)
        (out_dir / f"result_{len(results):02d}_{pay.label.replace('.', '_').replace('/', '_')}.json").write_text(json.dumps(C.to_jsonable(r), indent=1))
        print(f"[e7a]   done in {r['elapsed_s']:.1f}s", flush=True)
    for spec in args.synthetic:
        kind, _, arg = spec.partition(":")
        parents = {"chain": lambda: C.chain_parents(int(arg)), "binary": lambda: C.binary_parents(int(arg)),
                   "caterpillar": lambda: list(C.CATERPILLAR_10)}[kind]()
        pay = C.synth_payload(parents, args.seed, regime=args.regime)
        print(f"[e7a] synthetic {spec} regime={args.regime}", flush=True)
        r = evaluate_payload(pay, mods, args, dev)
        results.append(r)
        (out_dir / f"result_{len(results):02d}_synthetic_{kind}{arg}_{args.regime}.json").write_text(json.dumps(C.to_jsonable(r), indent=1))
        print(f"[e7a]   done in {r['elapsed_s']:.1f}s", flush=True)
    if args.b4:
        r = run_b4(args, mods, dev)
        (out_dir / "result_b4_synthetic.json").write_text(json.dumps(C.to_jsonable(r), indent=1))
        results.append(r)
    man = manifest(args, {**src_paths, **{k: v for k, v in mods_paths.items() if k.startswith("_")}}, dev)
    (out_dir / "manifest.json").write_text(json.dumps(C.to_jsonable(man), indent=1))
    # compact stdout digest
    for r in results:
        if "stage1_verify" not in r:
            continue
        s1, s3 = r["stage1_verify"], r["stage3_own_factors_own_commit"]
        print(f"\n== {r['label']} depth={r['depth']} P_min={r['gates']['P_min']:.3g}")
        for k in ("A_prod_out32_vs_oracle", "B_legacyWY_fp32closed_out32_vs_oracle", "B_fs[ieee]_out32_vs_oracle", "C_nm[ieee]_out32_vs_oracle", "B_fs[tf32]_out32_vs_oracle", "C_nm[tf32]_out32_vs_oracle"):
            if k in s1:
                print(f"  verify {k:44s} max_abs={s1[k]['max_abs']:.3e} rel={s1[k]['max_rel_to_refmax']:.3e} ulp32sig_max={s1[k].get('ulp32_max_sig')}")
        for k in ("A_prod_out16_vs_native_sg16_exact_frac", "B_fs[ieee]_out16_vs_native_sg16_exact_frac", "C_nm[ieee]_out16_vs_native_sg16_exact_frac"):
            if k in s1:
                print(f"  out16 exact vs native_sg {k:40s} {s1[k]:.4f}")
        for k in ("A_prod_replay_vs_oracle", "B_fs[ieee]_compact_own_vs_oracle", "C_nm[ieee]_compact_own_vs_oracle", "B_fs[tf32]_compact_own_vs_oracle", "C_nm[tf32]_compact_own_vs_oracle"):
            if k in s3:
                print(f"  commit {k:44s} max_abs={s3[k]['max_abs']:.3e} rel={s3[k]['max_rel_to_refmax']:.3e} ulp32sig_max={s3[k].get('ulp32_max_sig')}")
        if "timing_us" in r:
            for k, v in r["timing_us"].items():
                if isinstance(v, dict):
                    print(f"  time {k:48s} median={v['sync_median_us']:8.1f}us pipelined={v['pipelined_mean_us']:8.1f}us")
                else:
                    print(f"  time {k:48s} {v:.1f}us")
    print("[e7a] wrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
