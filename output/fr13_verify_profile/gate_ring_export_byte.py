#!/usr/bin/env python3
"""Offline byte gate for FR13_RING_EXPORT (B1 in-kernel replay-ring staging).

Same-process, same inputs, two paths:
  REF: launch scan with rings=None, stage rings via the 4 aten .copy_() calls
       (byte-for-byte what the served default does today).
  NEW: launch scan with ring tensors (RING_EXPORT=True), no aten copies.
Gate: (1) scan out byte-identical REF vs NEW; (2) ring contents byte-identical
      REF-staged vs NEW kernel-staged, for rows [0:n_actual] on all 4 rings.
Served dims: 21-node tail6 (n_pad=32, BV=8 via FR13_TREE_GDN_GEOM_OVERRIDE),
plus a BV=16 16-row case (cat9-class). RAW_GATING path (A_log/dt_bias), bf16
k/v, fp32 a/b -- matching the deployed ring dtypes.
"""
import os
import sys

import torch

sys.path.insert(0, "/home/mark/shared/lumoFlyWheel/src")
from lumo_flywheel_serving.fr10_gdn_tree_kernel import launch_tree_gdn_prepared


def build_tree_masks(parent, n_pad, device):
    n = len(parent)
    strict = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=device)
    for i, p in enumerate(parent):
        j = p
        while j >= 0:
            strict[i, j] = 1
            j = parent[j]
    visible = strict.clone()
    for i in range(n):
        visible[i, i] = 1
    return strict, visible


def run_case(name, parent, n_pad, geom_env, seed=1313):
    device = "cuda"
    os.environ.pop("FR13_TREE_GDN_GEOM_OVERRIDE", None)
    if geom_env:
        os.environ["FR13_TREE_GDN_GEOM_OVERRIDE"] = geom_env
    torch.manual_seed(seed)
    n = len(parent)
    KH, VH, K, V = 16, 32, 128, 128
    strict, visible = build_tree_masks(parent, n_pad, device)
    q = torch.randn(n, KH, K, dtype=torch.bfloat16, device=device)
    k = torch.randn(n, KH, K, dtype=torch.bfloat16, device=device)
    v = torch.randn(n, VH, V, dtype=torch.bfloat16, device=device)
    g = torch.randn(n, VH, dtype=torch.float32, device=device)
    beta = torch.rand(n, VH, dtype=torch.float32, device=device)
    raw_a = torch.randn(n, VH, dtype=torch.float32, device=device)
    raw_b = torch.randn(n, VH, dtype=torch.float32, device=device)
    A_log = torch.randn(VH, dtype=torch.float32, device=device)
    dt_bias = torch.randn(VH, dtype=torch.float32, device=device)
    h0 = torch.randn(VH, V, K, dtype=torch.float32, device=device)

    def rings():
        return (
            torch.zeros(n_pad, KH, K, dtype=k.dtype, device=device),
            torch.zeros(n_pad, VH, V, dtype=v.dtype, device=device),
            torch.zeros(n_pad, VH, dtype=raw_a.dtype, device=device),
            torch.zeros(n_pad, VH, dtype=raw_b.dtype, device=device),
        )

    common = dict(
        q=q, k=k, v=v, g=g, beta=beta, h0=h0,
        n_actual=n, n_pad=n_pad, strict_mask=strict, visible_mask=visible,
        output_scale=K ** -0.5, use_qk_l2norm_in_kernel=True,
        raw_a=raw_a, raw_b=raw_b, A_log=A_log, dt_bias=dt_bias,
    )

    # REF: no ring export; stage rings the aten way (the served default)
    out_ref = torch.zeros(n, VH, V, dtype=torch.bfloat16, device=device)
    launch_tree_gdn_prepared(out=out_ref, **common)
    rk_ref, rv_ref, ra_ref, rb_ref = rings()
    rk_ref[:n].copy_(k)
    rv_ref[:n].copy_(v)
    ra_ref[:n].copy_(raw_a)
    rb_ref[:n].copy_(raw_b)

    # NEW: in-kernel ring export
    out_new = torch.zeros(n, VH, V, dtype=torch.bfloat16, device=device)
    rk, rv, ra, rb = rings()
    launch_tree_gdn_prepared(
        out=out_new, ring_k=rk, ring_v=rv, ring_a=ra, ring_b=rb, **common
    )
    torch.cuda.synchronize()

    ok = True
    if not torch.equal(out_ref[:n], out_new[:n]):
        d = (out_ref[:n].float() - out_new[:n].float()).abs().max().item()
        print(f"[{name}] FAIL out mismatch max_abs={d:.3e}")
        ok = False
    for tag, a_t, b_t in (
        ("ring_k", rk_ref[:n], rk[:n]),
        ("ring_v", rv_ref[:n], rv[:n]),
        ("ring_a", ra_ref[:n], ra[:n]),
        ("ring_b", rb_ref[:n], rb[:n]),
    ):
        if not torch.equal(a_t, b_t):
            d = (a_t.float() - b_t.float()).abs().max().item()
            nz = int((a_t != b_t).sum().item())
            print(f"[{name}] FAIL {tag} mismatch max_abs={d:.3e} n_diff={nz}")
            ok = False
    # padded rows must stay untouched (zeros) in the NEW path too
    for tag, r in (("ring_k", rk), ("ring_v", rv), ("ring_a", ra), ("ring_b", rb)):
        if n_pad > n and float(r[n:].abs().sum().item()) != 0.0:
            print(f"[{name}] FAIL {tag} padded rows written")
            ok = False
    print(f"[{name}] {'PASS' if ok else 'FAIL'} (n={n}, n_pad={n_pad}, geom={geom_env or 'default BV16'})")
    return ok


def main():
    # tail6 21-node: 3-wide comb depths 1-5 + chain tail depths 6-11
    parent21 = [-1, -1, -1, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9, 12, 15, 16, 17, 18, 19]
    # cat9-class 9-node: spine 5 + one sibling at depths 1-4
    parent9 = [-1, 0, 1, 2, 3, -1, 0, 1, 2]
    ok = True
    ok &= run_case("tail6-21n-BV8", parent21, 32, "BV=8")
    ok &= run_case("cat9-9n-BV16", parent9, 16, None)
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
