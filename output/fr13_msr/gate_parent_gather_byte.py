"""Offline in-process byte gate: PARENT_GATHER=True scan vs deployed False scan."""
import os, sys, torch
sys.path.insert(0, "/workspace/src")
import lumo_flywheel_serving.fr10_gdn_tree_kernel as K

def run(flag, parent21, n_pad, geom):
    os.environ["FR13_PARENT_GATHER"] = "1" if flag else "0"
    os.environ.pop("FR13_PARENT_GATHER_SELFCHECK", None)
    if geom: os.environ["FR13_TREE_GDN_GEOM_OVERRIDE"] = geom
    else: os.environ.pop("FR13_TREE_GDN_GEOM_OVERRIDE", None)
    torch.manual_seed(1313)
    dev = "cuda"; n = len(parent21); KH, VH, Kd, Vd = 16, 32, 128, 128
    strict = torch.zeros((n_pad, n_pad), dtype=torch.int32, device=dev)
    for i, p in enumerate(parent21):
        j = p
        while j >= 0:
            strict[i, j] = 1; j = parent21[j]
    visible = strict.clone()
    for i in range(n): visible[i, i] = 1
    q = torch.randn(n, KH, Kd, dtype=torch.bfloat16, device=dev)
    k = torch.randn(n, KH, Kd, dtype=torch.bfloat16, device=dev)
    v = torch.randn(n, VH, Vd, dtype=torch.bfloat16, device=dev)
    g = torch.randn(n, VH, dtype=torch.float32, device=dev)
    beta = torch.rand(n, VH, dtype=torch.float32, device=dev)
    ra = torch.randn(n, VH, dtype=torch.float32, device=dev)
    rb = torch.randn(n, VH, dtype=torch.float32, device=dev)
    A = torch.randn(VH, dtype=torch.float32, device=dev)
    dtb = torch.randn(VH, dtype=torch.float32, device=dev)
    h0 = torch.randn(VH, Vd, Kd, dtype=torch.float32, device=dev)
    out = torch.zeros(n, VH, Vd, dtype=torch.bfloat16, device=dev)
    K.launch_tree_gdn_prepared(
        out=out, q=q, k=k, v=v, g=g, beta=beta, h0=h0, n_actual=n, n_pad=n_pad,
        strict_mask=strict, visible_mask=visible, output_scale=Kd**-0.5,
        use_qk_l2norm_in_kernel=True, raw_a=ra, raw_b=rb, A_log=A, dt_bias=dtb)
    torch.cuda.synchronize()
    return out.clone()

parent21 = [-1, -1, -1, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9, 12, 15, 16, 17, 18, 19]
ok = True
for name, tree, npad, geom in (("tail6-21n-BV8", parent21, 32, "BV=8"),):
    a = run(False, tree, npad, geom)
    b = run(True, tree, npad, geom)
    same = torch.equal(a, b)
    print(f"[{name}] byte-identical: {same}")
    if not same:
        d = (a.float()-b.float()).abs().max().item()
        print(f"  max_abs={d:.3e}"); ok = False
print("GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
