#!/usr/bin/env python3
"""Byte gate: _fr13_native_committer_all_layers_device's DEVICE-built layout
vs the CG body's host-built layout (acc.tolist + per-b python slicing), on
random inputs incl. zero-accept rows. Kernels stubbed (layout only). GPU.

Run: docker run --rm --gpus all --entrypoint python3 -v $PWD:/w \
       -e PYTHONPATH=/w/src <image> /w/scripts/fr13_s1_committer_device_fill_byte_gate.py
"""
import sys, types, torch

# stub the fused kernel import BEFORE importing the kernel lib caller path
_stub = types.ModuleType("vllm.model_executor.layers.fla.ops")
_stub.fused_sigmoid_gating_delta_rule_update = lambda **kw: None
for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers",
             "vllm.model_executor.layers.fla"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["vllm.model_executor.layers.fla.ops"] = _stub

from lumo_flywheel_serving import fr10_gdn_tree_kernel as tk  # noqa: E402

assert torch.cuda.is_available()
dev = torch.device("cuda")
g = torch.Generator().manual_seed(11)

L, RING, num_kh, dim_k, num_vh, dim_v = 4, 24, 2, 8, 2, 4
MAX_PATH = 16
fails = 0
for case in range(40):
    B = int(torch.randint(1, 5, (1,), generator=g))
    MAXT = B * MAX_PATH
    banks = [torch.zeros(64, num_vh, dim_k, dim_v, device=dev) for _ in range(L)]
    SCRATCH = 63
    k_rings = torch.randn(L, B, RING, num_kh, dim_k, generator=g).to(dev)
    v_rings = torch.randn(L, B, RING, num_vh, dim_v, generator=g).to(dev)
    a_rings = torch.randn(L, B, RING, num_vh, generator=g).to(dev)
    b_rings = torch.randn(L, B, RING, num_vh, generator=g).to(dev)
    paths = torch.randint(1, RING, (B, MAX_PATH), generator=g).to(dev)
    # lens 0..12 incl zero-accept rows
    lens = torch.randint(0, 13, (B,), generator=g).to(dev)
    ssi = torch.randint(0, 63, (L, B, 6), generator=g).to(dev, torch.int32)

    tk._FR13_GRAPH_COMMITTER.clear()
    tk._fr13_native_committer_all_layers_device(
        banks_list=banks, spec_state_indices=ssi,
        accepted_paths=paths, accepted_lens=lens,
        k_rings=k_rings, v_rings=v_rings, a_rings=a_rings, b_rings=b_rings,
        A_logs=[None] * L, dt_biases=[None] * L,
        num_layers=L, num_spec_decodes=B, output_scale=1.0,
        use_qk_l2norm_in_kernel=True, burn_node_bank=False, root_node=0)
    st = tk._FR13_GRAPH_COMMITTER[list(tk._FR13_GRAPH_COMMITTER.keys())[0]]

    # ---- host-layout expectation (CG body logic verbatim) ----
    acc = lens.tolist()
    seg = [1 + int(a) for a in acc]
    kexp = torch.zeros_like(st["kbuf"]); vexp = torch.zeros_like(st["vbuf"])
    aexp = torch.full_like(st["abuf"], -1e4); bexp = torch.zeros_like(st["bbuf"])
    ssiexp = torch.full_like(st["ssi"], SCRATCH)
    for b in range(B):
        nodes = torch.cat([
            torch.zeros(1, dtype=torch.long, device=dev),
            paths[b, : int(acc[b])].to(torch.long)])
        s0 = b * MAX_PATH
        nl = seg[b]
        kexp[:, s0:s0 + nl] = k_rings[:, b, nodes]
        vexp[:, s0:s0 + nl] = v_rings[:, b, nodes]
        aexp[:, s0:s0 + nl] = a_rings[:, b, nodes]
        bexp[:, s0:s0 + nl] = b_rings[:, b, nodes]
    ssiexp[:, :B, :] = ssi[:, :B, 0:1]
    cuexp = torch.arange(B + 1, device=dev, dtype=torch.int64) * MAX_PATH

    for name, gotten, want in (("kbuf", st["kbuf"], kexp), ("vbuf", st["vbuf"], vexp),
                               ("abuf", st["abuf"], aexp), ("bbuf", st["bbuf"], bexp),
                               ("ssi", st["ssi"], ssiexp),
                               ("cu", st["cu"].to(torch.int64), cuexp)):
        if not torch.equal(gotten, want):
            fails += 1
            d = (gotten != want)
            print(f"case {case} {name} MISMATCH B={B} lens={acc} "
                  f"nbad={int(d.sum())}")
            break

print(f"cases=40 fails={fails}")
assert fails == 0
print(">>> PASS — device layout byte-identical to CG host layout")
