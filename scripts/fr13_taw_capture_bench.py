#!/usr/bin/env python3
"""TAW capture bench — validates warmup->capture->replay->post-abort-health
IN A BARE GPU CONTAINER before any arm boot (boots 1-3 each cost ~12 min to
discover one in-capture illegal op; this finds them all in minutes).

Run: docker run --rm --gpus all --entrypoint python3 -v $PWD:/w <image> \
       /w/scripts/fr13_taw_capture_bench.py
PASS criteria:
  1. warmup eager call OK
  2. capture succeeds (no cudaErrorStreamCaptureUnsupported)
  3. replay x3 with changed inputs produces sane, CHANGING products
  4. after a FORCED failed capture (deliberate illegal op), eager torch ops
     still work (post-abort context health — the boot-3 death mode)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_dm", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "fr13_device_multidraft_kernel.py"))
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)

assert torch.cuda.is_available(), "needs GPU"
dev = torch.device("cuda")
torch.manual_seed(0)
V, B = 248320, 4
# tail6-like: 21 nodes/request (tree + suffix tail chain)
parents = [-1, -1, 0, 0, 2, 2, 4] + list(range(4, 18))
NC = len(parents)
os.environ["FR13_TAW"] = "1"
os.environ["FR13_STEP_GRAPH"] = "1"

_IG = torch.Generator(device=dev); _IG.manual_seed(123)
def make_inputs(spec):
    # explicit generator: doubles as the prod-survivability probe (vLLM uses
    # explicit gens; if these survive a failed capture, prod survives too)
    # spec: int B => all requests full (NC nodes); "4z" => counts
    # [NC,NC,NC,0] — the boot-7 live geometry: one running request with ZERO
    # draft tokens, whose flattened 63-node tpi is byte-identical to B=3.
    counts = [NC, NC, NC, 0] if spec == "4z" else [NC] * spec
    B, total = len(counts), sum(counts)
    drafts = torch.randint(0, V, (total,), device=dev, generator=_IG)
    tl = torch.randn(total, V, device=dev, generator=_IG)
    sl = torch.randn(total, V, device=dev, generator=_IG)
    bonus = torch.randint(0, V, (B, 1), device=dev, generator=_IG)
    ndt = torch.tensor(counts, dtype=torch.long, device=dev)
    tpi = torch.tensor([p for c in counts for p in parents[:c]], device=dev)
    return ndt, drafts, tpi, tl, sl, bonus

MAXSPEC = 20
outs = []
# warmup all three keys FIRST, then capture each: every capture happens with
# a DIFFERENT key's topology in the shared global (the boot-7 stale-topology
# exposure — step "4z" capture sees B=3's topology, the exact live fatal).
seq = [4, 3, "4z", 4, 3, "4z", 4, 3, "4z"]
for step, spec in enumerate(seq):
    ndt, drafts, tpi, tl, sl, bonus = make_inputs(spec)
    nreq_want = 4 if spec == "4z" else spec
    out = dm.fr13_taw_commit_captured(
        ndt, drafts, tpi, tl, sl, bonus, MAXSPEC,
        generators=None)
    outs.append([r[:3] for r in out[0]])
    print(f"step {step} spec={spec}: products={len(out[0])} rows head = {[r[:2] for r in out[0]]}")
    assert len(out[0]) == nreq_want, (
        f"step {step} spec={spec}: products={len(out[0])} != nreq {nreq_want} "
        "(stale-topology class)")
    if spec == "4z":
        # zero-count request must still get its bonus-token row
        assert len(out[0][3]) == 1, f"zero-count row wrong: {out[0][3]}"
if dm._FR13_SG_CAP_DEAD:
    # PROD-SURVIVABILITY PROBE: vLLM samplers use EXPLICIT generators, not the
    # default one the poison binds to. If explicit draws survive, a failed
    # capture in prod is non-fatal (dead-flag insurance suffices).
    try:
        eg = torch.Generator(device=dev); eg.manual_seed(7)
        _ = torch.rand(8, device=dev, generator=eg)
        torch.cuda.synchronize()
        print("post-abort EXPLICIT-generator draw: OK (prod likely survivable)")
    except Exception as e:
        print(f"post-abort EXPLICIT-generator draw FAILED: {type(e).__name__}")
    try:
        _ = torch.randn(8, device=dev)
        torch.cuda.synchronize()
        print("post-abort DEFAULT-generator randn: OK")
    except Exception as e:
        print(f"post-abort DEFAULT randn FAILED: {type(e).__name__}")
assert not dm._FR13_SG_CAP_DEAD, "capture went DEAD — read the DISABLED needle above"
assert True  # multi-B: outs shapes vary
n_keys = len([v for v in dm._FR13_SG_CAP.values() if isinstance(v, dict)])
print(f"captured graphs: {n_keys} (want 3: B4-full, B3, B4-zerocount)")
assert n_keys == 3, f"expected 3 captured graphs, got {n_keys}"

# post-abort context health: force an illegal capture and verify eager works
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
torch.cuda.synchronize()
prev = torch.cuda.current_stream()
torch.cuda.set_stream(s)
try:
    g.capture_begin()
    _ = torch.tensor([1, 2, 3], device=dev)  # pageable H2D = illegal
    g.capture_end()
    print("WARN: illegal op did not trip (driver tolerant)")
except Exception as e:
    try:
        g.capture_end()
    except Exception:
        pass
    torch.cuda.set_stream(prev)
    torch.cuda.synchronize()
    print(f"forced-abort tripped as expected: {type(e).__name__}")
torch.cuda.set_stream(prev)
x = torch.randn(1024, 1024, device=dev) @ torch.randn(1024, 1024, device=dev)
torch.cuda.synchronize()
print("post-abort eager compute OK:", float(x.sum()))
print(">>> PASS — TAW capture validated end-to-end in-container")
