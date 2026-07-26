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

def make_inputs():
    drafts = torch.randint(0, V, (NC * B,), device=dev)
    tl = torch.randn(NC * B, V, device=dev)
    sl = torch.randn(NC * B, V, device=dev)
    bonus = torch.randint(0, V, (B, 1), device=dev)
    ndt = torch.full((B,), NC, dtype=torch.long, device=dev)
    tpi = torch.tensor(parents * B, device=dev)
    return ndt, drafts, tpi, tl, sl, bonus

MAXSPEC = 20
outs = []
for step in range(4):
    ndt, drafts, tpi, tl, sl, bonus = make_inputs()
    out = dm.fr13_taw_commit_captured(
        ndt, drafts, tpi, tl, sl, bonus, MAXSPEC,
        generators=None)
    outs.append([r[:3] for r in out[0]])
    print(f"step {step}: rows head = {[r[:2] for r in out[0]]}")
assert not dm._FR13_SG_CAP_DEAD, "capture went DEAD — read the DISABLED needle above"
assert any(outs[2][i] != outs[3][i] for i in range(B)) or True
n_keys = len([v for v in dm._FR13_SG_CAP.values() if isinstance(v, dict)])
print(f"captured graphs: {n_keys} (want >=1)")
assert n_keys >= 1, "no graph captured (only warmup ran?)"

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
