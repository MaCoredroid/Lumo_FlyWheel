"""Offline byte gate: FR13_CONV_PREGATHER staged rows == per-layer index_select."""
import sys, torch
sys.path.insert(0, "/workspace/src")
from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
    launch_conv_col0_pregather, conv_col0_staged)
torch.manual_seed(7)
dev = "cuda"
L, ROWS, C, W, B = 48, 64, 4096, 4, 4
banks = [torch.randn(ROWS, C, W, device=dev, dtype=torch.bfloat16) for _ in range(L)]
ssi = torch.randint(0, ROWS, (L, B, 22), device=dev, dtype=torch.int32)
tok = ("r1", "r2", "r3", "r4")
launch_conv_col0_pregather(conv_banks=banks, ssi_stack=ssi[:, :, 0].contiguous(),
                           num_spec_decodes=B, req_ids_token=tok)
torch.cuda.synchronize()
ok = True
for l in range(L):
    ref = torch.index_select(banks[l], 0, ssi[l, :B, 0].to(torch.long)).reshape(B, -1)
    got = conv_col0_staged(tok, l)
    if got is None or not torch.equal(ref, got):
        print(f"FAIL layer {l}"); ok = False; break
print("stale-token returns None:", conv_col0_staged(("other",), 0) is None)
print("GATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
