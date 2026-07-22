#!/usr/bin/env python3
"""Offline byte gate for FR13_CONV_WB_FUSED (B2a fused conv-state write-back).

REF: new_state = fused_tree_conv_state_rows(source_z, state_src).to(dtype)
     conv_state.index_copy_(0, dst_rows.long(), new_state)
NEW: launch_conv_state_writeback(...)  (one kernel, gather->page-write)

Gate: ENTIRE conv_state backing buffer byte-identical (catches stray writes),
on a page-strided as_strided view mimicking the mamba cache layout, plus a
plain contiguous case. tail6 dims: tree_n=22 rows incl root? -- the served
call uses tree_n nodes (22 incl anchor per the map's [22, conv_dim, 24]);
we gate n=22 and n=9, conv_dim=8192, state_len=24, bf16.
"""
import sys

import torch

sys.path.insert(0, "/home/mark/shared/lumoFlyWheel/src")
from lumo_flywheel_serving.fr13_tree_conv_fused import (
    fused_tree_conv_state_rows,
    launch_conv_state_writeback,
)


def run_case(name, n, C, L, paged, seed=1313):
    torch.manual_seed(seed)
    device = "cuda"
    S = n + 4  # prior(3) + x(n) + zero(1)
    source_z = torch.randn(S, C, dtype=torch.bfloat16, device=device)
    state_src = torch.randint(0, S, (n * L,), dtype=torch.int64, device=device)
    num_rows = 64
    if paged:
        # page-strided view: each row lives at the head of a larger page
        page_elems = C * L * 3 + 512  # deliberately non-tight + misaligned-ish
        backing = torch.zeros(num_rows * page_elems, dtype=torch.bfloat16, device=device)
        def view_of(b):
            return torch.as_strided(b, (num_rows, C, L), (page_elems, L, 1))
    else:
        backing = torch.zeros(num_rows * C * L, dtype=torch.bfloat16, device=device)
        def view_of(b):
            return b.view(num_rows, C, L)
    # random pre-fill so untouched-row corruption is detectable
    backing.uniform_(-1, 1)
    backing_ref = backing.clone()
    backing_new = backing.clone()
    perm = torch.randperm(num_rows, device=device)[:n].to(torch.int32)

    cs_ref = view_of(backing_ref)
    new_state = fused_tree_conv_state_rows(
        source_z=source_z, state_src=state_src, tree_n=n, state_len=L
    ).to(dtype=cs_ref.dtype)
    cs_ref.index_copy_(0, perm.to(torch.long), new_state)

    cs_new = view_of(backing_new)
    launch_conv_state_writeback(
        source_z=source_z, state_src=state_src, dst_rows=perm,
        conv_state=cs_new, tree_n=n, state_len=L,
    )
    torch.cuda.synchronize()

    ok = torch.equal(backing_ref, backing_new)
    if not ok:
        d = (backing_ref.float() - backing_new.float()).abs()
        print(f"[{name}] FAIL max_abs={d.max().item():.3e} n_diff={int((d != 0).sum().item())}")
    else:
        print(f"[{name}] PASS (n={n}, C={C}, L={L}, paged={paged})")
    return ok


def main():
    ok = True
    ok &= run_case("tail6-22n-paged", 22, 8192, 24, True)
    ok &= run_case("tail6-22n-contig", 22, 8192, 24, False)
    ok &= run_case("cat9-9n-paged", 9, 8192, 24, True)
    ok &= run_case("odd-dims-paged", 21, 1536, 24, True)
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
