#!/usr/bin/env python3
"""Offline byte gate: launch_attn_kv_linear_remap_syncfree vs legacy.

Whole-cache byte comparison across every legacy code path:
  1. normal branching accepted paths (foreign copies happen)
  2. all-contiguous paths (legacy early-return: no foreign)
  3. non-uniform spans (legacy early-return guard)
  4. span-too-small overflow (legacy early-return guard)
  5. dst_pi permutation armed (SLOT_REORDER interplay)
  6. zero accepted (acc=0 rows)
Multiple kv cache layouts (NHD-ish 5D + 4D) in one kv_caches list.
"""
import sys

import torch

sys.path.insert(0, "/home/mark/shared/lumoFlyWheel/src")
from lumo_flywheel_serving.fr10_gdn_tree_kernel import (
    launch_attn_kv_linear_remap,
    launch_attn_kv_linear_remap_syncfree,
)


def mk_env(seed, b, span, path_cols, n_blocks=8, bs=16, H=4, D=32, spans_override=None):
    torch.manual_seed(seed)
    device = "cuda"
    total = b * span + 7
    slot_mapping = torch.randperm(n_blocks * bs, device=device)[:total].to(torch.long)
    qsl = [0]
    for i in range(b):
        qsl.append(qsl[-1] + (spans_override[i] if spans_override else span))
    query_start_loc = torch.tensor(qsl + [qsl[-1] + 3], dtype=torch.int32, device=device)
    kv1 = torch.randn(2, n_blocks, bs, H, D, dtype=torch.bfloat16, device=device)
    kv2 = torch.randn(2, n_blocks, bs, H * D, dtype=torch.bfloat16, device=device)
    return slot_mapping, query_start_loc, [kv1, kv2]


def run_case(name, b, span, path_cols, paths, accs, dst_pi=None, spans_override=None, seed=7):
    sm, qsl, kv_ref = mk_env(seed, b, span, path_cols, spans_override=spans_override)
    kv_new = [k.clone() for k in kv_ref]
    ap = torch.tensor(paths, dtype=torch.int32, device="cuda")
    acc = torch.tensor(accs, dtype=torch.int32, device="cuda")
    pi = None if dst_pi is None else torch.tensor(dst_pi, dtype=torch.int64, device="cuda")
    launch_attn_kv_linear_remap(
        kv_caches=kv_ref, slot_mapping=sm, query_start_loc=qsl,
        accepted_paths=ap, num_accepted_tokens=acc, num_spec_decodes=b, dst_pi=pi,
    )
    launch_attn_kv_linear_remap_syncfree(
        kv_caches=kv_new, slot_mapping=sm, query_start_loc=qsl,
        accepted_paths=ap, num_accepted_tokens=acc, num_spec_decodes=b, dst_pi=pi,
    )
    torch.cuda.synchronize()
    ok = all(torch.equal(a, c) for a, c in zip(kv_ref, kv_new))
    if not ok:
        for i, (a, c) in enumerate(zip(kv_ref, kv_new)):
            n = int((a != c).sum().item())
            if n:
                print(f"[{name}] FAIL kv[{i}] n_diff={n}")
    else:
        print(f"[{name}] PASS")
    return ok


def main():
    ok = True
    # 1. branching: spine [1,3,5,7,9] style non-contiguous paths, span 22
    ok &= run_case("branching", 3, 22,  6,
                   [[1, 3, 6, 9, 12, 15], [1, 4, 7, 10, 13, 16], [2, 5, 8, 11, 14, 17]],
                   [5, 3, 6])
    # 2. all-contiguous (ap == m+1) -> legacy no-op
    ok &= run_case("contiguous-noop", 2, 22, 6,
                   [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]], [6, 4])
    # 3. non-uniform spans -> legacy guard no-op
    ok &= run_case("nonuniform-spans", 2, 22, 6,
                   [[1, 3, 6, 9, 12, 15], [1, 2, 3, 4, 5, 6]], [5, 4],
                   spans_override=[22, 19])
    # 4. span too small (max offset >= span) -> legacy guard no-op
    ok &= run_case("span-overflow", 2, 5, 6,
                   [[1, 3, 6, 9, 12, 15], [1, 4, 7, 10, 13, 16]], [5, 5])
    # 5. dst_pi permutation (identity except swap 2<->3 to exercise the remap)
    pi = list(range(23)); pi[2], pi[3] = pi[3], pi[2]
    ok &= run_case("dst-pi", 2, 22, 6,
                   [[1, 3, 6, 9, 12, 15], [2, 5, 8, 11, 14, 17]], [5, 4], dst_pi=pi)
    # 6. zero accepted rows mixed in
    ok &= run_case("zero-accept", 3, 22, 6,
                   [[1, 3, 6, 9, 12, 15], [1, 2, 3, 4, 5, 6], [2, 5, 8, 11, 14, 17]],
                   [0, 0, 4])
    print("GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
