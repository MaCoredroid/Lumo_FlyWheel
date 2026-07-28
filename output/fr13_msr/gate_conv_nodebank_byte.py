#!/usr/bin/env python3
"""FR13_CONV_NODEBANK offline byte gate: bank path vs pool path, bit-exact.

Runs BOTH conv routes on identical synthetic inputs in ONE process (the
same-boot in-process discipline; cross-boot byte gates fork on GB10
autotune) and requires the POOL OBSERVABLES to match bit-for-bit:
  - pool linear cols 0..nacc-1 after the commit remap,
  - pool col0 (anchor deposit),
  - the committer's col0 write (leaf window),
including: invalid lanes (nacc < path_cols), nacc=0 rows, and an
ordinal-PERMUTED consume (composition-change modeling: deposits under prev
ordinals, remap under current with ordinal_perm).

Run in-container (GPU) or host (CPU): pure torch ops, no Triton needed —
the fused Triton writeback is byte-gated separately by the existing
FR13_CONV_WB_FUSED gates; here we gate the ROUTE (who reads/writes where).
"""
import sys

import torch

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, "src")
from lumo_flywheel_serving.fr13_tree_conv_fused import (  # noqa: E402
    prepare_replay_conv_remap_rows,
    prepare_replay_conv_remap_rows_from_bank,
    replay_conv_state_linear_remap_from_bank,
    replay_conv_state_linear_remap_prepared,
)


def run_case(dev, dtype, B, n_tree, spec_cols, seed, perm=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    C, L = 16, 4
    pool_rows = 512
    # ssi: unique pool rows per (b, col)
    ssi = torch.randperm(pool_rows, generator=g)[: B * spec_cols].view(
        B, spec_cols
    ).to(torch.int32).to(dev)
    # accepted paths: node cols (+1-anchored space) within tree width
    paths = torch.randint(1, n_tree, (B, spec_cols), generator=g).to(dev)
    lens = torch.tensor([spec_cols - 1, 0, 3, 1][:B], dtype=torch.int32).to(dev)
    deposits = torch.randn(B, n_tree, C, L, generator=g, dtype=torch.float32).to(dev).to(dtype)
    base_pool = torch.randn(pool_rows, C, L, generator=g, dtype=torch.float32).to(dev).to(dtype)

    # ---- ARM A (pool route): deposits land in pool node cols, remap pool->pool.
    # Pool pages FOLLOW the request, so request-at-ordinal-b's content
    # (deposits[b]) is always in its own pages — ordinals never matter here.
    pool_a = base_pool.clone()
    write_ord = perm if perm is not None else list(range(B))
    for b in range(B):
        rows = ssi[b, :n_tree].to(torch.long) if spec_cols >= n_tree else None
        if rows is None:
            # capped ssi cannot hold node cols -> pool route inapplicable
            return None
        pool_a[rows] = deposits[b]
    src, dst = prepare_replay_conv_remap_rows(
        spec_state_indices=ssi, accepted_paths=paths,
        num_accepted_tokens=lens, num_spec_decodes=B,
        max_path_len=spec_cols,
    )
    replay_conv_state_linear_remap_prepared(conv_state=pool_a, src_rows=src, dst_rows=dst)

    # ---- ARM B (bank route): request-at-ordinal-b deposited LAST step at
    # bank row write_ord[b] (its prev ordinal); pool gets col0 only.
    pool_b = base_pool.clone()
    bank = torch.zeros(B, n_tree, C * L, dtype=dtype, device=dev)
    for b in range(B):
        bank[write_ord[b]] = deposits[b].reshape(n_tree, C * L)
        pool_b[ssi[b, 0].to(torch.long)] = deposits[b][0]
    perm_t = None
    if perm is not None:
        # In serving, "request now at ordinal b deposited at bank row
        # write_ord[b] last step" — perm points ordinal b at that row.
        perm_t = torch.tensor(write_ord, dtype=torch.int32, device=dev)
    bsrc, bdst, valid = prepare_replay_conv_remap_rows_from_bank(
        spec_state_indices=ssi, accepted_paths=paths,
        num_accepted_tokens=lens, num_spec_decodes=B,
        max_path_len=spec_cols, n_tree=n_tree, ordinal_perm=perm_t,
    )
    bank_flat = bank.view(-1, C, L)
    replay_conv_state_linear_remap_from_bank(
        conv_state=pool_b, bank_view=bank_flat,
        bank_src_rows=bsrc, dst_rows=bdst, valid=valid,
    )

    # ---- Compare pool observables: linear cols + col0 per request
    fails = []
    for b in range(B):
        nacc = int(lens[b])
        cols = list(range(min(max(nacc, 1), spec_cols)))  # col0 always
        for k in cols:
            row = ssi[b, k].to(torch.long)
            a = pool_a[row]
            bb = pool_b[row]
            if a.dtype in (torch.bfloat16, torch.float16):
                same = torch.equal(a.view(torch.int16), bb.view(torch.int16))
            else:
                same = torch.equal(a, bb)
            if not same:
                fails.append((b, k, float((a.float() - bb.float()).abs().max())))
    return fails


def main() -> int:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    total_fail = 0
    cases = 0
    for dtype in (torch.bfloat16, torch.float32):
        for seed in range(6):
            for perm in (None, [2, 0, 3, 1], [1, 0, 2, 3]):
                r = run_case(dev, dtype, 4, 22, 22, 1000 + seed, perm=perm)
                if r is None:
                    continue
                cases += 1
                if r:
                    total_fail += len(r)
                    print(f"FAIL dtype={dtype} seed={seed} perm={perm}: {r[:4]}")
    if total_fail == 0:
        print(f"FR13_CONV_NODEBANK byte gate: ALL-IDENTICAL ({cases} cases, dev={dev})")
        return 0
    print(f"FR13_CONV_NODEBANK byte gate: {total_fail} mismatches over {cases} cases")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
