#!/usr/bin/env python3
"""FR13_DM_DEPTHSYNC byte gate: legacy per-node walk vs depth-synchronous walk
must produce BYTE-IDENTICAL five products at the same seeds (CPU-only, boot-free).

Stronger than the distribution gate: the depthsync walk claims the exact same
per-request tensor ops, sizes, order, and generators — so with identical seeds
the products must match exactly, across a stress matrix of tree shapes
(cat8 / cat6root / t33333), batch sizes, seeds, and adversarial logits
(dominant target token, near-uniform rows => rejects + residual draws + the
zero-overlap corner via a masked-vocab case).

Exit 0 = PASS.
"""
import importlib.util
import os
import sys
from pathlib import Path

import torch

REPO = Path("/home/mark/shared/lumoFlyWheel")
spec = importlib.util.spec_from_file_location(
    "_dm", REPO / "scripts/fr13_device_multidraft_kernel.py"
)
dm = importlib.util.module_from_spec(spec)
sys.modules["_dm"] = dm
spec.loader.exec_module(dm)

TREES = {
    "cat8": [(0,), (1,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0, 0)],
    "cat6root": [(0,), (1,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)],
    "t33333": [(0,), (1,), (2,), (0, 0), (0, 1), (0, 2), (0, 0, 0), (0, 0, 1), (0, 0, 2),
               (0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 0, 2), (0, 0, 0, 0, 0), (0, 0, 0, 0, 1), (0, 0, 0, 0, 2)],
}
VOCAB = 512


def tree_parents(choices):
    ch = sorted(choices, key=lambda p: (len(p), p))
    idx = {c: i for i, c in enumerate(ch)}
    parents = []
    for c in ch:
        parents.append(-1 if len(c) == 1 else idx[c[:-1]])
    return parents


def build_case(tree, batch, seed, mode):
    g = torch.Generator().manual_seed(seed)
    parents1 = tree_parents(TREES[tree])
    n = len(parents1)
    counts = [n] * batch
    parents = torch.tensor(parents1 * batch, dtype=torch.int32)
    drafts = torch.randint(0, VOCAB, (n * batch,), generator=g)
    tgt = torch.randn(n * batch, VOCAB, generator=g)
    slf = torch.randn(n * batch, VOCAB, generator=g)
    if mode == "dominant":
        # target strongly prefers the drafted tokens => deep accepts
        for i in range(n * batch):
            tgt[i, drafts[i]] += 8.0
    elif mode == "adversarial":
        # target avoids drafted tokens => rejects + residual draws
        for i in range(n * batch):
            tgt[i, drafts[i]] -= 12.0
    elif mode == "zero-overlap":
        # target mass exactly zero on ALL drafted children of the root level
        for i in range(n * batch):
            tgt[i, drafts[i]] = -torch.inf
    bonus = torch.randint(0, VOCAB, (batch,), generator=g)
    gens = {i: torch.Generator().manual_seed(seed * 1000 + i) for i in range(batch)}
    return counts, drafts, parents, tgt, slf, bonus, gens


def run(path_flag, case, max_spec_len):
    counts, drafts, parents, tgt, slf, bonus, gens = case
    os.environ["FR13_DM_DEPTHSYNC"] = path_flag
    # fresh generator states per run (clone so both paths see identical streams)
    gens2 = {}
    for k, v in gens.items():
        g = torch.Generator()
        g.set_state(v.get_state())
        gens2[k] = g
    return dm.fr13_device_multidraft_commit(
        counts, drafts, parents, tgt.clone(), slf.clone(), None, bonus,
        max_spec_len, generators=gens2,
    )


def main():
    fails = 0
    cases = 0
    for tree in TREES:
        msl = max(len(c) for c in TREES[tree])
        for batch in (1, 4):
            for seed in (1, 7, 42, 1234):
                for mode in ("random", "dominant", "adversarial", "zero-overlap"):
                    case = build_case(tree, batch, seed, mode)
                    a = run("0", case, msl)
                    b = run("1", case, msl)
                    cases += 1
                    if a != b:
                        fails += 1
                        print(f"FAIL {tree} b{batch} seed{seed} {mode}:")
                        for name, x, y in zip(
                            ("out_rows", "acc_rows", "acc_lens", "paths", "tok_rows"),
                            a, b,
                        ):
                            if x != y:
                                print(f"  {name}:\n    legacy={x}\n    depth ={y}")
    print(f"\n{cases - fails}/{cases} cases byte-identical")
    if fails:
        print(">>> FAIL")
        return 1
    print(">>> PASS — depthsync products byte-identical to legacy at same seeds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
