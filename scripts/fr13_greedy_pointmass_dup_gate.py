#!/usr/bin/env python3
"""CPU dup-sibling gate: the case fr13_greedy_pointmass_byte_gate.py DEFERS.

The main byte-gate enforces DISTINCT sibling drafts (the drafter's _pick_distinct
dedupes). The ONE way duplicate siblings arise in the deployed tree is the
last-resort pad (fr13_mtp_suffix_assembly.py:104-106) repeating the SPINE token
=> two siblings carrying the SAME token, and if that token == the greedy argmax
they BOTH match => the committer must pick ONE (whose leaf-state commits).

This gate constructs those dup-matched trees explicitly and checks the device
point-mass committer picks the FIRST matching sibling in source order (== the
independent top-down walk convention), across two sub-cases:
  (A) deep-then-match: the first (spine) sibling continues and its child ALSO
      matches greedy  => spine strictly wins (longer path).
  (B) LCP-TIE: the first (spine) sibling continues but its child MISMATCHES
      greedy, while the second (pad) sibling is a leaf => both paths have the
      SAME accepted length (1). This is the genuine tie; we verify the device
      picks the FIRST (child-0), which is the source-order convention.

Settles the DEVICE side. Whether the OLD path-LCP-max committer breaks the (B)
tie the same way is the live in-process gate's job (FR13_GREEDY_UNIFY_GATE) --
the old committer is entangled with injected globals and not CPU-extractable.

Run: PYTHONPATH=src:scripts .venv/bin/python scripts/fr13_greedy_pointmass_dup_gate.py
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "scripts")
import fr13_device_multidraft_kernel as K  # noqa: E402
from fr13_greedy_pointmass_byte_gate import independent_greedy_walk  # noqa: E402

VOCAB = 16
MAX_SPEC_LEN = 8


def _one_hot_rows(argmax_per_node, n):
    """Point-mass target/self logit rows: +10 at the chosen argmax, 0 elsewhere."""
    lg = np.zeros((n, VOCAB), dtype=np.float32)
    for i, a in enumerate(argmax_per_node):
        lg[i, a % VOCAB] = 10.0
    return lg


def dup_tree(rng, sub_case):
    """A tree whose root-child P has two children with the SAME draft == greedy.
    sub_case 'A' => spine child then a matching grandchild (spine strictly wins).
    sub_case 'B' => spine child's grandchild MISMATCHES greedy (LCP tie vs pad leaf).
    Node layout: 0=P(root child), 1=spine(child of P), 2=pad-leaf(child of P),
    3=grandchild(child of spine). g is the shared greedy token of the P-children.
    """
    g = int(rng.integers(0, VOCAB))
    gp = int(rng.integers(0, VOCAB))          # greedy token to ACCEPT P (root level)
    while gp == g:
        gp = int(rng.integers(0, VOCAB))
    gg = g if sub_case == "A" else ((g + 1) % VOCAB)  # grandchild greedy (match/mismatch)
    parents = [-1, 0, 0, 1]
    drafts = [gp, g, g, gg if sub_case == "A" else ((gg + 2) % VOCAB)]
    # target rows: node0 greedy==gp (accept P); nodes 1,2 decision row = node1's
    # target, greedy==g; node3 greedy==gg. Siblings share the decision at the
    # FIRST child index (node1), matching the committer convention.
    target_argmax = [gp, g, g, gg]
    self_argmax = [int(rng.integers(0, VOCAB)) for _ in range(4)]
    return parents, drafts, _one_hot_rows(target_argmax, 4), _one_hot_rows(self_argmax, 4)


def main():
    rng = np.random.default_rng(20260718)
    mism = 0
    trials = 2000
    for t in range(trials):
        sub = "A" if (t % 2 == 0) else "B"
        parents, drafts, target, self_l = dup_tree(rng, sub)
        n = len(parents)
        bonus = int(rng.integers(0, VOCAB))
        out = K.fr13_device_multidraft_commit(
            [n], torch.tensor(drafts, dtype=torch.long),
            torch.tensor(parents, dtype=torch.long),
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(self_l, dtype=torch.float32),
            None, torch.tensor([bonus], dtype=torch.long),
            MAX_SPEC_LEN, generators=None, all_greedy=True,
        )
        d_row, d_arow, d_alen, d_path, d_tok = (
            out[0][0], out[1][0], out[2][0], out[3][0], out[4][0]
        )
        r_row, r_arow, r_alen, r_path, r_tok = independent_greedy_walk(
            parents, drafts, target, self_l, bonus, MAX_SPEC_LEN
        )
        if (d_row != r_row or d_arow != r_arow or d_alen != r_alen
                or list(d_path) != list(r_path) or list(d_tok) != list(r_tok)):
            mism += 1
            if mism <= 6:
                print(f"MISMATCH t={t} sub={sub}")
                print("  parents", parents, "drafts", drafts)
                print("  device ", d_row, d_arow, d_alen, list(d_path), list(d_tok))
                print("  ref    ", r_row, r_arow, r_alen, list(r_path), list(r_tok))
    print(f"dup-sibling trials={trials} mismatches={mism}")
    print("PASS: device picks FIRST matching sibling (source order) on dup + LCP-tie"
          if mism == 0 else "FAIL: device dup tie-break diverges from top-down-first")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
