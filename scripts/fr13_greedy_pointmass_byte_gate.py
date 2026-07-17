#!/usr/bin/env python3
"""CPU byte-gate: device multidraft committer with all_greedy=True (point-mass
rows) == an independent greedy top-down argmax walk, over randomized trees.

This validates the temp-0 unification ("use rejection sampling also at temp 0"):
feeding a POINT MASS on the argmax to ``device_multidraft_node_step`` must make
the multidraft accept rule byte-identical to greedy longest-prefix, with ZERO
rng consumption. We stress edge cases: no child matches the argmax, several
children match DIFFERENT tokens, exact-tie logits, single-child chains, deep
trees. A mismatch here means the point-mass row does not reduce to greedy.

Run: PYTHONPATH=src:scripts .venv/bin/python scripts/fr13_greedy_pointmass_byte_gate.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "scripts")
import fr13_device_multidraft_kernel as K  # noqa: E402


def independent_greedy_walk(parents, drafts, target_logits, self_logits, bonus, max_spec_len):
    """Reference greedy committer, written independently of the device walk.

    Convention mirrors the served committer: the decision row for a set of
    siblings is the target row at the FIRST child index; the greedy token is its
    argmax; accept whichever sibling's draft == that argmax. Bonus after a fully
    accepted path is the argmax of the accepted leaf's SELF row.
    Returns (row, accepted_row, accepted_len, accepted_path, accepted_tokens).
    """
    current_parent = -1
    accepted_row = 0
    accepted_path = []
    row = []
    for _ in range(max_spec_len + 1):
        children = [n for n, p in enumerate(parents) if p == current_parent]
        if not children:
            if current_parent >= 0:
                row.append(int(np.argmax(self_logits[current_parent])))
            else:
                row.append(int(bonus))
            break
        g = int(np.argmax(target_logits[children[0]]))
        # pick the FIRST sibling whose draft == greedy token (source order)
        match = [c for c in children if drafts[c] == g]
        row.append(g)
        if not match:
            break  # rejected: no child carries the greedy token
        accepted = match[0]
        # committed token equals draft only when they agree (they do here: g==draft)
        if g != drafts[accepted]:
            break
        current_parent = accepted
        accepted_row = accepted
        accepted_path.append(accepted)
    return row[: max_spec_len + 1], accepted_row, len(accepted_path), accepted_path, [drafts[x] for x in accepted_path]


def random_tree(rng, vocab, max_nodes, tie_frac):
    """Random single-request tree. parents are node-local, root children = -1."""
    n = int(rng.integers(2, max_nodes + 1))
    parents = [-1]
    for node in range(1, n):
        # parent is any earlier node OR root (-1), biased toward chains/branches
        cand = [-1] + list(range(node))
        parents.append(int(rng.choice(cand)))
    logits = rng.normal(size=(n, vocab)).astype(np.float32)
    self_logits = rng.normal(size=(n, vocab)).astype(np.float32)
    # Introduce exact-tie LOGIT rows sometimes (argmax tie-break = first index).
    if rng.random() < tie_frac:
        r = int(rng.integers(0, n))
        logits[r, :] = 0.0
        logits[r, : min(3, vocab)] = 5.0  # tie among first few
    # Drafts. REALISTIC-TREE CONSTRAINT: siblings (same parent) propose DISTINCT
    # tokens -- the drafter emits distinct top-k candidates per branch, so two
    # children of one node never carry the same token. We honor that here; the
    # duplicate-sibling case (which makes the point-mass source pick ambiguous)
    # is validated separately by the live in-process gate on real trees.
    drafts = [0] * n
    by_parent = {}
    for node in range(n):
        by_parent.setdefault(parents[node], []).append(node)
    for par, kids in by_parent.items():
        used = set()
        # bias one sibling toward the greedy token so accepts happen
        greedy_tok = int(np.argmax(logits[kids[0]]))
        for i, node in enumerate(kids):
            if i == 0 and rng.random() < 0.6:
                tok = greedy_tok
            else:
                tok = int(rng.integers(0, vocab))
                while tok in used or (i != 0 and tok == greedy_tok):
                    tok = int(rng.integers(0, vocab))
            used.add(tok)
            drafts[node] = tok
    return parents, drafts, logits, self_logits


def main():
    rng = np.random.default_rng(20260717)
    vocab = 16
    max_spec_len = 8
    trials = 4000
    mism = 0
    for t in range(trials):
        parents, drafts, logits, self_logits = random_tree(rng, vocab, max_nodes=10, tie_frac=0.15)
        n = len(parents)
        bonus = int(rng.integers(0, vocab))

        num_draft_tokens = [n]
        draft_t = torch.tensor(drafts, dtype=torch.long)
        parent_t = torch.tensor(parents, dtype=torch.long)
        target_t = torch.tensor(logits, dtype=torch.float32)
        self_t = torch.tensor(self_logits, dtype=torch.float32)
        bonus_t = torch.tensor([bonus], dtype=torch.long)

        out = K.fr13_device_multidraft_commit(
            num_draft_tokens, draft_t, parent_t, target_t, self_t,
            None, bonus_t, max_spec_len, generators=None, all_greedy=True,
        )
        d_rows, d_arows, d_alens, d_paths, d_toks = out
        d_row, d_arow, d_alen, d_path, d_tok = (
            d_rows[0], d_arows[0], d_alens[0], d_paths[0], d_toks[0]
        )
        r_row, r_arow, r_alen, r_path, r_tok = independent_greedy_walk(
            parents, drafts, logits, self_logits, bonus, max_spec_len
        )
        if (d_row != r_row or d_arow != r_arow or d_alen != r_alen
                or list(d_path) != list(r_path) or list(d_tok) != list(r_tok)):
            mism += 1
            if mism <= 6:
                print("MISMATCH", t)
                print("  parents", parents, "drafts", drafts)
                print("  device ", d_row, d_arow, d_alen, d_path, d_tok)
                print("  ref    ", r_row, r_arow, r_alen, r_path, r_tok)

    print(f"trials={trials} mismatches={mism}")
    print("PASS: device all_greedy==greedy longest-prefix (byte)" if mism == 0 else "FAIL")
    return 0 if mism == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
