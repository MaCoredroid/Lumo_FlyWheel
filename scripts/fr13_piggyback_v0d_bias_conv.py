#!/usr/bin/env python3
"""FR13_PIGGYBACK V0(d) part 2 -- bias-ghost isomorphism + position remap + conv tables.

HOST-RUNNABLE (pure CPU torch/numpy; no container needed):

    .venv/bin/python scripts/fr13_piggyback_v0d_bias_conv.py

WHAT THIS SETTLES (FR13_PIGGYBACK_PHASE3_APPLY_REPORT.md V0(d); S1 spec P3):
  (a) BIAS-GHOST ISOMORPHISM: build the 18x18 extended-tree attention bias the
      way tree_attn does (ancestor-visibility: -inf everywhere, 0 at
      (row, ancestors+self)), apply the LANDED A2 + S1-P3 mutation statements,
      and assert: chain cols 1..7 dead for every row; chain rows 1..7 dead on
      every tree col; row 0 + col 0 fully dead (S1-P3); the live block
      {8..17} is EXACTLY the base-cat9 10-stream bias under the isomorphism
      phi(8)=0, phi(9+i)=1+i; no live row is fully masked. A SOURCE-DRIFT
      GUARD asserts the exact landed mutation statements still exist in the
      patcher -- if the landed code changes, this fixture FAILS instead of
      validating a stale replica.
  (a2) SEMANTIC PUSH-THROUGH: replicate apply_tree_bias's documented
      semantics (fr13_patch_fa2_tree_bias.py: bias == -INFINITY -> hard mask,
      else score += bias; bias covers ONLY the last-18 suffix cols) in torch
      on a random [18, ctx+18] score block; assert ghost rows (0..7) end up
      with ALL probability mass on context cols (finite softmax, no NaN) and
      hard-masked cells contribute exactly zero. The REAL CUDA path is
      exercised live by the cat9_pb V2 arm (it cannot be run offline).
  (b) POSITION REMAP UNIT: depth offsets of the sorted extended tree, clamped
      per the landed A1 formula np.maximum(offsets - 8, 0), must equal
      [0]*9 + [1,2,2,3,3,4,4,5,5]; the sort key must match A1's
      (len(choice), choice).
  (c) CONV TABLE GATE: replicate CONV-1a' (conv_parent = parent with
      conv_parent[8] = -1 iff the 18-node extended tree is detected) and
      assert the conv WINDOW STRUCTURE: window(8) == [8] (= prior ++ x_8);
      phi(window_ext(s)) == window_base(phi(s)) for every live subtree node
      (subtree windows = prior ++ x_8 ++ path == base cat9 windows); chain
      nodes' windows never appear in any live window; n != 18 trees are
      untouched (conv_parent == parent). BYTE-level conv-state equality
      remains a live V2.5 carrier gate (this is the structural half).

Deliberate scope: no GPU, no kernel launches -- the byte-level halves of
these properties are V0(d)-part-1 (ring induction, GPU) + V2/V2.5 (live).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# topology (same constants as the GPU validators; import asserts run)
import fr13_piggyback_v0c_validate as V0C  # noqa: E402

N = 18
NEG = float("-inf")
PATCHER = os.path.join(_HERE, "fr10_phase4_patch_vllm_tree_gdn.py")

# the LANDED mutation statements (A2 rules [1][2][3] + S1-P3) -- drift-guarded
LANDED_STMTS = (
    "tree_attn_mask[..., :, 1:8] = _fr13_pb_ninf",
    "tree_attn_mask[..., 1:8, :] = _fr13_pb_ninf",
    "tree_attn_mask[..., 8:, 0] = _fr13_pb_ninf",
    "tree_attn_mask[0, :] = -torch.inf",
    "tree_attn_mask[:, 0] = -torch.inf",
)


def _ancestor_bias(stream_parents) -> torch.Tensor:
    """tree_attn-style bias: -inf everywhere; 0 at (i, ancestors(i) + self)."""
    n = len(stream_parents)
    bias = torch.full((n, n), NEG, dtype=torch.float32)
    for i in range(n):
        j = i
        while j != -1:
            bias[i, j] = 0.0
            j = stream_parents[j]
    return bias


def _apply_landed_mutations(bias: torch.Tensor) -> torch.Tensor:
    """Exactly the landed A2 + S1-P3 statements (see LANDED_STMTS drift guard)."""
    b = bias.clone()
    _fr13_pb_ninf = NEG
    b[..., :, 1:8] = _fr13_pb_ninf
    b[..., 1:8, :] = _fr13_pb_ninf
    b[..., 8:, 0] = _fr13_pb_ninf
    b[0, :] = -torch.inf
    b[:, 0] = -torch.inf
    return b


def _conv_windows(parents) -> dict:
    """window(node) = root-to-node walk over the given parent table."""
    out = {}
    for i in range(len(parents)):
        w, j = [], i
        while j != -1:
            w.append(j)
            j = parents[j]
        out[i] = list(reversed(w))
    return out


def main() -> int:
    rows_out: list[tuple[str, bool, str]] = []

    def add(name, ok, detail=""):
        rows_out.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail and not ok else ""), flush=True)

    # ---- drift guard ---------------------------------------------------------
    src = open(PATCHER).read()
    for stmt in LANDED_STMTS:
        add(f"(guard) landed stmt present: {stmt[:46]}...", stmt in src,
            "landed mutation changed -- update this fixture before trusting it")

    # ---- (a) bias-ghost isomorphism -----------------------------------------
    print("[check a] bias-ghost isomorphism (A2 + S1-P3 on the 18x18 bias)", flush=True)
    ext = _apply_landed_mutations(_ancestor_bias(V0C.STREAM_PARENTS))
    base_parents = (-1,) + tuple(p + 1 for p in V0C.BASE_PARENTS)  # 10-stream cat9
    base = _ancestor_bias(base_parents)

    add("(a) chain cols 1..7 dead for EVERY row",
        bool((ext[:, 1:8] == NEG).all()))
    add("(a) chain rows 1..7 dead on EVERY tree col",
        bool((ext[1:8, :] == NEG).all()))
    add("(a) row 0 fully dead + col 0 fully dead (S1-P3)",
        bool((ext[0, :] == NEG).all() and (ext[:, 0] == NEG).all()))
    live = ext[8:, 8:]
    add("(a) live block {8..17} == base cat9 bias under phi(8)=0, phi(9+i)=1+i",
        bool(torch.equal(live, base)),
        f"{int((live != base).sum())} cells differ")
    add("(a) no live row fully masked (each attends >= self)",
        bool((live == 0.0).any(dim=1).all()))

    # ---- (a2) semantic push-through -----------------------------------------
    print("[check a2] apply_tree_bias torch-replica push-through", flush=True)
    g = torch.Generator().manual_seed(20260718)
    ctx = 7
    scores = torch.randn(N, ctx + N, generator=g)
    biased = scores.clone()
    tree = biased[:, ctx:]
    hard = ext == NEG
    tree[hard] = NEG                     # bias == -INFINITY -> hard mask
    tree[~hard] = tree[~hard] + ext[~hard]  # else score += bias
    probs = torch.softmax(biased, dim=-1)
    add("(a2) softmax finite everywhere (no NaN rows from ghosting)",
        bool(torch.isfinite(probs).all()))
    add("(a2) ghost rows 0..7: ALL mass on context cols",
        bool(torch.allclose(probs[:8, :ctx].sum(dim=-1),
                            torch.ones(8), atol=1e-6)))
    add("(a2) hard-masked cells contribute exactly zero",
        bool((probs[:, ctx:][hard] == 0.0).all()))

    # ---- (b) position remap --------------------------------------------------
    print("[check b] A1 position-remap unit", flush=True)
    choices = sorted(V0C.EXT_CHOICES, key=lambda p: (len(p), p))  # A1's sort key
    offsets = np.array([0] + [len(c) for c in choices], dtype=np.int64)
    clamped = np.maximum(offsets - 8, 0)
    expected = np.array([0] * 9 + [1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
    add("(b) armed offsets == [0]*9 + [1,2,2,3,3,4,4,5,5]",
        bool((clamped == expected).all()), f"got {clamped.tolist()}")
    add("(b) sort key matches A1 ((len, choice)) == V0C canonical order",
        tuple(choices) == V0C.EXT_CHOICES,
        "sorted() order diverges from the scan-stream order")

    # ---- (c) conv tables -----------------------------------------------------
    print("[check c] CONV-1a' conv_parent tables", flush=True)
    tree_choices = list(V0C.EXT_CHOICES)
    parent = list(V0C.STREAM_PARENTS)
    conv_parent = list(parent)
    _fr13_pb_ext_tree = (
        N == 18
        and all(tree_choices[_pbk] == tuple([0] * (_pbk + 1)) for _pbk in range(8))
    )
    add("(c) extended-tree detect fires (n==18, chain choices 0..7)",
        _fr13_pb_ext_tree)
    if _fr13_pb_ext_tree:
        conv_parent[8] = -1
    wins_ext = _conv_windows(conv_parent)
    wins_base = _conv_windows(list(base_parents))
    add("(c) window(8) == [8]  (= prior ++ x_8, the conv-root bonus window)",
        wins_ext[8] == [8], f"got {wins_ext[8]}")
    phi = {8: 0}
    phi.update({9 + i: 1 + i for i in range(9)})
    iso_ok, iso_why = True, ""
    for s in range(8, N):
        mapped = [phi[x] for x in wins_ext[s]]
        if mapped != wins_base[phi[s]]:
            iso_ok, iso_why = False, f"stream {s}: {mapped} != {wins_base[phi[s]]}"
            break
    add("(c) phi(window_ext(s)) == window_base(phi(s)) for all live nodes", iso_ok, iso_why)
    add("(c) no chain node (1..7) in any live window",
        all(all(x not in range(1, 8) for x in wins_ext[s]) for s in range(8, N)))
    # negative: a non-extended tree must be untouched
    base_conv = list(base_parents)
    neg_detect = (
        len(base_parents) == 18
        and all(tuple(V0C.BASE_CHOICES[k]) == tuple([0] * (k + 1)) for k in range(8))
    )
    add("(c) negative: base cat9 (n=10) does NOT trip the detect / stays untouched",
        (not neg_detect) and base_conv == list(base_parents))

    # ---- table ---------------------------------------------------------------
    print("=" * 92)
    print(f"{'FR13_PIGGYBACK V0(d) BIAS/CONV CHECK':74s} RESULT")
    print("-" * 92)
    for name, ok, detail in rows_out:
        print(f"{name:74s} {'PASS' if ok else 'FAIL'}")
        if detail and not ok:
            print(f"    -> {detail}")
    n_fail = sum(1 for _, ok, _ in rows_out if not ok)
    print("-" * 92)
    print("=== VERDICT: " + ("PASS -- ghosting/positions/conv tables match the LIVE-8 "
                             "isomorphism ===" if n_fail == 0 else
                             f"FAIL -- {n_fail}/{len(rows_out)} failed; do NOT arm ==="))
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
