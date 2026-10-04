#!/usr/bin/env python3
"""Tree shapes for the tree-size sensitivity sweep (pure Python; CPU-testable).

Shapes are compact sub-trees of the deployed Hydra27 fixed32 tree, so every node keeps its real operand row:
  chain12 : depth-only chain, root + 11 drafts (= the deepest accepted spine 0-1-4-9-14-19-24-26-28-29-30-31)
  tree8   : root + the first 8 active drafts in physical (level) order
  tree16  : root + the first 16 active drafts
  tree27  : root + all 27 active Hydra27 drafts (inactive rows 18,23,25,27 removed -> no padded work)
Node ids are compact level-order ids (parent[i] < i); `rows[i]` is the physical row whose operands node i uses."""
from __future__ import annotations

PHYS_PARENT = (-1, 0, 0, 0, 1, 1, 1, 2, 3, 4, 4, 4, 7, 8, 9, 9, 9, 12, 13, 14, 14, 14, 17, 18, 19, 23, 24, 25, 26, 28, 29, 30)
INACTIVE = (18, 23, 25, 27)
SPINE = (0, 1, 4, 9, 14, 19, 24, 26, 28, 29, 30, 31)
SHAPES = ("chain12", "tree8", "tree16", "tree27")


def _compact(rows, phys_parent=PHYS_PARENT):
    idx = {r: i for i, r in enumerate(rows)}
    parent = []
    for r in rows:
        p = phys_parent[r]
        if p >= 0 and p not in idx:
            raise ValueError(f"row {r}: parent {p} not in the sub-tree")
        parent.append(-1 if p < 0 else idx[p])
    return parent


def make_shape(name, phys_parent=PHYS_PARENT, inactive=INACTIVE):
    active = [r for r in range(len(phys_parent)) if r not in inactive]
    if name == "chain12":
        rows = list(SPINE)
    elif name.startswith("tree") and name[4:].isdigit():
        k = int(name[4:])
        if not 1 <= k <= len(active) - 1:
            raise ValueError(f"{name}: need 1..{len(active) - 1} drafts")
        rows = active[:k + 1]
    else:
        raise ValueError(f"unknown shape {name!r}")
    parent = _compact(rows, phys_parent)
    validate(parent)
    return {"name": name, "rows": rows, "parent": parent, "n": len(rows), "drafts": len(rows) - 1, "depth": max(depths(parent))}


def validate(parent):
    if not parent or parent[0] != -1 or any(p < 0 for p in parent[1:]) or any(not (parent[i] < i) for i in range(1, len(parent))):
        raise ValueError(f"parent vector must be root-first and topological: {parent}")


def depths(parent):
    d = []
    for i, p in enumerate(parent):
        d.append(0 if p < 0 else d[p] + 1)
    return d


def ancestors(parent, node):
    out = []
    while node >= 0:
        out.append(node)
        node = parent[node]
    return out[::-1]


def dfs_preorder(parent):
    """Compact ids in DFS pre-order (siblings ascending) -- the TreeWY author representation (parent[t] < t)."""
    ch = {i: [] for i in range(len(parent))}
    for i, p in enumerate(parent):
        if p >= 0:
            ch[p].append(i)
    out, stack = [], [0]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(sorted(ch[n], reverse=True))
    return out


def relabel(parent, order):
    """parent vector in the ids given by `order` (order[new] = old)."""
    new_of = {old: new for new, old in enumerate(order)}
    out = [-1 if parent[old] < 0 else new_of[parent[old]] for old in order]
    validate(out)
    return out


def depth_levels(parent):
    """Nodes grouped by depth (level-synchronous schedule)."""
    d = depths(parent)
    return [[i for i in range(len(parent)) if d[i] == k] for k in range(max(d) + 1)]


def next_pow2(n, floor=1):
    p = floor
    while p < n:
        p *= 2
    return p


def ancestor_masks(parent, n_pad=None):
    """(strict, visible) 0/1 lists [n_pad][n_pad] as the Q1 runner builds them (masks_from_parent), padded rows zero."""
    n = len(parent)
    n_pad = n if n_pad is None else n_pad
    strict = [[0] * n_pad for _ in range(n_pad)]
    vis = [[0] * n_pad for _ in range(n_pad)]
    for i in range(n):
        vis[i][i] = 1
        c = parent[i]
        while c >= 0:
            strict[i][c] = 1
            vis[i][c] = 1
            c = parent[c]
    return strict, vis
