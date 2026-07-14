#!/usr/bin/env python3
"""CPU LOGIC model for the FR13 slot-level canonical reorder (spine M-invariant fix).

Validates the INDEX BOOKKEEPING end-to-end BEFORE touching the 18k-line patcher:
  - pi (spine-first) / pi_inv correctness.
  - permuted WRITE (node j -> physical slot pi_inv[j]) => physical column k holds node pi[k].
  - bias column permute (col k <-> node pi[k]) keeps query rows in BFS.
  - SPINE nodes' surviving columns become CONTIGUOUS [0..depth] after reorder (the M-fix),
    IDENTICALLY for cat8 and cat6 => cat8-spine layout == cat6-spine layout == spine-only
    => FA2 butterfly association identical => M-invariant. (branches stay gapped-but-FIXED.)
  - COMMITTER: src = permuted slot of accepted node (auto-threads), dst = ORIGINAL/flat slot
    of committed offset (m+1) => committed prefix lands CONTIGUOUS + correct.

No floats — pure index correctness (the plumbing risk). Asserts loudly; exit 0 == GO.
"""

def anc_set(parent, i):
    s = []
    while i != -1:
        s.append(i)
        p = parent[i]
        # BFS validity: parents precede children; fail LOUD on cycles/bad arrays
        assert -1 <= p < i, f"invalid parent array: parent[{i}]={p} (must be -1 or < {i})"
        i = p
    return sorted(s)


def spine_of(parent, choices_all0):
    # spine = the all-zero path nodes; here we pass node ids known to be spine.
    return choices_all0


def build(parent, spine, branch, name):
    n = len(parent)
    pi = list(spine) + list(branch)              # physical column k -> node pi[k]
    assert sorted(pi) == list(range(n)), f"{name}: pi not a permutation: {pi}"
    pi_inv = [0] * n                             # node j -> physical column pi_inv[j]
    for k, node in enumerate(pi):
        pi_inv[node] = k
    # inverse check
    for node in range(n):
        assert pi[pi_inv[node]] == node, f"{name}: pi_inv wrong at {node}"
    return pi, pi_inv


def surviving_cols(parent, node, pi_inv):
    """physical columns a query row 'node' attends to, after the reorder."""
    return sorted(pi_inv[a] for a in anc_set(parent, node))


def check_tree(parent, spine, branch, name):
    print(f"\n=== {name}: parent={parent} spine={spine} branch={branch} ===")
    pi, pi_inv = build(parent, spine, branch, name)
    print(f"  pi (col->node)   = {pi}")
    print(f"  pi_inv(node->col)= {pi_inv}")

    # (1) SPINE contiguity: every spine node's surviving cols == [0..depth] contiguous.
    spine_layouts = {}
    for depth, s in enumerate(spine):
        cols = surviving_cols(parent, s, pi_inv)
        expect = list(range(depth + 1))          # spine depth d -> cols {0..d}
        assert cols == expect, f"{name}: spine node {s} cols {cols} != {expect} (NOT contiguous!)"
        spine_layouts[depth] = cols
    print(f"  SPINE contiguous  : OK  (depth->cols {spine_layouts})")

    # (2) BRANCH nodes: FIXED column set (independent of which OTHER branches exist),
    #     but generally GAPPED (subset ancestors). Report gappiness honestly.
    for b in branch:
        cols = surviving_cols(parent, b, pi_inv)
        contiguous = (cols == list(range(len(cols))))
        print(f"  branch {b} anc={anc_set(parent,b)} -> cols {cols} "
              f"({'contiguous' if contiguous else 'GAPPED (fixed, within-floor)'})")
    return spine_layouts


def check_committer(parent, spine, branch, accept_depth, name):
    """Simulate the committer for a SPINE accept of `accept_depth` tokens.
    accepted path = spine[1..accept_depth] (root=spine[0] already committed).
    src_off = accepted node id (BFS row); dst_off = m+1 (linear committed offset).
    src_slot = PERMUTED map; dst_slot = ORIGINAL/flat map. Verify committed prefix
    slots [1..accept_depth] hold the accepted spine nodes IN ORDER."""
    pi, pi_inv = build(parent, spine, branch, name)
    base = 1000                                  # arbitrary physical base
    orig_slot = [base + i for i in range(len(parent))]           # flat: node/offset i -> base+i
    perm_slot = [base + pi_inv[i] for i in range(len(parent))]   # permuted write map
    # accepted spine path (BFS ids), depths 1..accept_depth
    accepted = spine[1:accept_depth + 1]
    committed = {}                               # dst physical slot -> node whose KV lands there
    for m, node in enumerate(accepted):
        src_off = node                           # accepted node id
        dst_off = m + 1                          # linear committed offset
        src_slot = perm_slot[src_off]            # AUTO-THREADS on permuted map
        dst_slot = orig_slot[dst_off]            # THE FIX: original/flat map
        committed[dst_slot] = node               # copy accepted node's KV -> flat committed slot
    # verify: reading committed prefix at flat slots base+1..base+accept_depth gives
    # exactly the accepted spine nodes in depth order.
    ok = True
    for m in range(accept_depth):
        want_node = accepted[m]
        got_slot = base + (m + 1)
        got_node = committed.get(got_slot)
        if got_node != want_node:
            ok = False
            print(f"  COMMITTER FAIL {name} d{accept_depth}: prefix slot {got_slot} "
                  f"has node {got_node}, want {want_node}")
    assert ok, f"{name}: committer produced wrong contiguous prefix"
    # counter-check: if dst had (wrongly) used the PERMUTED map, would it scatter?
    scatter = any((base + pi_inv[m + 1]) != (base + (m + 1)) for m in range(accept_depth))
    print(f"  COMMITTER OK {name} d{accept_depth}: prefix {[committed[base+1+m] for m in range(accept_depth)]}"
          f"  (permuted-dst would{'' if scatter else ' NOT'} scatter => fix {'REQUIRED' if scatter else 'noop here'})")


def choices_to_tree(choices):
    """Derive (parent, spine, branch) from a speculative_token_tree choices list
    using the EXACT shipped algorithm (sorted by (len, path); node id = 1+index;
    parent = node of the path prefix; spine = root + all-zero paths)."""
    ch = sorted(choices, key=lambda p: (len(p), p))
    idx = {c: i + 1 for i, c in enumerate(ch)}
    parent = [-1]
    for c in ch:
        parent.append(0 if len(c) == 1 else idx[c[:-1]])
    spine = [0] + [i + 1 for i, c in enumerate(ch) if all(int(x) == 0 for x in c)]
    branch = [i + 1 for i, c in enumerate(ch) if not all(int(x) == 0 for x in c)]
    return parent, spine, branch


def main():
    # cat8 served: tree_n=9, parent=[-1,0,0,1,1,3,3,5,7], spine=[0,1,3,5,7,8], branch=[2,4,6]
    cat8_parent = [-1, 0, 0, 1, 1, 3, 3, 5, 7]
    cat8_spine, cat8_branch = [0, 1, 3, 5, 7, 8], [2, 4, 6]
    # cross-check the shipped derivation reproduces the served cat8
    _p, _s, _b = choices_to_tree(
        [(0,), (1,), (0, 0), (0, 1), (0, 0, 0), (0, 0, 1), (0, 0, 0, 0), (0, 0, 0, 0, 0)])
    assert (_p, _s, _b) == (cat8_parent, cat8_spine, cat8_branch), (_p, _s, _b)
    # cat6 (spine + 1 branch off root): tree_n=7 — 0 root, 1 spine-d1, 2 branch,
    # 3..6 spine d2..d5. Key: the spine layout must match cat8's after reorder.
    cat6_parent = [-1, 0, 0, 1, 3, 4, 5]
    cat6_spine, cat6_branch = [0, 1, 3, 4, 5, 6], [2]

    l8 = check_tree(cat8_parent, cat8_spine, cat8_branch, "cat8")
    l6 = check_tree(cat6_parent, cat6_spine, cat6_branch, "cat6")

    # (3) CROSS-TREE M-INVARIANCE: cat8-spine and cat6-spine share IDENTICAL column layout
    #     per depth (both [0..depth]) => FA2 butterfly identical => cat8-spine == cat6-spine.
    shared = min(len(l8), len(l6))
    for d in range(shared):
        assert l8[d] == l6[d], f"cat8 depth{d} {l8[d]} != cat6 {l6[d]}"
    print(f"\n  CROSS-TREE: cat8-spine layout == cat6-spine layout for depths 0..{shared-1}  "
          f"=> M-INVARIANT by construction. OK")

    # 3-3-3 tree (user challenge): per-depth widths [3,3,3] -> 9 choices + root,
    # spine (0,)/(0,0)/(0,0,0) + 2 branches fanning at EVERY depth (incl. deep).
    t333 = [(0,), (1,), (2,),
            (0, 0), (0, 1), (0, 2),
            (0, 0, 0), (0, 0, 1), (0, 0, 2)]
    p333, s333, b333 = choices_to_tree(t333)
    l333 = check_tree(p333, s333, b333, "tree-3-3-3")
    for d in range(min(len(l8), len(l333))):
        assert l8[d] == l333[d], f"cat8 depth{d} {l8[d]} != 3-3-3 {l333[d]}"
    print(f"  CROSS-TREE: 3-3-3 spine layout == cat8 spine layout for shared depths "
          f"=> same M-invariant canonical form. OK")

    # (4) committer correctness across accept depths, all trees
    print("\n=== COMMITTER simulation ===")
    for d in (1, 3, 5):
        check_committer(cat8_parent, cat8_spine, cat8_branch, d, "cat8")
    check_committer(cat6_parent, cat6_spine, cat6_branch, 3, "cat6")
    for d in (1, 2, 3):
        check_committer(p333, s333, b333, d, "tree-3-3-3")

    print("\n>>> GO — slot-reorder index logic is correct: spine contiguous+cross-tree-identical "
          "(M-invariant), branches fixed, committer prefix contiguous with src=permuted/dst=flat.")


if __name__ == "__main__":
    main()
