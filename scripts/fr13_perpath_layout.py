"""Prototype + self-test the per-node-path layout for the garble fix (offline, no GPU).

Builds, from a tree `parent` list, the tensors a per-path fused_sigmoid_gating call needs:
  - gather_idx : flat [T] node-id per path-token (so q_path = q_nodes[gather_idx])
  - cu_seqlens : [n_nodes+1] sub-sequence boundaries (one path per node)
  - out_idx    : [n_nodes] index into the packed output = LAST token of each node's path
  - state_idx  : [n_nodes] initial-state bank row per path (all = committed h0 row)
Validates against strict_ancestor_mask: each path == the node's ancestors-to-root, in order.
"""
import sys
sys.path.insert(0, "src")
from lumo_flywheel_serving.fr13_tree_conv_fused import tree_paths_from_parent
from lumo_flywheel_serving.fr13_replay_reference import strict_ancestor_mask


def build_perpath_layout(parent, h0_row=0):
    paths = tree_paths_from_parent(parent)          # paths[i] = [root..i] node ids
    n = len(parent)
    gather_idx, cu, out_idx, state_idx = [], [0], [], []
    for i in range(n):
        gather_idx.extend(paths[i])
        cu.append(len(gather_idx))
        out_idx.append(len(gather_idx) - 1)         # last token of node i's path
        state_idx.append(h0_row)
    return {"gather_idx": gather_idx, "cu_seqlens": cu, "out_idx": out_idx,
            "state_idx": state_idx, "paths": paths}


def selftest(parent, name):
    lay = build_perpath_layout(parent)
    paths = lay["paths"]
    n = len(parent)
    print(f"\n=== {name}  parent={parent} ===", flush=True)
    print(f"  paths: {paths}", flush=True)
    print(f"  T(total tokens)={len(lay['gather_idx'])} vs {n} unique  "
          f"(dup factor {len(lay['gather_idx'])/n:.2f}x)", flush=True)
    print(f"  cu_seqlens={lay['cu_seqlens']}", flush=True)
    print(f"  out_idx={lay['out_idx']}  (each == last token of its path)", flush=True)

    # invariant 1: path[i] ends at node i and starts at a root (parent==-1)
    ok = True
    for i in range(n):
        if paths[i][-1] != i:
            print(f"  FAIL: path[{i}] does not end at {i}"); ok = False
        if parent[paths[i][0]] != -1:
            print(f"  FAIL: path[{i}] does not start at a root"); ok = False
    # invariant 2: consecutive path nodes are parent-child
    for i in range(n):
        for a, b in zip(paths[i][:-1], paths[i][1:]):
            if parent[b] != a:
                print(f"  FAIL: {a}->{b} not parent-child in path[{i}]"); ok = False
    # invariant 3: path node-set == strict ancestors of i (incl i)
    mask = strict_ancestor_mask(parent, n)   # mask[i][j] True if j is a STRICT ancestor of i
    for i in range(n):
        anc = {j for j in range(n) if mask[i][j]}
        anc.add(i)
        if set(paths[i]) != anc:
            print(f"  FAIL: path[{i}] set {set(paths[i])} != ancestors {anc}"); ok = False
    # invariant 4: out_idx picks node i's OWN token (gather_idx[out_idx[i]]==i)
    for i in range(n):
        if lay["gather_idx"][lay["out_idx"][i]] != i:
            print(f"  FAIL: out_idx[{i}] points to node {lay['gather_idx'][lay['out_idx'][i]]}"); ok = False
    print(f"  invariants: {'ALL PASS' if ok else 'FAILED'}", flush=True)
    return ok


if __name__ == "__main__":
    # cat8: [(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]
    cat8 = [-1, -1, 0, 0, 2, 2, 4, 6]
    # cat6: [(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,0,0)] -> parents
    cat6 = [-1, -1, 0, 0, 2, 4]
    a = selftest(cat8, "cat8")
    b = selftest(cat6, "cat6")
    print("\nRESULT:", "ALL PASS" if (a and b) else "FAILURES", flush=True)
