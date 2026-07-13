"""Offline proof: is routing greedy through the deterministic multidraft reject == the greedy LCP-max
committer? (Tier-2 unify losslessness.) CPU-only, no server. If they always agree over random cat9-style
branching trees + random deterministic targets, Tier-2 (one-hot p) is lossless => one reject-sampling path.

Greedy LCP-max (abstract): commit the root->leaf PATH whose draft tokens match the target ARGMAX sequence
for the longest common prefix; the committed tokens = that path's matched prefix + the bonus (target argmax
at the first divergence). accepted_len = matched prefix length.

Deterministic multidraft descent (one-hot p = target argmax): at each node, among the children, accept the
child whose draft token == target argmax at this node (descend); if none matches, reject + commit bonus
(target argmax). This is sample_deterministic_multidraft_rejection_step with p=one-hot(argmax).

Compare: accepted PATH (node sequence) + committed TOKENS + accepted_len, over many random trees/targets.
"""
import random

# cat9 tree (paths sorted by (len,path)); node i -> parent. Spine=0,1,3,5,7; branches=2,4,6,8.
# paths: 0=(0,) 1=(0,0) 2=(0,1) 3=(0,0,0) 4=(0,0,1) 5=(0,0,0,0) 6=(0,0,0,1) 7=(0,0,0,0,0) 8=(0,0,0,0,1)
CAT9_PARENT = [-1, 0, 0, 1, 1, 3, 3, 5, 5]  # node -> parent node (root's parent = -1)

def children(parent):
    ch = {i: [] for i in range(len(parent))}
    for i, p in enumerate(parent):
        if p >= 0:
            ch[p].append(i)
    return ch

def root_to_leaf_paths(parent):
    ch = children(parent)
    leaves = [i for i in range(len(parent)) if not ch[i]]
    paths = []
    for lf in leaves:
        p = []
        cur = lf
        while cur >= 0:
            p.append(cur)
            cur = parent[cur]
        paths.append(list(reversed(p)))
    return paths, ch

def lcp_max_commit(parent, draft_tok, target_argmax):
    """draft_tok[node] = the token this node proposes. target_argmax[node] = argmax of target at that node's
    position (i.e. the correct next token AFTER descending to this node). Greedy accepts node child c iff
    draft_tok[c] == target_argmax[parent_position]. Returns (accepted_path_nodes, committed_tokens)."""
    paths, ch = root_to_leaf_paths(parent)
    best = None
    for path in paths:
        # match length: how many nodes along the path have draft == target-argmax-at-their-parent-slot
        m = 0
        for depth, node in enumerate(path):
            # target argmax at slot `depth` (the token expected after `depth` accepts)
            if draft_tok[node] == target_argmax[depth]:
                m += 1
            else:
                break
        if best is None or m > best[0]:
            best = (m, path)
    m, path = best
    accepted = path[:m]
    committed = [draft_tok[n] for n in accepted] + [target_argmax[m]]  # + bonus
    return accepted, committed

def descent_commit(parent, draft_tok, target_argmax):
    """Deterministic multidraft descent with p=one-hot(target argmax). At each accepted depth, among the
    current node's children pick the one whose draft==target argmax; descend. Stop at first miss."""
    ch = children(parent)
    accepted = []
    cur = -1  # virtual root parent; children of root = node 0
    depth = 0
    node_children = [0]  # the root node is node 0 (only child of virtual root)
    while node_children:
        want = target_argmax[depth]
        match = [c for c in node_children if draft_tok[c] == want]
        if not match:
            break
        c = match[0]  # deterministic leftmost tie-break
        accepted.append(c)
        depth += 1
        node_children = ch[c]
    committed = [draft_tok[n] for n in accepted] + [target_argmax[depth]]
    return accepted, committed

def main():
    rng = random.Random(1313)
    V = 20  # small vocab
    N = len(CAT9_PARENT)
    trials = 200000
    diverge = 0
    examples = []
    for t in range(trials):
        draft_tok = [rng.randrange(V) for _ in range(N)]
        # target argmax at each depth 0..maxdepth (5 for cat9)
        target_argmax = [rng.randrange(V) for _ in range(N + 1)]
        # bias: often make the argmax match a drafted child (so accepts happen)
        if rng.random() < 0.7:
            # pick a random path and make target argmax follow it for a random prefix
            paths, _ = root_to_leaf_paths(CAT9_PARENT)
            path = rng.choice(paths)
            k = rng.randrange(len(path) + 1)
            for depth in range(k):
                target_argmax[depth] = draft_tok[path[depth]]
        a1, c1 = lcp_max_commit(CAT9_PARENT, draft_tok, target_argmax)
        a2, c2 = descent_commit(CAT9_PARENT, draft_tok, target_argmax)
        if a1 != a2 or c1 != c2:
            diverge += 1
            if len(examples) < 5:
                examples.append((draft_tok, target_argmax, (a1, c1), (a2, c2)))
    print("trials=%d  DIVERGENCES=%d (%.4f%%)" % (trials, diverge, 100.0 * diverge / trials))
    if diverge == 0:
        print("=> LCP-max == deterministic-descent on cat9 for ALL random targets => Tier-2 unify is LOSSLESS.")
    else:
        print("=> They DIFFER. The greedy/temp>0 accept rules are genuinely distinct. Examples:")
        for dt, ta, r1, r2 in examples:
            print("   draft=%s targ=%s  LCP=%s  descent=%s" % (dt, ta[:6], r1, r2))

if __name__ == "__main__":
    main()
