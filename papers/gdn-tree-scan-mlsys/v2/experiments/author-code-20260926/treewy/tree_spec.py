# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
/ E1a — static tree topology + a reference tree-draft builder for
TREE speculative decoding of the MTP drafter.

The whole of rides on ONE data structure: a static draft tree laid out
in **DFS pre-order**, so that for every node `t` its parent satisfies
`parent[t] < t`. That invariant is what (a) keeps the per-request draft buffer
request-contiguous, (b) makes the GDN tree-WY rollback's strict-ancestor mask
lower-triangular (see tree_wy_ref.py), and (c) lets a single integer `parent`
array carry the entire structure through proposer -> verify -> sampler -> commit.

`TreeTopology` precomputes, from a per-depth fanout schedule `widths`:
  * parent[t], depth[t]   (DFS pre-order, parent[t] < t, root = -1)
  * the per-level groupings needed to expand the tree breadth-first (one model
    forward per depth, batched over all live nodes at that depth) while still
    scattering results into the DFS-ordered flat buffer.

`build_tree_reference` is a pure-python/torch reference that drives a given
`expand_fn` (token, hidden) -> (logits, hidden_out) through the topology and
returns the flat draft tokens + parent/depth arrays. It is the offline
correctness gate for the tree-construction logic, BEFORE it is wired into
vLLM's cudagraph-shaped proposer loop (E1b). Width-1 must reproduce the chain.

KEY ORDERING FACT (proved in the unit test): with DFS-pre-order ids, the level-L
nodes sorted by id equal the concatenation, over level-(L-1) nodes in id order,
of each parent's children in rank order. So the (parent_idx, child_rank)
flattening of a top-k expansion lands EXACTLY in sorted level-L order — no
permutation needed between the batched expansion and the DFS buffer.
"""

from dataclasses import dataclass

import torch


class TreeTopology:
    """Static draft-tree shape from a per-depth fanout schedule.

    widths[L] = number of children each live node spawns at depth L.
    depth D = len(widths). Total draft nodes T = sum_L prod_{j<=L} widths[j].
    Node ids are DFS pre-order; the implicit root (the last accepted/bonus
    token) is id -1 and is NOT a draft node.
    """

    def __init__(self, widths: list[int]):
        assert len(widths) >= 1 and all(w >= 1 for w in widths)
        self.widths = list(widths)
        self.depth = len(widths)

        parent: list[int] = []
        depth: list[int] = []
        # Recursive DFS pre-order id assignment: emit a node, then recurse into
        # its children. Because a child is always assigned after its parent,
        # parent[t] < t holds for every node.
        next_id = [0]

        def rec(d: int, par_id: int) -> None:
            if d == self.depth:
                return
            for _ in range(self.widths[d]):
                nid = next_id[0]
                next_id[0] += 1
                parent.append(par_id)
                depth.append(d)
                rec(d + 1, nid)

        rec(0, -1)

        self.num_nodes = len(parent)
        # device="cpu" EXPLICITLY on every tensor here: this object is HOST-side
        # topology metadata, and consumers read it on the host (.tolist(), .numpy(),
        # per-level Python indexing) as well as copying it to the device with an
        # explicit .to(device). A bare torch.tensor() honours the ambient default
        # device, so building the shared cache from inside a
        # `with torch.device("cuda")` block (which is where vLLM constructs model
        # layers) would silently make it a CUDA tensor and break every host
        # consumer -- e.g. gpu_model_runner._apply_tree_verify_mrope_positions'
        # node_depth.numpy(). Pinning to CPU makes the cache
        # construction-context-independent.
        self.parent = torch.tensor(
            parent, dtype=torch.int32, device="cpu"
        )  # [T], root = -1
        self.node_depth = torch.tensor(depth, dtype=torch.int32, device="cpu")

        # Per-level groupings (id-sorted within a level).
        # level_ids[L]          : LongTensor of DFS ids at depth L (ascending)
        # level_parent_local[L] : for each node in level_ids[L] (L>=1), the index
        #                         of its parent WITHIN level_ids[L-1]. For L==0 the
        #                         parent is the root, represented as index 0.
        self.level_ids: list[torch.Tensor] = []
        self.level_parent_local: list[torch.Tensor] = []
        for L in range(self.depth):
            ids = [t for t in range(self.num_nodes) if depth[t] == L]
            self.level_ids.append(torch.tensor(ids, dtype=torch.long, device="cpu"))
            if L == 0:
                self.level_parent_local.append(
                    torch.zeros(len(ids), dtype=torch.long, device="cpu")
                )
            else:
                prev = self.level_ids[L - 1].tolist()
                prev_pos = {nid: i for i, nid in enumerate(prev)}
                self.level_parent_local.append(
                    torch.tensor(
                        [prev_pos[parent[t]] for t in ids],
                        dtype=torch.long,
                        device="cpu",
                    )
                )

    def ancestors(self, node: int) -> list[int]:
        """Strict ancestors of `node` (DFS ids), nearest-first, root excluded."""
        out: list[int] = []
        p = int(self.parent[node])
        while p >= 0:
            out.append(p)
            p = int(self.parent[p])
        return out

    def __repr__(self) -> str:
        return (
            f"TreeTopology(widths={self.widths}, depth={self.depth}, "
            f"num_nodes={self.num_nodes})"
        )


def build_ancestor_mask(parent: torch.Tensor) -> torch.Tensor:
    """Dense intra-tree attention mask from a DFS-pre-order parent array.

    Returns bool `M[q, k]` = True iff key node `k` is the query node `q` itself
    or a strict ancestor of `q` — i.e. each tree node attends only up its own
    branch, never to siblings/cousins. Because `parent[t] < t`, `M` is lower-
    triangular. (The full committed prefix is attended separately by the backend
    and is NOT part of this T×T intra-tree block.)

    This is the relation applied (a) in the full-attention DRAFTER while it fans
    out the tree, (b) in the target's softmax-attention layers at verify, and
    (c) to pick the accepted leaf's ancestor set for the GDN tree-WY commit.
    """
    par = parent.tolist()
    T = len(par)
    # Follow `parent`'s device (CPU for the shared TreeTopology) rather than the
    # ambient default device, so this is safe to call from inside a
    # `with torch.device("cuda")` model-init block.
    M = torch.eye(T, dtype=torch.bool, device=parent.device)
    for q in range(T):
        p = par[q]
        while p >= 0:
            M[q, p] = True
            p = par[p]
    return M


def tree_attention_dense(
    q_node_id: torch.Tensor,  # [Nq] node id of each query (all tree nodes, >=0)
    kv_node_id: torch.Tensor,  # [Nkv] node id if a tree node, else -1 (prefix/cache)
    ancestor_mask: torch.Tensor,  # [T,T] from build_ancestor_mask
    q_pos: torch.Tensor,  # [Nq] logical position of each query
    kv_pos: torch.Tensor,  # [Nkv] logical position of each key
) -> torch.Tensor:
    """Reference dense [Nq, Nkv] attention mask for a tree forward (drafter OR
    target verify). Rule per (query q, key k):
      * k is a PREFIX/cache token (kv_node_id<0): allow iff causal (kv_pos<=q_pos)
      * k is a TREE node: allow iff k is q itself or an ancestor of q
    Same-depth siblings share a position, so the tree branch is resolved by NODE
    ID via ancestor_mask, not by position. This is the relation a FlexAttention
    mask_mod must implement (see make_tree_mask_mod)."""
    Nq, Nkv = q_node_id.shape[0], kv_node_id.shape[0]
    is_prefix = (kv_node_id < 0).view(1, Nkv).expand(Nq, Nkv)
    causal = kv_pos.view(1, Nkv) <= q_pos.view(Nq, 1)
    # tree-vs-tree ancestry; clamp kv ids so the gather is in-range where prefix.
    kv_safe = kv_node_id.clamp(min=0)
    tree_ok = ancestor_mask[q_node_id][:, kv_safe]  # [Nq, Nkv]
    return torch.where(is_prefix, causal, tree_ok)


def make_tree_mask_mod(
    q_node_id: torch.Tensor,
    kv_node_id: torch.Tensor,
    ancestor_mask: torch.Tensor,
    q_pos: torch.Tensor,
    kv_pos: torch.Tensor,
):
    """FlexAttention `mask_mod` over PHYSICAL query/key indices for a tree forward.
    Backed by the per-index lookup tables above (trees are tiny). Indexing into
    precomputed tensors by the scalar q/kv indices is the documented FlexAttention
    pattern (cf. FlexAttentionMetadata._convert_physical_to_logical)."""

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001
        is_prefix = kv_node_id[kv_idx] < 0
        causal = kv_pos[kv_idx] <= q_pos[q_idx]
        tree_ok = ancestor_mask[q_node_id[q_idx], kv_node_id[kv_idx].clamp(min=0)]
        return torch.where(is_prefix, causal, tree_ok)

    return mask_mod


def tree_logical_mask_mod(base: int, ancestor_mask: torch.Tensor):
    """FlexAttention `logical_mask_mod` for a tree forward — the INJECTION-READY
    form. Set it on the attention layer as `layer.logical_mask_mod = ...` and
    FlexAttention's forward picks it up and rebuilds the block_mask
    (flex_attention.py:1122).

    Relies on vLLM's logical index = decode_offset + local_query_index
    (flex_attention.py:436) being decoupled from the RoPE `positions` tensor. If
    the query/KV buffer is laid out in DFS node order, then for a draft/verify
    forward `logical_idx - base` IS the DFS node id, distinct per node — so the
    same-depth-position collision never arises. RoPE positions stay `base+depth`
    independently.

    Rule: a tree query (logical >= base) sees key `k`:
      * prefix/cache key (logical < base): always (causal; prefix precedes tree)
      * tree key: iff k is the query itself or an ancestor (ancestor_mask)."""

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ANN001
        kv_is_tree = kv_idx >= base
        qn = (q_idx - base).clamp(min=0)
        kn = (kv_idx - base).clamp(min=0)
        return torch.where(
            kv_is_tree,
            ancestor_mask[qn, kn],
            torch.ones((), dtype=torch.bool, device=ancestor_mask.device),
        )

    return mask_mod


def build_level_attn_mask(
    topo: "TreeTopology", level: int, prefix_len: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """drafter forward: flat 1-D bool custom_mask + slot DFS-ids for the
    level-`level` expansion (1..depth-1).

    At iteration `level` the query rows are the level-(level-1) input nodes
    (their KV is written this forward at slot-position base+dfs_id). Keys =
    `prefix_len` committed tokens + all `T` reserved draft slots (dfs-id order,
    seq_len = base+T). Query node q attends every prefix token + (self + its
    ancestors) among the draft slots — `build_ancestor_mask[q]`. Siblings,
    cousins, and unwritten future slots are masked (garbage excluded).

    Returns (custom_mask flat bool [n_in*(prefix_len+T)] row-major query-major,
    input_dfs_ids LongTensor [n_in] = the level-(level-1) node ids, whose
    slot-positions are base+input_dfs_ids).
    """
    A = build_ancestor_mask(topo.parent)  # [T, T] self + ancestors
    in_ids = topo.level_ids[level - 1]  # [n_in] query node dfs ids
    prefix_row = torch.ones(prefix_len, dtype=torch.bool)
    rows = [torch.cat([prefix_row, A[q]]) for q in in_ids.tolist()]
    return torch.cat(rows), in_ids.clone()


def tree_slot_mapping(
    block_table_row: torch.Tensor, base: int, dfs_ids: torch.Tensor, block_size: int
) -> torch.Tensor:
    """sibling KV slots: physical slot for each input node, placing its KV
    at logical position `base + dfs_id` (distinct per node → no sibling collision).
    Mirrors the eagle slot kernel (utils.py:88) slot = block_table[pos//bs]*bs +
    pos%bs, but one slot PER NODE at its DFS-id position instead of one +1 step.

    block_table_row: [n_blocks] physical block ids for the request.
    Returns [n_in] int slot ids aligned with dfs_ids.
    """
    pos = base + dfs_ids.to(torch.long)
    return block_table_row[pos // block_size].to(torch.long) * block_size + (
        pos % block_size
    )


def build_verify_attn_mask(topo: "TreeTopology", prefix_len: int) -> torch.Tensor:
    """E2 target-verify forward: flat 1-D bool custom_mask for ONE request.

    The target verifies all T draft nodes in a single forward. Its query (and the
    newly-written KV) is laid out as [anchor, node_0 .. node_{T-1}] = T+1 tokens,
    where `anchor` is the token that seeded the drafts (the tree root, re-fed as
    query 0). Keys = `prefix_len` committed tokens + those T+1 new tokens
    (seq_len = prefix_len + T+1). Rule per query row:
      * anchor (row 0): all prefix + itself.
      * draft node t (row 1+t): all prefix + the anchor + (self + its ancestors).
    Siblings/cousins/future-branch nodes are masked — so each verify slot's logits
    are computed over exactly the path root->...->node, never sibling-contaminated.

    Built by lifting the draft parent array into a (T+1)-node space where node 0 is
    the anchor/root: draft t (DFS id) becomes verify node 1+t, and its parent is
    1+parent[t] (or the anchor 0 when parent[t] == -1, i.e. a level-0 node). The
    ancestor mask over that lifted array gives self+ancestors-incl-anchor per row.

    Returns the flat bool [ (T+1) * (prefix_len + T + 1) ] in FlashInfer's
    query-major layout (concat over query rows of [prefix-row | tree-tail]).
    """
    A = build_ancestor_mask(verify_lifted_parent(topo.parent))  # [(T+1)²]
    prefix_row = torch.ones(prefix_len, dtype=torch.bool)
    rows = [torch.cat([prefix_row, A[q]]) for q in range(topo.num_nodes + 1)]
    return torch.cat(rows)


def verify_lifted_parent(parent: torch.Tensor) -> torch.Tensor:
    """Lift the T-node draft `parent` into the (T+1)-node space where node 0 is the
    ANCHOR (the tree root / committed-state base, re-fed as the first spec token)
    and draft node t becomes node 1+t with parent 1+parent[t] (or the anchor 0 when
    parent[t] == -1, i.e. a level-0 node).

    This is the space the spec-verify forward AND the GDN spec forward actually
    operate in: both process num_spec+1 = T+1 positions [anchor, node_0..node_{T-1}].
    Used by build_verify_attn_mask (softmax mask) and the E5 GDN commit `keep`. At
    width-1 the lift is a chain of length T+1, so an ancestor path == a prefix.
    """
    par = parent.tolist()
    T = len(par)
    # CPU-pinned host metadata (see TreeTopology.__init__): callers move it to the
    # device explicitly, and several read it on the host.
    return torch.tensor(
        [-1] + [(1 + par[t]) if par[t] >= 0 else 0 for t in range(T)],
        dtype=torch.int32,
        device="cpu",
    )


def ancestor_path_mask(parent: torch.Tensor, leaf: int) -> torch.Tensor:
    """Bool [T] mask of the path root->leaf: True iff a node is `leaf` itself or a
    strict ancestor of it (equivalently build_ancestor_mask(parent)[leaf], but
    O(depth)).

    This is the `keep` set for the E5 GDN tree-WY commit: exactly the draft nodes
    whose rank-1 corrections enter the accepted leaf's recurrent state. For a
    width-1 (chain) tree it is the contiguous prefix {0..leaf}, so the commit
    reduces to the proven D2b chain reconstruct; for a branch leaf it is the
    (possibly non-contiguous) DFS ids along that branch.
    """
    par = parent.tolist()
    m = torch.zeros(len(par), dtype=torch.bool, device=parent.device)
    c = leaf
    while c >= 0:
        m[c] = True
        c = par[c]
    return m


def tree_greedy_accept(
    parent: torch.Tensor,  # [T] DFS-pre-order parent (root = -1)
    draft_row: torch.Tensor,  # [T] draft token id per node (DFS order)
    pred_row: torch.Tensor,  # [T+1] target greedy argmax per verify slot
) -> tuple[list[int], int]:
    """E4 greedy TREE verification for ONE request: descend from the root,
    following the target's greedy argmax as far as the draft tree contains it.

    pred_row = [pred_anchor, pred_node_0, .., pred_node_{T-1}] where pred_anchor
    is the target's greedy token after the prefix (predicts a level-0 child) and
    pred_node_t is the target's greedy token after node t (predicts t's child).

    At each step, among the children of the current node, accept the (unique —
    children are distinct top-k tokens) child whose draft token equals the
    target's greedy prediction; stop at the first node with no matching child.

    Returns (accepted_nodes, bonus_token):
      * accepted_nodes: DFS ids along the accepted path, root->...->leaf (possibly
        empty if even the first token diverges).
      * bonus_token: the target's greedy token at the divergence point (= the
        recovered token). The emitted sequence is
        [draft_row[c] for c in accepted_nodes] + [bonus_token]; because each
        draft_row[c] equals the argmax that selected it, this IS the target's
        greedy continuation — identical acceptance semantics to the chain sampler,
        which this reduces to exactly when every node has one child (width 1).
    """
    par = parent.tolist()
    draft = draft_row.tolist()
    pred = pred_row.tolist()
    # Bound the descent by the actual number of draft nodes for THIS req, not the
    # static topo size: in a mixed prefill+spec batch a prefill req has an EMPTY
    # draft_row (and a 1-elem pred_row = just its sampled token), so len(draft)=0
    # -> no children matched -> (accepted=[], bonus=pred[0]). For a spec req
    # len(draft)==len(par), so this is unchanged.
    T = len(draft)
    cur = -1  # root / anchor
    want = pred[0]  # target's greedy first token
    accepted: list[int] = []
    while True:
        found = -1
        for c in range(T):
            if par[c] == cur and draft[c] == want:
                found = c  # DFS order => leftmost matching child first
                break
        if found < 0:
            break
        accepted.append(found)
        cur = found
        want = pred[1 + cur]  # target's greedy token after the accepted node
    return accepted, want


def tree_sampled_accept(
    parent: torch.Tensor,  # [T] DFS-pre-order parent (root = -1)
    draft_row: torch.Tensor,  # [T] draft token id per node (DFS), each i.i.d.~q
    q_slots: torch.Tensor,  # [T+1, V] draft dist per slot (0=anchor, 1+t=node t)
    p_slots: torch.Tensor,  # [T+1, V] (constrained) target dist per slot
    generator: "torch.Generator | None" = None,
) -> tuple[list[int], list[int]]:
    """LOSSLESS temp>0 TREE verification for ONE request (SpecInfer / SpecTr
    recursive multi-candidate rejection sampling). The temp>0 analogue of
    `tree_greedy_accept`: it produces an emitted token distributed EXACTLY as the
    (constrained) target p, so tree spec-decode does not change the output
    distribution — the same losslessness the chain rejection sampler guarantees.

    Spec + unbiasedness proof (numpy Monte-Carlo):
    the per-request reference implementation in the tests — this
    mirrors it line-for-line. Descend one root->leaf path:

      at the current node (slot s): q = q_slots[s], p_res = p_slots[s]
      for each drafted child c (DFS order), token x = draft_row[c]:
          accept x with prob min(1, p_res[x] / q[x])
          on reject:  p_res <- normalize(relu(p_res - q))   ; try next child
      accepted child -> emit x, descend into c ; all reject -> emit recovered ~ p_res

    Invariants (MUST hold — see the ref's header): q and p are constrained by the
    SAME temperature/top-k/top-p, and each child was sampled i.i.d. WITH
    replacement from its parent slot's q (so q[x] > 0 for every drafted x). At
    width 1 this reduces to single-draft chain speculative sampling; with
    point-mass p,q (temperature 0) it reduces to `tree_greedy_accept`.

    Returns (accepted_nodes, emitted_tokens): DFS ids along the accepted path, and
    the emitted [accepted-path tokens] + one recovered/bonus token (the lossless
    continuation) — identical output structure to the greedy path.

    NOTE (perf): a per-node Python walk with scalar `.item()` reads; correct and
    cheap on CPU. On GPU the scalar reads sync — acceptable because verify is not
    the bottleneck (the target forward dominates); do NOT pre-fuse this. Batch /
    kernelize ONLY if profiling later shows the verify step is material.
    """
    T = int(parent.shape[0])
    par = parent.tolist()
    draft = draft_row.tolist()
    device = q_slots.device

    def _sample(dist: torch.Tensor) -> "int | None":
        tot = dist.sum()
        if float(tot) <= 0.0:
            return None
        return int(torch.multinomial(dist / tot, 1, generator=generator).item())

    def _sample_int(dist: torch.Tensor, fallback: "torch.Tensor | None" = None) -> int:
        """`_sample` that always yields a token id.

        `_sample` returns None for a degenerate (all-zero) distribution, which the
        residual can become once enough mass has been subtracted. Emitting that
        None into `emitted: list[int]` would put a None in the returned token
        stream; fall back to the untouched slot distribution, then to argmax.
        """
        got = _sample(dist)
        if got is None and fallback is not None:
            got = _sample(fallback)
        if got is None:
            got = int(torch.argmax(dist).item())
        return got

    def _children(cur: int) -> list[int]:
        # DFS-ordered children of `cur` (cur == -1 is the anchor/root).
        return [c for c in range(T) if par[c] == cur]

    accepted: list[int] = []
    emitted: list[int] = []
    cur = -1
    while True:
        s = 0 if cur == -1 else 1 + cur
        q = q_slots[s]
        # float64 residual: matches the numpy ref and keeps relu(p-q) normalization
        # stable when p and q nearly cancel.
        p_res = p_slots[s].to(torch.float64).clone()
        q64 = q.to(torch.float64)
        acc = -1
        for c in _children(cur):
            x = draft[c]
            u = float(torch.rand((), generator=generator, device=device).item())
            qx = float(q64[x])
            ratio = (float(p_res[x]) / qx) if qx > 0.0 else 0.0
            if u < min(1.0, ratio):
                acc = c
                break
            p_res = torch.clamp(p_res - q64, min=0.0)
            tot = p_res.sum()
            if float(tot) > 0.0:
                p_res = p_res / tot
        if acc < 0:
            emitted.append(_sample_int(p_res, p_slots[s]))
            break
        emitted.append(draft[acc])
        accepted.append(acc)
        cur = acc
        if not _children(cur):  # leaf -> bonus token from the target at its slot
            emitted.append(_sample_int(p_slots[1 + cur]))
            break
    return accepted, emitted


def build_children_table(parent: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Static [T+1, maxw] child table for the graph-safe batched verify. Row s is
    the DFS ids of the children of the node at verify-slot s (slot 0 = anchor/root
    cur=-1, slot 1+t = node t), ascending, -1 padded to maxw = max fanout. Built
    once from the static topology (no per-step host work)."""
    T = int(parent.shape[0])
    par = parent.tolist()
    kids: dict[int, list[int]] = {s: [] for s in range(T + 1)}
    for c in range(T):
        cur = par[c]
        kids[0 if cur < 0 else 1 + cur].append(c)
    maxw = max((len(v) for v in kids.values()), default=1)
    tab = torch.full((T + 1, maxw), -1, dtype=torch.long)
    for s, v in kids.items():
        for j, c in enumerate(v):  # already ascending (DFS ids appended in order)
            tab[s, j] = c
    return tab, maxw


def tree_sampled_verify_batched(
    parent: torch.Tensor,  # [T] DFS parent (root -1)  (for depth/leaf bounds)
    depth: int,  # topo.depth (number of levels)
    draft_ids: torch.Tensor,  # [B, T] drafted token per node (DFS)
    q_slots: torch.Tensor,  # [B, T+1, V] draft dist per slot
    p_slots: torch.Tensor,  # [B, T+1, V] (constrained) target dist per slot
    children_table: torch.Tensor,  # [T+1, maxw] from build_children_table
    maxw: int,
    u_accept: torch.Tensor,  # [B, depth+1, maxw] uniforms for the accept tests
    u_recover: torch.Tensor,  # [B, depth+1] uniforms for the recovered-token draws
    placeholder_id: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CUDA-GRAPH-CAPTURABLE lossless tree verify: the exact algorithm of
    `tree_sampled_accept`, reformulated as FIXED-shape vectorized ops with NO host
    syncs and NO data-dependent host control flow (all branching via torch.where /
    masks; all randomness passed in as tensors). Loops a fixed depth+1 steps; the
    per-node candidate loop is a fixed maxw. Safe to run under a CUDA graph.

    The leaf BONUS token falls out naturally: after accepting to a leaf, the next
    fixed step lands on a slot whose children are all -1 -> no candidate accepts ->
    the recovered draw from p_res (== that slot's target p, no rejections) IS the
    bonus. Returns (out_tokens [B, depth+1] int32 with placeholder padding,
    leaf_ids [B] long, -1 if the first token diverged). Validated == the per-request
    per-request reference implementation in the tests.
    """
    B, S, V = q_slots.shape
    device = q_slots.device
    ar = torch.arange(B, device=device)
    cur_slot = torch.zeros(B, dtype=torch.long, device=device)  # anchor
    done = torch.zeros(B, dtype=torch.bool, device=device)
    emit_idx = torch.zeros(B, dtype=torch.long, device=device)
    leaf = torch.full((B,), -1, dtype=torch.long, device=device)
    out = torch.full((B, depth + 1), placeholder_id, dtype=torch.int32, device=device)

    for d in range(depth + 1):
        q = q_slots[ar, cur_slot]  # [B, V]
        p_res = p_slots[ar, cur_slot].to(torch.float32).clone()
        kids = children_table.to(device)[cur_slot]  # [B, maxw]
        picked = torch.zeros(B, dtype=torch.bool, device=device)
        chosen = torch.full((B,), -1, dtype=torch.long, device=device)
        for j in range(maxw):
            c = kids[:, j]  # [B] child id or -1
            active = (c >= 0) & (~done) & (~picked)
            cc = c.clamp(min=0)
            x = draft_ids[ar, cc]  # [B] token id
            qx = q[ar, x].clamp_min(1e-30)
            px = p_res[ar, x]
            acc = active & (u_accept[:, d, j] < (px / qx).clamp(max=1.0))
            chosen = torch.where(acc, c, chosen)
            picked = picked | acc
            rej = (active & (~acc)).unsqueeze(1)  # [B,1]
            newp = torch.relu(p_res - q)
            newp = newp / newp.sum(-1, keepdim=True).clamp_min(1e-30)
            p_res = torch.where(rej, newp, p_res)
        newly_active = ~done
        # recovered token (used where no child accepted): inverse-CDF of p_res.
        cdf = torch.cumsum(p_res, dim=-1)
        thr = (u_recover[:, d] * cdf[:, -1]).unsqueeze(1)
        rec_tok = torch.searchsorted(cdf, thr).squeeze(1).clamp(max=V - 1)
        tok_acc = draft_ids[ar, chosen.clamp(min=0)]
        emit_tok = torch.where(picked, tok_acc, rec_tok).to(torch.int32)
        ei = emit_idx.clamp(max=depth)
        out[ar, ei] = torch.where(newly_active, emit_tok, out[ar, ei])
        emit_idx = emit_idx + newly_active.long()
        acc_mask = newly_active & picked
        leaf = torch.where(acc_mask, chosen, leaf)
        cur_slot = torch.where(acc_mask, 1 + chosen, cur_slot)
        done = done | (newly_active & (~picked))  # recovered -> finished
    return out, leaf


@dataclass
class TreeDraftResult:
    # [B, T] draft token ids, DFS pre-order within each request row.
    draft_token_ids: torch.Tensor
    # [T] request-local DFS parent (root = -1); shared across the batch.
    parent: torch.Tensor
    # [T] depth of each node (root child = depth 0).
    depth: torch.Tensor


def build_tree_reference(
    topo: TreeTopology,
    root_token: torch.Tensor,  # [B]   last accepted token per request
    root_hidden: torch.Tensor,  # [B,H] sampling hidden state for the root
    expand_fn,  # (tokens[M], hidden[M,H]) -> (logits, hidden_out)
) -> TreeDraftResult:
    """Reference breadth-first tree expansion (one expand per depth) that scatters
    into a DFS-ordered [B, T] buffer. Mirrors what E1b will do inside the proposer
    loop, but with an injectable model so it is testable on CPU."""
    B = root_token.shape[0]
    T = topo.num_nodes
    device = root_hidden.device
    draft = torch.full((B, T), -1, dtype=torch.long, device=device)

    # Current level's input: token + carried hidden, one row per live node.
    # Start with the single root per request.
    in_tokens = root_token.view(B, 1)  # [B, n_in]
    in_hidden = root_hidden.view(B, 1, -1)  # [B, n_in, H]

    for L in range(topo.depth):
        n_in = in_tokens.shape[1]
        H = in_hidden.shape[-1]
        flat_tok = in_tokens.reshape(B * n_in)
        flat_hid = in_hidden.reshape(B * n_in, H)
        logits, hidden_out = expand_fn(flat_tok, flat_hid)  # [B*n_in, V], [B*n_in, H]
        w = topo.widths[L]
        topk = logits.topk(w, dim=-1).indices  # [B*n_in, w]
        # (parent_idx, child_rank) flatten == sorted level-L id order (see header).
        children = topk.reshape(B, n_in * w)  # [B, n_children]
        ids_L = topo.level_ids[L].to(device)  # [n_children]
        draft[:, ids_L] = children

        if topo.depth > L + 1:
            # Next step's inputs are the level-L nodes just produced. Each carries
            # the hidden_out of its parent (a level-(L-1) node = a row of THIS
            # step, ordered as level_ids[L-1]). level_parent_local[L] maps each
            # level-L node to its parent's position among those rows.
            hid_out = hidden_out.reshape(B, n_in, H)
            par_local = topo.level_parent_local[L].to(device)  # [n_children_L]
            in_hidden = hid_out[:, par_local, :]  # [B, n_children_L, H]
            in_tokens = children  # [B, n_children_L] == next in

    return TreeDraftResult(
        draft_token_ids=draft,
        parent=topo.parent.to(device),
        depth=topo.node_depth.to(device),
    )
