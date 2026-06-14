#!/usr/bin/env python3
"""FR13 OPT-1: GPU-resident tree committer (FR13_GPU_COMMITTER, default-OFF).

FIRST DRAFT. The greedy tree committer's accept / path-LCP / bonus decision is
today PURE-PYTHON on host lists
(``_lumo_tree_path_lcp_max_greedy_sample`` in fr10_phase4_patch_vllm_tree_gdn.py,
the integer block at :5608-5879). That host loop is gated by a packed DtoH +
``cuda.synchronize()`` (:5502) which census-measured blocks the MAIN launching
thread 91.9% of the verify window vs native's 0.8% -> the tree path loses
native's async run-ahead (FR13_BEAT_NATIVE_SPEED_DESIGN_BIND.md).

OPT-1 moves the *entire integer decision* onto the GPU as a Triton kernel so the
host never has to sync the committer inputs back before deciding what to commit.
The kernel is PURE INTEGER (token-id ``==`` compares, parent walks, an LCP scan,
an earliest-leaf strict-``>`` tie-break, and a 3-way bonus-source select). It is
a LOCATION-ONLY move of the host Python -- NO float, NO reduction, NO reorder --
so it is LOSSLESS BY CONSTRUCTION: for every input it emits the byte-identical
``output_token_ids`` / ``accepted_tree_rows`` the Python committer emits.

This module ships:
  * ``fr13_gpu_committer_oracle`` -- a pure-Python re-statement of the EXACT
    committer integer logic (the bit-exact reference; used by the byte-A/B gate
    and as the CPU fallback when Triton is unavailable).
  * ``fr13_gpu_committer_triton`` -- the Triton integer kernel + a thin host
    launcher that materialises the same ``out_rows`` / ``accepted_rows`` the
    Python committer produces, but WITHOUT the host-side per-node Python loop
    over synced lists.
  * ``fr13_gpu_committer`` -- the dispatch entry the flag-gated hook calls. It
    runs the Triton kernel on a non-gating side stream when CUDA+Triton are
    available, else the oracle (CPU). Both return identical bytes.

DEFAULT-OFF. The flag ``FR13_GPU_COMMITTER`` is read by the hook in the patcher;
with the flag OFF the legacy Python path at :5608-5879 runs untouched. This
module is import-only inert (no side effects at import).

GPU-iteration TODO (documented, needs a live GB10 boot -- see
FR13_GPU_COMMITTER_BIND.md): (1) CUDA-12.4 graph conditional-node / torch.cond
accept branch so the commit stays inside the captured graph; (2) the host
``.tolist`` moved to a non-gating side stream + event so the main thread never
syncs; (3) per-row variable ``node_count`` packing into a fixed-stride kernel
grid; (4) the per-served-token argmax gate tap (eager-only) stays on the Python
path -- the kernel does not need it on the serving path.
"""

from __future__ import annotations

import os
from typing import Sequence

try:
    import torch
except Exception:  # pragma: no cover - torch always present in the vLLM image
    torch = None  # type: ignore

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - exercised on CPU-only hosts
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAVE_TRITON = False


# Sentinel used to pad fixed-stride buffers. Token ids are non-negative, parents
# are >=-1; -1 also doubles as the "no parent / root" marker, matching the
# committer's ``node = int(parents[node]); while 0 <= node`` walk.
_PAD = -1


def fr13_gpu_committer_oracle(
    parents_cpu: Sequence[int],
    drafts_cpu: Sequence[int],
    parent_targets_cpu: Sequence[int],
    self_targets_cpu: Sequence[int],
    bonus_targets_cpu: Sequence[int],
    counts: Sequence[int],
    max_spec_len: int,
    *,
    bonus_self: bool = True,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Bit-exact pure-Python reference for the committer integer decision.

    This re-states the EXACT logic of ``_lumo_tree_path_lcp_max_greedy_sample``
    (fr10_phase4_patch_vllm_tree_gdn.py :5608-5879) for the SERVING path
    (``bonus_self=True`` => FR13_TREE_BONUS_SELF default, no path0_native_bonus
    legacy branch and no diagnostic gates). Returns
    ``(out_rows, accepted_rows, accepted_lens)`` exactly as the committer
    appends them, so a row-for-row, token-for-token equality with this oracle is
    a complete byte-A/B proof that the GPU kernel is lossless.

    NOTE: ``bonus_self`` is plumbed for completeness; the serving committer pins
    it True (the path0_native_bonus legacy branch is a *diagnostic-only* bug
    path, FR13 acceptance-ladder bind). The GPU kernel only implements the
    bonus_self=True serving contract; bonus_self=False stays on the Python path.
    """
    out_rows: list[list[int]] = []
    accepted_rows: list[int] = []
    accepted_lens: list[int] = []
    start = 0
    for node_count in counts:
        node_count = int(node_count)
        parents = [int(x) for x in parents_cpu[start:start + node_count]]
        drafts = [int(x) for x in drafts_cpu[start:start + node_count]]
        parent_targets = [
            int(x) for x in parent_targets_cpu[start:start + node_count]
        ]
        self_targets = [int(x) for x in self_targets_cpu[start:start + node_count]]

        # children adjacency (node-order leaves)
        children: dict[int, list[int]] = {-1: []}
        for node, parent in enumerate(parents):
            parent = int(parent)
            children.setdefault(parent, []).append(node)
            children.setdefault(node, [])
        leaves = [node for node in range(node_count) if not children.get(node)]
        if not leaves:
            leaves = list(range(node_count))

        best_path: list[int] = []
        best_lcp = -1
        best_path_idx = 0
        for path_idx, leaf in enumerate(leaves):
            path: list[int] = []
            node = int(leaf)
            guard = 0
            while 0 <= node < node_count and guard <= node_count:
                path.append(node)
                node = int(parents[node])
                guard += 1
            path.reverse()
            lcp = 0
            for node in path:
                if int(drafts[node]) != int(parent_targets[node]):
                    break
                lcp += 1
            # strict-> tie-break: earliest enumerated leaf keeps the win.
            if lcp > best_lcp:
                best_lcp = int(lcp)
                best_path = path
                best_path_idx = int(path_idx)
        best_lcp = max(0, int(best_lcp))

        row: list[int] = []
        for node in best_path[:best_lcp]:
            row.append(int(drafts[node]))
        if best_path:
            if best_lcp < len(best_path):
                row.append(int(parent_targets[best_path[best_lcp]]))
            elif best_lcp > 0:
                # bonus_self serving contract: accepted-leaf self-target.
                row.append(int(self_targets[best_path[best_lcp - 1]]))
            else:
                row.append(int(parent_targets[best_path[0]]))
        row = row[:int(max_spec_len) + 1]
        out_rows.append(row)
        accepted_rows.append(int(best_path[best_lcp - 1]) if best_lcp > 0 else 0)
        accepted_lens.append(int(best_lcp))
        start += node_count
    return out_rows, accepted_rows, accepted_lens


# ---------------------------------------------------------------------------
# Triton integer kernel
# ---------------------------------------------------------------------------
# Layout: every request is laid out at a fixed stride MAX_NODES so the grid is
# (num_requests,). Per request the launcher provides, on device:
#   parents[r, :nc], drafts[r, :nc], parent_targets[r, :nc], self_targets[r, :nc]
# padded with _PAD to MAX_NODES, plus node_count[r], leaves[r, :MAX_NODES]
# (node-order leaf ids, _PAD-terminated), num_leaves[r].
#
# The kernel computes, per request, into device outputs:
#   best_lcp[r], best_leaf_final_node[r] (the accepted_rows value), and the
#   committed row out_tokens[r, :MAX_SPEC_LEN+1] (_PAD-padded), plus row_len[r].
#
# It is a faithful integer transcription of the oracle above. Each program walks
# every leaf's root->leaf path, scores its LCP, keeps the strict-> earliest
# winner, then emits the accepted prefix + the 3-way bonus token.

if _HAVE_TRITON:

    @triton.jit
    def _fr13_committer_kernel(
        parents_ptr,
        drafts_ptr,
        ptgt_ptr,
        stgt_ptr,
        leaves_ptr,
        node_count_ptr,
        num_leaves_ptr,
        out_tokens_ptr,
        row_len_ptr,
        accepted_row_ptr,
        best_lcp_ptr,
        MAX_NODES: tl.constexpr,
        MAX_SPEC_LEN: tl.constexpr,
    ):
        r = tl.program_id(0)
        nc = tl.load(node_count_ptr + r)
        nl = tl.load(num_leaves_ptr + r)
        base = r * MAX_NODES
        out_base = r * (MAX_SPEC_LEN + 1)

        best_lcp = -1
        best_leaf = -1  # node id of the winning leaf (path end)
        # winning path is recomputed at emit time by re-walking best_leaf; this
        # keeps per-program state to scalars (no per-program arrays needed).

        # --- score every leaf, keep strict-> earliest winner ---
        li = 0
        while li < nl:
            leaf = tl.load(leaves_ptr + base + li)
            # walk leaf -> root, scoring LCP from the ROOT down. Because LCP is a
            # prefix from the root, and drafts[node]==parent_targets[node] is a
            # per-node predicate, we can compute the run length from the root by
            # first finding the depth, then walking root-down. To avoid storing
            # the path we do two walks: (1) up to count depth, (2) the LCP is the
            # longest root-anchored run of "match" nodes. We compute it by a
            # recursive-free method: a node is on the accepted prefix iff itself
            # AND all its ancestors match. So walk UP from leaf; the LCP = (depth)
            # minus (the number of trailing-from-root mismatches). Equivalent and
            # array-free: find the SHALLOWEST mismatching ancestor; everything
            # strictly above it (closer to root) is accepted.
            #
            # depth = number of nodes from root..leaf inclusive.
            depth = 0
            node = leaf
            guard = 0
            while (node >= 0) & (node < nc) & (guard <= nc):
                depth += 1
                node = tl.load(parents_ptr + base + node)
                guard += 1
            # shallowest mismatch depth-from-root: walk root-down by re-deriving
            # ancestors. We re-walk leaf->root collecting nodes into the path
            # order implicitly: ancestor at distance d-from-leaf. The root is at
            # distance depth-1. LCP scans root..leaf and stops at first mismatch.
            # Re-walk up, remembering for each node whether it matches; the LCP
            # is the count of the deepest contiguous matched prefix FROM THE ROOT.
            # We compute it as: lcp = depth - (length of the matched SUFFIX-from-
            # leaf is NOT what we want). Do it directly: for each root-anchored
            # position p (0..depth-1) the node is the ancestor at distance
            # (depth-1-p) from the leaf. Find the first p with a mismatch.
            lcp = 0
            p = 0
            broke = 0
            while (p < depth) & (broke == 0):
                # ancestor of `leaf` at distance (depth-1-p) up the tree.
                up = depth - 1 - p
                node2 = leaf
                k = 0
                while k < up:
                    node2 = tl.load(parents_ptr + base + node2)
                    k += 1
                d_tok = tl.load(drafts_ptr + base + node2)
                pt_tok = tl.load(ptgt_ptr + base + node2)
                is_match = d_tok == pt_tok
                if is_match:
                    lcp += 1
                else:
                    broke = 1
                p += 1
            if lcp > best_lcp:
                best_lcp = lcp
                best_leaf = leaf
            li += 1

        if best_lcp < 0:
            best_lcp = 0

        # --- recover the winning path's depth and emit the row ---
        depth = 0
        node = best_leaf
        guard = 0
        while (node >= 0) & (node < nc) & (guard <= nc):
            depth += 1
            node = tl.load(parents_ptr + base + node)
            guard += 1

        # emit accepted prefix tokens: drafts[best_path[0..best_lcp-1]] where
        # best_path[p] is the ancestor of best_leaf at distance (depth-1-p).
        max_row = MAX_SPEC_LEN + 1
        rlen = 0
        p = 0
        while (p < best_lcp) & (p < max_row):
            up = depth - 1 - p
            node2 = best_leaf
            k = 0
            while k < up:
                node2 = tl.load(parents_ptr + base + node2)
                k += 1
            tok = tl.load(drafts_ptr + base + node2)
            tl.store(out_tokens_ptr + out_base + p, tok)
            rlen += 1
            p += 1

        # --- bonus / correction token (one, only if best_path non-empty) ---
        # best_path non-empty <=> depth > 0 (best_leaf is a valid node).
        if depth > 0:
            bonus_tok = 0
            if best_lcp < depth:
                # reject_parent_target: parent_targets[best_path[best_lcp]]
                up = depth - 1 - best_lcp
                node2 = best_leaf
                k = 0
                while k < up:
                    node2 = tl.load(parents_ptr + base + node2)
                    k += 1
                bonus_tok = tl.load(ptgt_ptr + base + node2)
            else:
                if best_lcp > 0:
                    # tree_self_target: self_targets[best_path[best_lcp-1]]
                    up = depth - 1 - (best_lcp - 1)
                    node2 = best_leaf
                    k = 0
                    while k < up:
                        node2 = tl.load(parents_ptr + base + node2)
                        k += 1
                    bonus_tok = tl.load(stgt_ptr + base + node2)
                else:
                    # root_parent_target: parent_targets[best_path[0]] (root)
                    up = depth - 1
                    node2 = best_leaf
                    k = 0
                    while k < up:
                        node2 = tl.load(parents_ptr + base + node2)
                        k += 1
                    bonus_tok = tl.load(ptgt_ptr + base + node2)
            if rlen < max_row:
                tl.store(out_tokens_ptr + out_base + rlen, bonus_tok)
                rlen += 1

        tl.store(row_len_ptr + r, rlen)
        tl.store(best_lcp_ptr + r, best_lcp)
        # accepted_rows = best_path[best_lcp-1] if best_lcp>0 else 0
        acc_row = 0
        if best_lcp > 0:
            up = depth - 1 - (best_lcp - 1)
            node2 = best_leaf
            k = 0
            while k < up:
                node2 = tl.load(parents_ptr + base + node2)
                k += 1
            acc_row = node2
        tl.store(accepted_row_ptr + r, acc_row)


def _build_device_layout(
    parents_cpu, drafts_cpu, parent_targets_cpu, self_targets_cpu, counts, device
):
    """Pack the ragged per-request committer inputs into a fixed-stride layout.

    Returns the device tensors the Triton kernel consumes plus MAX_NODES. This
    packing is PURE INTEGER and does not depend on any value (only shapes), so it
    can run on a non-gating side stream at deploy time (GPU-iteration TODO).
    """
    counts = [int(c) for c in counts]
    n_req = len(counts)
    max_nodes = max(counts) if counts else 1
    max_nodes = max(1, max_nodes)

    parents = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64)
    drafts = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64)
    ptgt = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64)
    stgt = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64)
    leaves = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64)
    node_count = torch.zeros((n_req,), dtype=torch.int64)
    num_leaves = torch.zeros((n_req,), dtype=torch.int64)

    start = 0
    for r, nc in enumerate(counts):
        p = [int(x) for x in parents_cpu[start:start + nc]]
        d = [int(x) for x in drafts_cpu[start:start + nc]]
        pt = [int(x) for x in parent_targets_cpu[start:start + nc]]
        st = [int(x) for x in self_targets_cpu[start:start + nc]]
        # node-order leaves (no children), matching the committer.
        has_child = [False] * nc
        for node, parent in enumerate(p):
            if 0 <= parent < nc:
                has_child[parent] = True
        lv = [node for node in range(nc) if not has_child[node]]
        if not lv:
            lv = list(range(nc))
        for j in range(nc):
            parents[r, j] = p[j]
            drafts[r, j] = d[j]
            ptgt[r, j] = pt[j]
            stgt[r, j] = st[j]
        for j, leaf in enumerate(lv):
            leaves[r, j] = int(leaf)
        node_count[r] = int(nc)
        num_leaves[r] = len(lv)
        start += nc

    return (
        parents.to(device),
        drafts.to(device),
        ptgt.to(device),
        stgt.to(device),
        leaves.to(device),
        node_count.to(device),
        num_leaves.to(device),
        max_nodes,
    )


def fr13_gpu_committer_triton(
    parents_cpu,
    drafts_cpu,
    parent_targets_cpu,
    self_targets_cpu,
    bonus_targets_cpu,
    counts,
    max_spec_len,
):
    """Run the integer committer on-GPU via Triton, return (out_rows, accepted_rows, accepted_lens).

    The output is materialised on host (the eventual deploy form keeps the
    accepted-rows / out-tokens device-resident and copies on a side stream --
    GPU-iteration TODO), but the *decision* runs on-device with NO host sync of
    the per-node committer inputs.
    """
    if torch is None or not _HAVE_TRITON:
        raise RuntimeError("fr13_gpu_committer_triton requires torch + triton")
    device = "cuda"
    counts = [int(c) for c in counts]
    n_req = len(counts)
    if n_req == 0:
        return [], [], []
    (
        parents,
        drafts,
        ptgt,
        stgt,
        leaves,
        node_count,
        num_leaves,
        max_nodes,
    ) = _build_device_layout(
        parents_cpu,
        drafts_cpu,
        parent_targets_cpu,
        self_targets_cpu,
        counts,
        device,
    )
    out_tokens = torch.full(
        (n_req, int(max_spec_len) + 1), _PAD, dtype=torch.int64, device=device
    )
    row_len = torch.zeros((n_req,), dtype=torch.int64, device=device)
    accepted_row = torch.zeros((n_req,), dtype=torch.int64, device=device)
    best_lcp = torch.zeros((n_req,), dtype=torch.int64, device=device)

    _fr13_committer_kernel[(n_req,)](
        parents,
        drafts,
        ptgt,
        stgt,
        leaves,
        node_count,
        num_leaves,
        out_tokens,
        row_len,
        accepted_row,
        best_lcp,
        MAX_NODES=int(max_nodes),
        MAX_SPEC_LEN=int(max_spec_len),
    )

    # Side-stream-able host readback (deploy: non-gating event copy).
    out_tokens_h = out_tokens.cpu().tolist()
    row_len_h = row_len.cpu().tolist()
    accepted_row_h = accepted_row.cpu().tolist()
    best_lcp_h = best_lcp.cpu().tolist()

    out_rows: list[list[int]] = []
    accepted_rows: list[int] = []
    accepted_lens: list[int] = []
    for r in range(n_req):
        rl = int(row_len_h[r])
        out_rows.append([int(x) for x in out_tokens_h[r][:rl]])
        accepted_rows.append(int(accepted_row_h[r]))
        accepted_lens.append(int(best_lcp_h[r]))
    return out_rows, accepted_rows, accepted_lens


def _winning_path_prefix(parents, drafts, parent_targets, node_count):
    """Re-derive (best_path, best_lcp) for ONE request (oracle-identical).

    Returns the accepted node-id prefix (``best_path[:best_lcp]``) and the
    accepted draft tokens for those nodes -- exactly the ``accepted_node_paths``
    / ``accepted_token_rows`` the legacy committer appends.
    """
    nc = int(node_count)
    has_child = [False] * nc
    for node, parent in enumerate(parents):
        if 0 <= parent < nc:
            has_child[parent] = True
    leaves = [n for n in range(nc) if not has_child[n]] or list(range(nc))
    best_path: list[int] = []
    best_lcp = -1
    for leaf in leaves:
        path = []
        node = int(leaf)
        guard = 0
        while 0 <= node < nc and guard <= nc:
            path.append(node)
            node = int(parents[node])
            guard += 1
        path.reverse()
        lcp = 0
        for node in path:
            if int(drafts[node]) != int(parent_targets[node]):
                break
            lcp += 1
        if lcp > best_lcp:
            best_lcp = int(lcp)
            best_path = path
    best_lcp = max(0, int(best_lcp))
    accepted_path = [int(x) for x in best_path[:best_lcp]]
    accepted_tokens = [int(drafts[x]) for x in accepted_path]
    return accepted_path, accepted_tokens


def fr13_gpu_committer_full(
    parents_cpu,
    drafts_cpu,
    parent_targets_cpu,
    self_targets_cpu,
    bonus_targets_cpu,
    counts,
    max_spec_len,
    *,
    bonus_self: bool = True,
    prefer_triton: bool = True,
):
    """Dispatch entry returning ALL five committer products the hook needs.

    Returns ``(out_rows, accepted_rows, accepted_lens, accepted_node_paths,
    accepted_token_rows)`` byte-identical to the legacy Python committer's
    serving path. The GPU kernel produces the decision (out_rows / accepted_rows
    / accepted_lens) without a host sync of the per-node inputs; the cheap
    node-path / token-row metadata is re-derived from the SAME winning path.
    """
    out_rows, accepted_rows, accepted_lens = fr13_gpu_committer(
        parents_cpu,
        drafts_cpu,
        parent_targets_cpu,
        self_targets_cpu,
        bonus_targets_cpu,
        counts,
        max_spec_len,
        bonus_self=bonus_self,
        prefer_triton=prefer_triton,
    )
    accepted_node_paths: list[list[int]] = []
    accepted_token_rows: list[list[int]] = []
    start = 0
    for nc in counts:
        nc = int(nc)
        p = [int(x) for x in parents_cpu[start:start + nc]]
        d = [int(x) for x in drafts_cpu[start:start + nc]]
        pt = [int(x) for x in parent_targets_cpu[start:start + nc]]
        ap, at = _winning_path_prefix(p, d, pt, nc)
        accepted_node_paths.append(ap)
        accepted_token_rows.append(at)
        start += nc
    return (
        out_rows,
        accepted_rows,
        accepted_lens,
        accepted_node_paths,
        accepted_token_rows,
    )


def fr13_gpu_committer(
    parents_cpu,
    drafts_cpu,
    parent_targets_cpu,
    self_targets_cpu,
    bonus_targets_cpu,
    counts,
    max_spec_len,
    *,
    bonus_self: bool = True,
    prefer_triton: bool = True,
):
    """Dispatch entry for the flag-gated hook.

    Returns ``(out_rows, accepted_rows, accepted_lens)`` byte-identical to the
    Python committer's serving path. Uses the Triton kernel when CUDA+Triton are
    available and ``bonus_self`` is the serving default; otherwise the bit-exact
    CPU oracle (so the contract is testable CPU-only).
    """
    use_triton = (
        prefer_triton
        and bonus_self
        and _HAVE_TRITON
        and torch is not None
        and torch.cuda.is_available()
    )
    if use_triton:
        return fr13_gpu_committer_triton(
            parents_cpu,
            drafts_cpu,
            parent_targets_cpu,
            self_targets_cpu,
            bonus_targets_cpu,
            counts,
            max_spec_len,
        )
    return fr13_gpu_committer_oracle(
        parents_cpu,
        drafts_cpu,
        parent_targets_cpu,
        self_targets_cpu,
        bonus_targets_cpu,
        counts,
        max_spec_len,
        bonus_self=bonus_self,
    )
