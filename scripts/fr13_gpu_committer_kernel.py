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
        # int64 -1: `leaf` is loaded from the int64 leaves tensor, so `best_leaf = leaf`
        # reassigns to int64; a bare `-1` inits int32 and Triton rejects the branch-type
        # change ("initial value ... int32[] ... then block redefines it as int64[]").
        # Derive -1 from the int64 `nc` load (value-identical, no tl.full/tl.cast syntax risk).
        best_leaf = nc * 0 - 1  # node id of the winning leaf (path end)
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


# ---------------------------------------------------------------------------
# OPT-1 G2 (FR13_COMMITTER_SYNCKILL) device-arm kernels
# ---------------------------------------------------------------------------
# These are SUPERSET kernels of the byte-validated kernel above: they emit the
# SAME (out_tokens, row_len, accepted_row, best_lcp) decision AND additionally
# the accepted node-id PATH on device (acc_path[r, :best_lcp]). The accepted
# path is the only committer product the GDN durable-state advance re-derives on
# host (_winning_path_prefix); emitting it on device lets the SYNCKILL path do a
# device->device GDN-path writeback with NO host re-derivation. The integer
# decision is byte-for-byte identical to _fr13_committer_kernel (G2.c invariant:
# G2 moves transport, never the decision).
if _HAVE_TRITON:

    @triton.jit
    def _fr13_committer_leaves_kernel(
        parents_ptr,
        node_count_ptr,
        leaves_ptr,
        num_leaves_ptr,
        MAX_NODES: tl.constexpr,
    ):
        # One program per request: derive node-order leaves (nodes with no
        # child) array-free, matching the host _build_device_layout EXACTLY
        # (fallback to all-nodes when no leaf). Pure integer, no value dep.
        r = tl.program_id(0)
        nc = tl.load(node_count_ptr + r)
        base = r * MAX_NODES
        n_leaf = 0
        node = 0
        while node < nc:
            # node is a leaf iff no other node names it as parent.
            has_child = 0
            j = 0
            while j < nc:
                pj = tl.load(parents_ptr + base + j)
                if pj == node:
                    has_child = 1
                j += 1
            if has_child == 0:
                tl.store(leaves_ptr + base + n_leaf, node)
                n_leaf += 1
            node += 1
        if n_leaf == 0:
            # fallback: every node is a leaf (committer: leaves=range(nc)).
            node = 0
            while node < nc:
                tl.store(leaves_ptr + base + node, node)
                node += 1
            n_leaf = nc
        tl.store(num_leaves_ptr + r, n_leaf)

    @triton.jit
    def _fr13_committer_kernel_dev(
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
        acc_path_ptr,
        MAX_NODES: tl.constexpr,
        MAX_SPEC_LEN: tl.constexpr,
    ):
        # Byte-for-byte the decision of _fr13_committer_kernel (transcribed
        # verbatim) PLUS acc_path[r, :best_lcp] = best_path node ids (the
        # accepted prefix node ids, root..accepted, in committer order).
        r = tl.program_id(0)
        nc = tl.load(node_count_ptr + r)
        nl = tl.load(num_leaves_ptr + r)
        base = r * MAX_NODES
        out_base = r * (MAX_SPEC_LEN + 1)
        path_base = r * MAX_NODES

        best_lcp = -1
        best_leaf = nc * 0 - 1  # int64 -1 (match int64 leaf loads; see _fr13_committer_kernel)

        li = 0
        while li < nl:
            leaf = tl.load(leaves_ptr + base + li)
            depth = 0
            node = leaf
            guard = 0
            while (node >= 0) & (node < nc) & (guard <= nc):
                depth += 1
                node = tl.load(parents_ptr + base + node)
                guard += 1
            lcp = 0
            p = 0
            broke = 0
            while (p < depth) & (broke == 0):
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

        depth = 0
        node = best_leaf
        guard = 0
        while (node >= 0) & (node < nc) & (guard <= nc):
            depth += 1
            node = tl.load(parents_ptr + base + node)
            guard += 1

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

        # accepted node PATH: best_path[p] = ancestor of best_leaf at
        # distance (depth-1-p) = the same node2 the token emit walked, stored
        # into acc_path[r, :best_lcp].
        p = 0
        while p < best_lcp:
            up = depth - 1 - p
            node2 = best_leaf
            k = 0
            while k < up:
                node2 = tl.load(parents_ptr + base + node2)
                k += 1
            tl.store(acc_path_ptr + path_base + p, node2)
            p += 1

        if depth > 0:
            bonus_tok = 0
            if best_lcp < depth:
                up = depth - 1 - best_lcp
                node2 = best_leaf
                k = 0
                while k < up:
                    node2 = tl.load(parents_ptr + base + node2)
                    k += 1
                bonus_tok = tl.load(ptgt_ptr + base + node2)
            else:
                if best_lcp > 0:
                    up = depth - 1 - (best_lcp - 1)
                    node2 = best_leaf
                    k = 0
                    while k < up:
                        node2 = tl.load(parents_ptr + base + node2)
                        k += 1
                    bonus_tok = tl.load(stgt_ptr + base + node2)
                else:
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


# ---------------------------------------------------------------------------
# OPT-1 G2 (FR13_COMMITTER_SYNCKILL): device-resident committer + side-stream
# readback. The decision runs on the kernel reading DEVICE tensors (no host
# list = no committer-input :6761 sync); the outputs stay DEVICE-resident
# (no .cpu() = no committer-output sync); a dedicated side stream copies the
# packed outputs to a pinned host mirror with a CUDA event, so the host
# materialisation is LAZY (event.synchronize() only on first host access),
# letting the main thread launch the device->device writeback + next forward
# without blocking. Pure-integer, location-only: byte-identical to the host
# committer for every input.
# ---------------------------------------------------------------------------

# Module-level side stream (mirrors native AsyncGPUModelRunnerOutput's
# async_output_copy_stream). Allocated once, on first use, on the committer's
# device. Never the default compute stream, so the readback copy does not gate
# the main thread.
_FR13_SYNCKILL_SIDE_STREAM = None

# Persistent readback staging (device buffer, pinned host mirror, CUDA event,
# rec-flag), keyed by required element count. Reused across steps (B=1 steady
# state is a single fixed size), so the readback has NO per-step allocation; the
# event guards pinned-buffer reuse exactly like _fr13_eager_pack_stage. Grows
# only when capacity is exceeded.
_FR13_SYNCKILL_STAGE = {}


def _fr13_synckill_stage(total, device):
    entry = _FR13_SYNCKILL_STAGE.get("rb")
    if (
        entry is None
        or entry[0].numel() < total
        or entry[0].device != device
    ):
        cap = max(int(total), 256)
        entry = (
            torch.empty((cap,), dtype=torch.int64, device=device),
            torch.empty((cap,), dtype=torch.int64, pin_memory=True),
            torch.cuda.Event(),
            [False],
        )
        _FR13_SYNCKILL_STAGE["rb"] = entry
    return entry


def _fr13_synckill_side_stream(device):
    global _FR13_SYNCKILL_SIDE_STREAM
    if _FR13_SYNCKILL_SIDE_STREAM is None:
        _FR13_SYNCKILL_SIDE_STREAM = torch.cuda.Stream(device=device)
    return _FR13_SYNCKILL_SIDE_STREAM


def _build_device_layout_from_device(
    parents_dev, drafts_dev, ptgt_dev, stgt_dev, counts, device
):
    """Pack ragged DEVICE committer inputs into the fixed-stride kernel layout
    WITHOUT a host round-trip of the values.

    Only ``counts`` (the per-request node count, ``n_req`` ints — on the B=1
    serving path already a host list) drives the shape; the token-id / parent
    VALUES never touch the host. The per-request slices are scattered on-device
    (``out[r, :nc].copy_(src[start:start+nc])``), and the node-order leaves are
    derived ON DEVICE by ``_fr13_committer_leaves_kernel`` (matching the host
    ``_build_device_layout`` leaf rule exactly). Returns the device tensors the
    decision kernel consumes plus ``max_nodes``.
    """
    counts = [int(c) for c in counts]
    n_req = len(counts)
    max_nodes = max(counts) if counts else 1
    max_nodes = max(1, max_nodes)

    parents = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64, device=device)
    drafts = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64, device=device)
    ptgt = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64, device=device)
    stgt = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64, device=device)
    leaves = torch.full((n_req, max_nodes), _PAD, dtype=torch.int64, device=device)
    node_count = torch.tensor(counts, dtype=torch.int64, device=device)
    num_leaves = torch.zeros((n_req,), dtype=torch.int64, device=device)

    # Device-side gather: cast/reshape to int64 once, scatter per-request
    # slices on the default compute stream (no sync, no host copy).
    p_src = parents_dev.reshape(-1).to(torch.int64)
    d_src = drafts_dev.reshape(-1).to(torch.int64)
    pt_src = ptgt_dev.reshape(-1).to(torch.int64)
    st_src = stgt_dev.reshape(-1).to(torch.int64)
    start = 0
    for r, nc in enumerate(counts):
        if nc:
            parents[r, :nc].copy_(p_src[start:start + nc])
            drafts[r, :nc].copy_(d_src[start:start + nc])
            ptgt[r, :nc].copy_(pt_src[start:start + nc])
            stgt[r, :nc].copy_(st_src[start:start + nc])
        start += nc

    _fr13_committer_leaves_kernel[(n_req,)](
        parents, node_count, leaves, num_leaves, MAX_NODES=int(max_nodes),
    )
    return parents, drafts, ptgt, stgt, leaves, node_count, num_leaves, max_nodes


def fr13_gpu_committer_device(
    parents_dev,
    drafts_dev,
    ptgt_dev,
    stgt_dev,
    bonus_dev,  # accepted for signature parity; bonus_self serving = unused
    counts,
    max_spec_len,
):
    """Run the integer committer on-GPU from DEVICE inputs; return DEVICE outputs.

    NO ``.cpu()``, NO host list of the per-node committer inputs (that is the
    :6761 sync this kills). Returns the dict of device tensors the SYNCKILL
    writeback consumes device->device:
      ``out_tokens`` [n_req, max_spec_len+1] (_PAD padded)
      ``row_len``    [n_req]
      ``accepted_row`` [n_req]
      ``best_lcp``   [n_req]
      ``acc_path``   [n_req, max_nodes] accepted node-id path (_PAD padded)
    plus ``max_nodes`` / ``max_spec_len`` for downstream slicing.
    """
    if torch is None or not _HAVE_TRITON:
        raise RuntimeError("fr13_gpu_committer_device requires torch + triton")
    device = parents_dev.device
    counts = [int(c) for c in counts]
    n_req = len(counts)
    if n_req == 0:
        return {
            "out_tokens": torch.empty((0, int(max_spec_len) + 1), dtype=torch.int64, device=device),
            "row_len": torch.empty((0,), dtype=torch.int64, device=device),
            "accepted_row": torch.empty((0,), dtype=torch.int64, device=device),
            "best_lcp": torch.empty((0,), dtype=torch.int64, device=device),
            "acc_path": torch.empty((0, 1), dtype=torch.int64, device=device),
            "max_nodes": 1,
            "max_spec_len": int(max_spec_len),
        }
    (
        parents,
        drafts,
        ptgt,
        stgt,
        leaves,
        node_count,
        num_leaves,
        max_nodes,
    ) = _build_device_layout_from_device(
        parents_dev, drafts_dev, ptgt_dev, stgt_dev, counts, device
    )
    out_tokens = torch.full(
        (n_req, int(max_spec_len) + 1), _PAD, dtype=torch.int64, device=device
    )
    row_len = torch.zeros((n_req,), dtype=torch.int64, device=device)
    accepted_row = torch.zeros((n_req,), dtype=torch.int64, device=device)
    best_lcp = torch.zeros((n_req,), dtype=torch.int64, device=device)
    acc_path = torch.full(
        (n_req, int(max_nodes)), _PAD, dtype=torch.int64, device=device
    )

    _fr13_committer_kernel_dev[(n_req,)](
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
        acc_path,
        MAX_NODES=int(max_nodes),
        MAX_SPEC_LEN=int(max_spec_len),
    )
    return {
        "out_tokens": out_tokens,
        "row_len": row_len,
        "accepted_row": accepted_row,
        "best_lcp": best_lcp,
        "acc_path": acc_path,
        "max_nodes": int(max_nodes),
        "max_spec_len": int(max_spec_len),
    }


def fr13_gpu_committer_device_readback(dev_out):
    """Side-stream + CUDA-event readback of the device committer outputs.

    Packs the four scalar/row device outputs (out_tokens, row_len,
    accepted_row, best_lcp) plus acc_path into ONE staging device buffer on the
    default compute stream, then on a dedicated side stream copies it to a
    pinned host mirror with a CUDA event. Returns a LAZY materialiser: a closure
    that, on FIRST call, ``event.synchronize()`` then ``.tolist()`` (native
    get_output() shape). The main thread is free between the record() and the
    first materialise() call — that is the run-ahead the :6761 sync killed.

    Returns ``(materialise, event)``. ``materialise()`` -> the host
    ``(out_rows, accepted_rows, accepted_lens, acc_node_paths,
    accepted_token_rows)`` tuple byte-identical to the host committer
    (acc_node_paths = accepted node-id path = best_path[:best_lcp];
    accepted_token_rows = drafts on that path = out_rows[r][:best_lcp]).
    """
    if torch is None:
        raise RuntimeError("fr13_gpu_committer_device_readback requires torch")
    out_tokens = dev_out["out_tokens"]
    row_len = dev_out["row_len"]
    accepted_row = dev_out["accepted_row"]
    best_lcp = dev_out["best_lcp"]
    acc_path = dev_out["acc_path"]
    n_req = int(out_tokens.size(0))
    device = out_tokens.device
    if n_req == 0:
        def _empty():
            return [], [], [], [], []
        return _empty, None

    ot_cols = int(out_tokens.size(1))
    ap_cols = int(acc_path.size(1))
    # one contiguous staging buffer: [out_tokens | row_len | accepted_row |
    # best_lcp | acc_path]; device-side slice copies on the compute stream.
    # best_lcp doubles as accepted_lens (no redundant slot).
    base_ot = n_req * ot_cols
    base_rl = base_ot + n_req
    base_ar = base_rl + n_req
    base_bl = base_ar + n_req
    base_ap = base_bl  # best_lcp doubles as accepted_lens; no separate slot
    total = base_ap + n_req * ap_cols
    staging_buf, pinned_buf, event, rec = _fr13_synckill_stage(total, device)
    if rec[0]:
        # Pinned-reuse guard: never rewrite the pinned mirror while a prior
        # async readback may still be reading it (native event-guard shape).
        event.synchronize()
    staging = staging_buf[:total]
    pinned = pinned_buf[:total]
    staging[:base_ot].copy_(out_tokens.reshape(-1))
    staging[base_ot:base_rl].copy_(row_len)
    staging[base_rl:base_ar].copy_(accepted_row)
    staging[base_ar:base_bl].copy_(best_lcp)
    staging[base_ap:total].copy_(acc_path.reshape(-1))

    side = _fr13_synckill_side_stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        pinned.copy_(staging, non_blocking=True)
        event.record(side)
    rec[0] = True

    materialised = {}

    def _materialise():
        if "v" in materialised:
            return materialised["v"]
        event.synchronize()
        vals = pinned.tolist()
        # offsets mirror the packing above EXACTLY:
        #   [:base_ot]=out_tokens, [base_ot:base_rl]=row_len,
        #   [base_rl:base_ar]=accepted_row, [base_ar:base_bl]=best_lcp,
        #   [base_bl:base_ap]=accepted_lens(==best_lcp), [base_ap:]=acc_path
        ot = vals[:base_ot]
        rl = vals[base_ot:base_rl]
        ar = vals[base_rl:base_ar]
        bl = vals[base_ar:base_bl]
        ap = vals[base_ap:total]
        out_rows = []
        accepted_rows = []
        accepted_lens = []
        acc_node_paths = []
        accepted_token_rows = []
        for r in range(n_req):
            rln = int(rl[r])
            row = [int(x) for x in ot[r * ot_cols:r * ot_cols + rln]]
            out_rows.append(row)
            accepted_rows.append(int(ar[r]))
            lcp = int(bl[r])
            accepted_lens.append(lcp)
            acc_node_paths.append(
                [int(x) for x in ap[r * ap_cols:r * ap_cols + lcp]]
            )
            # accepted_token_rows = drafts on the accepted path = the accepted
            # prefix of out_rows (the first `lcp` committed tokens, before the
            # one bonus/correction token).
            accepted_token_rows.append([int(x) for x in row[:lcp]])
        v = (
            out_rows,
            accepted_rows,
            accepted_lens,
            acc_node_paths,
            accepted_token_rows,
        )
        materialised["v"] = v
        return v

    return _materialise, event


def fr13_gpu_committer_device_full(
    parents_dev,
    drafts_dev,
    ptgt_dev,
    stgt_dev,
    bonus_dev,
    counts,
    max_spec_len,
):
    """OPT-1 G2 entry: device-resident committer + side-stream readback.

    Runs the decision kernel on the DEVICE inputs (no committer-input :6761
    sync), launches the side-stream output readback (no inline .cpu()), and
    returns ``(dev_out, materialise)``:
      ``dev_out``     = the dict of device output tensors (for the device->device
                        same-step serving writeback);
      ``materialise`` = the lazy host materialiser -> ``(out_rows,
                        accepted_rows, accepted_lens, accepted_node_paths,
                        accepted_token_rows)`` byte-identical to the host
                        committer (event.synchronize() only on first call).
    The main thread is free between this return and the first materialise()
    call -- it launches the device->device writeback + the next forward without
    blocking on the committer-output sync.
    """
    dev_out = fr13_gpu_committer_device(
        parents_dev, drafts_dev, ptgt_dev, stgt_dev, bonus_dev, counts,
        max_spec_len,
    )
    materialise, _event = fr13_gpu_committer_device_readback(dev_out)
    return dev_out, materialise


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
