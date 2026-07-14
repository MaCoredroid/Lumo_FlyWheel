#!/usr/bin/env python3
"""FR13 SuffixSource — CPU-only, model-free draft-token source (ISOLATED).

Per-request "suffix decoding": match the request's current suffix (the last k
committed tokens) against earlier occurrences of the same k-gram in the
request's OWN history and propose the tokens that historically followed.
Retrieval / n-gram drafting: ZERO model forward, ZERO lm_head, pure host lookup.

Losslessness is NOT this module's job. The committer (rejection sampler) verifies
every candidate against the target, so a bad suffix candidate can only be
REJECTED, never wrong-accepted (FR13_DRAFTER_INTERFACE_DESIGN.md, Front 3). This
module optimises MATCH LOGIC + cheap amortised-O(1)-per-token host cost. A bug
here loses SPEED, never correctness.

Index (per request, per k in [min_k, max_k]):
    dict  k-gram(tuple of the k tokens)  ->  deque of positions p where that
    k-gram was FOLLOWED by tokens[p] in history (capped, most-recent kept).
update() registers only the newly-arrived follower positions (rolling; NEVER a
rebuild, NEVER an O(n) rescan). propose() does O(max_k) dict lookups + a bounded
gather over <=max_occurrences_per_key positions. Fully deterministic (no RNG):
longest suffix match wins; ties break by frequency then recency — the draw order
the committer replays must be reproducible.
"""
from collections import deque

__all__ = ["SuffixSource"]


def _to_int_list(token_ids):
    """Accept a python list or a 1-D int tensor / ndarray at the boundary."""
    tl = getattr(token_ids, "tolist", None)
    seq = tl() if tl is not None else token_ids
    return [int(t) for t in seq]


class _ReqState:
    __slots__ = ("tokens", "index")

    def __init__(self, min_k, max_k):
        self.tokens = []                               # full committed stream
        self.index = {k: {} for k in range(min_k, max_k + 1)}


class SuffixSource:
    def __init__(self, max_k=8, min_k=2, max_candidates_per_node=4,
                 max_occurrences_per_key=64):
        assert 1 <= min_k <= max_k, "need 1 <= min_k <= max_k"
        self.max_k = int(max_k)
        self.min_k = int(min_k)
        self.max_candidates_per_node = int(max_candidates_per_node)
        self.max_occurrences_per_key = int(max_occurrences_per_key)
        self.reqs = {}                                 # req_id -> _ReqState
        self.total_proposals = 0
        self.matched_proposals = 0

    # ---- engagement needle -------------------------------------------------
    @property
    def match_rate(self):
        """Fraction of propose() calls that returned matched=True (0.0 if none).
        0% => the source is vacuous / dead weight."""
        n = self.total_proposals
        return self.matched_proposals / n if n else 0.0

    # ---- ingest ------------------------------------------------------------
    def update(self, req_id, token_ids):
        """Incrementally ingest newly-committed tokens; amortised O(1)/token.
        Registers ONLY the new follower positions — no rebuild, no rescan."""
        toks = _to_int_list(token_ids)
        if not toks:
            return
        st = self.reqs.get(req_id)
        if st is None:
            st = self.reqs[req_id] = _ReqState(self.min_k, self.max_k)
        tokens = st.tokens
        index = st.index
        cap = self.max_occurrences_per_key
        min_k, max_k = self.min_k, self.max_k
        start = len(tokens)
        tokens.extend(toks)
        n = len(tokens)
        for i in range(start, n):        # token i is the follower of k-gram [i-k:i]
            for k in range(min_k, max_k + 1):
                j = i - k
                if j < 0:
                    break                # larger k is also out of range
                key = tuple(tokens[j:i])
                bucket = index[k].get(key)
                if bucket is None:
                    bucket = index[k][key] = deque(maxlen=cap)
                bucket.append(i)

    # ---- propose -----------------------------------------------------------
    def propose(self, req_id, num_nodes):
        """Up to num_nodes deep: a spine chain + per-depth branch candidates.
        {"spine":[..], "branch_candidates":[[..],..], "match_len":int, "matched":bool}."""
        self.total_proposals += 1
        empty = {"spine": [], "branch_candidates": [], "match_len": 0, "matched": False}
        st = self.reqs.get(req_id)
        if st is None or num_nodes < 1:
            return empty
        tokens = st.tokens
        n = len(tokens)
        if n < self.min_k:
            return empty

        # longest current-suffix k-gram that has a prior occurrence
        occ = None
        match_len = 0
        for k in range(min(self.max_k, n), self.min_k - 1, -1):
            bucket = st.index[k].get(tuple(tokens[n - k:n]))
            if bucket:                   # present => non-empty (created on append)
                occ, match_len = bucket, k
                break
        if occ is None:
            return empty

        positions = list(occ)            # ascending by insertion == ascending p

        def ranked(depth):
            """Distinct historical followers at `depth` past the match, ranked by
            (frequency, recency) — both strictly ordering => deterministic."""
            freq, last = {}, {}
            for p in positions:
                q = p + depth
                if q >= n:
                    continue
                t = tokens[q]
                freq[t] = freq.get(t, 0) + 1
                last[t] = q              # ascending p => last write == most recent
            order = sorted(freq, key=lambda t: (freq[t], last[t]), reverse=True)
            return order, last

        # spine: anchor on the most-frequent depth-0 token (tie: most recent),
        # then contiguous copy-forward of that historical run.
        order0, last0 = ranked(0)
        if not order0:
            return empty
        p_star = last0[order0[0]]
        spine_len = min(num_nodes, n - p_star)
        spine = tokens[p_star:p_star + spine_len]

        # branch candidates: extra tree width at each spine depth (committer-verified)
        cap_c = self.max_candidates_per_node
        branch = [(order0 if d == 0 else ranked(d)[0])[:cap_c] for d in range(spine_len)]

        self.matched_proposals += 1
        return {"spine": list(spine), "branch_candidates": branch,
                "match_len": match_len, "matched": True}

    # ---- lifecycle ---------------------------------------------------------
    def evict(self, req_id):
        """Drop a finished request's state; frees its tokens + index."""
        self.reqs.pop(req_id, None)
