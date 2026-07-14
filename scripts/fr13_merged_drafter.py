#!/usr/bin/env python3
"""FR13 merged drafter orchestration (MTP-k spine + Arctic-suffix grow-to-cat33333, ADAPTIVE Mode B).

Keeps the patcher edit THIN: the patcher injects hooks that call into this CPU-testable module,
mirroring the fr13_device_multidraft_kernel.py separate-module pattern. Ties together:
  - lifecycle: note_new_requests / ingest_committed / retire_requests (Arctic SuffixDecodingCache),
  - a rolling per-req committed-suffix buffer (for the speculate pattern),
  - decide_and_fill: per batch ROW (keyed by req_id) speculate -> adapter -> assemble -> fill, with
    the ADAPTIVE gate: SKIP the deep MTP spine forwards ONLY when ALL active rows have a full-depth
    Arctic match; otherwise return None => caller runs the full MTP loop (never-regress).

Gate-1 (committer contract) makes every filled candidate lossless regardless. The Arctic cache is
container-only (C++ ext); the module is import-free of arctic at top level and lets tests inject a
mock via set_cache_for_test(). All heavy correctness lives in the already-proven CPU components
(fr13_mtp_suffix_assembly 36/36, fr13_arctic_suffix_adapter 30/30, fr13_merged_fill 35/35).

v1 NOTE (flat-chain adapter): Arctic .token_ids is a flat continuation chain -> fills the deep
SPINE only; deep BRANCHES fall back to PAD when the MTP forwards were skipped (MTP topk for those
depths is unavailable). PAD is Gate-1 lossless (commits ~0) but the deep branch slots carry no
accept benefit in v1 -- the deep SPINE (Arctic) is the accept driver. A later version can widen
suffix_rel with Arctic's suffix-TREE alternatives to feed real deep branches. Root (d0) branches are
always MTP topk (computed pre-loop), so the near tree is unaffected.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_DEPTH = 5

_CACHE = None
_CACHE_INIT_FAILED = False
_COMMITTED: dict = {}     # req_id -> list[int] rolling recent-committed tokens (<= max_tree_depth)
STATS = {
    "speculate_fired": 0, "skip_fired": 0, "assembler_engaged": 0,
    "match_full": 0, "match_partial_norun": 0, "started": 0, "retired": 0,
}


def merged_on() -> bool:
    """Sidecar-gated (worker drops FR13_* env): /logs/fr13_draft_source_merged.arm written by the
    launcher when FR13_DRAFT_SOURCE=merged. Mirrors _fr13_committer_native_on()."""
    return os.path.exists("/logs/fr13_draft_source_merged.arm")


def get_cache(max_tree_depth: int = 24, max_cached_requests: int = 10000):
    """Lazily import Arctic + build the SuffixDecodingCache (container-only). Returns None (never
    raises) if arctic is unavailable -> caller falls back to pure MTP."""
    global _CACHE, _CACHE_INIT_FAILED
    if _CACHE is not None:
        return _CACHE
    if _CACHE_INIT_FAILED:
        return None
    try:
        from arctic_inference.suffix_decoding import SuffixDecodingCache
        _CACHE = SuffixDecodingCache(
            max_tree_depth=max_tree_depth, max_cached_requests=max_cached_requests
        )
    except Exception:
        _CACHE_INIT_FAILED = True
        _CACHE = None
    return _CACHE


def set_cache_for_test(mock):
    """Inject a mock SuffixDecodingCache (tests only)."""
    global _CACHE, _CACHE_INIT_FAILED
    _CACHE = mock
    _CACHE_INIT_FAILED = False


def reset_for_test():
    global _CACHE, _CACHE_INIT_FAILED, _COMMITTED
    _CACHE = None
    _CACHE_INIT_FAILED = False
    _COMMITTED = {}
    for k in STATS:
        STATS[k] = 0


# ---- lifecycle (RUNNER + COMMIT-SITE frames) --------------------------------
def note_new_requests(cache, req_id_to_prompt):
    """start_request for reqs not yet active (evict a stale cached response first).
    Port of vllm SuffixDecodingProposer :63-70."""
    if cache is None:
        return
    for req_id, prompt in req_id_to_prompt.items():
        try:
            if req_id not in cache.active_requests:
                if req_id in cache.cached_requests:
                    cache.evict_cached_response(req_id)
                cache.start_request(req_id, prompt)
                STATS["started"] += 1
        except Exception:
            pass


def ingest_committed(cache, req_id, accepted_ids, max_tree_depth=24):
    """COMMIT-SITE hook: feed the FULL accepted run to Arctic + update the rolling suffix buffer."""
    if not accepted_ids:
        return
    acc = [int(t) for t in accepted_ids]
    if cache is not None:
        try:
            cache.add_active_response(req_id, acc)
        except Exception:
            pass
    buf = _COMMITTED.setdefault(req_id, [])
    buf.extend(acc)
    if len(buf) > max_tree_depth:
        del buf[:-max_tree_depth]


def retire_requests(cache, gone_req_ids):
    """RUNNER frame: stop_request for reqs no longer in the batch. Port of :92-95."""
    for req_id in gone_req_ids:
        if cache is not None:
            try:
                cache.stop_request(req_id)
            except Exception:
                pass
        _COMMITTED.pop(req_id, None)
        STATS["retired"] += 1


# ---- the seam decision (:13361 speculate + :13860 fill) ---------------------
def decide_and_fill(cache, spec_row_req_ids, mtp_near_per_depth, mtp_topk_per_depth, mtp_k,
                    device, pad_token, max_spec_tokens=8, max_spec_factor=1.0, min_token_prob=0.0):
    """ADAPTIVE Mode B. Returns (spine_tokens, wide_topk, do_skip):
      * (spine_tokens, wide_topk, True)  when ALL active rows have a full-depth (>= N_DEPTH-mtp_k)
        Arctic match -> caller SKIPS the deep spine forwards and packs these columns.
      * (None, None, False)              otherwise -> caller runs the FULL MTP loop (never-regress);
        Arctic is IGNORED this step.
    mtp_near_per_depth[d]: an indexable of per-row MTP spine tokens for the mtp_k drafted depths
      (d in 0..mtp_k-1); mtp_topk_per_depth[d]: per-row [rank2,rank3] for the near depths.
    All ints; row order == spec_row_req_ids order (req_id-keyed by the caller)."""
    from fr13_arctic_suffix_adapter import arctic_draft_to_suffix_rel
    from fr13_mtp_suffix_assembly import assemble_cat33333
    from fr13_merged_fill import build_cat33333_columns

    B = len(spec_row_req_ids)
    if B == 0:
        return None, None, False
    need = N_DEPTH - mtp_k
    assembled = []
    all_full = True
    for b in range(B):
        req_id = spec_row_req_ids[b]
        near = [int(mtp_near_per_depth[d][b]) for d in range(mtp_k)]
        # deep MTP spine unknown (skipped) -> placeholder = last near token (only used if Arctic short)
        mtp_spine = near + [near[-1]] * (N_DEPTH - mtp_k)
        mtp_topk = {d: [int(x[b]) for x in mtp_topk_per_depth.get(d, [])] for d in range(N_DEPTH)}
        pattern = list(_COMMITTED.get(req_id, [])) + near
        draft = None
        if cache is not None:
            try:
                draft = cache.speculate(
                    req_id, pattern, max_spec_tokens=max_spec_tokens,
                    max_spec_factor=max_spec_factor, min_token_prob=min_token_prob,
                )
                STATS["speculate_fired"] += 1
            except Exception:
                draft = None
        suffix_rel = arctic_draft_to_suffix_rel(draft, max_rel=need)
        if len(suffix_rel) < need:
            all_full = False
        nodes, _ = assemble_cat33333(mtp_spine, mtp_topk, suffix_rel, mtp_k)
        assembled.append(nodes)

    if not all_full:
        STATS["match_partial_norun"] += 1
        return None, None, False   # never-regress: caller runs full MTP, Arctic ignored

    STATS["match_full"] += 1
    STATS["assembler_engaged"] += 1
    spine_tokens, wide_topk = build_cat33333_columns(assembled, device, pad_token)
    STATS["skip_fired"] += 1
    return spine_tokens, wide_topk, True
