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
_INGESTED_LEN: dict = {}  # req_id -> #tokens already fed to Arctic add_active_response (non-gappy)
STATS = {
    "speculate_fired": 0, "skip_fired": 0, "assembler_engaged": 0,
    "match_full": 0, "match_partial_norun": 0, "always_fill_miss": 0,
    "started": 0, "retired": 0, "ingested": 0,
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
    global _CACHE, _CACHE_INIT_FAILED, _COMMITTED, _INGESTED_LEN
    _CACHE = None
    _CACHE_INIT_FAILED = False
    _COMMITTED = {}
    _INGESTED_LEN = {}
    for k in STATS:
        STATS[k] = 0


# ---- lifecycle (RUNNER + COMMIT-SITE frames) --------------------------------
def note_new_requests(cache, req_id_to_prompt):
    """start_request for reqs not yet active (evict a stale cached response first).
    Port of vllm SuffixDecodingProposer :63-70. Sets _INGESTED_LEN to the prompt length so the
    subsequent delta-ingest does NOT re-feed the prompt (Arctic start_request already ingests it)."""
    if cache is None:
        return
    for req_id, prompt in req_id_to_prompt.items():
        try:
            if req_id not in cache.active_requests:
                if req_id in cache.cached_requests:
                    cache.evict_cached_response(req_id)
                cache.start_request(req_id, prompt)
                _INGESTED_LEN[req_id] = len(prompt)
                STATS["started"] += 1
        except Exception:
            pass


def ingest_from_sequence(cache, req_id, seq_list, num_tokens, max_tree_depth=24):
    """RUNNER hook (authoritative, NON-GAPPY): feed Arctic the NEW committed tokens since the last
    ingest, sourced from input_batch.token_ids_cpu[i, :num_tokens] (the real full sequence, incl the
    bonus token) -- NOT accepted-drafts-only (which would be gappy). Keeps _COMMITTED = the recent
    suffix for the seam's speculate pattern.
    seq_list: the full per-req token sequence (list/1-D); num_tokens: valid length."""
    # PERF-CRITICAL: convert ONLY the needed slices (delta + recent suffix), NEVER the whole
    # sequence. A prior O(num_tokens)-per-step `seq=[int(t) for t in seq_list[:num_tokens]]` stalled
    # the server on long agentic contexts (100k+ tok x B4 x hundreds of steps) -> dropped the agent
    # socket (UND_ERR_SOCKET) -> empty patches. This is O(delta + max_tree_depth) per step.
    last = _INGESTED_LEN.get(req_id, 0)
    if num_tokens > last:
        new = [int(x) for x in seq_list[last:num_tokens]]   # delta only
        if new and cache is not None:
            try:
                cache.add_active_response(req_id, new)
                STATS["ingested"] += 1
            except Exception:
                pass
        _INGESTED_LEN[req_id] = num_tokens
    start = max(0, num_tokens - max_tree_depth)
    _COMMITTED[req_id] = [int(x) for x in seq_list[start:num_tokens]]   # recent suffix only


def retire_requests(cache, gone_req_ids):
    """RUNNER frame: stop_request for reqs no longer in the batch. Port of :92-95."""
    for req_id in gone_req_ids:
        if cache is not None:
            try:
                cache.stop_request(req_id)
            except Exception:
                pass
        _COMMITTED.pop(req_id, None)
        _INGESTED_LEN.pop(req_id, None)
        STATS["retired"] += 1


# ---- the seam decision (:13361 speculate + :13860 fill) ---------------------
def decide_and_fill(cache, spec_row_req_ids, mtp_near_per_depth, mtp_topk_per_depth, mtp_k,
                    device, pad_token, max_spec_tokens=16, max_spec_factor=4.0, min_token_prob=0.0,
                    vocab_size=None):
    """ADAPTIVE Mode B. Returns (spine_tokens, wide_topk, do_skip):
      * (spine_tokens, wide_topk, True)  when ALL active rows have a full-depth (>= N_DEPTH-mtp_k)
        Arctic match -> caller SKIPS the deep spine forwards and packs these columns.
      * (None, None, False)              otherwise -> caller runs the FULL MTP loop (never-regress);
        Arctic is IGNORED this step.
    mtp_near_per_depth[d]: an indexable of per-row MTP spine tokens for the mtp_k drafted depths
      (d in 0..mtp_k-1); mtp_topk_per_depth[d]: per-row [rank2,rank3] for the near depths.
    All ints; row order == spec_row_req_ids order (req_id-keyed by the caller)."""
    from fr13_arctic_suffix_adapter import (arctic_draft_to_suffix_rel, arctic_tree_to_suffix_rel,
                                            arctic_flat_tree_to_suffix_rel, arctic_match_confidence)
    from fr13_mtp_suffix_assembly import assemble_cat33333
    from fr13_merged_fill import build_cat33333_columns

    # v2 (default): use_tree_spec=True -> the trie SUBTREE (parents/probs) fills cat33333 SPINE +
    # BRANCHES (the trie-branches->tree-branches mapping). v1 (FR13_MERGED_TREE_SPEC=0): flat chain,
    # spine only, branches = MTP fallback. Both Gate-1 lossless + never-regress.
    _use_tree = os.environ.get("FR13_MERGED_TREE_SPEC", "1") != "0"
    # FLAVOR: 'adaptive' (default, Flavor A) = skip deep MTP forwards ONLY on a batch-wide
    # full-depth Arctic match, else full MTP (never-regress). 'always' (Flavor B) = ALWAYS run
    # just mtp_k forwards + suffix-fill the whole deep tree unconditionally (max drafter speed;
    # accept degrades toward root-only on a miss -- still lossless via the committer).
    _flavor = os.environ.get("FR13_MERGED_FLAVOR", "adaptive")
    # CONFIDENCE-GATED skip (merge16d verdict): blanket skip fired on trie COVERAGE (match_full 96%) not
    # confidence -> accept collapsed 3.61->1.97 -> -17% speed. Gate the skip on the arctic match's MIN
    # per-node prob: skip ONLY when the suffix stats strongly expect the model to follow (>= threshold),
    # else full MTP (never-regress = baseline accept). Default 0.0 = OFF (blanket, back-compat).
    _min_prob = float(os.environ.get("FR13_MERGED_SKIP_MIN_PROB", "0.0"))

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
        suffix_rel = {}
        _row_conf = 1.0   # match confidence for the gated skip (1.0 = no signal -> don't block)
        if cache is not None:
            try:
                if _use_tree:
                    # HYBRID: FLAT (deep chain) drives the spine + depth-coverage (skip engagement),
                    # TREE (use_tree_spec) supplies branch alternatives -> best of both. OVERHEAD:
                    # only pay the 2nd (tree) speculate when the flat is a FULL-depth match -- on a
                    # non-full row (won't skip on Flavor A) the branches are discarded anyway, so
                    # skipping the tree call there ~halves the arctic latency that trips the emit-wedge.
                    flat_d = cache.speculate(
                        req_id, pattern, max_spec_tokens=max_spec_tokens,
                        max_spec_factor=max_spec_factor, min_token_prob=min_token_prob,
                        use_tree_spec=False)
                    _row_conf = arctic_match_confidence(flat_d)   # flat drives the spine -> its confidence
                    _flat_rel = arctic_draft_to_suffix_rel(flat_d, max_rel=need)
                    if len(_flat_rel) >= need:
                        tree_d = cache.speculate(
                            req_id, pattern, max_spec_tokens=max_spec_tokens,
                            max_spec_factor=max_spec_factor, min_token_prob=min_token_prob,
                            use_tree_spec=True)
                        suffix_rel = arctic_flat_tree_to_suffix_rel(flat_d, tree_d, max_rel=need)
                    else:
                        suffix_rel = _flat_rel   # non-full -> flat-only (discarded on Flavor A miss)
                else:
                    draft = cache.speculate(
                        req_id, pattern, max_spec_tokens=max_spec_tokens,
                        max_spec_factor=max_spec_factor, min_token_prob=min_token_prob,
                        use_tree_spec=False)
                    _row_conf = arctic_match_confidence(draft)
                    suffix_rel = arctic_draft_to_suffix_rel(draft, max_rel=need)
                STATS["speculate_fired"] += 1
            except Exception:
                suffix_rel = {}
        # Skip requires full depth AND (when gated) enough confidence -- a low-prob deep match is
        # exactly the over-confident-but-wrong case that collapsed accept, so fall back to full MTP.
        if len(suffix_rel) >= need and _min_prob > 0.0 and _row_conf < _min_prob:
            STATS["conf_gated"] = STATS.get("conf_gated", 0) + 1   # full-depth but blocked by low conf
        if len(suffix_rel) < need or (_min_prob > 0.0 and _row_conf < _min_prob):
            all_full = False
        nodes, _ = assemble_cat33333(mtp_spine, mtp_topk, suffix_rel, mtp_k)
        assembled.append(nodes)

    if not all_full and _flavor != "always":
        STATS["match_partial_norun"] += 1
        _maybe_log_engagement()
        return None, None, False   # Flavor A never-regress: caller runs full MTP, Arctic ignored
    # Flavor B ('always'): fill + skip UNCONDITIONALLY -- even a partial/empty match packs (missing
    # deep depths -> assemble MTP-placeholder + PAD branches; those nodes just don't accept). Max
    # drafter speed, accept degrades to root-only on a miss. Lossless via the committer regardless.
    if all_full:
        STATS["match_full"] += 1
    else:
        STATS["always_fill_miss"] += 1   # Flavor B skipped a non-full match (accept degrades)
    STATS["assembler_engaged"] += 1
    spine_tokens, wide_topk = build_cat33333_columns(assembled, device, pad_token, vocab_size=vocab_size)
    STATS["skip_fired"] += 1
    _maybe_log_engagement()
    return spine_tokens, wide_topk, True


_LOG_EVERY = 50
_LOG_N = 0


def _maybe_log_engagement():
    """Periodic engagement needle to the vLLM logger (-> docker log) for the ENGAGED gate.
    match_full>0 (not just speculate_fired>0) is the non-gappy proof; skip_fired is the speed win."""
    global _LOG_N
    _LOG_N += 1
    if _LOG_N % _LOG_EVERY != 0:
        return
    try:
        import logging
        try:
            from fr13_merged_fill import get_oob_stats
            _oob_n, _oob_last = get_oob_stats()
        except Exception:
            _oob_n, _oob_last = 0, None
        logging.getLogger("vllm.fr13_merged_drafter").info(
            "[FR13_MERGED ENGAGED] speculate_fired=%d match_full=%d match_partial=%d "
            "always_fill_miss=%d skip_fired=%d assembler_engaged=%d started=%d ingested=%d retired=%d "
            "arctic_oob_dropped=%d last_oob=%s conf_gated=%d",
            STATS["speculate_fired"], STATS["match_full"], STATS["match_partial_norun"],
            STATS["always_fill_miss"], STATS["skip_fired"], STATS["assembler_engaged"],
            STATS["started"], STATS["ingested"], STATS["retired"], _oob_n, _oob_last,
            STATS.get("conf_gated", 0),
        )
    except Exception:
        pass
