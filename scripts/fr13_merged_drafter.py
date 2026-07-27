#!/usr/bin/env python3
"""FR13 merged drafter orchestration — TAIL MODE (the shipped tail6 design).

Keeps the patcher edit THIN: the patcher injects hooks that call into this CPU-testable module,
mirroring the fr13_device_multidraft_kernel.py separate-module pattern. Ties together:
  - lifecycle: note_new_requests / ingest_from_sequence / retire_requests (Arctic
    SuffixDecodingCache) — runs at the runner every step,
  - a rolling per-req committed-suffix buffer (the speculate pattern seed),
  - decide_tail: the depth-6..11 Arctic suffix CHAIN appended past the native MTP head
    (head = depths 1-5, 100%% native/byte-identical; the tail only ADDS candidates —
    never-regress via the monotone committer; cold rows pad and simply never accept).

ENGAGEMENT PROOF (tail mode): TAIL[hit] > 0 in the needle. The tail costs ~0.3ms/step of
host trie-walk — it is NOT MTP-drafted (no extra sequential forwards).

The Arctic cache is container-only (C++ ext); the module is import-free of arctic at top
level and lets tests inject a mock via set_cache_for_test().

HISTORY: the head-MERGE path (decide_and_fill — MTP-k spine + Arctic grow-to-cat33333,
adaptive skip of the deep MTP forwards) was DELETED 2026-07-27 (cleanup+bake,
FR13_CLEANUP_BAKE_PLAN.md): Front-2/merge closed as a no-go, the path was dormant-by-design
in tail mode, and its zeroed counters in the shared needle caused a live misread
(boot-54 era: zeros mistaken for tail disengagement). git history has the code.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

N_DEPTH = 5

_CACHE = None
_CACHE_INIT_FAILED = False
_PREWARMED = False        # design §6b: harness-aware trie pre-warm done once at boot
_COMMITTED: dict = {}     # req_id -> list[int] rolling recent-committed tokens (<= max_tree_depth)
_INGESTED_LEN: dict = {}  # req_id -> #tokens already fed to Arctic add_active_response (non-gappy)
STATS = {
    "started": 0, "retired": 0, "ingested": 0, "prewarm_seeded": 0,
    "tail_speculate_fired": 0, "tail_hit": 0, "tail_all_cold": 0, "tail_branch_real": 0,
}


def merged_on() -> bool:
    """Sidecar-gated (worker drops FR13_* env): /logs/fr13_draft_source_merged.arm written by the
    launcher when FR13_DRAFT_SOURCE=merged. Mirrors _fr13_committer_native_on()."""
    return os.path.exists("/logs/fr13_draft_source_merged.arm")


# accept>5 TAIL mode: head_depth (default 5) = MTP-drafted head; the deep chain past it is Arctic.
TAIL_HEAD_DEPTH = 5


def tail_on() -> bool:
    """Sidecar-gated (worker drops FR13_* env): /logs/fr13_tail_mode.arm written by the launcher
    when FR13_TAIL_MODE=1. Tail mode caps native MTP forwards at the head + appends the Arctic tail
    chain. REQUIRES merged_on() too (the Arctic cache lifecycle rides on the merged runner hook)."""
    return os.path.exists("/logs/fr13_tail_mode.arm")


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
    global _CACHE, _CACHE_INIT_FAILED, _COMMITTED, _INGESTED_LEN, _PREWARMED
    _CACHE = None
    _CACHE_INIT_FAILED = False
    _PREWARMED = False
    _COMMITTED = {}
    _INGESTED_LEN = {}
    for k in STATS:
        STATS[k] = 0


# ---- PRE-WARM (design §6b: harness-aware trie) ------------------------------
def prewarm_trie(cache, corpus, prefix="prewarm"):
    """Pre-warm the Arctic CROSS-REQUEST trie with a harness-aware corpus of prior-trajectory token
    sequences so the per-request speculate() matches harness-structural spans (tool-call XML, system
    prompt, imports, boilerplate) from token 1 -- fixing Front-2's cold/task-local weakness. Each
    sequence is ingested as a completed cross-request pattern (start->add->stop => cached_requests).
    NEVER-REGRESS: pre-warm only ADDS candidates through the monotone committer (a non-matching pattern
    just doesn't accept). corpus = iterable of token-id lists. Returns #seeded. Never raises."""
    if cache is None:
        return 0
    seeded = 0
    for i, seq in enumerate(corpus):
        try:
            ids = [int(t) for t in seq]
        except Exception:
            continue
        if len(ids) < 2:
            continue
        rid = f"{prefix}_{i}"
        try:
            cache.start_request(rid, ids[:1])        # first token = prompt
            cache.add_active_response(rid, ids[1:])  # rest = the "response" ingested into the trie
            cache.stop_request(rid)                  # -> cached (available cross-request)
            seeded += 1
        except Exception:
            continue
    STATS["prewarm_seeded"] = STATS.get("prewarm_seeded", 0) + seeded
    return seeded


def maybe_prewarm(cache):
    """Boot-time hook: if FR13_PREWARM_TRIE=<path> is set, load the JSONL corpus (one token-id list per
    line, or {"token_ids":[...]}) and prewarm the trie ONCE. Gated + non-fatal (missing file => no-op)."""
    global _PREWARMED
    if _PREWARMED or cache is None:
        return
    # worker STRIPS FR13_* env (like merged_on's sidecar) -> read a /logs sidecar first (launcher copies
    # the corpus there when FR13_PREWARM_TRIE is set at host launch), then env, then a fixed repo path.
    path = None
    for cand in ("/logs/fr13_prewarm_corpus.jsonl",
                 os.environ.get("FR13_PREWARM_TRIE") or "",
                 "/workspace/output/fr13_prewarm/corpus_active.jsonl"):
        if cand and os.path.exists(cand):
            path = cand
            break
    if not path:
        _PREWARMED = True   # nothing to load -> mark done so we don't re-stat every step
        return
    corpus = []
    try:
        import json as _json
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = _json.loads(line)
                seq = obj if isinstance(obj, list) else (obj.get("token_ids") or obj.get("ids") or [])
                if seq:
                    corpus.append(seq)
        n = prewarm_trie(cache, corpus)
        import logging
        logging.getLogger("vllm.fr13_merged_drafter").info(
            "[FR13_PREWARM] seeded %d/%d corpus sequences from %s", n, len(corpus), path)
    except Exception:
        pass
    _PREWARMED = True


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


# ---- head-merge seam DELETED 2026-07-27 -------------------------------------
# decide_and_fill (the adaptive MTP-k + Arctic grow-to-cat33333 head-merge path) lived
# here. Removed with the patcher's FR13_MERGED_DRAFTER_SEAM (cleanup+bake). git history
# has the code; do not resurrect without a fresh gate — Front-2/merge was a closed no-go.

_TAIL_WIDE_TOPK = {}


def get_tail_wide_topk():
    """Direction-2 d6-branch: the tail-branch wide_topk {parent_pos: [B, width]} from the LAST decide_tail
    call ({} when the spine-only tail ran). The patcher tail-append merges this into _fr10_wide_topk."""
    return _TAIL_WIDE_TOPK


def decide_tail(cache, spec_row_req_ids, mtp_head_per_depth, head_depth, tail_len,
                device, pad_token, max_spec_tokens=32, max_spec_factor=4.0, min_token_prob=0.0,
                vocab_size=None):
    """accept>5 TAIL (mtp_k=head_depth mode): the MTP head (depths 0..head_depth-1) is 100% native
    (byte-identical baseline, filled by the drafter's own forwards); this ONLY produces the deep Arctic
    CHAIN past the head (depths head_depth..head_depth+tail_len-1) to APPEND to _fr10_spine_tokens.
    Pattern per row = _COMMITTED[req] + the native MTP head tokens (seeds the walk from MTP's confident
    prefix). Returns a list of `tail_len` int64 [batch] tensors on `device`, or None (empty batch / no
    cache). LOSSLESS + never-regress: the head is unchanged and the tail only ADDS candidates (cold ->
    pad, never matches past the head). The committer (accept=p(S), source/depth-blind) does the rest.

    mtp_head_per_depth: list length head_depth; entry d is a per-row-indexable of the native MTP spine
                        token at depth d (e.g. _fr10_spine_tokens[d].cpu().tolist()), row order ==
                        spec_row_req_ids order (req_id-keyed by the caller)."""
    from fr13_arctic_suffix_adapter import arctic_draft_to_suffix_rel, arctic_tree_to_suffix_rel
    from fr13_merged_fill import build_tail_columns, build_tail_branch_columns
    # Direction-2 d6-handoff repair (default 0 => spine-only == shipped tail6, byte-identical, NO drift).
    # The WORKER DROPS FR13_* env, so read the launcher-written /logs sidecar (like tail_on/merged_on) for
    # the branch value "<tail_branches> <tail_branch_depths>". When on: use the arctic TREE adapter (ranked
    # top-k per depth) + add tail_branches sibling candidates at the first tail_branch_depths tail depths
    # from suffix_rel[j][1:] (the runner-ups we discard today). Branch wide_topk stashed for the patcher.
    _tb, _tbd = 0, 0
    try:
        with open("/logs/fr13_tail_branches.cfg") as _bcf:
            _bp = _bcf.read().split()
            _tb, _tbd = int(_bp[0]), (int(_bp[1]) if len(_bp) > 1 else 0)
    except Exception:
        _tb, _tbd = 0, 0
    _branched = _tb > 0 and _tbd > 0
    _to_rel = arctic_tree_to_suffix_rel if _branched else arctic_draft_to_suffix_rel

    B = len(spec_row_req_ids)
    if B == 0 or cache is None:
        return None
    tail_rows = []
    branch_rows = [None] * B
    any_hit = False
    for b in range(B):
        req_id = spec_row_req_ids[b]
        head = [int(mtp_head_per_depth[d][b]) for d in range(head_depth)]
        pattern = list(_COMMITTED.get(req_id, [])) + head
        row = None
        try:
            draft = cache.speculate(
                req_id, pattern, max_spec_tokens=max_spec_tokens,
                max_spec_factor=max_spec_factor, min_token_prob=min_token_prob,
                use_tree_spec=_branched)
            STATS["tail_speculate_fired"] = STATS.get("tail_speculate_fired", 0) + 1
            rel = _to_rel(draft, max_rel=tail_len)  # {0..: [ranked tok,...]} = depths head+..
            row = [(rel[j][0] if rel.get(j) else None) for j in range(tail_len)]
            if _branched:
                branch_rows[b] = {j: [int(t) for t in rel[j][1:1 + _tb]]
                                  for j in range(min(_tbd, tail_len))
                                  if rel.get(j) and len(rel[j]) > 1}
                # engagement proof: count REAL arctic runner-up branch tokens (vs pad). br_real>0 in the
                # needle == the branched path ran + the tree adapter yielded runner-ups (NOT vacuous pad).
                STATS["tail_branch_real"] = STATS.get("tail_branch_real", 0) + sum(
                    len(v) for v in branch_rows[b].values())
            if any(t is not None for t in row):
                any_hit = True
                STATS["tail_hit"] = STATS.get("tail_hit", 0) + 1
        except Exception:
            row = None
        tail_rows.append(row)
    if not any_hit:
        STATS["tail_all_cold"] = STATS.get("tail_all_cold", 0) + 1
    _maybe_log_engagement()
    _spine = build_tail_columns(tail_rows, device, pad_token, tail_len, vocab_size=vocab_size)
    global _TAIL_WIDE_TOPK
    _TAIL_WIDE_TOPK = (build_tail_branch_columns(branch_rows, device, pad_token, head_depth, _tbd, _tb,
                                                 vocab_size=vocab_size) if _branched else {})
    return _spine


_LOG_EVERY = 50
_LOG_N = 0


def _maybe_log_engagement():
    """Periodic engagement needle to the vLLM logger (-> docker log) for the ENGAGED gate.
    TAIL[hit]>0 is THE engagement proof in tail mode (the only mode since the head-merge
    deletion). Leads with TAIL so a truncated log line can never hide engagement again
    (boot-54-era misread: merge-counter zeros at the front, TAIL[...] truncated off)."""
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
            "[FR13_MERGED ENGAGED] TAIL[fired=%d hit=%d cold=%d br_real=%d] "
            "started=%d ingested=%d retired=%d arctic_oob_dropped=%d last_oob=%s",
            STATS.get("tail_speculate_fired", 0), STATS.get("tail_hit", 0),
            STATS.get("tail_all_cold", 0), STATS.get("tail_branch_real", 0),
            STATS["started"], STATS["ingested"], STATS["retired"], _oob_n, _oob_last,
        )
    except Exception:
        pass
