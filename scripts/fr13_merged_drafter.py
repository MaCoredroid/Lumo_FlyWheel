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
_FIXED32_PENDING_STEP = None
_FIXED32_LIFECYCLE_POISON = None
STATS = {
    "started": 0, "retired": 0, "ingested": 0, "prewarm_seeded": 0,
    "tail_speculate_fired": 0, "tail_hit": 0, "tail_all_cold": 0, "tail_branch_real": 0,
    "hydra_speculate_fired": 0, "hydra_hit": 0, "hydra_real": 0,
    "hydra_rank2_hit": 0, "hydra_rank3_hit": 0,
    "fixed32_events": 0, "fixed32_rows": 0, "fixed32_carry_slots": 0,
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
    global _TAIL_WIDE_TOPK, _TAIL_PATH_TOKENS
    global _FIXED32_WORK_SERIAL, _FIXED32_LAST_WORK, _FIXED32_PENDING_STEP
    global _FIXED32_LIFECYCLE_POISON
    _CACHE = None
    _CACHE_INIT_FAILED = False
    _PREWARMED = False
    _COMMITTED = {}
    _INGESTED_LEN = {}
    _TAIL_WIDE_TOPK = {}
    _TAIL_PATH_TOKENS = {}
    _FIXED32_WORK_SERIAL = 0
    _FIXED32_LAST_WORK = None
    _FIXED32_PENDING_STEP = None
    _FIXED32_LIFECYCLE_POISON = None
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
def _fixed32_require_unpoisoned():
    if _FIXED32_LIFECYCLE_POISON is not None:
        raise RuntimeError(
            "fixed32 Arctic lifecycle is poisoned: "
            + repr(_FIXED32_LIFECYCLE_POISON)
        )


def _fixed32_poison(phase, step_seq=None):
    global _FIXED32_LIFECYCLE_POISON
    if _FIXED32_LIFECYCLE_POISON is None:
        _FIXED32_LIFECYCLE_POISON = {
            "phase": str(phase),
            "step_seq": None if step_seq is None else int(step_seq),
        }


def note_new_requests(cache, req_id_to_prompt, *, strict=False):
    """start_request for reqs not yet active (evict a stale cached response first).
    Port of vllm SuffixDecodingProposer :63-70. Sets _INGESTED_LEN to the prompt length so the
    subsequent delta-ingest does NOT re-feed the prompt (Arctic start_request already ingests it)."""
    if strict and cache is None:
        raise RuntimeError("fixed32 Arctic cache is unavailable")
    if cache is None:
        return
    if strict:
        _fixed32_require_unpoisoned()
        plans = []
        for req_id, prompt in req_id_to_prompt.items():
            prompt = [int(token) for token in prompt]
            prompt_len = len(prompt)
            is_active = req_id in cache.active_requests
            is_cached = req_id in cache.cached_requests
            prior = _INGESTED_LEN.get(req_id)
            if is_active and (
                not is_cached
                or prior is None
                or int(prior) < prompt_len
            ):
                raise RuntimeError(
                    "fixed32 Arctic active request has no consistent owner"
                )
            if not is_active and (
                prior is not None or req_id in _COMMITTED
            ):
                raise RuntimeError(
                    "fixed32 Arctic inactive request retained local state"
                )
            plans.append(
                (req_id, prompt, prompt_len, is_active, is_cached, prior)
            )
        try:
            for req_id, prompt, _prompt_len, is_active, is_cached, _prior in plans:
                if is_active:
                    continue
                if is_cached:
                    cache.evict_cached_response(req_id)
                    if req_id in cache.cached_requests:
                        raise RuntimeError(
                            "fixed32 Arctic cached response survived eviction"
                        )
                cache.start_request(req_id, prompt)
                if (
                    req_id not in cache.active_requests
                    or req_id not in cache.cached_requests
                ):
                    raise RuntimeError(
                        "fixed32 Arctic request start did not publish ownership: "
                        + repr(req_id)
                    )
        except Exception:
            _fixed32_poison("note_new_requests")
            raise
        for req_id, _prompt, prompt_len, is_active, _is_cached, _prior in plans:
            if not is_active:
                _INGESTED_LEN[req_id] = prompt_len
                STATS["started"] += 1
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


def ingest_from_sequence(
    cache,
    req_id,
    seq_list,
    num_tokens,
    max_tree_depth=24,
    *,
    strict=False,
):
    """RUNNER hook (authoritative, NON-GAPPY): feed Arctic the NEW committed tokens since the last
    ingest, sourced from input_batch.token_ids_cpu[i, :num_tokens] (the real full sequence, incl the
    bonus token) -- NOT accepted-drafts-only (which would be gappy). Keeps _COMMITTED = the recent
    suffix for the seam's speculate pattern.
    seq_list: the full per-req token sequence (list/1-D); num_tokens: valid length."""
    # PERF-CRITICAL: convert ONLY the needed slices (delta + recent suffix), NEVER the whole
    # sequence. A prior O(num_tokens)-per-step `seq=[int(t) for t in seq_list[:num_tokens]]` stalled
    # the server on long agentic contexts (100k+ tok x B4 x hundreds of steps) -> dropped the agent
    # socket (UND_ERR_SOCKET) -> empty patches. This is O(delta + max_tree_depth) per step.
    if strict:
        _fixed32_require_unpoisoned()
        if (
            cache is None
            or req_id not in cache.active_requests
            or req_id not in cache.cached_requests
            or req_id not in _INGESTED_LEN
        ):
            raise RuntimeError(
                "fixed32 Arctic ingest has no active cached owner: "
                + repr(req_id)
            )
        num_tokens = int(num_tokens)
        max_tree_depth = int(max_tree_depth)
        if num_tokens < 0 or max_tree_depth <= 0 or num_tokens > len(seq_list):
            raise RuntimeError(
                "fixed32 Arctic ingest sequence bounds drift: "
                + repr((req_id, num_tokens, len(seq_list), max_tree_depth))
            )
        last = int(_INGESTED_LEN[req_id])
        new = (
            [int(x) for x in seq_list[last:num_tokens]]
            if num_tokens > last
            else []
        )
        start = max(0, num_tokens - max_tree_depth)
        suffix = [int(x) for x in seq_list[start:num_tokens]]
        if len(new) != max(0, num_tokens - last):
            raise RuntimeError("fixed32 Arctic ingest delta is incomplete")
        if new:
            try:
                cache.add_active_response(req_id, new)
                if (
                    req_id not in cache.active_requests
                    or req_id not in cache.cached_requests
                ):
                    raise RuntimeError(
                        "fixed32 Arctic owner disappeared after ingest"
                    )
            except Exception:
                _fixed32_poison("ingest_from_sequence")
                raise
            _INGESTED_LEN[req_id] = num_tokens
            STATS["ingested"] += 1
        _COMMITTED[req_id] = suffix
        return

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


def retire_requests(cache, gone_req_ids, *, strict=False):
    """RUNNER frame: stop_request for reqs no longer in the batch. Port of :92-95."""
    if strict:
        _fixed32_require_unpoisoned()
        gone_req_ids = tuple(gone_req_ids)
        if len(set(gone_req_ids)) != len(gone_req_ids):
            raise RuntimeError("fixed32 Arctic retire request IDs are duplicated")
        for req_id in gone_req_ids:
            if cache is None or req_id not in cache.active_requests:
                raise RuntimeError(
                    "fixed32 Arctic retire has no active owner: "
                    + repr(req_id)
                )
        try:
            for req_id in gone_req_ids:
                cache.stop_request(req_id)
                if req_id in cache.active_requests:
                    raise RuntimeError(
                        "fixed32 Arctic request remained active after retire"
                    )
        except Exception:
            _fixed32_poison("retire_requests")
            raise
        for req_id in gone_req_ids:
            _COMMITTED.pop(req_id, None)
            _INGESTED_LEN.pop(req_id, None)
            STATS["retired"] += 1
        return
    for req_id in gone_req_ids:
        if cache is not None:
            try:
                cache.stop_request(req_id)
            except Exception:
                pass
        _COMMITTED.pop(req_id, None)
        _INGESTED_LEN.pop(req_id, None)
        STATS["retired"] += 1


def stage_fixed32_step(
    cache,
    request_ids,
    token_rows,
    prompt_lengths,
    computed_lengths,
    scheduled_lengths,
    scheduled_draft_lengths,
    committed_lengths,
    discard_rows,
    step_seq,
    *,
    restart_request_ids=(),
    max_tree_depth=24,
):
    """Publish the exact pre-forward committed boundary for one fixed32 step."""
    global _FIXED32_PENDING_STEP
    _fixed32_require_unpoisoned()
    request_ids = tuple(str(req_id) for req_id in request_ids)
    restart_ids = {str(req_id) for req_id in restart_request_ids}
    batch = len(request_ids)
    fields = (
        token_rows,
        prompt_lengths,
        computed_lengths,
        scheduled_lengths,
        scheduled_draft_lengths,
        committed_lengths,
        discard_rows,
    )
    if (
        cache is None
        or not hasattr(cache, "active_requests")
        or not hasattr(cache, "cached_requests")
        or _FIXED32_PENDING_STEP is not None
        or not 1 <= batch <= 4
        or len(set(request_ids)) != batch
        or any(len(field) < batch for field in fields)
        or type(step_seq) is not int
        or step_seq <= 0
        or int(max_tree_depth) <= 0
        or not restart_ids.issubset(request_ids)
    ):
        raise RuntimeError("fixed32 Arctic step staging ownership drift")

    rows = []
    active_ids = {str(req_id) for req_id in cache.active_requests}
    cached_ids = {str(req_id) for req_id in cache.cached_requests}
    for index, req_id in enumerate(request_ids):
        prompt_len = int(prompt_lengths[index])
        computed = int(computed_lengths[index])
        scheduled = int(scheduled_lengths[index])
        drafts = int(scheduled_draft_lengths[index])
        committed = int(committed_lengths[index])
        discard = bool(discard_rows[index])
        safe_end = computed + scheduled - drafts
        sequence = token_rows[index]
        if (
            prompt_len < 0
            or computed < 0
            or scheduled <= 0
            or drafts not in (0, 31)
            or drafts > scheduled
            or (
                drafts == 31
                and (
                    scheduled != 32
                    or safe_end != committed
                    or discard
                )
            )
            or not computed <= safe_end <= committed <= len(sequence)
            or prompt_len > committed
            or discard is not (safe_end < committed)
        ):
            raise RuntimeError(
                "fixed32 Arctic staged sequence geometry drift: "
                + repr(
                    (
                        req_id,
                        prompt_len,
                        computed,
                        scheduled,
                        drafts,
                        safe_end,
                        committed,
                        discard,
                        len(sequence),
                    )
                )
            )
        restart = req_id in restart_ids
        is_active = req_id in active_ids
        if not restart and not is_active and (
            req_id in _INGESTED_LEN or req_id in _COMMITTED
        ):
            raise RuntimeError(
                "fixed32 Arctic staged inactive row retained local state: "
                + repr(req_id)
            )
        if not restart and is_active and req_id not in cached_ids:
            raise RuntimeError(
                "fixed32 Arctic staged row is active but not cached: "
                + repr(req_id)
            )
        prompt = (
            None
            if is_active and not restart
            else [int(token) for token in sequence[:prompt_len]]
        )
        prior = (
            prompt_len
            if restart
            else (
                int(_INGESTED_LEN[req_id])
                if is_active and req_id in _INGESTED_LEN
                else prompt_len if not is_active else None
            )
        )
        if prior is None:
            raise RuntimeError(
                "fixed32 Arctic staged row has no ingest watermark: "
                + repr(req_id)
            )
        if prior < prompt_len:
            raise RuntimeError(
                "fixed32 Arctic staged row has a short ingest watermark"
            )
        if safe_end < prior and not (
            safe_end < prompt_len and prior == prompt_len
        ):
            raise RuntimeError(
                "fixed32 Arctic staged row regressed behind its watermark: "
                + repr((req_id, safe_end, prior, prompt_len))
            )
        delta = (
            [int(token) for token in sequence[prior:safe_end]]
            if safe_end > prior
            else []
        )
        suffix_start = max(0, safe_end - int(max_tree_depth))
        suffix = [
            int(token) for token in sequence[suffix_start:safe_end]
        ]
        if len(delta) != max(0, safe_end - prior):
            raise RuntimeError("fixed32 Arctic staged delta is incomplete")
        rows.append(
            {
                "request_id": req_id,
                "restart": restart,
                "was_active": is_active,
                "prompt": prompt,
                "prompt_len": prompt_len,
                "was_cached": req_id in cached_ids,
                "delta": delta,
                "suffix": suffix,
                "safe_end": safe_end,
                "scheduled_drafts": drafts,
                "discard": discard,
            }
        )

    live = set(request_ids)
    retired = tuple(req_id for req_id in active_ids if req_id not in live)
    try:
        for row in rows:
            req_id = row["request_id"]
            prompt = row["prompt"]
            if prompt is None:
                continue
            if row["restart"] and row["was_active"]:
                cache.stop_request(req_id)
                if req_id in cache.active_requests:
                    raise RuntimeError(
                        "fixed32 Arctic staged restart kept an active owner"
                    )
            if req_id in cache.cached_requests:
                cache.evict_cached_response(req_id)
                if req_id in cache.cached_requests:
                    raise RuntimeError(
                        "fixed32 Arctic staged cache eviction failed"
                    )
            cache.start_request(req_id, prompt)
            if (
                req_id not in cache.active_requests
                or req_id not in cache.cached_requests
            ):
                raise RuntimeError(
                    "fixed32 Arctic staged request start lost ownership"
                )
        for row in rows:
            delta = row["delta"]
            if not delta:
                continue
            req_id = row["request_id"]
            if (
                req_id not in cache.active_requests
                or req_id not in cache.cached_requests
            ):
                raise RuntimeError(
                    "fixed32 Arctic staged delta lost request ownership"
                )
            cache.add_active_response(req_id, delta)
            if (
                req_id not in cache.active_requests
                or req_id not in cache.cached_requests
            ):
                raise RuntimeError(
                    "fixed32 Arctic staged owner disappeared after ingest"
                )
        for req_id in retired:
            cache.stop_request(req_id)
            if req_id in cache.active_requests:
                raise RuntimeError(
                    "fixed32 Arctic staged retirement kept an active owner"
                )
    except Exception:
        _fixed32_poison("stage_fixed32_step", step_seq)
        raise

    for req_id in retired:
        _COMMITTED.pop(req_id, None)
        _INGESTED_LEN.pop(req_id, None)
        STATS["retired"] += 1
    for row in rows:
        req_id = row["request_id"]
        if row["prompt"] is not None:
            _INGESTED_LEN[req_id] = int(row["prompt_len"])
            STATS["started"] += 1
        if row["delta"]:
            _INGESTED_LEN[req_id] = int(row["safe_end"])
            STATS["ingested"] += 1
        _COMMITTED[req_id] = list(row["suffix"])
    _FIXED32_PENDING_STEP = {
        "step_seq": step_seq,
        "request_ids": request_ids,
        "rows": tuple(rows),
        "max_tree_depth": int(max_tree_depth),
    }


def finalize_fixed32_step(
    cache,
    request_ids,
    accepted_output,
    next_token_ids,
    step_seq,
    vocab_size,
):
    """Append accepted tree-path tokens plus the current sampled token."""
    global _FIXED32_PENDING_STEP
    _fixed32_require_unpoisoned()
    pending = _FIXED32_PENDING_STEP
    request_ids = tuple(str(req_id) for req_id in request_ids)
    next_token_ids = tuple(int(token) for token in next_token_ids)
    if (
        not isinstance(pending, dict)
        or pending.get("step_seq") != int(step_seq)
        or pending.get("request_ids") != request_ids
        or len(next_token_ids) != len(request_ids)
        or int(vocab_size) <= 0
    ):
        raise RuntimeError("fixed32 Arctic finalization ownership drift")

    accepted_by_req = {}
    if accepted_output is not None:
        if (
            not isinstance(accepted_output, dict)
            or set(accepted_output)
            != {
                "step_seq",
                "request_ids",
                "full_request_ids",
                "output_rows",
                "output_lens",
            }
            or accepted_output.get("step_seq") != int(step_seq)
            or accepted_output.get("full_request_ids") != request_ids
        ):
            raise RuntimeError(
                "fixed32 Arctic accepted-output freshness drift"
            )
        spec_ids = tuple(
            str(req_id)
            for req_id in accepted_output.get("request_ids", ())
        )
        output_rows = tuple(accepted_output.get("output_rows", ()))
        output_lens = tuple(accepted_output.get("output_lens", ()))
        if (
            len(spec_ids) != len(output_rows)
            or len(spec_ids) != len(output_lens)
            or len(set(spec_ids)) != len(spec_ids)
            or any(req_id not in request_ids for req_id in spec_ids)
        ):
            raise RuntimeError(
                "fixed32 Arctic accepted-output row map drift"
            )
        for req_id, output_row, output_len in zip(
            spec_ids, output_rows, output_lens
        ):
            output_len = int(output_len)
            output_row = tuple(output_row)
            values = [int(token) for token in output_row[:output_len]]
            padding = [int(token) for token in output_row[output_len:]]
            if (
                len(output_row) != 32
                or not 1 <= output_len <= 32
                or len(values) != output_len
                or any(token != -1 for token in padding)
            ):
                raise RuntimeError(
                    "fixed32 Arctic accepted-output length drift"
                )
            accepted_by_req[str(req_id)] = values

    max_tree_depth = int(pending["max_tree_depth"])
    actions = []
    for index, row in enumerate(pending["rows"]):
        req_id = row["request_id"]
        output = accepted_by_req.get(req_id)
        if (
            req_id not in cache.active_requests
            or req_id not in cache.cached_requests
        ):
            raise RuntimeError(
                "fixed32 Arctic finalization lost request ownership"
            )
        if row["discard"]:
            if output is not None:
                raise RuntimeError(
                    "fixed32 Arctic discard row received tree output"
                )
            continue
        if row["scheduled_drafts"] > 0 and output is None:
            raise RuntimeError(
                "fixed32 Arctic spec row has no accepted tree output"
            )
        if row["scheduled_drafts"] == 0 and output is not None:
            raise RuntimeError(
                "fixed32 Arctic non-spec row received tree output"
            )
        next_token = next_token_ids[index]
        accepted = [] if output is None else output[:-1]
        if output is not None and output[-1] != next_token:
            raise RuntimeError(
                "fixed32 Arctic committer bonus disagrees with Eagle root"
            )
        if len(accepted) > int(row["scheduled_drafts"]):
            raise RuntimeError(
                "fixed32 Arctic accepted more drafts than scheduled"
            )
        extra = accepted + [next_token]
        if any(not 0 <= token < int(vocab_size) for token in extra):
            raise RuntimeError(
                "fixed32 Arctic finalized token is outside the vocabulary"
            )
        safe_end = int(row["safe_end"])
        if (
            _INGESTED_LEN.get(req_id) != safe_end
            or req_id not in cache.active_requests
            or req_id not in cache.cached_requests
        ):
            raise RuntimeError(
                "fixed32 Arctic finalization watermark drift: "
                + repr((req_id, _INGESTED_LEN.get(req_id), safe_end))
            )
        actions.append((req_id, safe_end, extra))

    try:
        for req_id, _safe_end, extra in actions:
            cache.add_active_response(req_id, extra)
            if (
                req_id not in cache.active_requests
                or req_id not in cache.cached_requests
            ):
                raise RuntimeError(
                    "fixed32 Arctic owner disappeared after finalization"
                )
    except Exception:
        _fixed32_poison("finalize_fixed32_step", step_seq)
        raise

    for req_id, safe_end, extra in actions:
        _INGESTED_LEN[req_id] = safe_end + len(extra)
        _COMMITTED[req_id] = (
            list(_COMMITTED.get(req_id, ())) + extra
        )[-max_tree_depth:]
        STATS["ingested"] += 1
    _FIXED32_PENDING_STEP = None


# ---- head-merge seam DELETED 2026-07-27 -------------------------------------
# decide_and_fill (the adaptive MTP-k + Arctic grow-to-cat33333 head-merge path) lived
# here. Removed with the patcher's FR13_MERGED_DRAFTER_SEAM (cleanup+bake). git history
# has the code; do not resurrect without a fresh gate — Front-2/merge was a closed no-go.

_TAIL_WIDE_TOPK = {}
_TAIL_PATH_TOKENS = {}
_FIXED32_WORK_SERIAL = 0
_FIXED32_LAST_WORK = None


def get_tail_wide_topk():
    """Direction-2 d6-branch: the tail-branch wide_topk {parent_pos: [B, width]} from the LAST decide_tail
    call ({} when the spine-only tail ran). The patcher tail-append merges this into _fr10_wide_topk."""
    return _TAIL_WIDE_TOPK


def get_tail_path_tokens():
    """Hydra conditional-chain columns keyed by full tree path from the last call."""
    return _TAIL_PATH_TOKENS


def decide_tail(cache, spec_row_req_ids, mtp_head_per_depth, head_depth, tail_len,
                device, pad_token, max_spec_tokens=32, max_spec_factor=4.0, min_token_prob=0.0,
                vocab_size=None, hydra_seed_per_rank=None,
                hydra_branch_chains=((1, 4),), physical_branch_chains=None,
                carry_short_chains=False, strict=False, work_counters=None):
    """accept>5 TAIL (mtp_k=head_depth mode): the MTP head (depths 0..head_depth-1) is 100% native
    (byte-identical baseline, filled by the drafter's own forwards); this ONLY produces the deep Arctic
    CHAIN past the head (depths head_depth..head_depth+tail_len-1) to APPEND to _fr10_spine_tokens.
    Pattern per row = _COMMITTED[req] + the native MTP head tokens (seeds the walk from MTP's confident
    prefix). Returns a list of `tail_len` int64 [batch] tensors on `device`, or None (empty batch / no
    cache). LOSSLESS + never-regress: the head is unchanged and the tail only ADDS candidates (cold ->
    pad, never matches past the head). The committer (accept=p(S), source/depth-blind) does the rest.

    mtp_head_per_depth: list length head_depth; entry d is a per-row-indexable of the native MTP spine
                        token at depth d (e.g. _fr10_spine_tokens[d].cpu().tolist()), row order ==
                        spec_row_req_ids order (req_id-keyed by the caller).

    ``hydra_seed_per_rank`` additionally runs flat Arctic walks conditioned
    only on committed tokens plus each root sibling. Those candidates remain
    distribution-lossless through the parent-aware committer, but Hydra23 is
    a candidate trade and is not monotone in acceptance versus tail6."""
    from fr13_arctic_suffix_adapter import arctic_draft_to_suffix_rel, arctic_tree_to_suffix_rel
    from fr13_merged_fill import (
        build_tail_branch_columns,
        build_tail_and_hydra_columns,
    )
    global _TAIL_WIDE_TOPK, _TAIL_PATH_TOKENS
    _TAIL_WIDE_TOPK = {}
    _TAIL_PATH_TOKENS = {}
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
    _hydra_on = bool(hydra_seed_per_rank)
    if _hydra_on and _branched:
        raise ValueError("hydra conditional chains and tail sibling branches are mutually exclusive")
    _hydra_chains = tuple(
        (int(root_rank), int(chain_len))
        for root_rank, chain_len in hydra_branch_chains
    )
    _physical_chains = (
        _hydra_chains
        if physical_branch_chains is None
        else tuple(
            (int(root_rank), int(chain_len))
            for root_rank, chain_len in physical_branch_chains
        )
    )
    B = len(spec_row_req_ids)
    if _hydra_on:
        if any(root_rank <= 0 or chain_len < 0 for root_rank, chain_len in _hydra_chains):
            raise ValueError(f"invalid hydra branch chains: {_hydra_chains!r}")
        lookup_lengths = dict(_hydra_chains)
        if (
            len(lookup_lengths) != len(_hydra_chains)
            or any(root_rank <= 0 or chain_len < 0
                   for root_rank, chain_len in _physical_chains)
            or any(
                root_rank not in lookup_lengths
                or chain_len < lookup_lengths[root_rank]
                for root_rank, chain_len in _physical_chains
            )
            or set(lookup_lengths) != {root_rank for root_rank, _ in _physical_chains}
        ):
            raise ValueError(
                "physical hydra chains must contain every lookup chain at "
                f"equal-or-greater length: lookup={_hydra_chains!r} "
                f"physical={_physical_chains!r}"
            )
        for root_rank, chain_len in _hydra_chains:
            seeds = hydra_seed_per_rank.get(root_rank)
            if chain_len and (seeds is None or len(seeds) != B):
                raise ValueError(
                    f"hydra root rank {root_rank} needs {B} row-aligned seeds"
                )
    _to_rel = arctic_tree_to_suffix_rel if _branched else arctic_draft_to_suffix_rel

    if B == 0 or cache is None:
        return None
    tail_rows = []
    branch_rows = [None] * B
    hydra_rows = []
    any_hit = False
    for b in range(B):
        req_id = spec_row_req_ids[b]
        head = [int(mtp_head_per_depth[d][b]) for d in range(head_depth)]
        committed = list(_COMMITTED.get(req_id, []))
        pattern = committed + head
        row = None
        try:
            draft = cache.speculate(
                req_id, pattern, max_spec_tokens=max_spec_tokens,
                max_spec_factor=max_spec_factor, min_token_prob=min_token_prob,
                use_tree_spec=_branched)
            if work_counters is not None:
                work_counters["main_lookup_calls"] = (
                    int(work_counters.get("main_lookup_calls", 0)) + 1
                )
                work_counters["main_lookup_tokens"] = (
                    int(work_counters.get("main_lookup_tokens", 0))
                    + int(max_spec_tokens)
                )
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
            if strict:
                raise
            row = None
        tail_rows.append(row)
        hydra_row = {}
        if _hydra_on:
            for root_rank, chain_len in _hydra_chains:
                seeds = hydra_seed_per_rank.get(root_rank)
                if chain_len == 0:
                    hydra_row[root_rank] = []
                    continue
                seed = int(seeds[b])
                try:
                    hydra_draft = cache.speculate(
                        req_id,
                        committed + [seed],
                        max_spec_tokens=chain_len,
                        max_spec_factor=max_spec_factor,
                        min_token_prob=min_token_prob,
                        use_tree_spec=False,
                    )
                    if work_counters is not None:
                        call_key = f"rank{int(root_rank)}_lookup_calls"
                        token_key = f"rank{int(root_rank)}_lookup_tokens"
                        work_counters[call_key] = (
                            int(work_counters.get(call_key, 0)) + 1
                        )
                        work_counters[token_key] = (
                            int(work_counters.get(token_key, 0))
                            + int(chain_len)
                        )
                    STATS["hydra_speculate_fired"] = (
                        STATS.get("hydra_speculate_fired", 0) + 1
                    )
                    hydra_rel = arctic_draft_to_suffix_rel(
                        hydra_draft, max_rel=chain_len
                    )
                    raw_chain = [
                        hydra_rel[j][0] if hydra_rel.get(j) else None
                        for j in range(chain_len)
                    ]
                    real = sum(token is not None for token in raw_chain)
                    STATS["hydra_real"] = STATS.get("hydra_real", 0) + real
                    if real:
                        STATS["hydra_hit"] = STATS.get("hydra_hit", 0) + 1
                        rank_key = f"hydra_rank{root_rank + 1}_hit"
                        STATS[rank_key] = STATS.get(rank_key, 0) + 1
                    # A conditional path must never borrow the main spine's
                    # token. On a short/cold Arctic result, carry this branch's
                    # own latest valid token (starting at its root seed).
                    hydra_chain = []
                    prev = seed
                    for token in raw_chain:
                        if token is None:
                            hydra_chain.append(prev)
                            continue
                        token = int(token)
                        hydra_chain.append(token)
                        if vocab_size is None or 0 <= token < int(vocab_size):
                            prev = token
                    hydra_row[root_rank] = hydra_chain
                except Exception:
                    if strict:
                        raise
                    hydra_row[root_rank] = [seed] * chain_len
        hydra_rows.append(hydra_row if hydra_row else None)
    if not any_hit:
        STATS["tail_all_cold"] = STATS.get("tail_all_cold", 0) + 1
    _maybe_log_engagement()
    _spine, _TAIL_PATH_TOKENS = build_tail_and_hydra_columns(
        tail_rows,
        hydra_rows if _hydra_on else None,
        device,
        pad_token,
        tail_len,
        branch_chains=_physical_chains,
        carry_short_chains=carry_short_chains,
        vocab_size=vocab_size,
        work_counters=work_counters,
    )
    _TAIL_WIDE_TOPK = (build_tail_branch_columns(branch_rows, device, pad_token, head_depth, _tbd, _tb,
                                                 vocab_size=vocab_size) if _branched else {})
    return _spine


def decide_fixed32(
    cache,
    spec_row_req_ids,
    mtp_head_per_depth,
    hydra_seed_per_rank,
    device,
    pad_token,
    *,
    vocab_size=None,
    max_spec_factor=4.0,
    min_token_prob=0.0,
):
    """Build the common 31-draft payload for both fixed-32 logical arms.

    Every request executes exactly three Arctic calls: main/6, rank-1/4, and
    rank-2/2. The physical rank-2 chain has six columns, so its final four
    columns carry the last valid rank-2 token without another lookup. Together
    with the native 15-node MTP head this always publishes 31 draft columns.
    Logical Tail6 versus Hydra27 selection happens later through the sampler
    validity mask and cannot alter drafter work.
    """
    from fr13_fixed32_topology import (
        ARCTIC_LOOKUP_CHAINS,
        ARCTIC_LOOKUP_CALLS_PER_REQUEST,
        ARCTIC_LOOKUP_TOKENS_PER_REQUEST,
        ARCTIC_MAIN_TAIL_LENGTH,
        PHYSICAL_BRANCH_CHAINS,
        PHYSICAL_DRAFTS,
        RESCUE_CARRY_SLOTS_PER_REQUEST,
        branch_paths,
        validate_contract,
    )

    validate_contract()
    B = len(spec_row_req_ids)
    if not 1 <= B <= 4:
        raise RuntimeError(f"fixed32 drafter requires B in [1,4], got {B}")
    if cache is None:
        raise RuntimeError("fixed32 drafter requires a live Arctic cache")
    if len(mtp_head_per_depth) != N_DEPTH:
        raise RuntimeError(
            f"fixed32 drafter requires exactly {N_DEPTH} MTP head depths, "
            f"got {len(mtp_head_per_depth)}"
        )
    if set(hydra_seed_per_rank or {}) != {
        rank for rank, _length in ARCTIC_LOOKUP_CHAINS
    }:
        raise RuntimeError(
            "fixed32 drafter requires row-aligned rank-1 and rank-2 root seeds"
        )

    work = {
        "batch_size": B,
        "main_lookup_calls": 0,
        "main_lookup_tokens": 0,
        "rank1_lookup_calls": 0,
        "rank1_lookup_tokens": 0,
        "rank2_lookup_calls": 0,
        "rank2_lookup_tokens": 0,
        "rescue_carry_slots": 0,
    }
    tail = decide_tail(
        cache,
        spec_row_req_ids,
        mtp_head_per_depth,
        head_depth=N_DEPTH,
        tail_len=ARCTIC_MAIN_TAIL_LENGTH,
        device=device,
        pad_token=pad_token,
        max_spec_tokens=ARCTIC_MAIN_TAIL_LENGTH,
        max_spec_factor=max_spec_factor,
        min_token_prob=min_token_prob,
        vocab_size=vocab_size,
        hydra_seed_per_rank=hydra_seed_per_rank,
        hydra_branch_chains=ARCTIC_LOOKUP_CHAINS,
        physical_branch_chains=PHYSICAL_BRANCH_CHAINS,
        carry_short_chains=True,
        strict=True,
        work_counters=work,
    )
    paths = get_tail_path_tokens()
    expected_paths = set(branch_paths(PHYSICAL_BRANCH_CHAINS))
    if tail is None or len(tail) != ARCTIC_MAIN_TAIL_LENGTH:
        raise RuntimeError("fixed32 main tail did not produce exactly six columns")
    if set(paths) != expected_paths:
        raise RuntimeError(
            "fixed32 rescue pack path mismatch: "
            f"expected={sorted(expected_paths)} actual={sorted(paths)}"
        )
    if N_DEPTH * 3 + len(tail) + len(paths) != PHYSICAL_DRAFTS:
        raise RuntimeError("fixed32 drafter pack is not exactly 31 columns")
    work.update(
        {
            "arctic_lookup_calls": (
                int(work["main_lookup_calls"])
                + int(work["rank1_lookup_calls"])
                + int(work["rank2_lookup_calls"])
            ),
            "arctic_requested_tokens": (
                int(work["main_lookup_tokens"])
                + int(work["rank1_lookup_tokens"])
                + int(work["rank2_lookup_tokens"])
            ),
            "main_tail_columns": len(tail),
            "rescue_path_columns": len(paths),
            "merge_fill_calls": 1,
            "merge_fill_columns": len(tail) + len(paths),
            "merge_fill_rows": B * (len(tail) + len(paths)),
        }
    )
    expected_work = {
        "batch_size": B,
        "main_lookup_calls": B,
        "main_lookup_tokens": ARCTIC_MAIN_TAIL_LENGTH * B,
        "rank1_lookup_calls": B,
        "rank1_lookup_tokens": ARCTIC_LOOKUP_CHAINS[0][1] * B,
        "rank2_lookup_calls": B,
        "rank2_lookup_tokens": ARCTIC_LOOKUP_CHAINS[1][1] * B,
        "rescue_carry_slots": RESCUE_CARRY_SLOTS_PER_REQUEST * B,
        "arctic_lookup_calls": ARCTIC_LOOKUP_CALLS_PER_REQUEST * B,
        "arctic_requested_tokens": ARCTIC_LOOKUP_TOKENS_PER_REQUEST * B,
        "main_tail_columns": ARCTIC_MAIN_TAIL_LENGTH,
        "rescue_path_columns": sum(length for _rank, length in PHYSICAL_BRANCH_CHAINS),
        "merge_fill_calls": 1,
        "merge_fill_columns": (
            ARCTIC_MAIN_TAIL_LENGTH
            + sum(length for _rank, length in PHYSICAL_BRANCH_CHAINS)
        ),
        "merge_fill_rows": B
        * (
            ARCTIC_MAIN_TAIL_LENGTH
            + sum(length for _rank, length in PHYSICAL_BRANCH_CHAINS)
        ),
    }
    if work != expected_work:
        raise RuntimeError(
            "fixed32 drafter observed work drift: "
            f"actual={work!r} expected={expected_work!r}"
        )

    STATS["fixed32_events"] += 1
    STATS["fixed32_rows"] += B
    STATS["fixed32_carry_slots"] += RESCUE_CARRY_SLOTS_PER_REQUEST * B
    global _FIXED32_WORK_SERIAL, _FIXED32_LAST_WORK
    _FIXED32_WORK_SERIAL += 1
    _FIXED32_LAST_WORK = {
        "serial": _FIXED32_WORK_SERIAL,
        **work,
    }
    return tail


def get_fixed32_drafter_last_work():
    """Return the last successfully completed fixed32 call-boundary census."""
    if _FIXED32_LAST_WORK is None:
        return None
    return dict(_FIXED32_LAST_WORK)


def get_fixed32_drafter_work_serial():
    """Return the successful fixed32 decision serial without exposing mutable work."""
    return int(_FIXED32_WORK_SERIAL)


def get_fixed32_drafter_work(batch_size):
    """Return the exact per-event drafter census expected by fixed-32."""
    from fr13_fixed32_topology import (
        ARCTIC_LOOKUP_CHAINS,
        ARCTIC_LOOKUP_CALLS_PER_REQUEST,
        ARCTIC_LOOKUP_TOKENS_PER_REQUEST,
        ARCTIC_MAIN_TAIL_LENGTH,
        MTP_FORWARD_CALLS,
        PHYSICAL_DRAFTS,
        RESCUE_CARRY_SLOTS_PER_REQUEST,
    )

    B = int(batch_size)
    if not 1 <= B <= 4:
        raise RuntimeError(f"fixed32 drafter census requires B in [1,4], got {B}")
    return {
        "mtp_forward_calls": MTP_FORWARD_CALLS,
        "mtp_forward_rows": MTP_FORWARD_CALLS * B,
        "arctic_lookup_calls": ARCTIC_LOOKUP_CALLS_PER_REQUEST * B,
        "arctic_requested_tokens": ARCTIC_LOOKUP_TOKENS_PER_REQUEST * B,
        "main_tail_length": ARCTIC_MAIN_TAIL_LENGTH,
        "rescue_chains": [list(chain) for chain in ARCTIC_LOOKUP_CHAINS],
        "carry_fill_slots": RESCUE_CARRY_SLOTS_PER_REQUEST * B,
        "pack_columns": PHYSICAL_DRAFTS,
        "packed_rows": PHYSICAL_DRAFTS * B,
    }


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
            "HYDRA[fired=%d hit=%d rank2_hit=%d rank3_hit=%d real=%d] "
            "started=%d ingested=%d retired=%d "
            "arctic_oob_dropped=%d last_oob=%s",
            STATS.get("tail_speculate_fired", 0), STATS.get("tail_hit", 0),
            STATS.get("tail_all_cold", 0), STATS.get("tail_branch_real", 0),
            STATS.get("hydra_speculate_fired", 0), STATS.get("hydra_hit", 0),
            STATS.get("hydra_rank2_hit", 0), STATS.get("hydra_rank3_hit", 0),
            STATS.get("hydra_real", 0),
            STATS["started"], STATS["ingested"], STATS["retired"], _oob_n, _oob_last,
        )
    except Exception:
        pass
