#!/usr/bin/env python3
"""FR13 Arctic SuffixDecodingCache.speculate() -> relative suffix_rel adapter (CPU, isolated).

Chosen architecture (FR13_DRAFTER_INTERFACE_DESIGN.md, 2026-07-14, Path B): native MTP drafts the
confident near-spine (mtp_k in {1,2}); Arctic (arctic_inference.suffix_decoding.SuffixDecodingCache)
GROWS the deep spine + branches into the fixed 16-node cat33333 tree; OUR multidraft committer
verifies (lossless by Gate 1, source-agnostic). This module is the small, isolated bridge (SEAM
step (d): ".speculate().token_ids -> relative suffix_rel") between Arctic's output and
scripts/fr13_mtp_suffix_assembly.assemble_cat33333().

Arctic can't run on the host (C++ ext + torch build -> lives in the vLLM container image), so this
adapter is deliberately import-free and CPU-unit-testable against a MOCK draft object. In the live
drafter it consumes the real object returned by:

    draft = suffix_cache.speculate(req_id, pattern, max_spec_tokens=, max_spec_factor=, min_token_prob=)

The authoritative shape (vLLM v1 SuffixDecodingProposer, vllm/v1/spec_decode/suffix_decoding.py:89):
`draft.token_ids` is a FLAT list[int] -- a dynamic-length speculation *continuation chain*. The
proposer appends it verbatim. So this FIRST VERSION treats it as a flat chain and emits a
spine-only suffix_rel:

    suffix_rel = {i: [token_ids[i]] for i in range(len(token_ids))}

`suffix_rel[i]` = ranked candidate token ids for the i-th continuation position PAST the MTP prefix
(i=0 == the FIRST token after the mtp_k-th MTP spine token; absolute spine depth == mtp_k + i).
One candidate per position => assemble_cat33333 fills the deep SPINE from Arctic and FALLS BACK to
MTP topk for every BRANCH (Arctic supplies no per-position alternatives yet). A later version can
widen each list with Arctic's tree alternatives (min_token_prob-ranked siblings) to feed branch
slots too; the assembly already consumes suffix_rel[i][1:] as ranked branch candidates.

CONTRACT (must stay exact so assemble_cat33333 consumes it verbatim):
  * keys are ints 0..len-1 (relative continuation index).
  * values are non-empty lists of python ints, ranked best-first.
  * empty / cold / None draft -> {} (assembly then does pure-MTP fallback = the baseline).
"""
from __future__ import annotations

__all__ = ["arctic_draft_to_suffix_rel", "extract_draft_token_ids", "arctic_match_confidence"]


def arctic_match_confidence(draft) -> float:
    """Confidence of an Arctic match for the CONFIDENCE-GATED skip (merge16d verdict: the blanket skip
    fires on trie COVERAGE, not confidence -> skips over-confident-but-wrong deep matches -> accept
    collapse 3.61->1.97). This returns the MIN per-node continuation probability along the drafted chain
    -- the weakest link: if any deep node is low-prob, the suffix statistics do NOT strongly expect the
    model to follow, so the caller should NOT skip (fall back to full MTP = never-regress). Falls back to
    `.score`, then to 1.0 (no signal -> don't block = back-compat blanket behaviour). None -> 0.0.

    Higher = the suffix stats more strongly expect the model to follow the whole chain, so the deep-forward
    skip is more likely to hold accept. arctic prob = suffix FREQUENCY (context repetition), not the model
    prob, but high-frequency continuations are exactly the ones the model tends to follow (the win case)."""
    if draft is None:
        return 0.0
    probs = getattr(draft, "probs", None)
    if probs is not None:
        try:
            vals = [float(p) for p in probs]
            if vals:
                return min(vals)
        except Exception:
            pass
    score = getattr(draft, "score", None)
    if score is not None:
        try:
            return float(score)
        except Exception:
            pass
    return 1.0   # no confidence signal -> don't block (back-compat = blanket skip)


def extract_draft_token_ids(draft) -> list[int]:
    """Pull the flat continuation chain out of an Arctic .speculate() result, robustly.

    Accepts:
      * an object exposing `.token_ids` (the real arctic draft; also our mock) -> uses that,
      * a plain list/tuple of ids (or any iterable) -> uses it directly,
      * a tensor / ndarray (via `.tolist()`),
      * None, or a draft whose `.token_ids` is None/empty -> [].

    Every element is coerced to a python int (arctic yields python ints; tensors yield np/torch
    scalars which int() flattens) so downstream keys/values are clean ints.
    """
    if draft is None:
        return []
    # Object exposing .token_ids (real arctic draft OR mock) takes precedence over "is a list".
    tok = draft.token_ids if hasattr(draft, "token_ids") else draft
    if tok is None:
        return []
    # tensor / ndarray -> python list
    if hasattr(tok, "tolist"):
        tok = tok.tolist()
    return [int(t) for t in tok]


def arctic_tree_to_suffix_rel(draft, max_rel: int | None = None) -> dict[int, list[int]]:
    """v2: convert an Arctic use_tree_spec=True draft into PER-DEPTH RANKED suffix_rel.

    The tree draft (from speculate(..., use_tree_spec=True)) exposes:
      draft.token_ids : flat list, one token per tree node (topological order).
      draft.parents   : parent index per node (-1 for the root's children = depth-0 continuation).
      draft.probs     : per-node probability (for ranking siblings into spine vs branches).
    Walking the MTP prefix down the suffix trie lands at the root; each node's depth = chain length
    from a `-1` parent. suffix_rel[d] = the tokens at depth d, RANKED best-first by prob -- so
    assemble_cat33333 takes suffix_rel[d][0] as the deep spine and suffix_rel[d][1:] as the 2
    cat33333 branches (exactly your trie-branches -> tree-branches mapping). Empty/None -> {} (cold
    fallback = pure MTP baseline).

    max_rel caps the relative depth (N_DEPTH - mtp_k). Robust to a flat draft (parents absent ->
    every node depth 0? no -- falls back to the flat spine adapter).
    """
    if draft is None:
        return {}
    token_ids = extract_draft_token_ids(draft)
    if not token_ids:
        return {}
    parents = getattr(draft, "parents", None)
    if parents is None or not hasattr(draft, "probs"):
        # no tree structure -> flat spine-only (v1 behaviour)
        return arctic_draft_to_suffix_rel(draft, max_rel=max_rel)
    parents = list(parents)
    probs = [float(p) for p in draft.probs]
    n = min(len(token_ids), len(parents), len(probs))
    # depth of each node = chain length from a -1 parent (root children are depth 0)
    depth = [0] * n
    for i in range(n):
        p = int(parents[i])
        depth[i] = 0 if p < 0 else depth[p] + 1
    # group (token, prob) by depth
    by_depth: dict = {}
    for i in range(n):
        by_depth.setdefault(depth[i], []).append((probs[i], int(token_ids[i])))
    out: dict = {}
    for d, items in by_depth.items():
        if max_rel is not None and d >= max_rel:
            continue
        # rank best-first by prob; de-dup tokens keeping the highest-prob occurrence
        seen = set()
        ranked = []
        for _pr, tok in sorted(items, key=lambda x: x[0], reverse=True):
            if tok in seen:
                continue
            seen.add(tok)
            ranked.append(tok)
        out[d] = ranked
    return out


def arctic_flat_tree_to_suffix_rel(flat_draft, tree_draft, max_rel: int | None = None) -> dict[int, list[int]]:
    """HYBRID (v2-correct): SPINE from the flat deep chain (use_tree_spec=False) + BRANCHES from the
    trie subtree (use_tree_spec=True). Solves the depth-vs-breadth tension: use_tree_spec spends its
    budget on breadth -> shallow tree -> rarely reaches the depth the skip gate needs (v2 engaged
    ~0.7% live); the FLAT chain always goes deep (v1 engaged ~47%). So take the spine (and the depth
    coverage that drives the skip gate) from FLAT, and add the trie's per-depth siblings as branches.

    Returns {i: [flat_spine_i, trie_branch, ...]} for i over the flat chain's depth (capped max_rel).
    assemble_cat33333 takes [0] as the deep spine and [1:] as the cat33333 branches -- so the SPINE
    reaches full depth (high engagement) AND the branches come from the trie (your trie-walk).
    """
    flat = extract_draft_token_ids(flat_draft)
    if not flat:
        return {}
    tree_rel = arctic_tree_to_suffix_rel(tree_draft, max_rel=max_rel)
    out: dict = {}
    for i, spine_tok in enumerate(flat):
        if max_rel is not None and i >= max_rel:
            break
        spine_tok = int(spine_tok)
        branches = [t for t in tree_rel.get(i, []) if t != spine_tok]   # trie siblings, dedup vs spine
        out[i] = [spine_tok] + branches
    return out


def arctic_draft_to_suffix_rel(draft, max_rel: int | None = None) -> dict[int, list[int]]:
    """Convert an Arctic .speculate() draft into the RELATIVE suffix_rel dict.

    draft:    the object returned by SuffixDecodingCache.speculate() (has `.token_ids`), OR a plain
              list of token ids, OR None (cold / no match).
    max_rel:  optional cap on the number of relative positions to emit. The live drafter only needs
              the deep spine positions it is growing (N_DEPTH - mtp_k of them), so it may pass
              max_rel = N_DEPTH - mtp_k to keep the dict tight; assemble_cat33333 harmlessly ignores
              any extra positions, so None (no cap = spec-exact FIRST VERSION) is also fine.

    Returns {i: [token_ids[i]]} for i in range(len) -- spine-only ranked-candidate lists. Empty when
    the draft is cold/None (assembly -> pure-MTP fallback = baseline, never regresses).
    """
    token_ids = extract_draft_token_ids(draft)
    if max_rel is not None:
        if max_rel < 0:
            raise ValueError(f"max_rel must be >= 0, got {max_rel}")
        token_ids = token_ids[:max_rel]
    return {i: [token_ids[i]] for i in range(len(token_ids))}
