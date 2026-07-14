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

__all__ = ["arctic_draft_to_suffix_rel", "extract_draft_token_ids"]


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
