#!/usr/bin/env python3
"""FR13 merged-drafter fill: assembled cat33333 nodes -> _fr10_spine_tokens / _fr10_wide_topk tensors.

The correctness-critical bridge from the (CPU, per-request) assembly output to the exact tensor
format the FR10 WIDE packer consumes (:13905): for each (pp, rk) in _fr10_wide_plan the packer reads
`_fr10_spine_tokens[pp]` (rk==0) or `_fr10_wide_topk[pp][:, rk]` (rk>0). This module builds those
tensors with the red-team's REQUIRED tensor discipline (wf_5ab28060 review):
  * DEVICE-matched: every column on the incumbent draft tensors' device (int64), one H2D per depth
    -- NOT torch.tensor(list) (CPU) which crashes the packer's torch.stack on device mismatch.
  * ROW-MAJOR by req_id: the caller passes assembled_rows ORDERED BY BATCH ROW b (row b == the
    drafter batch row whose req_id is spec_row_req_ids[b]); this module never reorders. Positional
    misalignment = garble-free but wrong-row drafts = accept collapse, so the caller MUST key the
    per-row assembly by req_id (this module just consumes the row-ordered list).
  * WIDTH >= 3: _fr10_wide_topk[d] is [batch, width>=3]; cat33333 consumes ranks {1,2} and the packer
    fail-loud-checks rk < shape[1] (:13928). col 0 = the spine token (free self-dedup placeholder).
  * PAD-densified: a row with no assembly (None -> cold/no-match fallback for that row) is filled with
    pad_token at every slot (Gate-1 lossless: p[pad]~0 commits at rate ~0, never wrong-accepted).

cat33333 spine depth = 5; assembled_rows[b] is the 15-int CAT33333_ORDER list from
fr13_mtp_suffix_assembly.assemble_cat33333 (== [spine_d, branch_a_d, branch_b_d] for d in 0..4).
"""
from __future__ import annotations

import torch

N_DEPTH = 5

# Module counter: arctic-sourced candidate ids that fell OUTSIDE [0, vocab_size) and were dropped to
# pad. An OOB draft token indexes the embedding/gather out of range -> CUDA device-side assert that
# kills EngineCore (observed merge16c 2026-07-15, ~skip #357). Arctic's suffix output is the ONLY
# unguarded token source (draft_token_ids + torch.topk candidates are valid by construction). Dropping
# OOB -> pad is lossless (Gate-1: p[pad]~0, MTP-equivalent fallback for that slot) AND fail-loud-visible.
_OOB_DROPPED = 0
_OOB_LAST = None  # (depth, slot, bad_value) of the most recent drop, for the needle/log


def get_oob_stats():
    """Return (total_dropped, last_offending) so the drafter needle can surface OOB arctic ids."""
    return _OOB_DROPPED, _OOB_LAST


def build_cat33333_columns(assembled_rows, device, pad_token, width=3, vocab_size=None):
    """Build (spine_tokens, wide_topk) for the wide packer from row-ordered assembled nodes.

    assembled_rows: list of length batch. Each entry is either a 15-int list (CAT33333_ORDER from
                    assemble_cat33333) OR None (that row falls back -> PAD every slot).
    device:         the device of the incumbent draft tensors (draft_token_ids.device).
    pad_token:      int token id for missing/None rows (e.g. a benign token; Gate-1 makes it lossless).
    width:          columns of each _fr10_wide_topk[d] tensor (>=3 required; default 3).

    Returns (spine_tokens, wide_topk):
      spine_tokens: list of N_DEPTH int64 [batch] tensors on `device` (spine token per depth).
      wide_topk:    dict {d: int64 [batch, width] tensor on `device`} for d in 0..N_DEPTH-1;
                    col 0 = spine token (dedup-safe placeholder), cols 1,2 = the two branch tokens.
    """
    assert width >= 3, f"wide_topk width must be >= 3 (packer reads ranks 1,2), got {width}"
    B = len(assembled_rows)
    assert B > 0, "empty batch"
    vs = int(vocab_size) if vocab_size is not None else None
    pad = int(pad_token)
    if vs is not None and not (0 <= pad < vs):
        pad = 0   # pad itself OOB (should never happen: pad=root MTP token) -> id 0 is always valid

    def node(row, i):
        if row is None:
            return pad
        v = int(row[i])
        # BOUNDS GUARD: an OOB arctic candidate would index the embedding out of range ->
        # CUDA device-side assert (kills EngineCore). Drop to pad (lossless MTP-equiv fallback).
        if vs is not None and not (0 <= v < vs):
            global _OOB_DROPPED, _OOB_LAST
            _OOB_DROPPED += 1
            _OOB_LAST = (i // 3, i % 3, v)
            return pad
        return v

    spine_tokens = []
    wide_topk = {}
    for d in range(N_DEPTH):
        # one CPU-built row-ordered list per depth, then ONE H2D to `device`. Read each node ONCE
        # (spine col derived from the triple's col0) so the OOB counter isn't double-counted.
        spine_col = []
        wt_rows = []
        for b in range(B):
            row = assembled_rows[b]
            s = node(row, 3 * d)
            ba = node(row, 3 * d + 1)
            bb = node(row, 3 * d + 2)
            spine_col.append(s)
            wt_rows.append([s, ba, bb] + [pad] * (width - 3))   # col0=spine, cols1,2=branches, pad extra
        spine_tokens.append(torch.tensor(spine_col, dtype=torch.int64, device=device))
        wide_topk[d] = torch.tensor(wt_rows, dtype=torch.int64, device=device)
    return spine_tokens, wide_topk
