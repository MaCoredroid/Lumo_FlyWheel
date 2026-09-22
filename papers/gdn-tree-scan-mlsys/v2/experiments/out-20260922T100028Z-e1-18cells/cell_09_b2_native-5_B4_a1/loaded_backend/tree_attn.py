# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with TreeAttention."""

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


# FR13_FA2_MAB: in-process forked-FA2 M-invariance A/B (default OFF).
#
# Decisive controlled test for the cat9 22-flip SPINE_PERTURBATION carrier.
# At the carrier full_attention layers, on the SAME live captured K/V from one
# cat9 boot's deep-accept event, re-call OUR forked-FA2 op (the tree-bias
# varlen forward) TWICE and compare the deep-spine row's attn_out:
#   (a) M=9: the full tree (all 9 query rows + the 9x9 ancestry bias).
#   (b) M=5: the SPINE-SLICE (only the spine query rows [0,1,2,4,6] + the 5x5
#            spine-ancestry sub-bias; the spine-suffix KV rows [0,1,2,4,6]).
# The spine rows attend ONLY to spine ancestors (strict mask), so the M=5
# slice's spine bias == the M=9 spine bias for those keys; the ONLY thing that
# changes between (a) and (b) is M (query-row occupancy / kBlockM MMA fragment
# tile). RAW max_abs (NOT atol) on the deep-spine row (cat9 flat row 6 = node5)
# is computed PER full_attention layer.
#
# VERDICT: RAW != 0 at the full-attn layers => the forked-FA2 IS M-dependent
# (the query-tile fragment realization) => the query-pad fix (FR13_FA2_QPAD) is
# REAL.  RAW == 0 (bit-exact) => the fork is M-invariant => NOT the carrier =>
# the 22 is genuinely DIFFUSE (per-layer GDN ~1-ULP) => reshape is the route.
#
# This re-uses the captured K/V (NO second served stream => no stream
# decoherence). The re-call uses CONTIGUOUS (non-paged) KV built from the live
# block-gathered dense K/V (no block_table) -- exactly how the banked FA2 path
# oracles already call the fork -- so it is unaffected by paged-KV block_table
# or varlen cu_seqlens. Default OFF; locked cat9 default path is byte-identical
# when FR13_FA2_MAB != "1".
_FR13_MAB_TREE_PARENT = (-1, 0, 1, 1, 2, 2, 4, 4, 6, 6)
_FR13_MAB_SPINE_ROWS = (0, 1, 2, 4, 6)
_FR13_MAB_DEEP_ROW = 6


def _fr13_fa2_mab_recall(
    impl,
    layer_name,
    capture_call_index,
    capture_saved_index,
    query,
    dense_key,
    dense_value,
    seq_lens,
    tree_attn_bias,
    output,
    spine_nodes=None,
):
    """In-process M_full-vs-M_spine spine-slice A/B on OUR forked-FA2 op.

    spine_nodes (attn_metadata.fr10_tree_path0_nodes) generalizes the A/B to ANY
    served tree (cat8 spine=[0,1,3,5,7,8] deep=8, cat6, cat9); falls back to the
    hardcoded cat9 caterpillar constants only when it is None.
    """
    if os.environ.get("FR13_FA2_MAB", "0") != "1":
        return
    dump = os.environ.get(
        "FR13_FA2_MAB_DUMP", "/logs/fr13_fa2_mab.jsonl"
    )
    try:
        import importlib

        # Resolve the forked varlen op via importlib (NOT a bare import line) so
        # this helper text never collides with the fork patch's import-insertion
        # anchors in fr13_patch_fa2_tree_bias.py::_patch_tree_attn.
        flash_attn_varlen_func = getattr(
            importlib.import_module("vllm.v1.attention.backends.fa_utils"),
            "flash_attn_varlen_func",
        )

        if tree_attn_bias is None:
            return
        bias = tree_attn_bias.to(torch.float32).cpu()
        m_full = int(query.shape[0])
        if m_full <= 0 or bias.shape[0] < m_full or bias.shape[1] < m_full:
            return
        # Derive the spine for the LIVE served tree (cat8 [0,1,3,5,7,8] deep=8,
        # cat6, cat9), NOT the hardcoded cat9 caterpillar.  IMPORTANT: the FA2
        # hook receives TreeAttentionMetadata, which carries tree_attn_bias but
        # NOT fr10_tree_parent/path0 (those are on the SEPARATE GDN metadata) --
        # so spine_nodes is reliably None here.  Derive the spine from the bias
        # itself = the DEEPEST node's ancestor set (its visible keys); that is
        # the longest root->leaf path = the all-zeros spine, and it is causally
        # closed (every spine node's ancestors are in the spine).  Threshold-free
        # visibility: finite & > -1e30 handles both -inf and finfo.min masks.
        spine_rows = None
        if spine_nodes is not None:
            try:
                sr = [int(x) for x in spine_nodes.reshape(-1).tolist()]
                sr = sorted(r for r in sr if 0 <= r < m_full)
                if len(sr) >= 2:
                    spine_rows = sr
            except Exception:
                spine_rows = None
        sub = bias[:m_full, :m_full]
        bmin = float(sub.min().item())
        bmax = float(sub.max().item())
        if spine_rows is not None:
            deep_row = spine_rows[-1]
        else:
            # Data-driven visible/masked split robust to ANY mask convention
            # (-inf, finfo.min, -1e9, -1e4): the mask is the global min (hugely
            # negative); visible ancestor entries (~0) sit far above HALF the
            # min, masked entries sit below it.  0.5*(-inf) = -inf handles the
            # -inf case (bias > -inf keeps finite, drops -inf).
            thresh = 0.5 * bmin if bmin < -1.0 else -1e30
            vis = sub > thresh
            counts = vis.sum(dim=1)
            deep_row = int(counts.argmax().item())
            spine_rows = sorted(
                j for j in range(m_full) if bool(vis[deep_row, j].item())
            )
        if deep_row not in spine_rows or len(spine_rows) < 2:
            return

        # Single-sequence assumption for the carrier capture (the deep-accept
        # event is one decode request); use seq 0.
        seq_len = int(seq_lens.reshape(-1)[0].item())
        context_len = seq_len - m_full
        if context_len < 0:
            return

        dev = torch.device("cuda")
        scale = float(getattr(impl, "scale", 0.0))
        softcap = float(getattr(impl, "logits_soft_cap", 0.0))
        # Use bf16 KV/Q for the fork (the deployed decode dtype); the dense
        # K/V capture is fp32, so cast down once for an apples-to-apples call.
        # Move all operands to the SAME device up front so the row index_select
        # (index tensors built on dev) never crosses the cpu/cuda boundary; the
        # cast+placement is common-mode across the M=9 and M=5 arms so the
        # M9-vs-M5 RAW max_abs verdict is unaffected.
        q_all = query.to(dev).to(torch.bfloat16)
        k_all = dense_key[0].to(dev).to(torch.bfloat16)
        v_all = dense_value[0].to(dev).to(torch.bfloat16)
        bias_full = bias[:m_full, :m_full].contiguous()

        def _call(q_rows, suffix_rows, bias_sub, pad_to=0, kv_pad_to=0, causal=True):
            m = len(q_rows)
            q = q_all.index_select(
                0, torch.tensor(q_rows, dtype=torch.long, device=dev)
            ).contiguous()
            # KV = full context prefix + the selected tree-suffix nodes.
            kv_idx = list(range(context_len)) + [
                context_len + r for r in suffix_rows
            ]
            tb = bias_sub.to(dev).contiguous()
            # FR13 KV-SUFFIX-PAD validation (kv_pad_to): pad the tree-SUFFIX KV
            # (and the tree_bias columns) to a fixed width with -inf-masked dummy
            # keys, so the flash online-softmax block iteration over the suffix is
            # M-invariant (M9 has 9 suffix keys, M6 has 6 -> both -> kv_pad_to).
            # Dummy keys repeat suffix key 0 but are bias=-inf for ALL queries =>
            # contribute 0 => math unchanged; only the iteration count changes,
            # which is exactly the axis under test.  Near-no-HBM-tax (suffix <<
            # cached context).
            _S = len(suffix_rows)
            if kv_pad_to and kv_pad_to > _S:
                kv_idx = kv_idx + [context_len + suffix_rows[0]] * (
                    kv_pad_to - _S
                )
                neg = torch.full(
                    (tb.shape[0], kv_pad_to - _S), float("-inf"),
                    dtype=tb.dtype, device=dev,
                )
                tb = torch.cat([tb, neg], dim=1).contiguous()
            kv_sel = torch.tensor(kv_idx, dtype=torch.long, device=dev)
            k = k_all.index_select(0, kv_sel).contiguous()
            v = v_all.index_select(0, kv_sel).contiguous()
            # FR13_FA2_QPAD validation: pad the QUERY to a fixed pad_to so
            # max_seqlen_q (and the kBlockM tile occupancy / q_offset) is
            # M-invariant.  Dummy rows repeat row 0's q + bias (valid attn, no
            # NaN); their outputs are discarded (we return only the real m rows).
            # Attention rows are independent, so padding cannot corrupt real
            # rows -- it ONLY changes max_seqlen_q, which is exactly the axis
            # under test.
            mq = m
            if pad_to and pad_to > m:
                npad = pad_to - m
                # FRONT-pad: dummy rows at the TOP, real rows at the BOTTOM.  The
                # forked FA2 places query rows bottom-aligned (q_offset =
                # max_seqlen_q - rows), so bottom-aligning the real rows PRESERVES
                # their bias/causal alignment (self-check ~0), AND lands the
                # deep-spine node (last real row) at position pad_to-1 in EVERY arm
                # => same tile position => the actual QPAD M-invariance test.
                q = torch.cat(
                    [q[:1].expand(npad, *q.shape[1:]).contiguous(), q], dim=0
                ).contiguous()
                tb = torch.cat(
                    [tb[:1].expand(npad, tb.shape[1]).contiguous(), tb], dim=0
                ).contiguous()
                mq = pad_to
            cu_q = torch.tensor([0, q.shape[0]], dtype=torch.int32, device=dev)
            cu_k = torch.tensor(
                [0, k.shape[0]], dtype=torch.int32, device=dev
            )
            out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=mq,
                max_seqlen_k=int(k.shape[0]),
                softmax_scale=scale,
                causal=causal,
                window_size=[-1, -1],
                softcap=softcap,
                fa_version=2,
                tree_bias=tb,
            )
            torch.cuda.synchronize()
            # Real rows are the BOTTOM m (front-padded); slice them off preserving
            # the original q_rows order.  For pad_to==0 this is the whole output.
            return out.detach().to(torch.float32).cpu()[-m:]

        # (a) M=9 full tree: all rows, suffix = all tree nodes, full bias.
        full_rows = list(range(m_full))
        out_m9 = _call(full_rows, full_rows, bias_full)
        # (b) M=5 spine-slice: spine query rows attend to spine-suffix nodes
        # only; the 5x5 spine sub-bias is the M=9 bias restricted to spine
        # rows/cols (strict spine ancestry).
        bias_spine = bias_full.index_select(
            0, torch.tensor(spine_rows, dtype=torch.long)
        ).index_select(
            1, torch.tensor(spine_rows, dtype=torch.long)
        ).contiguous()
        out_m5 = _call(spine_rows, spine_rows, bias_spine)

        # Row alignment: the deep-spine row (flat row 6 = node5) is at
        # position deep_row in the M=9 output and at index
        # spine_rows.index(deep_row) in the M=5 spine-slice output.
        m5_deep = spine_rows.index(deep_row)
        row_m9 = out_m9[deep_row].reshape(-1)
        row_m5 = out_m5[m5_deep].reshape(-1)
        raw_max_abs = float((row_m9 - row_m5).abs().max().item())
        raw_mean_abs = float((row_m9 - row_m5).abs().mean().item())
        # Also the served kernel's own row 6 (cross-check the re-call's M=9
        # arm matches the live forked-FA2 output for the deep-spine row).
        served_row = output[deep_row].reshape(-1).to(torch.float32)
        recall_vs_served = float((row_m9 - served_row).abs().max().item())
        # FR13_FA2_QPAD VALIDATION (gated FR13_FA2_MAB_QPAD=1): re-run BOTH arms
        # with the query padded to each fixed pad_to, and compare the deep-spine
        # row.  If any pad_to drives deep_spine_raw_max_abs -> 0, then pinning
        # max_seqlen_q (the QPAD fix) makes the forked FA2 M-invariant on the
        # spine => QPAD is the fix.  If NO pad_to reaches 0, QPAD is refuted for
        # cat8 (the deep-spine row sits at a DIFFERENT tile position in the M=9
        # vs M=6 arm, so a uniform pad cannot align it) and another compute-only
        # route is needed.  16/32 keep one kBlockM tile; 64 = a full kBlockM
        # (Is_even_MN true).  Observe-only; NO live-path change.
        qpad_results = {}
        qpad_self = {}  # SELF-CHECK: padded-M9 deep row vs UNPADDED-M9 deep row.
        # If padding is CLEAN (attention rows independent), this is ~0 and the
        # qpad_results M9-vs-M6 comparison is a valid QPAD verdict.  If it is
        # LARGE, back-padding corrupts the real rows (bias/causal misalignment)
        # => the QPAD test itself is broken, NOT a valid refutation.
        if os.environ.get("FR13_FA2_MAB_QPAD", "0") == "1":
            for _pt in (16, 32, 64):
                if _pt < m_full:
                    continue
                try:
                    o9p = _call(full_rows, full_rows, bias_full, pad_to=_pt)
                    o5p = _call(spine_rows, spine_rows, bias_spine, pad_to=_pt)
                    r9p = o9p[deep_row].reshape(-1)
                    r5p = o5p[m5_deep].reshape(-1)
                    qpad_results[str(_pt)] = float(
                        (r9p - r5p).abs().max().item()
                    )
                    qpad_self[str(_pt)] = float(
                        (r9p - row_m9).abs().max().item()
                    )
                except Exception as _qe:
                    qpad_results[str(_pt)] = "err:" + repr(_qe)
                    qpad_self[str(_pt)] = "err"
        # FR13 KV-SUFFIX-PAD VALIDATION (gated FR13_FA2_MAB_KVPAD=1): QPAD refuted
        # (query dim irrelevant) => the carrier is the tree-SUFFIX WIDTH.  Pad the
        # suffix KV+bias of BOTH arms to a fixed width (masked dummy keys) so the
        # flash block iteration over the suffix is identical, and compare the deep
        # row.  kv_self = kvpad-M9 vs UNPADDED-M9: MUST be ~0 (masked dummies are
        # math-neutral) for the test to be valid.  If kvpad M9-vs-M6 -> 0 with
        # kv_self~0 => KV-suffix-pad is the fix (near-no-HBM-tax).
        kvpad_results = {}
        kvpad_self = {}
        if os.environ.get("FR13_FA2_MAB_KVPAD", "0") == "1":
            for _kt in (16, 32):
                if _kt < m_full:
                    continue
                try:
                    o9k = _call(
                        full_rows, full_rows, bias_full, kv_pad_to=_kt
                    )
                    o5k = _call(
                        spine_rows, spine_rows, bias_spine, kv_pad_to=_kt
                    )
                    r9k = o9k[deep_row].reshape(-1)
                    r5k = o5k[m5_deep].reshape(-1)
                    kvpad_results[str(_kt)] = float(
                        (r9k - r5k).abs().max().item()
                    )
                    kvpad_self[str(_kt)] = float(
                        (r9k - row_m9).abs().max().item()
                    )
                except Exception as _ke:
                    kvpad_results[str(_kt)] = "err:" + repr(_ke)
                    kvpad_self[str(_kt)] = "err"
        # FR13 FIX-A' VALIDATION (gated FR13_FA2_MAB_REORDER=1): CONTIGUOUS-SPINE
        # REORDER.  Mechanism: the 6.25e-2 is 1 bf16 ULP from butterfly reduction
        # REASSOCIATION -- interleaved branches place surviving spine keys in
        # different score-tile columns.  Fix: permute the suffix q/K/V + BOTH
        # tree_bias axes to pi = [spine (depth order), then branches (topological)]
        # so the deep node's ancestors sit in cols 0..depth (contiguous) in cat8
        # EXACTLY as in the spine-only arm -> identical lane partials -> bit-exact.
        # This preserves actual_seqlen_k/q (no anchor slide) and is a pure exact
        # relabeling (lossless).  reorder_deep_vs_m6 -> 0.0 => A' is the fix.
        reorder_results = {}
        if os.environ.get("FR13_FA2_MAB_REORDER", "0") == "1":
            try:
                _branch = [r for r in range(m_full) if r not in spine_rows]
                _pi = list(spine_rows) + _branch  # topological (parents precede)
                _pit = torch.tensor(_pi, dtype=torch.long)
                _bias_pi = bias_full.index_select(0, _pit).index_select(
                    1, _pit
                ).contiguous()
                _out_pi = _call(_pi, _pi, _bias_pi)  # [m_full] in permuted order
                _deep_pi = _out_pi[_pi.index(deep_row)].reshape(-1)
                # A' claim: reordered cat8 deep row == spine-only (M6) deep row
                reorder_results["deep_vs_m6"] = float(
                    (_deep_pi - row_m5).abs().max().item()
                )
                # the fix magnitude: reordered deep vs ORIGINAL cat8 deep (should
                # be ~1 ULP = the correction, NOT gross corruption like kv-pad)
                reorder_results["deep_vs_m9"] = float(
                    (_deep_pi - row_m9).abs().max().item()
                )
                # relabel-neutrality self-check: un-permute the NON-deep rows and
                # compare to unpadded M9 -- must be within floor (no corruption).
                _inv = [0] * m_full
                for _i, _p in enumerate(_pi):
                    _inv[_p] = _i
                _ndmax = 0.0
                for _r in range(m_full):
                    if _r == deep_row:
                        continue
                    _ndmax = max(_ndmax, float((
                        _out_pi[_inv[_r]].reshape(-1) - out_m9[_r].reshape(-1)
                    ).abs().max().item()))
                reorder_results["nondeep_relabel_max"] = _ndmax
                # WHOLE-SPINE check: EVERY reordered spine row (positions 0..S-1
                # in pi, depth order) vs the M6 spine-only row at the same depth.
                # All must be 0.0 for A' to M-invariantize the full spine (accept
                # depends on every spine node, not just the deepest).
                _spine_all = 0.0
                for _d in range(len(spine_rows)):
                    _spine_all = max(_spine_all, float((
                        _out_pi[_d].reshape(-1) - out_m5[_d].reshape(-1)
                    ).abs().max().item()))
                reorder_results["spine_all_vs_m6_max"] = _spine_all
                reorder_results["pi"] = list(_pi)
                # ---- BRANCH proper-fix check (user requirement: BOTH spine AND
                # branch). A branch b's ancestors are a SUBSET of the spine, so
                # non-ancestor spine nodes interleave b's columns even after the
                # reorder -> the reorder does NOT auto-M-invariantize branches.
                # The proper-fix gate for branches is TWO measurements:
                #  (1) coresident_vs_solo: b in full reordered cat8 vs b in a
                #      (spine + b ONLY) reordered tree. 0 => b is invariant to
                #      co-resident branches (the deliverable-critical property:
                #      branch rescues don't wobble with the branch set). ~1 ULP =>
                #      b's OWN suffix column still shifts (within-floor; does NOT
                #      touch the spine accept). Gross => branches NOT properly
                #      fixed -> need fixed-slot branch layout too.
                #  (2) ctx_preserved: b's output vs its context-only path (drop
                #      the tree suffix entirely) is UNTOUCHED by the reorder =>
                #      the fix never perturbs the branch context (the A1/hybrid
                #      failure we just refuted on CPU).
                _bc = []
                for _b in _branch:
                    _anc_b = sorted([_j for _j in range(m_full)
                                     if float(bias_full[_b][_j].item()) == 0.0])
                    _sb_pi = list(spine_rows) + [_b]  # spine-first, single branch
                    _sbt = torch.tensor(_sb_pi, dtype=torch.long)
                    _bias_sb = bias_full.index_select(0, _sbt).index_select(
                        1, _sbt).contiguous()
                    _out_sb = _call(_sb_pi, _sb_pi, _bias_sb)
                    _b_solo = _out_sb[_sb_pi.index(_b)].reshape(-1)
                    _b_full = _out_pi[_pi.index(_b)].reshape(-1)
                    _bc.append({
                        "branch": int(_b),
                        "ancestors": _anc_b,
                        "coresident_vs_solo_max": float(
                            (_b_full - _b_solo).abs().max().item()),
                    })
                reorder_results["branch_check"] = _bc
                reorder_results["branch_coresident_max"] = max(
                    (c["coresident_vs_solo_max"] for c in _bc), default=0.0)
            except Exception as _rre:
                reorder_results["err"] = repr(_rre)
        # FR13_SLOT_REORDER in-process gates (FR13_FA2_MAB_CAUSAL=1). Cross-boot
        # byte-identity is INVALID on GB10 (autotune floor) so both S1 claims are
        # proven same-boot on the SAME captured operands:
        #  CAUSAL arm (S1): causal=False vs the causal=True M9 baseline must be
        #    int-exact 0.0 -- causal is redundant for the decode tree call (all
        #    context cols precede all tree rows; BFS ancestry bias is already
        #    -inf at every col causal would hit).
        #  KPERM arm (live-fix semantics): BFS query rows + KEY-ONLY spine-first
        #    permutation + bias[:, pi] + causal=False == the slot-reorder layout.
        #    kperm_spine_all_vs_m6_max == 0.0 => the DELIVERED fix (not the q+k
        #    relabel above) M-invariantizes the whole spine.
        causal_results = {}
        if os.environ.get("FR13_FA2_MAB_CAUSAL", "0") == "1":
            try:
                _out_nc = _call(full_rows, full_rows, bias_full, causal=False)
                causal_results["causal_off_vs_m9_max"] = float(
                    (_out_nc - out_m9).abs().max().item()
                )
                _branch_c = [r for r in range(m_full) if r not in spine_rows]
                _pi_c = list(spine_rows) + _branch_c
                _pit_c = torch.tensor(_pi_c, dtype=torch.long)
                _bias_kp = bias_full.index_select(1, _pit_c).contiguous()
                _out_kp = _call(full_rows, _pi_c, _bias_kp, causal=False)
                # rows stay BFS: row j is node j in BOTH arms -> direct compares
                _kp_spine_all = 0.0
                for _d, _sr in enumerate(spine_rows):
                    _kp_spine_all = max(_kp_spine_all, float((
                        _out_kp[_sr].reshape(-1) - out_m5[_d].reshape(-1)
                    ).abs().max().item()))
                causal_results["kperm_spine_all_vs_m6_max"] = _kp_spine_all
                causal_results["kperm_deep_vs_m6"] = float((
                    _out_kp[deep_row].reshape(-1)
                    - out_m5[spine_rows.index(deep_row)].reshape(-1)
                ).abs().max().item())
                # branch rows vs the BFS M9 baseline (column move only; expect
                # ~1 ULP lateral, NOT gross; gross => causal/bias wiring bug)
                _kp_branch_max = 0.0
                for _b in _branch_c:
                    _kp_branch_max = max(_kp_branch_max, float((
                        _out_kp[_b].reshape(-1) - out_m9[_b].reshape(-1)
                    ).abs().max().item()))
                causal_results["kperm_branch_vs_m9_max"] = _kp_branch_max
                causal_results["kperm_pi"] = list(_pi_c)
            except Exception as _sce:
                causal_results["err"] = repr(_sce)
        # Per-spine-depth RAW (M=9 spine rows vs M=5 spine rows) for context.
        by_depth = []
        for depth, sr in enumerate(spine_rows):
            a = out_m9[sr].reshape(-1)
            b = out_m5[depth].reshape(-1)
            by_depth.append(
                {
                    "depth": int(depth),
                    "m9_row": int(sr),
                    "m5_row": int(depth),
                    "max_abs": float((a - b).abs().max().item()),
                    "mean_abs": float((a - b).abs().mean().item()),
                }
            )
        rec = {
            "schema": "fr13.fa2_mab.v1",
            "layer_name": layer_name,
            "capture_call_index": int(capture_call_index),
            "capture_saved_index": int(capture_saved_index),
            "m_full": int(m_full),
            "context_len": int(context_len),
            "seq_len": int(seq_len),
            "spine_rows": list(spine_rows),
            "deep_row": int(deep_row),
            "bias_min": bmin,
            "bias_max": bmax,
            "scale": scale,
            "softcap": softcap,
            "deep_spine_raw_max_abs": raw_max_abs,
            "deep_spine_raw_mean_abs": raw_mean_abs,
            "recall_m9_vs_served_deep_max_abs": recall_vs_served,
            "qpad_deep_raw_max_abs": qpad_results,
            "qpad_self_m9_vs_unpadded": qpad_self,
            "kvpad_deep_raw_max_abs": kvpad_results,
            "kvpad_self_m9_vs_unpadded": kvpad_self,
            "reorder_a_prime": reorder_results,
            "slot_reorder_causal": causal_results,
            "by_spine_depth": by_depth,
        }
        out_path = Path(dump)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        logger.warning(
            "FR13_FA2_MAB layer=%s call=%s M%d-vs-M%d deep-spine(row%d) "
            "RAW max_abs=%.3e (recall_full_vs_served=%.3e)",
            layer_name,
            capture_call_index,
            m_full,
            len(spine_rows),
            deep_row,
            raw_max_abs,
            recall_vs_served,
        )
    except Exception as exc:  # pragma: no cover - diagnostic only
        logger.warning("FR13_FA2_MAB A/B re-call failed: %s", exc)


# FR13_TREE_ATTN_OP_CAPTURE
def _fr13_tree_attn_op_capture(
    impl,
    layer,
    query,
    key,
    value,
    output,
    key_cache,
    value_cache,
    attn_metadata,
):
    # Fixed32 accounting is part of the acceptance route, not the optional
    # capture diagnostic below. Keep it outside that diagnostic's fail-open
    # try/except and execute it only after unified_attention returned.
    layer_name = str(getattr(layer, "layer_name", ""))
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _fr13_f32_gdn
    if getattr(_fr13_f32_gdn, "_FR13_FIXED32_MODE", ""):
        _fr13_f32_capturing = bool(
            torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing()
        )
        if (
            _fr13_f32_capturing
            or _fr13_f32_gdn._fr13_fixed32_observed_event_active()
        ):
            _fr13_f32_bias = getattr(attn_metadata, "tree_attn_bias", None)
            if _fr13_f32_bias is None:
                raise RuntimeError("FR13 fixed32 tree attention has no bias tensor")
            _fr13_f32_gdn._fr13_fixed32_observed_tree_attn(
                layer_name,
                int(query.shape[0]),
                tuple(int(value) for value in _fr13_f32_bias.shape),
                _fr13_f32_capturing,
            )
    path = os.environ.get("FR13_TREE_ATTN_OP_CAPTURE")
    # FR13_FA2_MAB shares this hook's dense-KV gather + carrier-event gating but
    # is an independent diagnostic: it runs even when the op-capture SAVE path
    # is unset.  Both default OFF; when neither is set the hook is a no-op and
    # the locked cat9 default path is byte-identical.
    mab_on = os.environ.get("FR13_FA2_MAB", "0") == "1"
    if not path and not mab_on:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return
    except Exception:
        return
    try:
        # The A/B can sweep all 16 full_attention layers via its own LAYER
        # filter (default "*"); the save path keeps its own layer default.
        if mab_on and not path:
            want = os.environ.get("FR13_FA2_MAB_LAYER", "*")
        else:
            want = os.environ.get(
                "FR13_TREE_ATTN_OP_CAPTURE_LAYER",
                "language_model.model.layers.3.self_attn",
            )
        if want and want != "*" and not layer_name.startswith(want):
            return
        seen = int(globals().get("_FR13_TREE_ATTN_OP_CAPTURE_SEEN", 0))
        skip = int(os.environ.get("FR13_TREE_ATTN_OP_CAPTURE_SKIP", "0"))
        limit = int(os.environ.get("FR13_TREE_ATTN_OP_CAPTURE_LIMIT", "1"))
        saved = int(globals().get("_FR13_TREE_ATTN_OP_CAPTURE_SAVED", 0))
        # FR13_FA2_MAB carrier-event gating: an independent per-(layer) counter
        # so the A/B fires on the FR13_FA2_MAB_SKIP-th tree-verify forward (the
        # deep-accept carrier event) and at most FR13_FA2_MAB_LIMIT times per
        # layer.  Keyed by layer_name so a "*" sweep gates each of the 16
        # full-attn layers at the SAME event ordinal.
        if mab_on:
            mab_skip = int(os.environ.get("FR13_FA2_MAB_SKIP", "0"))
            mab_limit = int(os.environ.get("FR13_FA2_MAB_LIMIT", "1"))
            mab_counts = globals().setdefault("_FR13_FA2_MAB_COUNTS", {})
            mab_seen = int(mab_counts.get(layer_name, 0))
            mab_counts[layer_name] = mab_seen + 1
            run_mab = mab_skip <= mab_seen < (mab_skip + mab_limit)
        else:
            run_mab = False
        globals()["_FR13_TREE_ATTN_OP_CAPTURE_SEEN"] = seen + 1
        save_now = bool(path) and not (seen < skip or saved >= limit)
        if not save_now and not run_mab:
            return

        block_size = int(key_cache.shape[1])
        q_start = attn_metadata.query_start_loc.detach().to(torch.long).cpu()
        seq_lens = attn_metadata.seq_lens.detach().to(torch.long).cpu()
        block_table_cpu = attn_metadata.block_table.detach().to(torch.long).cpu()
        dense_k = []
        dense_v = []
        used_blocks = []
        for seq_idx in range(int(seq_lens.numel())):
            seq_len = int(seq_lens[seq_idx].item())
            n_blocks = (seq_len + block_size - 1) // block_size
            seq_k_parts = []
            seq_v_parts = []
            seq_blocks = []
            remaining = seq_len
            for local_block_idx in range(n_blocks):
                block_id = int(block_table_cpu[seq_idx, local_block_idx].item())
                take = min(block_size, remaining)
                if block_id >= 0 and take > 0:
                    seq_blocks.append(block_id)
                    seq_k_parts.append(
                        key_cache[block_id, :take].detach().to(torch.float32).cpu()
                    )
                    seq_v_parts.append(
                        value_cache[block_id, :take].detach().to(torch.float32).cpu()
                    )
                remaining -= take
            dense_k.append(torch.cat(seq_k_parts, dim=0) if seq_k_parts else torch.empty(0))
            dense_v.append(torch.cat(seq_v_parts, dim=0) if seq_v_parts else torch.empty(0))
            used_blocks.append(seq_blocks)

        root, ext = os.path.splitext(path)
        call_path = root + ".call" + str(saved) + (ext or ".pt")
        payload = {
            "schema": "fr13.tree_attn_op_capture.v1",
            "source": "TreeAttentionImpl.forward",
            "path": path,
            "call_path": call_path,
            "layer_name": layer_name,
            "capture_call_index": int(seen),
            "capture_saved_index": int(saved),
            "scale": float(getattr(impl, "scale", 0.0)),
            "num_heads": int(getattr(impl, "num_heads", 0)),
            "num_kv_heads": int(getattr(impl, "num_kv_heads", 0)),
            "num_queries_per_kv": int(getattr(impl, "num_queries_per_kv", 0)),
            "head_size": int(getattr(impl, "head_size", 0)),
            "sliding_window": tuple(int(x) for x in getattr(impl, "sliding_window", (-1, -1))),
            "logits_soft_cap": float(getattr(impl, "logits_soft_cap", 0.0)),
            "query_start_loc": q_start,
            "seq_lens": seq_lens,
            "block_table": block_table_cpu,
            "used_blocks": used_blocks,
            "tree_attn_bias": (
                attn_metadata.tree_attn_bias.detach().to(torch.float32).cpu()
                if attn_metadata.tree_attn_bias is not None
                else None
            ),
            "query": query.detach().to(torch.float32).cpu(),
            "key_input": key.detach().to(torch.float32).cpu()
            if key is not None
            else None,
            "value_input": value.detach().to(torch.float32).cpu()
            if value is not None
            else None,
            "output": output.detach().to(torch.float32).cpu(),
            "dense_key": dense_k,
            "dense_value": dense_v,
        }
        if save_now:
            out = Path(call_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, out)
            if saved == 0:
                torch.save(payload, Path(path))
            globals()["_FR13_TREE_ATTN_OP_CAPTURE_SAVED"] = saved + 1
        if run_mab:
            _fr13_fa2_mab_recall(
                impl,
                layer_name,
                int(seen),
                int(saved),
                query,
                dense_k,
                dense_v,
                seq_lens,
                payload["tree_attn_bias"],
                output.detach().to(torch.float32).cpu(),
                spine_nodes=getattr(attn_metadata, "fr10_tree_path0_nodes", None),
            )
    except Exception as exc:
        logger.warning("FR13 tree attention op capture failed: %s", exc)




class TreeAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "TREE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TreeAttentionImpl"]:
        return TreeAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["TreeAttentionMetadataBuilder"]:
        return TreeAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class TreeAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    tree_attn_bias: torch.Tensor | None = None
    max_prefill_query_len: int = 0
    max_prefill_seq_len: int = 0
    max_decode_query_len: int = 0
    max_decode_seq_len: int = 0

    # Cached Prefill/decode metadata.
    _cached_prefill_metadata: "TreeAttentionMetadata | None" = None
    _cached_decode_metadata: "TreeAttentionMetadata | None" = None

    @property
    def prefill_metadata(self) -> "TreeAttentionMetadata | None":
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        q_start_loc = self.query_start_loc[self.num_decodes :]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[self.num_decodes :]
        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_prefill_tokens,
            max_query_len=self.max_prefill_query_len,
            query_start_loc=q_start_loc - q_start_loc[0],
            max_seq_len=self.max_prefill_seq_len,
            seq_lens=kv_seqlens,
            block_table=self.block_table[self.num_decodes :],
            slot_mapping=self.slot_mapping[self.num_decode_tokens :],
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> "TreeAttentionMetadata | None":
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata

        q_start_loc = self.query_start_loc[: self.num_decodes + 1]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[: self.num_decodes]
        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            max_query_len=self.max_decode_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=self.max_decode_seq_len,
            seq_lens=kv_seqlens,
            block_table=self.block_table[: self.num_decodes],
            slot_mapping=self.slot_mapping[: self.num_decode_tokens],
            tree_attn_bias=self.tree_attn_bias,
        )
        return self._cached_decode_metadata


class TreeAttentionMetadataBuilder(AttentionMetadataBuilder[TreeAttentionMetadata]):
    # FR13_TREE_ATTN_CUDAGRAPH_METADATA: spec decode is uniform tree-token batches.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        spec_config = vllm_config.speculative_config
        # FR10_SPEC_CONFIG_TREE_OVERRIDE: keep attention tree identical to the FR10 launch descriptor.
        spec_token_tree: str | None = None
        try:
            spec_env = os.environ.get("SPEC_CONFIG")
            if spec_env:
                spec_token_tree = json.loads(spec_env).get("speculative_token_tree")
        except Exception:
            spec_token_tree = None
        if spec_token_tree is None and (spec := spec_config):
            spec_token_tree = spec.speculative_token_tree
        tree_choices: list[tuple[int, ...]] = (
            sorted(ast.literal_eval(spec_token_tree), key=lambda _p: (len(_p), _p)) if spec_token_tree is not None else [(0,)]
        )
        # Construct the tree attention bias.
        depth_counts = _get_depth_counts(tree_choices)
        self.tree_attn_bias = _prepare_tree_attn_bias(
            tree_choices,
            depth_counts,
            dtype=torch.float32,
            device=device,
        )

        self.reorder_batch_threshold = self.tree_attn_bias.shape[0]

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TreeAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TreeAttentionMetadata:
        decode_threshold = self.tree_attn_bias.shape[0]
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=decode_threshold
            )
        )

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = common_attn_metadata.max_seq_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # FR13: cached prefill/decode metadata is built during the
        # Python metadata phase.  Do not derive these values in
        # TreeAttentionImpl.forward(), where GPU tensor .item() syncs
        # break CUDA graph capture.
        if num_decodes == 0:
            max_decode_query_len = 0
            max_decode_seq_len = 0
        elif num_prefills == 0:
            max_decode_query_len = max_query_len
            max_decode_seq_len = max_seq_len
        else:
            max_decode_query_len = min(max_query_len, decode_threshold)
            max_decode_seq_len = max_seq_len
        max_prefill_query_len = max_query_len if num_prefills else 0
        max_prefill_seq_len = max_seq_len if num_prefills else 0

        return TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            tree_attn_bias=self.tree_attn_bias,
            max_prefill_query_len=max_prefill_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_query_len=max_decode_query_len,
            max_decode_seq_len=max_decode_seq_len,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> TreeAttentionMetadata:
        # Cache the original tree attention bias.
        orig_tree_attn_bias = self.tree_attn_bias

        if draft_index == 0:
            # Use prefill for drafting at the root level.
            self.tree_attn_bias = torch.empty(0)
        else:
            # Slice the tree attention bias for drafting. Exclude
            # the root level.
            start, end = 1, 1 + common_attn_metadata.max_query_len
            self.tree_attn_bias = self.tree_attn_bias[start:end, start:end].contiguous()

        # Build attention bias.
        attn_metadata = self.build(0, common_attn_metadata, fast_build=True)

        # Reset the tree attention bias to the original value.
        self.tree_attn_bias = orig_tree_attn_bias
        return attn_metadata


def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    # Count the number of choices at each depth of the tree.
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _prepare_tree_attn_bias(
    sorted_tree_choices: list[tuple[int, ...]],
    depth_counts: list[int],
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> torch.Tensor:
    # +1 comes from the additional root node.
    tree_len = len(sorted_tree_choices) + 1
    tree_attn_mask = torch.full(
        (tree_len, tree_len), -torch.inf, device=device, dtype=dtype
    )

    # Set diagonal to all zeros. Each token should
    # attend to itself.
    mask_val = 0
    for i in range(tree_len):
        tree_attn_mask[i, i] = mask_val

    # Set root to all zeros. All tokens attend to it.
    tree_attn_mask[:, 0] = mask_val

    # Set all ancestors to zeros.
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # Retrieve ancestor position.
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(
                    sorted_tree_choices.index(cur_tree_choice[: c + 1]) + 1
                )
            tree_attn_mask[j + start + 1, ancestor_idx] = mask_val
        start += depth_counts[i]
    # FR10_ROOT_ATTENTION_BIAS_CAPTURE: dump the runtime root/bonus attention bias row.
    try:
        _fr10_mask_path = os.environ.get("FR10_ROOT_HIDDEN_CAPTURE")
        if _fr10_mask_path and not os.path.exists(_fr10_mask_path + ".tree_attn_bias.pt"):
            torch.save(
                {
                    "root_row": tree_attn_mask[0].detach().cpu(),
                    "full_bias": tree_attn_mask.detach().cpu(),
                    "sorted_tree_choices": [tuple(_p) for _p in sorted_tree_choices],
                    "depth_counts": [int(_x) for _x in depth_counts],
                },
                _fr10_mask_path + ".tree_attn_bias.pt",
            )
    except Exception:
        pass
    return tree_attn_mask


class TreeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if logits_soft_cap is None:
            # Setting logits_soft_cap to 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TreeAttentionImpl."
            )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TreeAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with TreeAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for TreeAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        key_cache, value_cache = kv_cache.unbind(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, key.shape[1])
        if prefill_meta := attn_metadata.prefill_metadata:
            unified_attention(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        if decode_meta := attn_metadata.decode_metadata:
            unified_attention(
                q=query[:num_decode_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_decode_tokens],
                cu_seqlens_q=decode_meta.query_start_loc,
                max_seqlen_q=decode_meta.max_query_len,
                seqused_k=decode_meta.seq_lens,
                max_seqlen_k=decode_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                qq_bias=decode_meta.tree_attn_bias,
                window_size=self.sliding_window,
                block_table=decode_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )
            _fr13_tree_attn_op_capture(
                self,
                layer,
                query[:num_decode_tokens],
                key[:num_decode_tokens] if key is not None else None,
                value[:num_decode_tokens] if value is not None else None,
                output[:num_decode_tokens],
                key_cache,
                value_cache,
                decode_meta,
            )
        return output
