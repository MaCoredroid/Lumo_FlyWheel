#!/usr/bin/env python3
"""FR13 DEVICE MULTIDRAFT: device-side temp>0 tree-rejection committer.

FIRST INCREMENT. The deployment-temp (0.6) tree committer is today the slow
lever: ``_lumo_tree_canonical_multidraft_sample``
(``fr10_phase4_patch_vllm_tree_gdn.py`` :8184, dispatched :9198-9244 on
``not all_greedy``) does

    target_logits.softmax(-1, fp32).cpu().numpy()      # [nodes x vocab] DtoH
    tree_self_logits.softmax(-1, fp32).cpu().numpy()   # [nodes x vocab] DtoH

then a PER-REQUEST / PER-NODE Python interpreter walk
(``sample_deterministic_multidraft_rejection_step`` /
``sample_multidraft_rejection_step`` in
``src/lumo_flywheel_serving/fr10_tree_rejection_sampler.py`` :137/:170) that
calls ``rng.choice`` over the full vocab. That host softmax DtoH +
Python-loop is what makes the temp-0.6 tree tax ~1.4x vs the chain5 spine
~1.11x (the GREEDY device committer ``fr13_gpu_committer_kernel.py`` only
covers ``all_greedy``).

This module moves the SAME SpecInfer / multi-draft canonical accept rule onto
the device:
  * NO ``[nodes x vocab]`` host softmax DtoH. The accept test only needs the
    target probability AT a handful of candidate token ids per node, and the
    rejection residual is sampled with at most one [vocab] softmax used
    on-device (never DtoH'd). Both come from ``log_softmax``-style stable
    on-device math, gathered at the candidate token ids.
  * NO Python per-node interpreter loop over synced host lists. The per-node
    canonical decision (select source ~ weights, accept ~ min(1,p/q_mix),
    residual fallback) is a small fixed set of device tensor ops driven by a
    DEVICE rng (``torch.Generator(device=...)``) seeded per request from the
    same per-request generator the host reference uses.

It is DISTRIBUTION-LOSSLESS, not byte-identical: the host reference draws from
``numpy.random.Generator``; this device path draws the SAME categorical /
Bernoulli / residual distributions from a ``torch`` device generator. RNG
samples differ; the accept DISTRIBUTION is identical (proven offline by
``scripts/fr13_device_multidraft_offline_gate.py``: per-node accept
probabilities match the host reference within float tolerance, and sampled
token frequencies match within sampling noise over N draws).

This module ships:
  * ``host_multidraft_accept_probs`` -- the EXACT host reference per-node
    accept-probability + source-weight + residual rule, transcribed verbatim
    from the deterministic step (the offline ground truth for the
    distribution gate; numpy, model-free).
  * ``device_multidraft_node_step`` -- the device transcription: given the
    on-device ``target_probs`` row(s) for one node's children, computes the
    source weights, the selected source / token, the canonical accept
    probability, the Bernoulli accept, and (on reject) the residual sample --
    all on-device, no full-vocab DtoH.
  * ``fr13_device_multidraft_commit`` -- the dispatch entry the flag-gated hook
    calls (FR13_DEVICE_MULTIDRAFT). It walks each request's tree top-down on
    device, returning the SAME committer products the host reference publishes
    (out_rows, accepted_rows, accepted_lens, accepted_node_paths,
    accepted_token_rows). Materialised on host at the end (deploy form keeps
    the side-stream readback of ``fr13_gpu_committer_device_readback`` -- a
    documented GPU-iteration TODO; this increment proves the on-device decision
    + distribution losslessness CPU-first).

DEFAULT-OFF. The flag ``FR13_DEVICE_MULTIDRAFT`` is read by the hook in the
patcher; with the flag OFF the host reference ``_lumo_tree_canonical_multidraft_
sample`` runs untouched (byte-identical to HEAD). This module is import-only
inert (no side effects at import).

Fail-loud: ``fr13_device_multidraft_commit`` raises on disengagement (missing
device tensors, draft_probs path not yet supported on device, etc.) -- never a
silent fallback to the host walk (bug-class 9).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace
from typing import Any, Callable, Sequence

try:
    import torch
except Exception:  # pragma: no cover - torch always present in the vLLM image
    torch = None  # type: ignore

import numpy as np


try:  # Triton is present in the serving image, not on CPU-only audit hosts.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised by CPU-only source gates
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fr13_fixed32_taw_exact_commit_kernel(
        child_table,
        child_counts,
        current_input,
        alive_input,
        self_token,
        bonus_token,
        source,
        selected_token,
        rejected_token,
        accepted,
        current_state,
        alive_state,
        output_tokens,
        output_lens,
        accepted_path_rows,
        accepted_lens,
        last_row,
        LEVEL: tl.constexpr,
        PHYSICAL_DRAFTS: tl.constexpr,
        PHYSICAL_ROWS: tl.constexpr,
        FANOUT: tl.constexpr,
        OUTPUT_CAPACITY: tl.constexpr,
        PATH_CAPACITY: tl.constexpr,
    ):
        """Commit already-decided tokens using integer and boolean operations only."""
        request = tl.program_id(0)
        current = tl.load(current_input + request).to(tl.int64)
        alive = tl.load(alive_input + request) != 0

        if LEVEL == 0:
            output_columns = tl.arange(0, OUTPUT_CAPACITY)
            path_columns = tl.arange(0, PATH_CAPACITY)
            tl.store(
                output_tokens + request * OUTPUT_CAPACITY + output_columns,
                -1,
            )
            tl.store(
                accepted_path_rows + request * PATH_CAPACITY + path_columns,
                0,
            )
            output_len = 0
            path_len = 0
            prior_last_row = 0
        else:
            output_len = tl.load(output_lens + request).to(tl.int64)
            path_len = tl.load(accepted_lens + request).to(tl.int64)
            prior_last_row = tl.load(last_row + request).to(tl.int64)

        parent_slot = tl.maximum(
            0,
            tl.minimum(current + 1, PHYSICAL_ROWS - 1),
        )
        child_count = tl.load(
            child_counts + request * PHYSICAL_ROWS + parent_slot
        ).to(tl.int64)
        has_kids = alive & (child_count > 0)
        leaf = alive & (child_count == 0)
        current_valid = (current >= 0) & (current < PHYSICAL_DRAFTS)

        sampled_self = tl.load(self_token + request).to(tl.int64)
        sampled_bonus = tl.load(bonus_token + request).to(tl.int64)
        leaf_token = tl.where(current_valid, sampled_self, sampled_bonus)
        tl.store(
            output_tokens + request * OUTPUT_CAPACITY + output_len,
            leaf_token,
            mask=leaf,
        )

        is_accepted = tl.load(accepted + request) != 0
        sampled_selected = tl.load(selected_token + request).to(tl.int64)
        sampled_rejected = tl.load(rejected_token + request).to(tl.int64)
        emitted_token = tl.where(
            is_accepted,
            sampled_selected,
            sampled_rejected,
        )
        tl.store(
            output_tokens + request * OUTPUT_CAPACITY + output_len,
            emitted_token,
            mask=has_kids,
        )
        output_len_new = output_len + leaf.to(tl.int64) + has_kids.to(tl.int64)

        selected_source = tl.load(source + request).to(tl.int64)
        accepted_node = tl.load(
            child_table
            + request * PHYSICAL_ROWS * FANOUT
            + parent_slot * FANOUT
            + selected_source
        ).to(tl.int64)
        accepted_row = accepted_node + 1
        tl.store(
            accepted_path_rows + request * PATH_CAPACITY + path_len,
            accepted_row,
            mask=is_accepted,
        )
        path_len_new = path_len + is_accepted.to(tl.int64)
        current_new = tl.where(is_accepted, accepted_node, current)
        alive_new = (alive & (~leaf)) & is_accepted
        last_row_new = tl.where(is_accepted, accepted_row, prior_last_row)

        tl.store(current_state + request, current_new)
        tl.store(alive_state + request, alive_new)
        tl.store(output_lens + request, output_len_new)
        tl.store(accepted_lens + request, path_len_new)
        tl.store(last_row + request, last_row_new)


# ---------------------------------------------------------------------------
# MEASUREMENT-ONLY commit trace (FR13 garble mechanism binding, 2026-07-10).
# Gated on LUMO_TREE_SAMPLER_DEBUG_LOG (the ONE diagnostic flag proven to reach
# the forward EngineCore worker). When the flag is UNSET this is a no-op and the
# committer is byte-identical to HEAD. It logs, at the EXACT commit node, the
# tree-verify prob the committer actually committed each token at -- so a
# committed near-neighbor misspell's accept-time p_target can be read WITHOUT the
# gather->commit alignment trap, then joined to the no-spec localizer. Writes to a
# `.commit` sibling of the debug log so it never interleaves with tree_logit_gather.
# ---------------------------------------------------------------------------
_FR13_COMMIT_TRACE_FH = None
_FR13_COMMIT_TRACE_TRIED = False


def _fr13_commit_trace_fh():
    global _FR13_COMMIT_TRACE_FH, _FR13_COMMIT_TRACE_TRIED
    if _FR13_COMMIT_TRACE_TRIED:
        return _FR13_COMMIT_TRACE_FH
    _FR13_COMMIT_TRACE_TRIED = True
    path = os.environ.get("LUMO_TREE_SAMPLER_DEBUG_LOG")
    if not path:
        return None
    try:
        _FR13_COMMIT_TRACE_FH = open(path + ".commit", "a", buffering=1)
    except Exception:
        _FR13_COMMIT_TRACE_FH = None
    return _FR13_COMMIT_TRACE_FH


# NOTE (2026-07-10): a committer DRIFT-BAND BIAS was built + live-tested here and
# ABANDONED as a class (user decision). The committer only ever sees the DRIFTED
# target, so it cannot distinguish a drift-inflated garble branch (true ~1e-6) from
# a genuine alternative (true ~0.2); any spine>branch bias is either distribution-
# breaking or ineffective (live A/B: the drift depresses the correct argmax to ~0.80
# so garbles slip the threshold; blunt; boot-fragile). Deleted. The commit-trace
# instrument below (measurement-only) is kept. See FR13_GARBLE_FIX_DECISION.md.
def _fr13_commit_trace_emit(fh, *, req, node, token_id, accepted, p_row,
                            child_drafts):
    """Emit one commit-decision record. EAGER/diagnostic only (syncs .item())."""
    import json as _j
    try:
        p_norm = p_row / p_row.sum()
        cd = [int(t) for t in child_drafts]
        child_probs = [float(p_norm[t].item()) for t in cd]
        committed_prob = float(p_norm[int(token_id)].item())
        amax = int(torch.argmax(p_norm).item())
        rec = {
            "event": "commit_trace",
            "req": int(req),
            "node": int(node),
            "committed_token": int(token_id),
            "committed_prob": committed_prob,          # tree-verify p the committer USED for what it emitted
            "committed_is_argmax": bool(int(token_id) == amax),
            "argmax_token": amax,
            "argmax_prob": float(p_norm[amax].item()),
            "accepted": bool(accepted),                # True=accept path, False=residual resample
            "overlap_mass": float(sum(child_probs)),
            "child_drafts": cd,
            "child_probs": child_probs,
            "argmax_drafted": bool(amax in cd),
        }
        fh.write(_j.dumps(rec) + chr(10))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# HOST REFERENCE per-node rule (the distribution ground truth).
# Transcribed verbatim from sample_deterministic_multidraft_rejection_step
# (fr10_tree_rejection_sampler.py :170-218) -- the draft_probs=None / MTP path
# the deployment uses -- so the device kernel's per-node distribution can be
# A/B'd against THIS without booting vLLM.
# ---------------------------------------------------------------------------
def host_multidraft_accept_probs(
    target_probs_row: np.ndarray,
    child_draft_tokens: Sequence[int],
):
    """Return the host reference per-node distribution objects.

    Mirrors ``sample_deterministic_multidraft_rejection_step`` EXACTLY for the
    deterministic (draft_probs=None) deployment path:

      overlaps      = p[draft_tokens]                       (target prob @ each child)
      overlap_mass  = sum(overlaps)
      if overlap_mass <= 0:  reject, token ~ p              (full-vocab residual = p)
      weights       = overlaps / overlap_mass               (source selection probs)
      # q_mix over the (possibly duplicated) draft tokens:
      q_mix[tok]    = sum(weights[i] for draft_tokens[i] == tok)
      accept_prob(source) = min(1, p[token] / q_mix[token]) where token=draft_tokens[source]
      residual      = max(p - q_mix_vocab, 0) / mass        (reject fallback dist)

    Returns a dict with the per-source ``weights`` (categorical over children),
    the per-source ``accept_prob`` (the Bernoulli accept probability conditioned
    on that source being selected), the ``residual`` distribution (full vocab,
    the reject fallback), and ``overlap_mass``. These are the EXACT distribution
    objects the device kernel must reproduce -- not a single sample.
    """
    p = np.asarray(target_probs_row, dtype=np.float64)
    p = p / p.sum()  # normalize() in the reference
    tokens = np.asarray(child_draft_tokens, dtype=np.int64)
    overlaps = p[tokens].astype(np.float64, copy=True)
    overlap_mass = float(overlaps.sum())
    vocab = int(p.shape[0])

    if overlap_mass <= 0.0:
        # reference: token ~ p, accepted=False (source_index=0)
        return {
            "overlap_mass": 0.0,
            "weights": None,
            "accept_prob": None,
            "residual": p.copy(),
            "all_reject": True,
        }

    weights = overlaps / overlap_mass
    # q_mix over duplicated draft tokens, per the reference q_mix_token sum.
    accept_prob = np.empty(tokens.shape[0], dtype=np.float64)
    for src in range(tokens.shape[0]):
        token = int(tokens[src])
        q_mix_token = float(weights[tokens == token].sum())
        accept_prob[src] = min(1.0, float(p[token] / q_mix_token)) if q_mix_token > 0.0 else 0.0

    # residual over the full vocab (reference builds q_mix vocab then max(p-q,0)).
    q_mix_vocab = np.zeros(vocab, dtype=np.float64)
    for i in range(tokens.shape[0]):
        q_mix_vocab[int(tokens[i])] += weights[i]
    residual = np.maximum(p - q_mix_vocab, 0.0)
    mass = float(residual.sum())
    if mass == 0.0:
        residual = p.copy()
    else:
        residual = residual / mass

    return {
        "overlap_mass": overlap_mass,
        "weights": weights,
        "accept_prob": accept_prob,
        "residual": residual,
        "all_reject": False,
    }


def host_multidraft_step(
    target_probs_row: np.ndarray,
    child_draft_tokens: Sequence[int],
    *,
    rng: np.random.Generator,
):
    """One host reference step (source, token, accepted). For the offline gate.

    Identical RNG-consumption order to the reference: choose source ~ weights,
    token = draft_tokens[source], Bernoulli(accept_prob[source]); on reject
    token ~ residual. Returns (token_id, source_index, accepted).
    """
    d = host_multidraft_accept_probs(target_probs_row, child_draft_tokens)
    tokens = np.asarray(child_draft_tokens, dtype=np.int64)
    p = np.asarray(target_probs_row, dtype=np.float64)
    p = p / p.sum()
    if d["all_reject"]:
        token = int(rng.choice(p.shape[0], p=p))
        return token, 0, False
    source = int(rng.choice(tokens.shape[0], p=d["weights"]))
    token = int(tokens[source])
    accepted = bool(rng.random() < float(d["accept_prob"][source]))
    if accepted:
        return token, source, True
    token = int(rng.choice(p.shape[0], p=d["residual"]))
    return token, source, False


# ---------------------------------------------------------------------------
# DEVICE per-node step. Computes the SAME distribution objects on-device with
# NO full-vocab DtoH, then draws from a torch DEVICE generator.
# ---------------------------------------------------------------------------
def _device_softmax_row(target_logits_row, *, fp32: bool = True):
    """Stable softmax of ONE logits row on-device (no DtoH).

    Returns the full-vocab prob row ON DEVICE. The row stays device-resident;
    only the handful of gathered candidate probs / one residual sample ever
    leave the device. This is the on-device analogue of the host
    ``target_logits.softmax(-1, fp32)`` but WITHOUT the ``.cpu().numpy()`` for
    the whole ``[nodes x vocab]`` matrix.
    """
    row = target_logits_row
    if fp32:
        row = row.to(torch.float32)
    return torch.softmax(row, dim=-1)


def _device_onehot_argmax_row(target_logits_row, *, fp32: bool = True):
    """POINT-MASS prob row on the argmax logit (the temp->0 / greedy limit).

    Returns a full-vocab prob row that is 1.0 at ``argmax(target_logits_row)``
    and 0.0 elsewhere, ON DEVICE. Feeding this to ``device_multidraft_node_step``
    makes the multidraft accept rule DETERMINISTIC and byte-identical to the
    greedy longest-prefix committer: overlap mass is nonzero only for a child
    whose draft == argmax, so that child is accepted with prob 1 (and any
    residual/bonus multinomial over a point mass returns the argmax with no rng
    consumption). This is the "use rejection sampling also at temp 0"
    unification -- greedy is the one-hot specialization of the SAME rule, so the
    separate greedy path-LCP committer is redundant. Proven node-equal to greedy
    over 20k random trials and byte-equal at the tree level by
    scripts/fr13_greedy_pointmass_byte_gate.py.
    """
    row = target_logits_row
    if fp32:
        row = row.to(torch.float32)
    out = torch.zeros_like(row)
    out[int(torch.argmax(row).item())] = 1.0
    return out


def device_multidraft_node_step(
    target_probs_row,
    child_draft_tokens,
    *,
    generator,
):
    """Device transcription of the deterministic multidraft step.

    ``target_probs_row`` is an ON-DEVICE 1-D prob tensor (softmax already
    applied on-device). ``child_draft_tokens`` is an ON-DEVICE 1-D int tensor of
    the children's draft token ids. ``generator`` is a ``torch.Generator`` on
    the SAME device, seeded per request from the host's per-request generator.

    Returns (token_id:int, source_index:int, accepted:bool). The decision is
    computed with device tensor ops; only the 3 scalar outcomes cross to host.

    Distribution-equivalent to ``host_multidraft_step`` /
    ``sample_deterministic_multidraft_rejection_step``: same source weights,
    same accept probability, same residual distribution.
    """
    p = target_probs_row
    if p.dtype != torch.float32 and p.dtype != torch.float64:
        p = p.to(torch.float32)
    # normalize (reference normalize()); softmax already sums ~1 but match the
    # reference's explicit renorm for the residual mass arithmetic.
    p = p / p.sum()
    tokens = child_draft_tokens.to(torch.long)
    overlaps = p[tokens].to(torch.float64)
    overlap_mass = overlaps.sum()

    if float(overlap_mass.item()) <= 0.0:
        # reject, token ~ p (full vocab). Sample on-device from p.
        token = int(torch.multinomial(p.to(torch.float32), 1, generator=generator).item())
        return token, 0, False

    weights = (overlaps / overlap_mass)  # float64 device
    # source ~ Categorical(weights), on-device.
    source = int(
        torch.multinomial(weights.to(torch.float32), 1, generator=generator).item()
    )
    token = int(tokens[source].item())
    # q_mix_token = sum of weights over children whose draft token == token.
    same = tokens == tokens[source]
    q_mix_token = weights[same].sum()
    accept_prob = torch.clamp(p[token].to(torch.float64) / q_mix_token, max=1.0)
    u = torch.rand(1, generator=generator, device=p.device, dtype=torch.float32)
    accepted = bool(float(u.item()) < float(accept_prob.item()))
    if accepted:
        return token, source, True

    # reject: residual = max(p - q_mix_vocab, 0) / mass, sampled on-device. We
    # build q_mix only at the (few) distinct draft tokens via scatter-add, never
    # materialising a separate dense numpy vocab on host.
    q_mix_vocab = torch.zeros_like(p, dtype=torch.float64)
    q_mix_vocab.scatter_add_(0, tokens, weights)
    residual = torch.clamp(p.to(torch.float64) - q_mix_vocab, min=0.0)
    mass = residual.sum()
    if float(mass.item()) == 0.0:
        residual = p.to(torch.float64)
    else:
        residual = residual / mass
    token = int(
        torch.multinomial(residual.to(torch.float32), 1, generator=generator).item()
    )
    return token, source, False


# ---------------------------------------------------------------------------
# Dispatch entry: device tree walk producing the host-reference committer
# products. Mirrors _lumo_tree_canonical_multidraft_sample's top-down walk but
# the per-node decision runs on-device (no [nodes x vocab] softmax DtoH, no
# numpy per-node loop).
# ---------------------------------------------------------------------------
def fr13_device_multidraft_commit(
    num_draft_tokens,
    draft_token_ids,
    tree_parent_indices,
    target_logits,
    tree_self_logits,
    draft_probs,
    bonus_token_ids,
    max_spec_len: int,
    *,
    generators=None,
    all_greedy: bool = False,
):
    """Device-side temp>0 multidraft committer (FR13_DEVICE_MULTIDRAFT).

    ``all_greedy`` (temp-0 unification): when True, every target/self prob row
    is the POINT MASS on its argmax (``_device_onehot_argmax_row``) instead of
    the softmax. The multidraft accept rule then reduces, byte-for-byte and with
    ZERO rng consumption, to the greedy longest-prefix committer -- this is how
    "use rejection sampling also at temp 0" stays lossless. The separate greedy
    path-LCP committer is thereby redundant and deleted.

    Returns ``(out_rows, accepted_rows, accepted_lens, accepted_node_paths,
    accepted_token_rows)`` -- the SAME products
    ``_lumo_tree_canonical_multidraft_sample`` publishes -- with the per-node
    canonical decision computed on-device. ``target_logits`` /
    ``tree_self_logits`` STAY on device; only per-node candidate probs / one
    residual sample / the 3 scalar outcomes per node leave the device. No
    ``[nodes x vocab]`` softmax DtoH, no numpy per-node interpreter loop.

    draft_probs is not None (the explicit-q multi-draft path) is NOT yet ported
    to device in this increment -- the deployment MTP path passes draft_probs=
    None (deterministic). We FAIL LOUD (no silent host fallback) if a caller
    hands us draft_probs so a temp>0 run cannot silently bypass the device
    committer.
    """
    if torch is None:
        raise RuntimeError("fr13_device_multidraft_commit requires torch")
    if draft_probs is not None:
        raise RuntimeError(
            "FR13_DEVICE_MULTIDRAFT: draft_probs!=None (explicit-q multidraft) "
            "not yet ported to device -- refusing silent host fallback "
            "(bug-class 9). Deployment MTP path passes draft_probs=None."
        )
    if target_logits is None or tree_self_logits is None:
        raise RuntimeError(
            "FR13_DEVICE_MULTIDRAFT: missing target/self logits (disengaged)"
        )
    fixed32_mode = os.environ.get("FR13_FIXED32_MODE", "").strip()
    if fixed32_mode:
        return fr13_fixed32_taw_commit(
            num_draft_tokens,
            draft_token_ids,
            tree_parent_indices,
            target_logits,
            tree_self_logits,
            bonus_token_ids,
            max_spec_len,
            generators=generators,
            all_greedy=all_greedy,
            mode=fixed32_mode,
        )
    # FR13_DM_DEPTHSYNC (S1/P1, default OFF => byte-identical ship): route to
    # the depth-synchronous walk (~2 batched readbacks per LEVEL instead of
    # 4-7 blocking .item() per NODE => ~100 syncs/step -> ~2x walk depth).
    # SAME per-request tensor ops, draw order, draw sizes, and generators =>
    # products BYTE-IDENTICAL at the same seeds (gated CPU-only by
    # scripts/fr13_dm_depthsync_byte_gate.py).
    # FR13_TAW (S1, default OFF): fully-tensorized zero-readback walk.
    # Distribution-equal (fr13_taw_equiv_gate.py PASS), NOT byte-equal to the
    # legacy rng stream; gates as its own live arm. Trace diagnostics and
    # all_greedy stay on the legacy path (same exclusions as depthsync).
    if (
        os.environ.get("FR13_TAW", "0") == "1"
        and _fr13_commit_trace_fh() is None
        and not all_greedy
    ):
        if (os.environ.get("FR13_STEP_GRAPH", "0") in ("1", "3")
                and not _FR13_SG_CAP_DEAD):
            return fr13_taw_commit_captured(
                num_draft_tokens, draft_token_ids, tree_parent_indices,
                target_logits, tree_self_logits, bonus_token_ids,
                max_spec_len, generators=generators)
        return fr13_taw_commit(
            num_draft_tokens,
            draft_token_ids,
            tree_parent_indices,
            target_logits,
            tree_self_logits,
            bonus_token_ids,
            max_spec_len,
            generators=generators,
        )
    if (
        os.environ.get("FR13_DM_DEPTHSYNC", "0") == "1"
        and _fr13_commit_trace_fh() is None
        and not all_greedy
    ):
        # (commit-trace diagnostics need per-node host values => the legacy
        # per-node path serves them; depthsync + trace never silently mix.)
        return _fr13_commit_depthsync(
            num_draft_tokens,
            draft_token_ids,
            tree_parent_indices,
            target_logits,
            tree_self_logits,
            bonus_token_ids,
            max_spec_len,
            generators=generators,
        )

    device = target_logits.device
    parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
    drafts_cpu = [int(x) for x in draft_token_ids.detach().cpu().tolist()]
    if hasattr(num_draft_tokens, "detach"):
        counts = [int(x) for x in num_draft_tokens.detach().cpu().tolist()]
    else:
        counts = [int(x) for x in num_draft_tokens]

    out_rows: list[list[int]] = []
    accepted_rows: list[int] = []
    accepted_lens: list[int] = []
    accepted_node_paths: list[list[int]] = []
    accepted_token_rows: list[list[int]] = []

    # all_greedy => POINT-MASS rows (temp-0 unification, byte-identical to the
    # greedy longest-prefix committer); else softmax rows (the temp>0 rule).
    _row_fn = _device_onehot_argmax_row if all_greedy else _device_softmax_row

    start = 0
    for req_i, node_count in enumerate(counts):
        node_count = int(node_count)
        # Per-request DEVICE generator seeded from the host per-request
        # generator -- the SAME seed source the host reference uses
        # (FR13_TREE_PER_REQ_GEN). Distribution-equivalent; the device draws
        # differ byte-for-byte from numpy but follow the same distributions.
        dev_gen = torch.Generator(device=device)
        if generators:
            host_gen = generators.get(req_i)
            if host_gen is not None:
                seed = int(
                    torch.randint(
                        0, 2 ** 31 - 1, (1,), device=host_gen.device,
                        generator=host_gen,
                    ).item()
                )
                dev_gen.manual_seed(seed)

        parents = parents_cpu[start:start + node_count]
        drafts = drafts_cpu[start:start + node_count]
        current_parent = -1
        accepted_row = 0
        accepted_path: list[int] = []
        row: list[int] = []
        for _step in range(int(max_spec_len) + 1):
            children = [
                node for node, parent in enumerate(parents)
                if int(parent) == int(current_parent)
            ]
            if not children:
                # dbg15: bound by node_count -- under pb current_parent starts
                # at the chain root (7); a zero-node row would index a foreign
                # flat row (same class as the batch-walk _bonus fix).
                if current_parent >= 0 and current_parent < int(node_count):
                    # self-target bonus: sample from the accepted node's self
                    # prob row, on-device (no [nodes x vocab] DtoH).
                    self_row = _row_fn(
                        tree_self_logits[start + current_parent]
                    )
                    tok = int(
                        torch.multinomial(
                            self_row.to(torch.float32), 1, generator=dev_gen
                        ).item()
                    )
                    row.append(tok)
                elif req_i < int(bonus_token_ids.numel()):
                    row.append(int(bonus_token_ids.reshape(-1)[req_i].item()))
                break

            target_row = int(start + children[0])
            child_drafts = [int(drafts[child]) for child in children]
            # On-device target prob row for this node (single row softmax, or a
            # point mass on the argmax when all_greedy => byte-identical greedy).
            p_row = _row_fn(target_logits[target_row])
            child_draft_tensor = torch.tensor(
                child_drafts, dtype=torch.long, device=device
            )
            token_id, source_index, accepted = device_multidraft_node_step(
                p_row, child_draft_tensor, generator=dev_gen,
            )
            selected_child = int(children[int(source_index)])
            _ct_fh = _fr13_commit_trace_fh()
            if _ct_fh is not None:
                _fr13_commit_trace_emit(
                    _ct_fh, req=req_i, node=selected_child, token_id=token_id,
                    accepted=accepted, p_row=p_row, child_drafts=child_drafts,
                )
            row.append(int(token_id))
            if not accepted:
                break
            accepted_child = selected_child
            if int(token_id) != int(drafts[accepted_child]):
                break
            current_parent = accepted_child
            accepted_row = int(current_parent)
            accepted_path.append(int(current_parent))

        out_rows.append(row[:int(max_spec_len) + 1])
        accepted_rows.append(int(accepted_row))
        accepted_lens.append(int(len(accepted_path)))
        accepted_node_paths.append([int(x) for x in accepted_path])
        accepted_token_rows.append([int(drafts[x]) for x in accepted_path])
        start += node_count

    return (
        out_rows,
        accepted_rows,
        accepted_lens,
        accepted_node_paths,
        accepted_token_rows,
    )


_FR13_DEPTHSYNC_ANNOUNCED = False


def _fr13_commit_depthsync(
    num_draft_tokens,
    draft_token_ids,
    tree_parent_indices,
    target_logits,
    tree_self_logits,
    bonus_token_ids,
    max_spec_len: int,
    *,
    generators=None,
):
    """Depth-SYNCHRONOUS multidraft walk (FR13_DM_DEPTHSYNC, S1/P1).

    Semantically the SAME walk as ``fr13_device_multidraft_commit``'s legacy
    per-node loop, restructured so host<->device round-trips are per LEVEL
    (batched over the active requests), not per node:

      legacy: 4-7 blocking ``.item()`` per node x ~10-15 nodes/step (~100 syncs)
      here:   readback A (overlap masses) + readback B (source/accept/continue)
              per level (+ readback C only on levels with rejects) + one seed
              batch + one final packed row DtoH  =>  ~2 x walk-depth syncs.

    BYTE-IDENTITY CONTRACT (gated by scripts/fr13_dm_depthsync_byte_gate.py):
    per request, the tensor ops, their SIZES, their ORDER, and the generator
    are exactly the legacy path's -- single-row softmax (never batched: a
    stacked [A,V] softmax could shift p by 1 ULP), exact-k weights for the
    source ``multinomial`` (replacement=False rng consumption depends on input
    size, so padding is banned), residual draws launched only for rejected
    requests as their LAST draw. Cross-request interleaving differs from the
    legacy sequential order, but every request draws from its OWN generator,
    so each request's stream is unchanged => identical products at same seeds.

    Control comparisons replicate the legacy python-float semantics exactly:
    ``u.item() < accept_prob.item()`` (f32 widened vs f64) becomes the device
    compare ``u.to(f64) < accept_prob``; ``overlap_mass.item() <= 0.0`` and
    ``mass.item() == 0.0`` are read back and compared as python floats.
    """
    device = target_logits.device
    global _FR13_DEPTHSYNC_ANNOUNCED
    if not _FR13_DEPTHSYNC_ANNOUNCED:
        _FR13_DEPTHSYNC_ANNOUNCED = True
        try:
            from vllm.logger import init_logger as _il
            _il("vllm.fr13_device_multidraft").info(
                "FR13_DM_DEPTHSYNC ENGAGED: depth-synchronous multidraft walk "
                "(batched per-level readbacks; byte gate 96/96)"
            )
        except Exception:  # noqa: BLE001
            pass
    parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
    drafts_cpu = [int(x) for x in draft_token_ids.detach().cpu().tolist()]
    if hasattr(num_draft_tokens, "detach"):
        counts = [int(x) for x in num_draft_tokens.detach().cpu().tolist()]
    else:
        counts = [int(x) for x in num_draft_tokens]
    nreq = len(counts)
    row_cap = int(max_spec_len) + 1

    # --- per-request generators: same seed derivation as legacy, but the seed
    # readbacks are batched into (at most) one sync per generator device.
    gens: list = [None] * nreq
    seed_pend: list = []  # (req_i, 0-dim seed tensor) needing readback
    for req_i in range(nreq):
        dev_gen = torch.Generator(device=device)
        gens[req_i] = dev_gen
        host_gen = generators.get(req_i) if generators else None
        if host_gen is not None:
            seed_t = torch.randint(
                0, 2 ** 31 - 1, (1,), device=host_gen.device, generator=host_gen
            )
            if seed_t.device.type == "cpu":
                dev_gen.manual_seed(int(seed_t.item()))  # cpu .item() = no sync
            else:
                seed_pend.append((req_i, seed_t.reshape(())))
    if seed_pend:
        vals = torch.stack([t for _, t in seed_pend]).cpu().tolist()
        for (req_i, _), v in zip(seed_pend, vals):
            gens[req_i].manual_seed(int(v))

    # --- per-request walk state (host ints; the tree structure is host-known)
    starts = []
    s = 0
    for c in counts:
        starts.append(s)
        s += int(c)
    cur_parent = [-1] * nreq
    accepted_row = [0] * nreq
    accepted_path: list[list[int]] = [[] for _ in range(nreq)]
    row_len = [0] * nreq
    # device row buffer: tokens land here asynchronously; ONE final DtoH.
    row_buf = torch.full((nreq, row_cap), -1, dtype=torch.long, device=device)
    active = [i for i in range(nreq)]

    def _children_of(req_i, parent):
        st, n = starts[req_i], int(counts[req_i])
        return [
            node for node in range(n)
            if int(parents_cpu[st + node]) == int(parent)
        ]

    def _emit_token(req_i, token_0dim):
        # async device-side append into the packed row buffer
        row_buf[req_i, row_len[req_i]] = token_0dim
        row_len[req_i] += 1

    def _bonus(req_i):
        # walk end for req_i: leaf bonus (self-row sample) or root bonus id.
        # dbg15: bound cp by the request's node count -- under pb cur_parent
        # STARTS at the chain root (7), so a zero-count row (bonus-only
        # commit) would index tree_self_logits[starts+7] into another
        # request's flat rows (the observed index-24-size-17 crash). A
        # non-empty pb tree keeps the root self-row sample (live-root bonus
        # application) unchanged: 7 < counts.
        cp = cur_parent[req_i]
        if cp >= 0 and cp < int(counts[req_i]):
            self_row = _device_softmax_row(
                tree_self_logits[starts[req_i] + cp]
            )
            tok = torch.multinomial(
                self_row.to(torch.float32), 1, generator=gens[req_i]
            ).reshape(())
            _emit_token(req_i, tok)
        elif req_i < int(bonus_token_ids.numel()):
            _emit_token(req_i, bonus_token_ids.reshape(-1)[req_i])

    for _level in range(row_cap):
        if not active:
            break
        # ---- phase A (async device work; per-request ops identical to legacy)
        stage = {}   # req_i -> dict of per-request tensors
        ended = []
        for req_i in list(active):
            children = _children_of(req_i, cur_parent[req_i])
            if not children:
                _bonus(req_i)
                ended.append(req_i)
                continue
            st = starts[req_i]
            child_drafts = [int(drafts_cpu[st + c]) for c in children]
            p_row = _device_softmax_row(target_logits[st + children[0]])
            p = p_row
            if p.dtype != torch.float32 and p.dtype != torch.float64:
                p = p.to(torch.float32)
            p = p / p.sum()
            tokens_t = torch.tensor(
                child_drafts, dtype=torch.long, device=device
            )
            overlaps = p[tokens_t].to(torch.float64)
            overlap_mass = overlaps.sum()
            stage[req_i] = {
                "children": children,
                "child_drafts": child_drafts,
                "p": p,
                "tokens": tokens_t,
                "overlaps": overlaps,
                "mass": overlap_mass,
            }
        for req_i in ended:
            active.remove(req_i)
        if not stage:
            continue
        # ---- readback A: overlap masses (one sync for all active requests)
        a_reqs = sorted(stage.keys())
        masses = torch.stack([stage[r]["mass"] for r in a_reqs]).cpu().tolist()
        # ---- phase B: draws (async; exact-k sizes; per-request generators)
        b_pend = {}
        for r, m in zip(a_reqs, masses):
            sg = stage[r]
            if float(m) <= 0.0:
                # zero-overlap reject: token ~ p (full vocab); LAST draw for r.
                tok = torch.multinomial(
                    sg["p"].to(torch.float32), 1, generator=gens[r]
                ).reshape(())
                _emit_token(r, tok)
                active.remove(r)
                continue
            weights = sg["overlaps"] / sg["mass"]
            source_t = torch.multinomial(
                weights.to(torch.float32), 1, generator=gens[r]
            ).reshape(())
            token_t = sg["tokens"][source_t]
            same = sg["tokens"] == token_t
            q_mix = weights[same].sum()
            accept_prob = torch.clamp(
                sg["p"][token_t].to(torch.float64) / q_mix, max=1.0
            )
            u = torch.rand(
                1, generator=gens[r], device=device, dtype=torch.float32
            ).reshape(())
            acc_t = u.to(torch.float64) < accept_prob
            _emit_token(r, token_t)
            b_pend[r] = {
                "weights": weights,
                "source": source_t,
                "token": token_t,
                "acc": acc_t,
            }
        if not b_pend:
            continue
        # ---- readback B: source index + accept flag (one sync)
        b_reqs = sorted(b_pend.keys())
        packed = torch.stack(
            [
                torch.stack(
                    [
                        b_pend[r]["source"].to(torch.int64),
                        b_pend[r]["acc"].to(torch.int64),
                    ]
                )
                for r in b_reqs
            ]
        ).cpu().tolist()
        # ---- phase C: rejected -> residual pipeline (values stay on device;
        # only the legacy `mass == 0` control float is read back, batched)
        c_pend = []
        for r, (src, acc) in zip(b_reqs, packed):
            sg, bp = stage[r], b_pend[r]
            if acc:
                child = int(sg["children"][int(src)])
                # legacy safety check `token != drafts[child]` is identically
                # true here (token == tokens[source] == drafts[child]); the
                # accepted path continues exactly as legacy.
                cur_parent[r] = child
                accepted_row[r] = child
                accepted_path[r].append(child)
                continue
            q_mix_vocab = torch.zeros_like(sg["p"], dtype=torch.float64)
            q_mix_vocab.scatter_add_(0, sg["tokens"], bp["weights"])
            residual = torch.clamp(
                sg["p"].to(torch.float64) - q_mix_vocab, min=0.0
            )
            rmass = residual.sum()
            c_pend.append((r, residual, rmass))
            active.remove(r)
        if c_pend:
            rmasses = torch.stack([m for _, _, m in c_pend]).cpu().tolist()
            for (r, residual, rmass_t), m in zip(c_pend, rmasses):
                if float(m) == 0.0:
                    residual = stage[r]["p"].to(torch.float64)
                else:
                    # legacy divides by the DEVICE mass tensor -- keep that op
                    residual = residual / rmass_t
                tok = torch.multinomial(
                    residual.to(torch.float32), 1, generator=gens[r]
                ).reshape(())
                # legacy appends the RESIDUAL token over the provisional
                # rejected token (same row position: legacy appends once per
                # node; the rejected node's row entry IS the residual token).
                row_len[r] -= 1
                _emit_token(r, tok)

    # any request still active hit the level cap exactly like the legacy
    # range(max_spec_len+1) loop; requests whose walk ended at a leaf got
    # their bonus in _bonus(). Requests that ran out of levels take no bonus
    # (legacy: loop exhausts without the children==[] branch firing).
    rows_host = row_buf.cpu().tolist()
    out_rows = [rows_host[i][: row_len[i]][:row_cap] for i in range(nreq)]
    return (
        out_rows,
        [int(x) for x in accepted_row],
        [int(len(p)) for p in accepted_path],
        [[int(x) for x in p] for p in accepted_path],
        [
            [int(drafts_cpu[starts[i] + n]) for n in accepted_path[i]]
            for i in range(nreq)
        ],
    )


# FR13_MULTIDRAFT_GPU_TIMER (diagnostic, default OFF => byte-identical): coarse GPU-time of the temp>0
# multidraft committer PER decode step. This is the REAL temp-0.6 committer (the greedy LCP / FR13_GPU_COMMITTER
# is a DIFFERENT, greedy-only path). Decomposes the 94ms committer_gpu span: multidraft_ms vs the rest
# (result DtoH + verify-wait). Non-invasive: the timed function body is UNCHANGED. Uses synchronize() (inflates
# other spans -- diagnostic run only, never deploy). Fresh import applies the wrapper (patcher exec_module's it).
_FR13_MD_GPU_SECONDS = 0.0
_FR13_MD_N = 0


def _fr13_md_gpu_timed(_orig):
    import functools

    @functools.wraps(_orig)
    def _w(*a, **k):
        import os
        if os.environ.get("FR13_MULTIDRAFT_GPU_TIMER", "0") != "1":
            return _orig(*a, **k)
        import json
        import torch
        _s = torch.cuda.Event(enable_timing=True)
        _e = torch.cuda.Event(enable_timing=True)
        _s.record()
        _r = _orig(*a, **k)
        _e.record()
        _e.synchronize()
        global _FR13_MD_GPU_SECONDS, _FR13_MD_N
        _FR13_MD_GPU_SECONDS += _s.elapsed_time(_e) / 1000.0
        _FR13_MD_N += 1
        if _FR13_MD_N % 50 == 0:
            try:
                json.dump(
                    {"gpu_seconds": _FR13_MD_GPU_SECONDS, "n_spans": _FR13_MD_N},
                    open(os.environ.get("FR13_MULTIDRAFT_GPU_TIMER_JSON", "/logs/fr13_multidraft_gpu.json"), "w"),
                )
            except Exception:
                pass
        return _r

    return _w


fr13_device_multidraft_commit = _fr13_md_gpu_timed(fr13_device_multidraft_commit)


# ---------------------------------------------------------------------------
# FR13 fixed-32 TAW: one physical 31-draft tree, fixed 12-iteration walk, and
# device-only fixed-capacity products. This route is completely separate from
# legacy TAW. It is selected only by FR13_FIXED32_MODE; an unset mode leaves
# every legacy branch above and below unchanged.
# ---------------------------------------------------------------------------
_FR13_FIXED32_TOPOLOGY = None
_FR13_FIXED32_TAW_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_FR13_FIXED32_TAW_WORK_CALLBACK: Callable[[dict[str, Any]], None] | None = None
_FR13_FIXED32_TAW_LAST_WORK: dict[str, Any] | None = None
_FR13_FIXED32_TAW_WARMUPS: dict[tuple[Any, ...], dict[str, Any]] = {}
_FR13_FIXED32_WORK_ANNOUNCED = False
_FR13_FIXED32_BATCHES = (1, 2, 3, 4)
_FR13_FIXED32_INTEGER_DTYPES = None
_FR13_FIXED32_TAW_SOURCE_SCHEMA = "fr13-fixed32-taw-exact-commit-v3"
_FR13_FIXED32_TAW_SOURCE_SHA256 = (
    "fe73ad35a916e41532575e29a5f9f6442d1081d0d1c0d0fc18210fdc8f0f56f8"
)
_FR13_FIXED32_TAW_SOURCE_CACHE: dict[str, Any] | None = None
_FR13_FIXED32_TAW_SOURCE_CODES: tuple[tuple[str, Any], ...] | None = None
_FR13_FIXED32_TAW_SOURCE_FUNCTIONS = (
    "_fr13_fixed32_topology",
    "_fr13_fixed32_device_key",
    "_fr13_fixed32_expected_active",
    "_fr13_fixed32_parse_int",
    "_fr13_fixed32_taw_native_precompute_enabled",
    "_fr13_fixed32_taw_tensor_call_census",
    "_fr13_fixed32_taw_source_contract",
    "_fr13_fixed32_runtime_contract",
    "fr13_fixed32_taw_cache_key",
    "fr13_fixed32_taw_preseed",
    "fr13_fixed32_taw_preseeded_counts",
    "_fr13_fixed32_integer_dtypes",
    "_fr13_fixed32_require_tensor",
    "_fr13_fixed32_device_assert",
    "_fr13_pin_uniforms",
    "_fr13_bulk_gen",
    "_fr13_fixed32_fill_uniforms",
    "_fr13_fixed32_validate_inputs",
    "_fr13_fixed32_tensor_layout",
    "_fr13_fixed32_layout_contract",
    "_fr13_fixed32_taw_probability_caches",
    "_fr13_fixed32_taw_execute_torch",
    "_fr13_fixed32_taw_execute_exact_cuda",
    "_fr13_fixed32_taw_execute",
    "_fr13_fixed32_publish_work",
    "fr13_fixed32_taw_commit",
    "_fr13_taw_inv_cdf",
)
_FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS = (
    "_fr13_fixed32_taw_exact_commit_kernel",
)
_FR13_FIXED32_TAW_GEOMETRY = {
    "physical_drafts": 31,
    "physical_rows": 32,
    "walk_cap": 12,
    "fanout": 3,
    "output_capacity": 32,
    "accepted_path_capacity": 16,
}
_FR13_FIXED32_TAW_TENSOR_CALL_CENSUS = {
    "walk_levels": 12,
    "full_vocab_row_gathers": 24,
    "full_vocab_fp32_casts": 24,
    "full_vocab_softmax_calls": 24,
    "full_vocab_normalizations": 36,
    "full_vocab_cdf_calls": 24,
    "source_cdf_calls": 12,
    "qmix_zero_fills": 12,
    "qmix_scatter_add_calls": 12,
    "residual_subtract_calls": 12,
    "residual_clamp_calls": 12,
    "residual_where_calls": 24,
    "output_scatter_calls": 0,
    "path_scatter_calls": 0,
    "exact_commit_launches": 12,
    "exact_commit_programs_per_request": 12,
    "floating_sampling_reimplementation": False,
}
_FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_TENSOR_CALL_CENSUS = {
    **_FR13_FIXED32_TAW_TENSOR_CALL_CENSUS,
    "walk_levels": 24,
    "full_vocab_row_gathers": 74,
    "full_vocab_fp32_casts": 26,
    "full_vocab_softmax_calls": 26,
    "full_vocab_normalizations": 72,
    "full_vocab_cdf_calls": 48,
    "source_cdf_calls": 24,
    "qmix_zero_fills": 24,
    "qmix_scatter_add_calls": 24,
    "residual_subtract_calls": 24,
    "residual_clamp_calls": 24,
    "residual_where_calls": 48,
    "exact_commit_launches": 24,
    "exact_commit_programs_per_request": 24,
}


def _fr13_fixed32_topology():
    """Load and validate the sibling fixed-32 topology contract once."""
    global _FR13_FIXED32_TOPOLOGY
    if _FR13_FIXED32_TOPOLOGY is not None:
        return _FR13_FIXED32_TOPOLOGY
    path = Path(__file__).resolve().with_name("fr13_fixed32_topology.py")
    if not path.is_file():
        raise RuntimeError(f"FR13 fixed32 topology contract is missing: {path}")
    module_name = "_fr13_fixed32_topology_contract"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load FR13 fixed32 topology: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    module.validate_contract()
    if (
        module.PHYSICAL_DRAFTS != 31
        or module.PHYSICAL_ROWS != 32
        or module.WALK_CAP != 12
        or module.COMMIT_PATH_CAP != 16
        or module.SAMPLER_MAX_FANOUT != 3
        or tuple(module.DRAFT_PARENT)
        != (
            -1,
            -1,
            -1,
            0,
            0,
            0,
            1,
            2,
            3,
            3,
            3,
            6,
            7,
            8,
            8,
            8,
            11,
            12,
            13,
            13,
            13,
            16,
            17,
            18,
            22,
            23,
            24,
            25,
            27,
            28,
            29,
        )
    ):
        raise RuntimeError("FR13 fixed32 topology constants drifted")
    _FR13_FIXED32_TOPOLOGY = module
    return module


def _fr13_fixed32_device_key(device) -> tuple[str, int | None]:
    normalized = torch.device(device)
    index = normalized.index
    if normalized.type == "cuda" and index is None:
        index = torch.cuda.current_device()
    return normalized.type, index


def _fr13_fixed32_expected_active(topology, mode: str) -> int:
    if mode == "tail6_fixed32":
        return int(topology.TAIL6_ACTIVE_DRAFTS)
    if mode == "hydra27_fixed32":
        return int(topology.HYDRA27_ACTIVE_DRAFTS)
    raise RuntimeError(f"unknown FR13_FIXED32_MODE {mode!r}")


def _fr13_fixed32_parse_int(raw: str, *, name: str) -> int:
    try:
        return int(raw, 0)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from error


def _fr13_fixed32_taw_native_precompute_enabled() -> bool:
    raw = os.environ.get("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE", "")
    if raw in ("", "0"):
        return False
    if raw == "1":
        return True
    raise RuntimeError(
        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE must be unset, 0, or 1"
    )


def _fr13_fixed32_taw_tensor_call_census() -> dict[str, Any]:
    census = (
        _FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_TENSOR_CALL_CENSUS
        if _fr13_fixed32_taw_native_precompute_enabled()
        else _FR13_FIXED32_TAW_TENSOR_CALL_CENSUS
    )
    return dict(census)


def _fr13_fixed32_taw_source_contract(topology) -> dict[str, Any]:
    """Bind the audited fixed TAW source and geometry without tracing a live event."""
    global _FR13_FIXED32_TAW_SOURCE_CACHE
    global _FR13_FIXED32_TAW_SOURCE_CODES

    functions = []
    codes = []
    for name in _FR13_FIXED32_TAW_SOURCE_FUNCTIONS:
        function = globals().get(name)
        code = getattr(function, "__code__", None)
        if code is None:
            raise RuntimeError(f"FR13 fixed32 TAW source function is missing: {name}")
        functions.append((name, function))
        codes.append((name, code))

    if _FR13_FIXED32_TAW_SOURCE_CACHE is not None:
        if _FR13_FIXED32_TAW_SOURCE_CODES != tuple(codes):
            raise RuntimeError("FR13 fixed32 TAW source objects changed after binding")
        digest = _FR13_FIXED32_TAW_SOURCE_CACHE["source_contract_sha256"]
        if digest != _FR13_FIXED32_TAW_SOURCE_SHA256:
            raise RuntimeError(
                "FR13 fixed32 TAW pinned source digest changed after binding"
            )
        return {
            **_FR13_FIXED32_TAW_SOURCE_CACHE,
            "tensor_call_census": _fr13_fixed32_taw_tensor_call_census(),
        }

    geometry = {
        "physical_drafts": int(topology.PHYSICAL_DRAFTS),
        "physical_rows": int(topology.PHYSICAL_ROWS),
        "walk_cap": int(topology.WALK_CAP),
        "fanout": int(topology.SAMPLER_MAX_FANOUT),
        "output_capacity": int(topology.OUTPUT_PUBLISH_CAPACITY),
        "accepted_path_capacity": int(topology.ACCEPTED_PATH_CAPACITY),
    }
    if geometry != _FR13_FIXED32_TAW_GEOMETRY:
        raise RuntimeError(
            "FR13 fixed32 TAW source geometry drift: " + repr(geometry)
        )

    normalized_sources = {}
    for name, function in functions:
        try:
            source = textwrap.dedent(inspect.getsource(function))
            parsed = ast.parse(source)
        except (OSError, TypeError, SyntaxError) as error:
            raise RuntimeError(
                f"FR13 fixed32 cannot inspect TAW source function {name}"
            ) from error
        definitions = [
            node
            for node in parsed.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if len(definitions) != 1 or definitions[0].name != name:
            raise RuntimeError(
                f"FR13 fixed32 TAW source parse was ambiguous for {name}"
            )
        normalized_sources[name] = ast.dump(
            definitions[0],
            annotate_fields=True,
            include_attributes=False,
        )

    try:
        module_tree = ast.parse(
            Path(__file__).resolve().read_text(encoding="utf-8")
        )
    except (OSError, SyntaxError) as error:
        raise RuntimeError("FR13 fixed32 cannot inspect exact commit kernel") from error
    kernel_definitions = {
        node.name: node
        for node in ast.walk(module_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in _FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS
    }
    if set(kernel_definitions) != set(_FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS):
        raise RuntimeError("FR13 fixed32 exact commit kernel source is incomplete")
    normalized_kernels = {
        name: ast.dump(
            kernel_definitions[name],
            annotate_fields=True,
            include_attributes=False,
        )
        for name in _FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS
    }

    canonical = json.dumps(
        {
            "schema": _FR13_FIXED32_TAW_SOURCE_SCHEMA,
            "functions": normalized_sources,
            "kernels": normalized_kernels,
            "geometry": geometry,
            "tensor_call_census_by_route": {
                "default": _FR13_FIXED32_TAW_TENSOR_CALL_CENSUS,
                "native_precompute": (
                    _FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_TENSOR_CALL_CENSUS
                ),
            },
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    if digest != _FR13_FIXED32_TAW_SOURCE_SHA256:
        raise RuntimeError(
            "FR13 fixed32 TAW source digest drift: "
            f"{digest} != {_FR13_FIXED32_TAW_SOURCE_SHA256}"
        )
    contract = {
        "source_contract_schema": _FR13_FIXED32_TAW_SOURCE_SCHEMA,
        "source_contract_sha256": digest,
    }
    _FR13_FIXED32_TAW_SOURCE_CODES = tuple(codes)
    _FR13_FIXED32_TAW_SOURCE_CACHE = contract
    return {
        **contract,
        "tensor_call_census": _fr13_fixed32_taw_tensor_call_census(),
    }


def _fr13_fixed32_runtime_contract(mode: str) -> tuple[Any, int]:
    """Validate all fixed-route pins without inspecting device values."""
    topology = _fr13_fixed32_topology()
    if mode not in topology.VALID_MASK_BY_MODE:
        raise RuntimeError(f"unknown FR13_FIXED32_MODE {mode!r}")
    env_mode = os.environ.get("FR13_FIXED32_MODE", "")
    if env_mode != mode:
        raise RuntimeError(
            f"FR13 fixed32 mode mismatch: argument={mode!r} environment={env_mode!r}"
        )
    mask_raw = os.environ.get("FR13_FIXED32_VALID_MASK")
    active_raw = os.environ.get("FR13_FIXED32_ACTIVE_NODES")
    walk_raw = os.environ.get("FR13_FIXED32_TAW_WALK_CAP")
    missing = [
        name
        for name, raw in (
            ("FR13_FIXED32_VALID_MASK", mask_raw),
            ("FR13_FIXED32_ACTIVE_NODES", active_raw),
            ("FR13_FIXED32_TAW_WALK_CAP", walk_raw),
        )
        if raw is None
    ]
    if missing:
        raise RuntimeError(f"FR13 fixed32 missing route pins: {missing}")
    mask = _fr13_fixed32_parse_int(
        mask_raw,
        name="FR13_FIXED32_VALID_MASK",
    )
    active = _fr13_fixed32_parse_int(
        active_raw,
        name="FR13_FIXED32_ACTIVE_NODES",
    )
    walk_cap = _fr13_fixed32_parse_int(
        walk_raw,
        name="FR13_FIXED32_TAW_WALK_CAP",
    )
    expected_mask = int(topology.VALID_MASK_BY_MODE[mode])
    expected_active = _fr13_fixed32_expected_active(topology, mode)
    if mask != expected_mask:
        raise RuntimeError(
            f"{mode}: validity mask {mask:#x} != contract {expected_mask:#x}"
        )
    if active != expected_active:
        raise RuntimeError(
            f"{mode}: active nodes {active} != contract {expected_active}"
        )
    if walk_cap != int(topology.WALK_CAP):
        raise RuntimeError(
            f"{mode}: TAW walk cap {walk_cap} != contract {topology.WALK_CAP}"
        )
    if os.environ.get("FR13_TAW") != "1":
        raise RuntimeError("FR13 fixed32 route requires FR13_TAW=1")
    if os.environ.get("LUMO_TREE_SAMPLER_DEBUG_LOG"):
        raise RuntimeError(
            "FR13 fixed32 route forbids host commit-trace materialization"
        )
    return topology, mask


def fr13_fixed32_taw_cache_key(
    mode: str,
    valid_mask: int,
    batch_size: int,
    device,
) -> tuple[Any, ...]:
    """Return the mode/mask/B/device cache key used by the fixed route."""
    if batch_size not in _FR13_FIXED32_BATCHES:
        raise RuntimeError(f"FR13 fixed32 batch must be 1..4, got {batch_size}")
    return (
        mode,
        int(valid_mask),
        int(batch_size),
        _fr13_fixed32_device_key(device),
    )


def fr13_fixed32_taw_preseed(
    device,
    *,
    mode: str | None = None,
    valid_mask: int | None = None,
) -> tuple[tuple[Any, ...], ...]:
    """Preseed fixed child tables and persistent products for B=1..4.

    This warmup API performs the only host-to-device construction of the
    topology tables. The measured commit route requires a cache hit and never
    creates, transfers, or falls back to a topology.
    """
    if torch is None:
        raise RuntimeError("FR13 fixed32 preseed requires torch")
    topology = _fr13_fixed32_topology()
    _fr13_fixed32_taw_source_contract(topology)
    if mode is None:
        mode = os.environ.get("FR13_FIXED32_MODE", "")
    if mode not in topology.VALID_MASK_BY_MODE:
        raise RuntimeError(f"unknown FR13 fixed32 preseed mode {mode!r}")
    expected_mask = int(topology.VALID_MASK_BY_MODE[mode])
    if valid_mask is None:
        raw_mask = os.environ.get("FR13_FIXED32_VALID_MASK")
        valid_mask = (
            expected_mask
            if raw_mask is None
            else _fr13_fixed32_parse_int(
                raw_mask,
                name="FR13_FIXED32_VALID_MASK",
            )
        )
    if int(valid_mask) != expected_mask:
        raise RuntimeError(
            f"{mode}: preseed mask {int(valid_mask):#x} != contract {expected_mask:#x}"
        )
    normalized_device = torch.device(device)
    if normalized_device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("FR13 fixed32 preseed is forbidden during capture")

    child_table, child_counts = topology.sampler_child_table(mode)
    if (
        len(child_table) != topology.PHYSICAL_ROWS
        or len(child_table[0]) != topology.SAMPLER_MAX_FANOUT
        or len(child_counts) != topology.PHYSICAL_ROWS
    ):
        raise RuntimeError(f"{mode}: fixed child-table contract drifted")
    table_base = torch.tensor(
        child_table,
        dtype=torch.long,
        device=normalized_device,
    )
    counts_base = torch.tensor(
        child_counts,
        dtype=torch.long,
        device=normalized_device,
    )
    parent_base = torch.tensor(
        topology.DRAFT_PARENT,
        dtype=torch.long,
        device=normalized_device,
    )
    active_children = topology.active_child_lists(mode)
    self_source_union = set()
    target_source_union = set()
    for fixed_mode in topology.VALID_BY_MODE:
        fixed_children = topology.active_child_lists(fixed_mode)
        fixed_active = (
            node
            for node, enabled in enumerate(topology.valid_for_mode(fixed_mode))
            if enabled
        )
        self_source_union.update(
            node for node in fixed_active if node not in fixed_children
        )
        target_source_union.update(
            children[0] for children in fixed_children.values()
        )
    self_source_nodes = tuple(sorted(self_source_union))
    target_source_nodes = tuple(sorted(target_source_union))
    if len(self_source_nodes) != 13 or len(target_source_nodes) != 17:
        raise RuntimeError(
            "FR13 native precompute fixed-row union drifted: "
            f"self={len(self_source_nodes)} target={len(target_source_nodes)}"
        )
    self_slot_by_node = [0] * int(topology.PHYSICAL_DRAFTS)
    for slot, node in enumerate(self_source_nodes):
        self_slot_by_node[node] = slot
    target_source_slot = {
        source_node: slot for slot, source_node in enumerate(target_source_nodes)
    }
    target_slot_by_parent = [0] * int(topology.PHYSICAL_ROWS)
    for parent, children in active_children.items():
        target_slot_by_parent[parent + 1] = target_source_slot[children[0]]
    self_source_base = torch.tensor(
        self_source_nodes,
        dtype=torch.long,
        device=normalized_device,
    )
    target_source_base = torch.tensor(
        target_source_nodes,
        dtype=torch.long,
        device=normalized_device,
    )
    self_slot_base = torch.tensor(
        self_slot_by_node,
        dtype=torch.long,
        device=normalized_device,
    )
    target_slot_base = torch.tensor(
        target_slot_by_parent,
        dtype=torch.long,
        device=normalized_device,
    )

    keys = []
    for batch_size in _FR13_FIXED32_BATCHES:
        key = fr13_fixed32_taw_cache_key(
            mode,
            int(valid_mask),
            batch_size,
            normalized_device,
        )
        starts = (
            torch.arange(
                batch_size,
                dtype=torch.long,
                device=normalized_device,
            )
            * int(topology.PHYSICAL_DRAFTS)
        )
        entry = {
            "mode": mode,
            "valid_mask": int(valid_mask),
            "batch_size": batch_size,
            "draft_counts": torch.full(
                (batch_size,),
                int(topology.PHYSICAL_DRAFTS),
                dtype=torch.int32,
                device=normalized_device,
            ),
            "child_table": table_base.unsqueeze(0).expand(batch_size, -1, -1).clone(),
            "child_counts": counts_base.unsqueeze(0).expand(batch_size, -1).clone(),
            "expected_parent": parent_base.unsqueeze(0).expand(batch_size, -1).clone(),
            "starts": starts,
            "native_self_rows_per_request": len(self_source_nodes),
            "native_target_rows_per_request": len(target_source_nodes),
            "native_self_source_indices": (
                starts.unsqueeze(1) + self_source_base.unsqueeze(0)
            ).reshape(-1),
            "native_target_source_indices": (
                starts.unsqueeze(1) + target_source_base.unsqueeze(0)
            ).reshape(-1),
            "native_self_slot_by_node": self_slot_base.clone(),
            "native_target_slot_by_parent": target_slot_base.clone(),
            "native_self_request_offsets": (
                torch.arange(
                    batch_size,
                    dtype=torch.long,
                    device=normalized_device,
                )
                * len(self_source_nodes)
            ),
            "native_target_request_offsets": (
                torch.arange(
                    batch_size,
                    dtype=torch.long,
                    device=normalized_device,
                )
                * len(target_source_nodes)
            ),
            "request_rows": torch.arange(
                batch_size,
                dtype=torch.long,
                device=normalized_device,
            ),
            "exact_initial_current": torch.full(
                (batch_size,),
                -1,
                dtype=torch.long,
                device=normalized_device,
            ),
            "exact_initial_alive": torch.ones(
                batch_size,
                dtype=torch.bool,
                device=normalized_device,
            ),
            "exact_current": torch.empty(
                batch_size,
                dtype=torch.long,
                device=normalized_device,
            ),
            "exact_alive": torch.empty(
                batch_size,
                dtype=torch.bool,
                device=normalized_device,
            ),
            "uniforms": torch.empty(
                (
                    batch_size,
                    int(topology.WALK_CAP),
                    3,
                ),
                dtype=torch.float32,
                device=normalized_device,
            ),
            "draft_tokens": torch.empty(
                (
                    batch_size,
                    int(topology.PHYSICAL_DRAFTS),
                ),
                dtype=torch.long,
                device=normalized_device,
            ),
            "bonus_tokens": torch.empty(
                batch_size,
                dtype=torch.long,
                device=normalized_device,
            ),
            "output_tokens": torch.full(
                (
                    batch_size,
                    int(topology.OUTPUT_PUBLISH_CAPACITY),
                ),
                -1,
                dtype=torch.long,
                device=normalized_device,
            ),
            "output_lens": torch.zeros(
                batch_size,
                dtype=torch.long,
                device=normalized_device,
            ),
            "accepted_path_rows": torch.zeros(
                (
                    batch_size,
                    int(topology.ACCEPTED_PATH_CAPACITY),
                ),
                dtype=torch.long,
                device=normalized_device,
            ),
            "accepted_lens": torch.zeros(
                batch_size,
                dtype=torch.long,
                device=normalized_device,
            ),
            "last_row": torch.zeros(
                batch_size,
                dtype=torch.long,
                device=normalized_device,
            ),
        }
        native_ab_entry = dict(entry)
        native_ab_entry.update(
            {
                "exact_current": torch.empty(
                    batch_size,
                    dtype=torch.long,
                    device=normalized_device,
                ),
                "exact_alive": torch.empty(
                    batch_size,
                    dtype=torch.bool,
                    device=normalized_device,
                ),
                "output_tokens": torch.full(
                    (
                        batch_size,
                        int(topology.OUTPUT_PUBLISH_CAPACITY),
                    ),
                    -1,
                    dtype=torch.long,
                    device=normalized_device,
                ),
                "output_lens": torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=normalized_device,
                ),
                "accepted_path_rows": torch.zeros(
                    (
                        batch_size,
                        int(topology.ACCEPTED_PATH_CAPACITY),
                    ),
                    dtype=torch.long,
                    device=normalized_device,
                ),
                "accepted_lens": torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=normalized_device,
                ),
                "last_row": torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=normalized_device,
                ),
            }
        )
        entry["native_ab_entry"] = native_ab_entry
        entry["native_ab_probability_mismatches"] = torch.zeros(
            (),
            dtype=torch.int64,
            device=normalized_device,
        )
        entry["native_ab_product_mismatches"] = torch.zeros(
            (),
            dtype=torch.int64,
            device=normalized_device,
        )
        entry["native_ab_root_checks"] = 0
        _FR13_FIXED32_TAW_CACHE[key] = entry
        keys.append(key)
    return tuple(keys)


def fr13_fixed32_taw_preseeded_counts(
    device,
    *,
    mode: str,
    valid_mask: int,
    batch_size: int,
):
    """Return the cache-owned fixed [31]*B tensor; never allocate on this route."""
    if torch is None:
        raise RuntimeError("FR13 fixed32 preseeded counts require torch")
    key = fr13_fixed32_taw_cache_key(
        mode,
        int(valid_mask),
        int(batch_size),
        device,
    )
    entry = _FR13_FIXED32_TAW_CACHE.get(key)
    if entry is None:
        raise RuntimeError(
            "FR13 fixed32 draft-count cache miss; preseed before requesting counts"
        )
    counts = entry.get("draft_counts")
    expected_device = torch.device(device)
    if (
        not isinstance(counts, torch.Tensor)
        or tuple(counts.shape) != (int(batch_size),)
        or counts.dtype != torch.int32
        or counts.device != expected_device
        or tuple(counts.stride()) != (1,)
        or not counts.is_contiguous()
    ):
        raise RuntimeError("FR13 fixed32 preseeded draft-count layout drift")
    return counts


def fr13_fixed32_taw_set_work_callback(
    callback: Callable[[dict[str, Any]], None] | None,
) -> None:
    """Set the complete-event TAW work callback used by the census owner."""
    if callback is not None and not callable(callback):
        raise TypeError("FR13 fixed32 TAW work callback must be callable")
    global _FR13_FIXED32_TAW_WORK_CALLBACK
    _FR13_FIXED32_TAW_WORK_CALLBACK = callback


def fr13_fixed32_taw_last_work() -> dict[str, Any] | None:
    """Return the last host-side fixed-work counter payload."""
    if _FR13_FIXED32_TAW_LAST_WORK is None:
        return None
    result = dict(_FR13_FIXED32_TAW_LAST_WORK)
    result["taw"] = dict(_FR13_FIXED32_TAW_LAST_WORK["taw"])
    return result


def _fr13_fixed32_taw_cache_lease(entry: dict[str, Any]) -> tuple[Any, ...]:
    """Bind every cache tensor object, storage, pointer, dtype, and layout."""
    tensor_names = (
        "draft_counts",
        "child_table",
        "child_counts",
        "expected_parent",
        "starts",
        "request_rows",
        "exact_initial_current",
        "exact_initial_alive",
        "exact_current",
        "exact_alive",
        "uniforms",
        "draft_tokens",
        "bonus_tokens",
        "output_tokens",
        "output_lens",
        "accepted_path_rows",
        "accepted_lens",
        "last_row",
    )
    lease = []
    for name in tensor_names:
        value = entry.get(name)
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(
                f"FR13 fixed32 TAW cache lease is missing tensor {name}"
            )
        lease.append(
            (
                name,
                id(value),
                int(value.data_ptr()),
                int(value.untyped_storage().data_ptr()),
                tuple(int(dimension) for dimension in value.shape),
                tuple(int(stride) for stride in value.stride()),
                str(value.dtype),
                str(value.device),
            )
        )
    return tuple(lease)


def fr13_fixed32_taw_warmup_counters(
    device,
    *,
    mode: str,
    valid_mask: int,
    max_batch_size: int,
    vocab_size: int,
) -> dict[str, Any]:
    """Return boot-warm evidence only while it still owns the live cache."""
    capacity = int(max_batch_size)
    key = (
        str(mode),
        int(valid_mask),
        capacity,
        int(vocab_size),
        _fr13_fixed32_device_key(device),
    )
    record = _FR13_FIXED32_TAW_WARMUPS.get(key)
    if record is None:
        return {
            "ready": False,
            "classification": "unmeasured_boot",
            "mode": str(mode),
            "valid_mask": int(valid_mask),
            "max_batch_size": capacity,
            "vocab_size": int(vocab_size),
            "batches": (),
            "executions": 0,
            "cache_lease_current": False,
            "rng_state_restored": False,
            "staging_state_restored": False,
            "measured_state_restored": False,
        }
    cache_entries = tuple(
        _FR13_FIXED32_TAW_CACHE.get(
            fr13_fixed32_taw_cache_key(
                str(mode),
                int(valid_mask),
                batch,
                device,
            )
        )
        for batch in range(1, capacity + 1)
    )
    cache_leases = tuple(
        _fr13_fixed32_taw_cache_lease(entry)
        for entry in cache_entries
        if entry is not None
    )
    result = {
        name: value
        for name, value in record.items()
        if name != "_cache_leases"
    }
    result["cache_lease_current"] = (
        len(cache_leases) == capacity
        and cache_leases == tuple(record["_cache_leases"])
    )
    result["ready"] = bool(
        result.get("ready")
        and result.get("classification") == "unmeasured_boot"
        and result.get("mode") == str(mode)
        and int(result.get("valid_mask", -1)) == int(valid_mask)
        and int(result.get("max_batch_size", -1)) == capacity
        and int(result.get("vocab_size", -1)) == int(vocab_size)
        and tuple(result.get("batches", ()))
        == tuple(range(1, capacity + 1))
        and int(result.get("executions", -1)) == capacity
        and result["cache_lease_current"]
        and result.get("rng_state_restored") is True
        and result.get("staging_state_restored") is True
        and result.get("measured_state_restored") is True
    )
    return result


def fr13_fixed32_taw_warm_products(
    device,
    *,
    mode: str,
    valid_mask: int,
    max_batch_size: int,
    vocab_size: int,
    batch_size: int,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return cache-owned boot products only under a current warm lease."""
    evidence = fr13_fixed32_taw_warmup_counters(
        device,
        mode=mode,
        valid_mask=valid_mask,
        max_batch_size=max_batch_size,
        vocab_size=vocab_size,
    )
    if evidence.get("ready") is not True:
        raise RuntimeError(
            "FR13 fixed32 TAW warm products require a current boot lease"
        )
    batch = int(batch_size)
    if not 1 <= batch <= int(max_batch_size):
        raise ValueError(
            "FR13 fixed32 TAW warm-product batch exceeds boot capacity"
        )
    entry = _FR13_FIXED32_TAW_CACHE[
        fr13_fixed32_taw_cache_key(
            mode,
            int(valid_mask),
            batch,
            device,
        )
    ]
    return (
        entry["output_tokens"],
        entry["output_lens"],
        entry["accepted_path_rows"],
        entry["accepted_lens"],
        entry["last_row"],
    )


def _fr13_fixed32_tensor_bits_equal(left, right) -> bool:
    return bool(
        torch.equal(
            left.contiguous().view(torch.uint8),
            right.contiguous().view(torch.uint8),
        )
    )


def fr13_fixed32_taw_warm_execute(
    device,
    *,
    mode: str,
    valid_mask: int,
    max_batch_size: int,
    vocab_size: int,
) -> dict[str, Any]:
    """Execute the production fixed TAW call graph outside measured events.

    All cache-owned tensors, RNG position, work callbacks, last-work evidence,
    and announcement state are restored exactly. The initialized bulk
    generator object and backend kernels remain available for serving.
    """
    if torch is None:
        raise RuntimeError("FR13 fixed32 TAW boot warm requires torch")
    topology, runtime_mask = _fr13_fixed32_runtime_contract(str(mode))
    capacity = int(max_batch_size)
    vocab = int(vocab_size)
    if int(valid_mask) != runtime_mask:
        raise RuntimeError(
            "FR13 fixed32 TAW boot-warm mask drift: "
            f"{int(valid_mask):#x} != {runtime_mask:#x}"
        )
    if capacity not in _FR13_FIXED32_BATCHES:
        raise ValueError(
            "FR13 fixed32 TAW boot-warm capacity must be 1..4, "
            f"got {capacity}"
        )
    if vocab <= 0:
        raise ValueError(
            f"FR13 fixed32 TAW boot-warm vocabulary must be positive, got {vocab}"
        )
    normalized_device = torch.device(device)
    if (
        normalized_device.type == "cuda"
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError("FR13 fixed32 TAW boot warm is forbidden during capture")

    prior = fr13_fixed32_taw_warmup_counters(
        normalized_device,
        mode=str(mode),
        valid_mask=runtime_mask,
        max_batch_size=capacity,
        vocab_size=vocab,
    )
    if prior["ready"]:
        return prior

    entries = []
    for batch in range(1, capacity + 1):
        cache_key = fr13_fixed32_taw_cache_key(
            str(mode),
            runtime_mask,
            batch,
            normalized_device,
        )
        entry = _FR13_FIXED32_TAW_CACHE.get(cache_key)
        if entry is None:
            raise RuntimeError(
                "FR13 fixed32 TAW boot warm requires every preseeded "
                f"occupancy through B={capacity}; missing B={batch}"
            )
        entries.append(entry)

    mutable_names = (
        "uniforms",
        "draft_tokens",
        "bonus_tokens",
        "output_tokens",
        "output_lens",
        "accepted_path_rows",
        "accepted_lens",
        "last_row",
        "exact_current",
        "exact_alive",
    )
    saved_cache = tuple(
        {
            name: entry[name].clone()
            for name in mutable_names
        }
        for entry in entries
    )
    generator = _fr13_bulk_gen(normalized_device)
    saved_generator_state = generator.get_state().clone()
    global _FR13_FIXED32_TAW_WORK_CALLBACK
    global _FR13_FIXED32_TAW_LAST_WORK
    global _FR13_FIXED32_WORK_ANNOUNCED
    saved_callback = _FR13_FIXED32_TAW_WORK_CALLBACK
    saved_last_work = _FR13_FIXED32_TAW_LAST_WORK
    saved_announced = _FR13_FIXED32_WORK_ANNOUNCED

    def discard_warm_work(_payload):
        return None

    _FR13_FIXED32_TAW_WORK_CALLBACK = discard_warm_work
    _FR13_FIXED32_WORK_ANNOUNCED = True
    executions = 0
    logits = None
    try:
        max_rows = capacity * int(topology.PHYSICAL_DRAFTS)
        logits = torch.zeros(
            (max_rows, vocab),
            dtype=torch.float32,
            device=normalized_device,
        )
        for batch, entry in enumerate(entries, start=1):
            rows = batch * int(topology.PHYSICAL_DRAFTS)
            draft_ids = (
                torch.arange(
                    int(topology.PHYSICAL_DRAFTS),
                    dtype=torch.int32,
                    device=normalized_device,
                )
                .remainder(vocab)
                .repeat(batch)
            )
            parents = entry["expected_parent"].to(torch.int32).reshape(-1)
            bonus = torch.zeros(
                (batch, 1),
                dtype=torch.int32,
                device=normalized_device,
            )
            products = fr13_fixed32_taw_commit(
                entry["draft_counts"],
                draft_ids,
                parents,
                logits[:rows],
                logits[:rows],
                bonus,
                int(topology.PHYSICAL_DRAFTS),
                generators=None,
                uniforms=None,
                all_greedy=False,
                mode=str(mode),
            )
            if (
                len(products) != 5
                or products[0] is not entry["output_tokens"]
                or products[1] is not entry["output_lens"]
                or products[2] is not entry["accepted_path_rows"]
                or products[3] is not entry["accepted_lens"]
                or products[4] is not entry["last_row"]
            ):
                raise RuntimeError(
                    "FR13 fixed32 TAW boot warm returned non-cache products"
                )
            executions += 1
        if normalized_device.type == "cuda":
            torch.cuda.synchronize(normalized_device)
    finally:
        try:
            generator.set_state(saved_generator_state)
        finally:
            try:
                for entry, saved in zip(entries, saved_cache, strict=True):
                    for name in mutable_names:
                        entry[name].copy_(saved[name])
            finally:
                _FR13_FIXED32_TAW_WORK_CALLBACK = saved_callback
                _FR13_FIXED32_TAW_LAST_WORK = saved_last_work
                _FR13_FIXED32_WORK_ANNOUNCED = saved_announced
                if normalized_device.type == "cuda":
                    torch.cuda.synchronize(normalized_device)
        del logits

    staging_state_restored = all(
        _fr13_fixed32_tensor_bits_equal(entry[name], saved[name])
        for entry, saved in zip(entries, saved_cache, strict=True)
        for name in mutable_names
    )
    rng_state_restored = _fr13_fixed32_tensor_bits_equal(
        generator.get_state(),
        saved_generator_state,
    )
    measured_state_restored = (
        _FR13_FIXED32_TAW_WORK_CALLBACK is saved_callback
        and _FR13_FIXED32_TAW_LAST_WORK is saved_last_work
        and _FR13_FIXED32_WORK_ANNOUNCED is saved_announced
    )
    if executions != capacity:
        raise RuntimeError(
            "FR13 fixed32 TAW boot warm did not execute every occupancy"
        )
    if (
        not staging_state_restored
        or not rng_state_restored
        or not measured_state_restored
    ):
        raise RuntimeError(
            "FR13 fixed32 TAW boot warm did not restore serving state"
        )
    record = {
        "ready": True,
        "classification": "unmeasured_boot",
        "mode": str(mode),
        "valid_mask": runtime_mask,
        "max_batch_size": capacity,
        "vocab_size": vocab,
        "batches": tuple(range(1, capacity + 1)),
        "executions": executions,
        "cache_lease_current": True,
        "rng_state_restored": rng_state_restored,
        "staging_state_restored": staging_state_restored,
        "measured_state_restored": measured_state_restored,
        "_cache_leases": tuple(
            _fr13_fixed32_taw_cache_lease(entry)
            for entry in entries
        ),
    }
    warm_key = (
        str(mode),
        runtime_mask,
        capacity,
        vocab,
        _fr13_fixed32_device_key(normalized_device),
    )
    _FR13_FIXED32_TAW_WARMUPS[warm_key] = record
    return fr13_fixed32_taw_warmup_counters(
        normalized_device,
        mode=str(mode),
        valid_mask=runtime_mask,
        max_batch_size=capacity,
        vocab_size=vocab,
    )


def _fr13_fixed32_integer_dtypes():
    global _FR13_FIXED32_INTEGER_DTYPES
    if _FR13_FIXED32_INTEGER_DTYPES is None:
        _FR13_FIXED32_INTEGER_DTYPES = frozenset(
            {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
        )
    return _FR13_FIXED32_INTEGER_DTYPES


def _fr13_fixed32_require_tensor(
    name: str,
    value,
    *,
    device,
    integer: bool = False,
    floating: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"FR13 fixed32 {name} must be a torch.Tensor")
    if value.device != device:
        raise RuntimeError(
            f"FR13 fixed32 {name} device {value.device} != {device}; "
            "host/device transfers are forbidden"
        )
    if integer and value.dtype not in _fr13_fixed32_integer_dtypes():
        raise RuntimeError(
            f"FR13 fixed32 {name} must have integer dtype, got {value.dtype}"
        )
    if floating and not value.dtype.is_floating_point:
        raise RuntimeError(
            f"FR13 fixed32 {name} must have floating dtype, got {value.dtype}"
        )


def _fr13_fixed32_device_assert(condition, message: str) -> None:
    assert_async = getattr(torch, "_assert_async", None)
    if assert_async is None:
        raise RuntimeError("FR13 fixed32 requires torch._assert_async")
    assert_async(condition, message)


def _fr13_fixed32_fill_uniforms(
    entry: dict[str, Any],
    *,
    generators=None,
    uniforms=None,
):
    target = entry["uniforms"]
    if generators:
        raise RuntimeError(
            "FR13 fixed32 forbids per-request generator maps; "
            "the fixed route requires one bulk device RNG call"
        )
    if uniforms is not None:
        _fr13_fixed32_require_tensor(
            "uniforms",
            uniforms,
            device=target.device,
            floating=True,
        )
        if (
            tuple(uniforms.shape) != tuple(target.shape)
            or uniforms.dtype != torch.float32
            or tuple(uniforms.stride()) != tuple(target.stride())
            or not uniforms.is_contiguous()
        ):
            raise RuntimeError(
                "FR13 fixed32 uniforms layout drift: "
                + repr(
                    {
                        "shape": tuple(uniforms.shape),
                        "dtype": str(uniforms.dtype),
                        "stride": tuple(uniforms.stride()),
                        "contiguous": bool(uniforms.is_contiguous()),
                        "expected_shape": tuple(target.shape),
                        "expected_stride": tuple(target.stride()),
                    }
                )
            )
        _fr13_fixed32_device_assert(
            torch.all((uniforms >= 0.0) & (uniforms < 1.0)),
            "FR13 fixed32 uniforms must be in [0,1)",
        )
        return uniforms, "provided_uniforms"

    target.uniform_(generator=_fr13_bulk_gen(target.device))
    if _fr13_pin_uniforms():
        target.fill_(0.5)
        return target, "bulk_device_generator_then_pin"
    return target, "bulk_device_generator"


def _fr13_fixed32_tensor_layout(value) -> dict[str, Any]:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("FR13 fixed32 layout value is not a tensor")
    return {
        "shape": [int(dimension) for dimension in value.shape],
        "dtype": str(value.dtype),
        "stride": [int(dimension) for dimension in value.stride()],
        "contiguous": bool(value.is_contiguous()),
    }


def _fr13_fixed32_validate_inputs(
    topology,
    entry: dict[str, Any],
    num_draft_tokens,
    draft_token_ids,
    tree_parent_indices,
    target_logits,
    tree_self_logits,
    bonus_token_ids,
    max_spec_len: int,
) -> tuple[Any, Any]:
    device = target_logits.device
    batch_size = entry["batch_size"]
    physical_drafts = int(topology.PHYSICAL_DRAFTS)
    flat_rows = batch_size * physical_drafts
    if (
        not isinstance(max_spec_len, int)
        or isinstance(max_spec_len, bool)
        or max_spec_len != physical_drafts
    ):
        raise RuntimeError(
            "FR13 fixed32 max_spec_len must equal 31 physical drafts, got "
            f"{max_spec_len!r}"
        )
    for name, value in (
        ("num_draft_tokens", num_draft_tokens),
        ("draft_token_ids", draft_token_ids),
        ("tree_parent_indices", tree_parent_indices),
        ("bonus_token_ids", bonus_token_ids),
    ):
        _fr13_fixed32_require_tensor(
            name,
            value,
            device=device,
            integer=True,
        )
    _fr13_fixed32_require_tensor(
        "target_logits",
        target_logits,
        device=device,
        floating=True,
    )
    _fr13_fixed32_require_tensor(
        "tree_self_logits",
        tree_self_logits,
        device=device,
        floating=True,
    )
    if target_logits.ndim != 2 or int(target_logits.shape[1]) <= 0:
        raise RuntimeError("FR13 fixed32 vocabulary must be nonempty")
    vocab_size = int(target_logits.shape[1])
    expected_layouts = {
        "draft_counts": {
            "shape": [batch_size],
            "dtype": "torch.int32",
            "stride": [1],
            "contiguous": True,
        },
        "draft_input": {
            "shape": [flat_rows],
            "dtype": "torch.int32",
            "stride": [1],
            "contiguous": True,
        },
        "parent_input": {
            "shape": [flat_rows],
            "dtype": "torch.int32",
            "stride": [1],
            "contiguous": True,
        },
        "target_logits": {
            "shape": [flat_rows, vocab_size],
            "dtype": "torch.float32",
            "stride": [vocab_size, 1],
            "contiguous": True,
        },
        "self_logits": {
            "shape": [flat_rows, vocab_size],
            "dtype": "torch.float32",
            "stride": [vocab_size, 1],
            "contiguous": True,
        },
        "bonus_input": {
            "shape": [batch_size, 1],
            "dtype": "torch.int32",
            "stride": [1, 1],
            "contiguous": True,
        },
    }
    actual_layouts = {
        "draft_counts": _fr13_fixed32_tensor_layout(num_draft_tokens),
        "draft_input": _fr13_fixed32_tensor_layout(draft_token_ids),
        "parent_input": _fr13_fixed32_tensor_layout(tree_parent_indices),
        "target_logits": _fr13_fixed32_tensor_layout(target_logits),
        "self_logits": _fr13_fixed32_tensor_layout(tree_self_logits),
        "bonus_input": _fr13_fixed32_tensor_layout(bonus_token_ids),
    }
    if num_draft_tokens is not entry.get("draft_counts"):
        raise RuntimeError(
            "FR13 fixed32 requires the cache-owned preseeded draft-count tensor"
        )
    if actual_layouts != expected_layouts:
        raise RuntimeError(
            "FR13 fixed32 live input layout drift: "
            + repr((actual_layouts, expected_layouts))
        )

    counts = num_draft_tokens.reshape(batch_size)
    parents = tree_parent_indices.reshape(batch_size, physical_drafts)
    drafts_input = draft_token_ids.reshape(batch_size, physical_drafts)
    bonus_input = bonus_token_ids.reshape(batch_size)
    _fr13_fixed32_device_assert(
        torch.all(counts == physical_drafts),
        "FR13 fixed32 requires every active request to have exactly 31 drafts",
    )
    _fr13_fixed32_device_assert(
        torch.all(parents == entry["expected_parent"]),
        "FR13 fixed32 physical parent vector mismatch",
    )
    _fr13_fixed32_device_assert(
        torch.all((drafts_input >= 0) & (drafts_input < vocab_size)),
        "FR13 fixed32 draft token id is outside the vocabulary",
    )
    _fr13_fixed32_device_assert(
        torch.all((bonus_input >= 0) & (bonus_input < vocab_size)),
        "FR13 fixed32 bonus token id is outside the vocabulary",
    )
    drafts = entry["draft_tokens"]
    bonus = entry["bonus_tokens"]
    drafts.copy_(drafts_input, non_blocking=True)
    bonus.copy_(bonus_input, non_blocking=True)
    return drafts, bonus


def _fr13_fixed32_layout_contract(
    topology,
    entry: dict[str, Any],
    num_draft_tokens,
    draft_token_ids,
    tree_parent_indices,
    target_logits,
    tree_self_logits,
    bonus_token_ids,
    uniforms,
    *,
    rng_route: str,
) -> dict[str, Any]:
    """Validate and publish the live layouts that select the audited call graph."""
    batch = int(entry["batch_size"])
    physical_drafts = int(topology.PHYSICAL_DRAFTS)
    vocab = int(target_logits.shape[1])
    expected_cache_layouts = {
        "draft_counts": ([batch], "torch.int32", [1]),
        "child_table": ([batch, 32, 3], "torch.int64", [96, 3, 1]),
        "child_counts": ([batch, 32], "torch.int64", [32, 1]),
        "expected_parent": ([batch, 31], "torch.int64", [31, 1]),
        "starts": ([batch], "torch.int64", [1]),
        "request_rows": ([batch], "torch.int64", [1]),
        "exact_initial_current": ([batch], "torch.int64", [1]),
        "exact_initial_alive": ([batch], "torch.bool", [1]),
        "exact_current": ([batch], "torch.int64", [1]),
        "exact_alive": ([batch], "torch.bool", [1]),
        "uniforms": ([batch, 12, 3], "torch.float32", [36, 3, 1]),
        "draft_tokens": ([batch, 31], "torch.int64", [31, 1]),
        "bonus_tokens": ([batch], "torch.int64", [1]),
        "output_tokens": ([batch, 32], "torch.int64", [32, 1]),
        "output_lens": ([batch], "torch.int64", [1]),
        "accepted_path_rows": ([batch, 16], "torch.int64", [16, 1]),
        "accepted_lens": ([batch], "torch.int64", [1]),
        "last_row": ([batch], "torch.int64", [1]),
    }
    actual_cache_layouts = {}
    for name in expected_cache_layouts:
        value = entry.get(name)
        layout = _fr13_fixed32_tensor_layout(value)
        actual_cache_layouts[name] = (
            layout["shape"],
            layout["dtype"],
            layout["stride"],
        )
        if not layout["contiguous"]:
            raise RuntimeError(
                f"FR13 fixed32 cached tensor is noncontiguous: {name}"
            )
    if actual_cache_layouts != expected_cache_layouts:
        raise RuntimeError(
            "FR13 fixed32 cached tensor layout drift: "
            + repr((actual_cache_layouts, expected_cache_layouts))
        )

    live = {
        "count": _fr13_fixed32_tensor_layout(num_draft_tokens),
        "draft": _fr13_fixed32_tensor_layout(draft_token_ids),
        "parent": _fr13_fixed32_tensor_layout(tree_parent_indices),
        "target": _fr13_fixed32_tensor_layout(target_logits),
        "self": _fr13_fixed32_tensor_layout(tree_self_logits),
        "bonus": _fr13_fixed32_tensor_layout(bonus_token_ids),
        "uniform": _fr13_fixed32_tensor_layout(uniforms),
    }
    expected_live = {
        "count": {
            "shape": [batch],
            "dtype": "torch.int32",
            "stride": [1],
            "contiguous": True,
        },
        "draft": {
            "shape": [batch * physical_drafts],
            "dtype": "torch.int32",
            "stride": [1],
            "contiguous": True,
        },
        "parent": {
            "shape": [batch * physical_drafts],
            "dtype": "torch.int32",
            "stride": [1],
            "contiguous": True,
        },
        "target": {
            "shape": [batch * physical_drafts, vocab],
            "dtype": "torch.float32",
            "stride": [vocab, 1],
            "contiguous": True,
        },
        "self": {
            "shape": [batch * physical_drafts, vocab],
            "dtype": "torch.float32",
            "stride": [vocab, 1],
            "contiguous": True,
        },
        "bonus": {
            "shape": [batch, 1],
            "dtype": "torch.int32",
            "stride": [1, 1],
            "contiguous": True,
        },
        "uniform": {
            "shape": [batch, 12, 3],
            "dtype": "torch.float32",
            "stride": [36, 3, 1],
            "contiguous": True,
        },
    }
    if live != expected_live:
        raise RuntimeError(
            "FR13 fixed32 published tensor layout drift: "
            + repr((live, expected_live))
        )
    if rng_route not in (
        "bulk_device_generator",
        "bulk_device_generator_then_pin",
        "provided_uniforms",
    ):
        raise RuntimeError(f"FR13 fixed32 unknown RNG route: {rng_route!r}")

    count_route = (
        "preseeded_cuda_fixed31"
        if num_draft_tokens.device.type == "cuda"
        else "preseeded_cpu_fixed31_test"
    )
    return {
        "count_route": count_route,
        "rng_route": rng_route,
        "vocab_size": vocab,
        "count_shape": live["count"]["shape"],
        "count_dtype": live["count"]["dtype"],
        "count_stride": live["count"]["stride"],
        "count_contiguous": live["count"]["contiguous"],
        "draft_shape": live["draft"]["shape"],
        "draft_dtype": live["draft"]["dtype"],
        "draft_stride": live["draft"]["stride"],
        "draft_contiguous": live["draft"]["contiguous"],
        "parent_shape": live["parent"]["shape"],
        "parent_dtype": live["parent"]["dtype"],
        "parent_stride": live["parent"]["stride"],
        "parent_contiguous": live["parent"]["contiguous"],
        "bonus_shape": live["bonus"]["shape"],
        "bonus_dtype": live["bonus"]["dtype"],
        "bonus_stride": live["bonus"]["stride"],
        "bonus_contiguous": live["bonus"]["contiguous"],
        "target_shape": live["target"]["shape"],
        "target_dtype": live["target"]["dtype"],
        "target_stride": live["target"]["stride"],
        "target_contiguous": live["target"]["contiguous"],
        "self_shape": live["self"]["shape"],
        "self_dtype": live["self"]["dtype"],
        "self_stride": live["self"]["stride"],
        "self_contiguous": live["self"]["contiguous"],
        "uniform_shape": live["uniform"]["shape"],
        "uniform_dtype": live["uniform"]["dtype"],
        "uniform_stride": live["uniform"]["stride"],
        "uniform_contiguous": live["uniform"]["contiguous"],
        "child_table_shape": list(expected_cache_layouts["child_table"][0]),
        "child_counts_shape": list(expected_cache_layouts["child_counts"][0]),
        "output_shape": list(expected_cache_layouts["output_tokens"][0]),
        "output_lens_shape": list(expected_cache_layouts["output_lens"][0]),
        "accepted_path_shape": list(
            expected_cache_layouts["accepted_path_rows"][0]
        ),
        "accepted_lens_shape": list(expected_cache_layouts["accepted_lens"][0]),
        "last_row_shape": list(expected_cache_layouts["last_row"][0]),
        "exact_current_shape": list(expected_cache_layouts["exact_current"][0]),
        "exact_alive_shape": list(expected_cache_layouts["exact_alive"][0]),
    }


def _fr13_fixed32_taw_probability_caches(
    entry: dict[str, Any],
    target_logits,
    tree_self_logits,
    *,
    native_precompute: bool | None = None,
) -> tuple[Any | None, Any | None]:
    """Batch immutable reachable rows with the native PyTorch softmax operator."""
    if native_precompute is None:
        native_precompute = _fr13_fixed32_taw_native_precompute_enabled()
    if not native_precompute:
        return None, None

    batch_size = int(entry["batch_size"])
    self_rows = int(entry["native_self_rows_per_request"])
    target_rows = int(entry["native_target_rows_per_request"])
    expected = {
        "native_self_source_indices": ((batch_size * self_rows,), torch.long),
        "native_target_source_indices": ((batch_size * target_rows,), torch.long),
        "native_self_slot_by_node": ((31,), torch.long),
        "native_target_slot_by_parent": ((32,), torch.long),
        "native_self_request_offsets": ((batch_size,), torch.long),
        "native_target_request_offsets": ((batch_size,), torch.long),
    }
    for name, (shape, dtype) in expected.items():
        value = entry.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != target_logits.device
            or not value.is_contiguous()
        ):
            raise RuntimeError(f"FR13 fixed32 native precompute state drift: {name}")

    self_prob_cache = torch.softmax(
        tree_self_logits[entry["native_self_source_indices"]].to(torch.float32),
        dim=-1,
    )
    target_prob_cache = torch.softmax(
        target_logits[entry["native_target_source_indices"]].to(torch.float32),
        dim=-1,
    )
    return self_prob_cache, target_prob_cache


def _fr13_fixed32_taw_execute_torch(
    topology,
    entry: dict[str, Any],
    drafts,
    target_logits,
    tree_self_logits,
    bonus_flat,
    uniforms,
    *,
    walk_cap: int,
    native_precompute: bool | None = None,
    probability_caches: tuple[Any, Any] | None = None,
    comparison_probability_caches: tuple[Any, Any] | None = None,
    probability_mismatches=None,
) -> tuple[Any, Any, Any, Any, Any, int]:
    """Run the exact PyTorch oracle; every loop body has the same B/V shapes."""
    batch_size = entry["batch_size"]
    physical_drafts = int(topology.PHYSICAL_DRAFTS)
    output_capacity = int(topology.OUTPUT_PUBLISH_CAPACITY)
    path_capacity = int(topology.ACCEPTED_PATH_CAPACITY)
    fanout = int(topology.SAMPLER_MAX_FANOUT)
    if walk_cap != int(topology.WALK_CAP):
        raise RuntimeError(f"FR13 fixed32 core walk {walk_cap} != {topology.WALK_CAP}")
    if walk_cap > path_capacity or walk_cap > output_capacity:
        raise RuntimeError("FR13 fixed32 walk can overflow fixed products")
    child_table = entry["child_table"]
    child_counts = entry["child_counts"]
    if tuple(child_table.shape) != (
        batch_size,
        int(topology.PHYSICAL_ROWS),
        fanout,
    ):
        raise RuntimeError(
            f"FR13 fixed32 child-table shape drifted: {tuple(child_table.shape)}"
        )
    if tuple(child_counts.shape) != (
        batch_size,
        int(topology.PHYSICAL_ROWS),
    ):
        raise RuntimeError(
            f"FR13 fixed32 child-count shape drifted: {tuple(child_counts.shape)}"
        )
    _fr13_fixed32_device_assert(
        torch.all((child_counts >= 0) & (child_counts <= fanout)),
        "FR13 fixed32 sampler fanout overflow",
    )

    device = target_logits.device
    starts = entry["starts"]
    output_tokens = entry["output_tokens"]
    output_lens = entry["output_lens"]
    accepted_path_rows = entry["accepted_path_rows"]
    accepted_lens = entry["accepted_lens"]
    last_row = entry["last_row"]
    output_tokens.fill_(-1)
    output_lens.zero_()
    accepted_path_rows.zero_()
    accepted_lens.zero_()
    last_row.zero_()

    current = torch.full(
        (batch_size,),
        -1,
        dtype=torch.long,
        device=device,
    )
    alive = torch.ones(batch_size, dtype=torch.bool, device=device)
    request_rows = torch.arange(batch_size, device=device)
    output_trash = torch.full(
        (batch_size,),
        output_capacity - 1,
        dtype=torch.long,
        device=device,
    )
    path_trash = torch.full(
        (batch_size,),
        path_capacity - 1,
        dtype=torch.long,
        device=device,
    )
    if native_precompute is None:
        native_precompute = _fr13_fixed32_taw_native_precompute_enabled()
    if native_precompute:
        if probability_caches is None:
            probability_caches = _fr13_fixed32_taw_probability_caches(
                entry,
                target_logits,
                tree_self_logits,
                native_precompute=True,
            )
        self_prob_cache, target_prob_cache = probability_caches
    else:
        self_prob_cache, target_prob_cache = None, None
    if (comparison_probability_caches is None) != (probability_mismatches is None):
        raise RuntimeError("FR13 fixed32 probability byte gate is incomplete")
    loop_iterations = 0

    for level in range(walk_cap):
        loop_iterations += 1
        parent_slots = current + 1
        kids = child_table[request_rows, parent_slots]
        child_count = child_counts[request_rows, parent_slots]
        has_kids = alive & (child_count > 0)
        leaf = alive & (child_count == 0)

        # Full-vocab self CDF rows execute for every request at every level.
        current_valid = (current >= 0) & (current < physical_drafts)
        if self_prob_cache is None:
            self_indices = starts + current.clamp(
                min=0,
                max=physical_drafts - 1,
            )
            self_prob = torch.softmax(
                tree_self_logits[self_indices].to(torch.float32),
                dim=-1,
            )
            if comparison_probability_caches is not None:
                self_slots = entry["native_self_slot_by_node"][
                    current.clamp(min=0, max=physical_drafts - 1)
                ]
                comparison_indices = (
                    entry["native_self_request_offsets"] + self_slots
                )
                comparison_prob = comparison_probability_caches[0][
                    comparison_indices
                ]
                probability_mismatches.add_(
                    torch.count_nonzero(
                        (
                            self_prob.view(torch.int32)
                            != comparison_prob.view(torch.int32)
                        )
                        & leaf.unsqueeze(1)
                    )
                )
        else:
            self_slots = entry["native_self_slot_by_node"][
                current.clamp(min=0, max=physical_drafts - 1)
            ]
            self_indices = entry["native_self_request_offsets"] + self_slots
            self_prob = self_prob_cache[self_indices]
        self_prob = self_prob / self_prob.sum(dim=-1, keepdim=True)
        self_token = _fr13_taw_inv_cdf(
            self_prob,
            uniforms[:, level, 2],
        )
        leaf_token = torch.where(current_valid, self_token, bonus_flat)
        output_tokens.scatter_(
            1,
            torch.where(leaf, output_lens, output_trash).unsqueeze(1),
            leaf_token.unsqueeze(1),
        )
        output_lens.add_(leaf.to(output_lens.dtype))
        alive = alive & ~leaf

        # Full-vocab target/source/qmix/residual rows also execute for every
        # request at every level. The validity mask exists only in child_table.
        first_child = kids[:, 0].clamp(min=0)
        if target_prob_cache is None:
            target_indices = starts + first_child
            target_prob = torch.softmax(
                target_logits[target_indices].to(torch.float32),
                dim=-1,
            )
            if comparison_probability_caches is not None:
                target_slots = entry["native_target_slot_by_parent"][parent_slots]
                comparison_indices = (
                    entry["native_target_request_offsets"] + target_slots
                )
                comparison_prob = comparison_probability_caches[1][
                    comparison_indices
                ]
                probability_mismatches.add_(
                    torch.count_nonzero(
                        (
                            target_prob.view(torch.int32)
                            != comparison_prob.view(torch.int32)
                        )
                        & has_kids.unsqueeze(1)
                    )
                )
        else:
            target_slots = entry["native_target_slot_by_parent"][parent_slots]
            target_indices = entry["native_target_request_offsets"] + target_slots
            target_prob = target_prob_cache[target_indices]
        target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True)
        kid_tokens = drafts.gather(1, kids.clamp(min=0))
        kid_mask = kids >= 0
        overlaps = torch.gather(target_prob, 1, kid_tokens.clamp(min=0)) * kid_mask
        overlap_mass = overlaps.sum(dim=-1)
        zero_mass = has_kids & (overlap_mass <= 0)
        source = _fr13_taw_inv_cdf(
            overlaps,
            uniforms[:, level, 0],
        )
        selected_token = kid_tokens[request_rows, source]
        same_token = (kid_tokens == selected_token.unsqueeze(1)) & kid_mask
        q_mix_token = (overlaps * same_token).sum(dim=-1) / overlap_mass.clamp(
            min=1e-30
        )
        target_at_token = torch.gather(
            target_prob,
            1,
            selected_token.unsqueeze(1),
        ).squeeze(1)
        accept_probability = (target_at_token / q_mix_token.clamp(min=1e-30)).clamp(
            max=1.0
        )
        accepted = has_kids & ~zero_mass & (uniforms[:, level, 1] < accept_probability)
        rejected = has_kids & ~accepted

        weights = overlaps / overlap_mass.clamp(min=1e-30).unsqueeze(1)
        q_mix_vocab = torch.zeros_like(target_prob)
        q_mix_vocab.scatter_add_(
            1,
            kid_tokens.clamp(min=0),
            weights * kid_mask,
        )
        residual = (target_prob - q_mix_vocab).clamp(min=0)
        residual_mass = residual.sum(dim=-1, keepdim=True)
        residual = torch.where(
            residual_mass > 0,
            residual / residual_mass.clamp(min=1e-30),
            target_prob,
        )
        residual = torch.where(
            zero_mass.unsqueeze(1),
            target_prob,
            residual,
        )
        rejected_token = _fr13_taw_inv_cdf(
            residual,
            uniforms[:, level, 2],
        )
        emitted_token = torch.where(
            rejected,
            rejected_token,
            selected_token,
        )
        output_tokens.scatter_(
            1,
            torch.where(has_kids, output_lens, output_trash).unsqueeze(1),
            emitted_token.unsqueeze(1),
        )
        output_lens.add_(has_kids.to(output_lens.dtype))

        accepted_node = kids[request_rows, source]
        accepted_row = accepted_node + 1
        accepted_path_rows.scatter_(
            1,
            torch.where(
                accepted,
                accepted_lens,
                path_trash,
            ).unsqueeze(1),
            accepted_row.unsqueeze(1),
        )
        accepted_lens.add_(accepted.to(accepted_lens.dtype))
        current = torch.where(accepted, accepted_node, current)
        alive = alive & accepted

    if loop_iterations != int(topology.WALK_CAP):
        raise RuntimeError(f"FR13 fixed32 executed {loop_iterations} walk iterations")
    _fr13_fixed32_device_assert(
        torch.all(output_lens <= output_capacity),
        "FR13 fixed32 output overflow",
    )
    _fr13_fixed32_device_assert(
        torch.all(accepted_lens <= path_capacity),
        "FR13 fixed32 accepted-path overflow",
    )

    output_columns = torch.arange(
        output_capacity,
        device=device,
    ).unsqueeze(0)
    output_tokens.copy_(
        torch.where(
            output_columns < output_lens.unsqueeze(1),
            output_tokens,
            torch.full_like(output_tokens, -1),
        )
    )
    path_columns = torch.arange(
        path_capacity,
        device=device,
    ).unsqueeze(0)
    accepted_path_rows.copy_(
        torch.where(
            path_columns < accepted_lens.unsqueeze(1),
            accepted_path_rows,
            torch.zeros_like(accepted_path_rows),
        )
    )
    last_index = (accepted_lens - 1).clamp(min=0)
    last_row.copy_(
        torch.where(
            accepted_lens > 0,
            accepted_path_rows.gather(
                1,
                last_index.unsqueeze(1),
            ).squeeze(1),
            torch.zeros_like(accepted_lens),
        )
    )
    return (
        output_tokens,
        output_lens,
        accepted_path_rows,
        accepted_lens,
        last_row,
        loop_iterations,
    )


def _fr13_fixed32_taw_execute_exact_cuda(
    topology,
    entry: dict[str, Any],
    drafts,
    target_logits,
    tree_self_logits,
    bonus_flat,
    uniforms,
    *,
    walk_cap: int,
    native_precompute: bool | None = None,
    probability_caches: tuple[Any, Any] | None = None,
    comparison_probability_caches: tuple[Any, Any] | None = None,
    probability_mismatches=None,
) -> tuple[Any, Any, Any, Any, Any, int]:
    """Preserve PyTorch sampling math and fuse only integer product commits."""
    if triton is None or tl is None:
        raise RuntimeError("FR13 fixed32 exact commit requires Triton")
    if not target_logits.is_cuda:
        raise RuntimeError("FR13 fixed32 exact commit requires CUDA tensors")

    batch_size = int(entry["batch_size"])
    physical_drafts = int(topology.PHYSICAL_DRAFTS)
    physical_rows = int(topology.PHYSICAL_ROWS)
    output_capacity = int(topology.OUTPUT_PUBLISH_CAPACITY)
    path_capacity = int(topology.ACCEPTED_PATH_CAPACITY)
    fanout = int(topology.SAMPLER_MAX_FANOUT)
    if walk_cap != int(topology.WALK_CAP):
        raise RuntimeError(f"FR13 fixed32 core walk {walk_cap} != {topology.WALK_CAP}")
    if walk_cap > path_capacity or walk_cap > output_capacity:
        raise RuntimeError("FR13 fixed32 walk can overflow fixed products")

    child_table = entry["child_table"]
    child_counts = entry["child_counts"]
    if tuple(child_table.shape) != (batch_size, physical_rows, fanout):
        raise RuntimeError(
            f"FR13 fixed32 child-table shape drifted: {tuple(child_table.shape)}"
        )
    if tuple(child_counts.shape) != (batch_size, physical_rows):
        raise RuntimeError(
            f"FR13 fixed32 child-count shape drifted: {tuple(child_counts.shape)}"
        )
    _fr13_fixed32_device_assert(
        torch.all((child_counts >= 0) & (child_counts <= fanout)),
        "FR13 fixed32 sampler fanout overflow",
    )

    expected_state = {
        "request_rows": ((batch_size,), torch.long),
        "exact_initial_current": ((batch_size,), torch.long),
        "exact_initial_alive": ((batch_size,), torch.bool),
        "exact_current": ((batch_size,), torch.long),
        "exact_alive": ((batch_size,), torch.bool),
    }
    for name, (shape, dtype) in expected_state.items():
        value = entry.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.dtype != dtype
            or value.device != target_logits.device
            or not value.is_contiguous()
        ):
            raise RuntimeError(f"FR13 fixed32 exact commit state drift: {name}")

    starts = entry["starts"]
    request_rows = entry["request_rows"]
    initial_current = entry["exact_initial_current"]
    initial_alive = entry["exact_initial_alive"]
    current_state = entry["exact_current"]
    alive_state = entry["exact_alive"]
    output_tokens = entry["output_tokens"]
    output_lens = entry["output_lens"]
    accepted_path_rows = entry["accepted_path_rows"]
    accepted_lens = entry["accepted_lens"]
    last_row = entry["last_row"]
    if native_precompute is None:
        native_precompute = _fr13_fixed32_taw_native_precompute_enabled()
    if native_precompute:
        if probability_caches is None:
            probability_caches = _fr13_fixed32_taw_probability_caches(
                entry,
                target_logits,
                tree_self_logits,
                native_precompute=True,
            )
        self_prob_cache, target_prob_cache = probability_caches
    else:
        self_prob_cache, target_prob_cache = None, None
    if (comparison_probability_caches is None) != (probability_mismatches is None):
        raise RuntimeError("FR13 fixed32 probability byte gate is incomplete")

    for level in range(walk_cap):
        current = initial_current if level == 0 else current_state
        alive = initial_alive if level == 0 else alive_state
        parent_slots = current + 1
        kids = child_table[request_rows, parent_slots]
        child_count = child_counts[request_rows, parent_slots]
        has_kids = alive & (child_count > 0)
        leaf = alive & (child_count == 0)

        # This floating path is intentionally identical to the PyTorch oracle.
        if self_prob_cache is None:
            self_indices = starts + current.clamp(
                min=0,
                max=physical_drafts - 1,
            )
            self_prob = torch.softmax(
                tree_self_logits[self_indices].to(torch.float32),
                dim=-1,
            )
            if comparison_probability_caches is not None:
                self_slots = entry["native_self_slot_by_node"][
                    current.clamp(min=0, max=physical_drafts - 1)
                ]
                comparison_indices = (
                    entry["native_self_request_offsets"] + self_slots
                )
                comparison_prob = comparison_probability_caches[0][
                    comparison_indices
                ]
                probability_mismatches.add_(
                    torch.count_nonzero(
                        (
                            self_prob.view(torch.int32)
                            != comparison_prob.view(torch.int32)
                        )
                        & leaf.unsqueeze(1)
                    )
                )
        else:
            self_slots = entry["native_self_slot_by_node"][
                current.clamp(min=0, max=physical_drafts - 1)
            ]
            self_indices = entry["native_self_request_offsets"] + self_slots
            self_prob = self_prob_cache[self_indices]
        self_prob = self_prob / self_prob.sum(dim=-1, keepdim=True)
        self_token = _fr13_taw_inv_cdf(
            self_prob,
            uniforms[:, level, 2],
        )

        first_child = kids[:, 0].clamp(min=0)
        if target_prob_cache is None:
            target_indices = starts + first_child
            target_prob = torch.softmax(
                target_logits[target_indices].to(torch.float32),
                dim=-1,
            )
            if comparison_probability_caches is not None:
                target_slots = entry["native_target_slot_by_parent"][parent_slots]
                comparison_indices = (
                    entry["native_target_request_offsets"] + target_slots
                )
                comparison_prob = comparison_probability_caches[1][
                    comparison_indices
                ]
                probability_mismatches.add_(
                    torch.count_nonzero(
                        (
                            target_prob.view(torch.int32)
                            != comparison_prob.view(torch.int32)
                        )
                        & has_kids.unsqueeze(1)
                    )
                )
        else:
            target_slots = entry["native_target_slot_by_parent"][parent_slots]
            target_indices = entry["native_target_request_offsets"] + target_slots
            target_prob = target_prob_cache[target_indices]
        target_prob = target_prob / target_prob.sum(dim=-1, keepdim=True)
        kid_tokens = drafts.gather(1, kids.clamp(min=0))
        kid_mask = kids >= 0
        overlaps = torch.gather(target_prob, 1, kid_tokens.clamp(min=0)) * kid_mask
        overlap_mass = overlaps.sum(dim=-1)
        zero_mass = has_kids & (overlap_mass <= 0)
        source = _fr13_taw_inv_cdf(
            overlaps,
            uniforms[:, level, 0],
        )
        selected_token = kid_tokens[request_rows, source]
        same_token = (kid_tokens == selected_token.unsqueeze(1)) & kid_mask
        q_mix_token = (overlaps * same_token).sum(dim=-1) / overlap_mass.clamp(
            min=1e-30
        )
        target_at_token = torch.gather(
            target_prob,
            1,
            selected_token.unsqueeze(1),
        ).squeeze(1)
        accept_probability = (target_at_token / q_mix_token.clamp(min=1e-30)).clamp(
            max=1.0
        )
        accepted = has_kids & ~zero_mass & (uniforms[:, level, 1] < accept_probability)

        weights = overlaps / overlap_mass.clamp(min=1e-30).unsqueeze(1)
        q_mix_vocab = torch.zeros_like(target_prob)
        q_mix_vocab.scatter_add_(
            1,
            kid_tokens.clamp(min=0),
            weights * kid_mask,
        )
        residual = (target_prob - q_mix_vocab).clamp(min=0)
        residual_mass = residual.sum(dim=-1, keepdim=True)
        residual = torch.where(
            residual_mass > 0,
            residual / residual_mass.clamp(min=1e-30),
            target_prob,
        )
        residual = torch.where(
            zero_mass.unsqueeze(1),
            target_prob,
            residual,
        )
        rejected_token = _fr13_taw_inv_cdf(
            residual,
            uniforms[:, level, 2],
        )

        _fr13_fixed32_taw_exact_commit_kernel[(batch_size,)](
            child_table,
            child_counts,
            current,
            alive,
            self_token,
            bonus_flat,
            source,
            selected_token,
            rejected_token,
            accepted,
            current_state,
            alive_state,
            output_tokens,
            output_lens,
            accepted_path_rows,
            accepted_lens,
            last_row,
            LEVEL=level,
            PHYSICAL_DRAFTS=physical_drafts,
            PHYSICAL_ROWS=physical_rows,
            FANOUT=fanout,
            OUTPUT_CAPACITY=output_capacity,
            PATH_CAPACITY=path_capacity,
            num_warps=1,
        )

    _fr13_fixed32_device_assert(
        torch.all(output_lens <= output_capacity),
        "FR13 fixed32 output overflow",
    )
    _fr13_fixed32_device_assert(
        torch.all(accepted_lens <= path_capacity),
        "FR13 fixed32 accepted-path overflow",
    )
    return (
        output_tokens,
        output_lens,
        accepted_path_rows,
        accepted_lens,
        last_row,
        walk_cap,
    )


def _fr13_fixed32_taw_execute(
    topology,
    entry: dict[str, Any],
    drafts,
    target_logits,
    tree_self_logits,
    bonus_flat,
    uniforms,
    *,
    walk_cap: int,
    native_precompute: bool | None = None,
    probability_caches: tuple[Any, Any] | None = None,
    comparison_probability_caches: tuple[Any, Any] | None = None,
    probability_mismatches=None,
) -> tuple[Any, Any, Any, Any, Any, int]:
    """Dispatch CUDA to exact integer commit fusion and retain the CPU oracle."""
    if target_logits.is_cuda:
        return _fr13_fixed32_taw_execute_exact_cuda(
            topology,
            entry,
            drafts,
            target_logits,
            tree_self_logits,
            bonus_flat,
            uniforms,
            walk_cap=walk_cap,
            native_precompute=native_precompute,
            probability_caches=probability_caches,
            comparison_probability_caches=comparison_probability_caches,
            probability_mismatches=probability_mismatches,
        )
    return _fr13_fixed32_taw_execute_torch(
        topology,
        entry,
        drafts,
        target_logits,
        tree_self_logits,
        bonus_flat,
        uniforms,
        walk_cap=walk_cap,
        native_precompute=native_precompute,
        probability_caches=probability_caches,
        comparison_probability_caches=comparison_probability_caches,
        probability_mismatches=probability_mismatches,
    )


def _fr13_fixed32_publish_work(
    topology,
    *,
    mode: str,
    valid_mask: int,
    batch_size: int,
    loop_iterations: int,
    source_contract: dict[str, Any],
    layout_contract: dict[str, Any],
) -> None:
    native_precompute = _fr13_fixed32_taw_native_precompute_enabled()
    work_multiplier = 2 if native_precompute else 1
    taw = {
        "route": (
            "fixed32_native_precompute_byte_ab_reference_return"
            if native_precompute
            else "fixed32_pytorch_exact_float_triton_integer_commit"
        ),
        "preseeded_batches": list(_FR13_FIXED32_BATCHES),
        "topology_cache_hit": True,
        "cache_misses": 0,
        "table_shape": [
            batch_size,
            int(topology.PHYSICAL_ROWS),
            int(topology.SAMPLER_MAX_FANOUT),
        ],
        "buffer_capacity": int(topology.OUTPUT_PUBLISH_CAPACITY),
        "loop_iterations": int(loop_iterations),
        "uniform_slots": int(topology.TAW_UNIFORM_SLOTS) * batch_size,
        "child_lanes": (
            int(topology.TAW_CHILD_LANES) * batch_size * work_multiplier
        ),
        "target_rows": int(topology.WALK_CAP) * batch_size * work_multiplier,
        "self_rows": int(topology.WALK_CAP) * batch_size * work_multiplier,
        "self_cdf_rows": int(topology.WALK_CAP) * batch_size * work_multiplier,
        "source_cdf_rows": int(topology.WALK_CAP) * batch_size * work_multiplier,
        "residual_cdf_rows": int(topology.WALK_CAP) * batch_size * work_multiplier,
        "qmix_rows": int(topology.WALK_CAP) * batch_size * work_multiplier,
        "residual_rows": int(topology.WALK_CAP) * batch_size * work_multiplier,
        "row_scatter_slots": (
            int(topology.TAW_ROW_SCATTER_SLOTS) * batch_size * work_multiplier
        ),
        "path_scatter_slots": (
            int(topology.TAW_PATH_SCATTER_SLOTS) * batch_size * work_multiplier
        ),
        "exact_commit_launches": int(topology.WALK_CAP) * work_multiplier,
        "exact_commit_programs": (
            int(topology.WALK_CAP) * batch_size * work_multiplier
        ),
        "floating_sampling_reimplementation": False,
    }
    if (
        set(source_contract)
        != {
            "source_contract_schema",
            "source_contract_sha256",
            "tensor_call_census",
        }
        or source_contract["source_contract_schema"]
        != _FR13_FIXED32_TAW_SOURCE_SCHEMA
        or source_contract["source_contract_sha256"]
        != _FR13_FIXED32_TAW_SOURCE_SHA256
        or source_contract["tensor_call_census"]
        != _fr13_fixed32_taw_tensor_call_census()
    ):
        raise RuntimeError("FR13 fixed32 TAW source contract drift at publish")
    overlap = (
        set(taw).intersection(source_contract)
        | set(taw).intersection(layout_contract)
        | set(source_contract).intersection(layout_contract)
    )
    if overlap:
        raise RuntimeError(
            "FR13 fixed32 TAW work payload keys overlap: " + repr(sorted(overlap))
        )
    taw.update(source_contract)
    taw.update(layout_contract)
    payload = {
        "mode": mode,
        "valid_mask": int(valid_mask),
        "batch_size": int(batch_size),
        "taw": taw,
    }
    global _FR13_FIXED32_TAW_LAST_WORK
    _FR13_FIXED32_TAW_LAST_WORK = payload
    callback = _FR13_FIXED32_TAW_WORK_CALLBACK
    if callback is None:
        if os.environ.get("FR13_FIXED32_WORK_CENSUS") == "1":
            raise RuntimeError(
                "FR13 fixed32 work census is armed without a TAW callback"
            )
        return
    callback(payload)


def _fr13_fixed32_announce(topology) -> None:
    global _FR13_FIXED32_WORK_ANNOUNCED
    if _FR13_FIXED32_WORK_ANNOUNCED:
        return
    _FR13_FIXED32_WORK_ANNOUNCED = True
    print(
        "[FR13_FIXED32_WORK] engaged: physical_drafts=31 rows=32 "
        f"gdn_launches={topology.GDN_LAUNCHES} "
        f"gdn_programs={topology.GDN_PATH_PROGRAMS} "
        f"gdn_slots={topology.GDN_PADDED_SLOTS} "
        f"taw_walk={topology.WALK_CAP} "
        f"taw_buffer={topology.OUTPUT_PUBLISH_CAPACITY} "
        f"output_slots={topology.OUTPUT_PUBLISH_CAPACITY} "
        f"path_slots={topology.ACCEPTED_PATH_CAPACITY} "
        f"reqkey_slots={topology.REQUEST_KEY_PATH_CAPACITY} "
        f"kv_slots={topology.KV_REMAP_PATH_CAPACITY} "
        f"conv_layers={topology.CONV_COMMIT_LAYERS} "
        f"committer_slots={topology.COMMIT_PATH_CAP}",
        flush=True,
    )


def fr13_fixed32_taw_commit(
    num_draft_tokens,
    draft_token_ids,
    tree_parent_indices,
    target_logits,
    tree_self_logits,
    bonus_token_ids,
    max_spec_len: int,
    *,
    generators=None,
    uniforms=None,
    all_greedy: bool = False,
    mode: str | None = None,
):
    """Run the fail-closed fixed-32 TAW and return five device tensors.

    Returns ``(output_tokens, output_lens, accepted_path_rows, accepted_lens,
    last_row)``. Output is int64 ``[B,32]`` padded with -1. Accepted paths are
    root-inclusive GDN row ids (draft-local node + 1), int64 ``[B,16]`` padded
    with zero. No scalar or tensor value is materialized on the host.
    """
    if torch is None:
        raise RuntimeError("FR13 fixed32 TAW requires torch")
    if mode is None:
        mode = os.environ.get("FR13_FIXED32_MODE", "")
    topology, valid_mask = _fr13_fixed32_runtime_contract(mode)
    source_contract = _fr13_fixed32_taw_source_contract(topology)
    if all_greedy:
        raise RuntimeError(
            "FR13 fixed32 acceptance route requires sampled temp>0 requests"
        )
    if target_logits is None or tree_self_logits is None:
        raise RuntimeError("FR13 fixed32 TAW requires target and self logits")
    if not isinstance(target_logits, torch.Tensor):
        raise RuntimeError("FR13 fixed32 target_logits must be a tensor")
    if not isinstance(num_draft_tokens, torch.Tensor):
        raise RuntimeError("FR13 fixed32 num_draft_tokens must be a tensor")
    batch_size = int(num_draft_tokens.shape[0])
    if batch_size not in _FR13_FIXED32_BATCHES:
        raise RuntimeError(f"FR13 fixed32 batch must be 1..4, got {batch_size}")
    key = fr13_fixed32_taw_cache_key(
        mode,
        valid_mask,
        batch_size,
        target_logits.device,
    )
    entry = _FR13_FIXED32_TAW_CACHE.get(key)
    if entry is None:
        raise RuntimeError(
            "FR13 fixed32 topology cache miss; call "
            "fr13_fixed32_taw_preseed(device, mode=...) before serving"
        )
    drafts, bonus_flat = _fr13_fixed32_validate_inputs(
        topology,
        entry,
        num_draft_tokens,
        draft_token_ids,
        tree_parent_indices,
        target_logits,
        tree_self_logits,
        bonus_token_ids,
        max_spec_len,
    )
    fixed_uniforms, rng_route = _fr13_fixed32_fill_uniforms(
        entry,
        generators=generators,
        uniforms=uniforms,
    )
    layout_contract = _fr13_fixed32_layout_contract(
        topology,
        entry,
        num_draft_tokens,
        draft_token_ids,
        tree_parent_indices,
        target_logits,
        tree_self_logits,
        bonus_token_ids,
        fixed_uniforms,
        rng_route=rng_route,
    )
    native_precompute = _fr13_fixed32_taw_native_precompute_enabled()
    if native_precompute:
        probability_mismatches = entry["native_ab_probability_mismatches"]
        product_mismatches = entry["native_ab_product_mismatches"]
        if target_logits.is_cuda and not torch.cuda.is_current_stream_capturing():
            root_checks = int(entry["native_ab_root_checks"])
            if root_checks:
                probability_bad = int(probability_mismatches.item())
                product_bad = int(product_mismatches.item())
                if probability_bad or product_bad:
                    raise AssertionError(
                        "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE byte mismatch: "
                        f"probabilities={probability_bad} products={product_bad}"
                    )
                if root_checks % 128 == 0:
                    print(
                        "[FR13_FIXED32_TAW_NATIVE_PRECOMPUTE] PASS "
                        f"root_checks={root_checks} probability_mismatches=0 "
                        "product_mismatches=0 reference_returned=1",
                        flush=True,
                    )
            entry["native_ab_root_checks"] = root_checks + 1
        probability_caches = _fr13_fixed32_taw_probability_caches(
            entry,
            target_logits,
            tree_self_logits,
            native_precompute=True,
        )
        reference = _fr13_fixed32_taw_execute(
            topology,
            entry,
            drafts,
            target_logits,
            tree_self_logits,
            bonus_flat,
            fixed_uniforms,
            walk_cap=int(topology.WALK_CAP),
            native_precompute=False,
            comparison_probability_caches=probability_caches,
            probability_mismatches=probability_mismatches,
        )
        candidate = _fr13_fixed32_taw_execute(
            topology,
            entry["native_ab_entry"],
            drafts,
            target_logits,
            tree_self_logits,
            bonus_flat,
            fixed_uniforms,
            walk_cap=int(topology.WALK_CAP),
            native_precompute=True,
            probability_caches=probability_caches,
        )
        for reference_product, candidate_product in zip(
            reference[:5],
            candidate[:5],
            strict=True,
        ):
            product_mismatches.add_(
                torch.count_nonzero(reference_product != candidate_product)
            )
        (
            output_tokens,
            output_lens,
            accepted_path_rows,
            accepted_lens,
            last_row,
            loop_iterations,
        ) = reference
    else:
        (
            output_tokens,
            output_lens,
            accepted_path_rows,
            accepted_lens,
            last_row,
            loop_iterations,
        ) = _fr13_fixed32_taw_execute(
            topology,
            entry,
            drafts,
            target_logits,
            tree_self_logits,
            bonus_flat,
            fixed_uniforms,
            walk_cap=int(topology.WALK_CAP),
            native_precompute=False,
        )
    _fr13_fixed32_publish_work(
        topology,
        mode=mode,
        valid_mask=valid_mask,
        batch_size=batch_size,
        loop_iterations=loop_iterations,
        source_contract=source_contract,
        layout_contract=layout_contract,
    )
    _fr13_fixed32_announce(topology)
    return (
        output_tokens,
        output_lens,
        accepted_path_rows,
        accepted_lens,
        last_row,
    )


# ---------------------------------------------------------------------------
# FR13_TAW (S1): fully-tensorized tree-accept walk — zero per-node readbacks.
# Distribution-equal to the legacy/depthsync walk (NOT byte-equal: draws are
# pre-drawn uniforms consumed by inverse-CDF, so the rng stream differs), the
# same standard used when device-multidraft replaced the numpy host walk.
# Capture-legal core: after topology prep, the walk is ~(max_spec_len+1)
# fixed iterations of batched tensor ops with NO .item()/.tolist(). The
# product materialization shim at the end does ONE batched DtoH — the S1
# wrapper phase moves consumers to the tensor products and drops the shim.
# ---------------------------------------------------------------------------
_FR13_TAW_TOPO_CACHE: dict = {}
_FR13_TAW_ANNOUNCED = False
_FR13_BULK_GEN = None


def _fr13_pin_uniforms():
    """FR13_SG_PIN_UNIFORMS=1 (selfcheck-only): pin uniform draws to 0.5 so
    accept/recovery are deterministic functions of logits alone — makes the
    eager-vs-replay pair byte-comparable (RNG sequencing differs by design)."""
    import os
    return os.environ.get("FR13_SG_PIN_UNIFORMS", "0") == "1"


def _fr13_bulk_gen(device):
    """POISON-IMMUNE bulk RNG (twin of the topk_topp port): a silently-
    aborted capture leaves the DEFAULT philox generator graph-registered and
    every default-gen draw dies ('Offset increment outside graph capture' —
    the boot-14 post-DISABLE death was THIS module's uniform fallback).
    Explicit generators survive; the unseeded bulk draw was never
    reproducible, so this is distribution-equal."""
    global _FR13_BULK_GEN
    if _FR13_BULK_GEN is None:
        g = torch.Generator(device=device)
        g.manual_seed(torch.initial_seed() & 0x7FFFFFFF)
        _FR13_BULK_GEN = g
    return _FR13_BULK_GEN


def _fr13_taw_topology(parents_key, parents_cpu, counts, max_spec_len, device):
    """Per-shape topology tables (STEP-CONSTANT for the fixed tree): for each
    request-local parent node id (-1=root), the padded child-node table and
    counts. Cached by (parents tuple, counts tuple)."""
    key = (parents_key, tuple(counts), int(max_spec_len))
    hit = _FR13_TAW_TOPO_CACHE.get(key)
    if hit is not None:
        return hit
    nreq = len(counts)
    nmax = max(int(c) for c in counts) if counts else 0
    wmax = 1
    child_lists: list = []
    starts = []
    s = 0
    for c in counts:
        starts.append(s)
        s += int(c)
    for req_i in range(nreq):
        st, n = starts[req_i], int(counts[req_i])
        per_parent = {}
        for node in range(n):
            par = int(parents_cpu[st + node])
            per_parent.setdefault(par, []).append(node)
        child_lists.append(per_parent)
        for v in per_parent.values():
            wmax = max(wmax, len(v))
    # tables indexed [req, parent+1] -> padded child node ids / count
    # (parent -1 = root maps to slot 0)
    ctab = torch.full((nreq, nmax + 1, wmax), -1, dtype=torch.long)
    ccnt = torch.zeros((nreq, nmax + 1), dtype=torch.long)
    for req_i in range(nreq):
        for par, kids in child_lists[req_i].items():
            ctab[req_i, par + 1, : len(kids)] = torch.tensor(kids)
            ccnt[req_i, par + 1] = len(kids)
    out = (
        ctab.to(device),
        ccnt.to(device),
        torch.tensor(starts, dtype=torch.long, device=device),
        wmax,
        torch.tensor(counts, dtype=torch.long, device=device),
    )
    _FR13_TAW_TOPO_CACHE[key] = out
    return out


def _fr13_taw_inv_cdf(weights, u):
    """source = inverse-CDF draw: first index where cumsum(weights) > u*mass.
    weights [B, W] (>=0, rows may be all-zero), u [B] in [0,1)."""
    cs = torch.cumsum(weights, dim=-1)
    total = cs[:, -1:]
    thresh = u.unsqueeze(-1).to(weights.dtype) * total
    return (cs <= thresh).sum(dim=-1).clamp(max=weights.shape[-1] - 1)


def fr13_taw_commit(
    num_draft_tokens,
    draft_token_ids,
    tree_parent_indices,
    target_logits,
    tree_self_logits,
    bonus_token_ids,
    max_spec_len: int,
    *,
    generators=None,
    uniforms=None,
    defer_materialize=False,
):
    """FR13_TAW: batched zero-readback walk. Returns the SAME products as
    fr13_device_multidraft_commit via a single materialization DtoH.
    defer_materialize=True (S1 capture mode): returns the DEVICE tensors
    (row_buf, row_len, path_buf, path_len) with NO DtoH — the S1 wrapper
    materializes post-replay. Zero syncs inside => capture-legal."""
    global _FR13_TAW_ANNOUNCED
    if not _FR13_TAW_ANNOUNCED:
        _FR13_TAW_ANNOUNCED = True
        print("[FR13_TAW] ENGAGED: tensorized tree-accept walk "
              "(zero per-node readbacks; distribution-equal gate)", flush=True)
    device = target_logits.device
    global _FR13_SG_TOPOLOGY
    if not defer_materialize and torch.cuda.is_available():
        defer_materialize = bool(
            torch.cuda.is_current_stream_capturing() or _FR13_SG_FORCE_DEFER)
    if defer_materialize and _FR13_SG_TOPOLOGY is not None:
        # S1 capture mode: topology pre-read OUTSIDE the capture (step-constant)
        parents_cpu, counts = _FR13_SG_TOPOLOGY
    elif defer_materialize:
        raise RuntimeError("FR13_TAW capture without cached topology (warmup step missing)")
    else:
        parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
        if hasattr(num_draft_tokens, "detach"):
            counts = [int(x) for x in num_draft_tokens.detach().cpu().tolist()]
        else:
            counts = [int(x) for x in num_draft_tokens]
        _FR13_SG_TOPOLOGY = (parents_cpu, counts)  # eager call caches for capture
    nreq = len(counts)
    row_cap = int(max_spec_len) + 1
    ctab, ccnt, starts_t, wmax, counts_t = _fr13_taw_topology(
        tuple(parents_cpu), parents_cpu, counts, max_spec_len, device
    )
    drafts_t = draft_token_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
    # S1 handoff: the STEP_GRAPH wrapper pre-fills _FR13_SG_UNIFORMS outside
    # the capture; consume it here so the captured region has zero seeding
    # syncs. (Shape must match; wrapper owns refill per replay.)
    if uniforms is None and _FR13_SG_UNIFORMS is not None and _FR13_SG_UNIFORMS.shape == (nreq, row_cap, 3):
        uniforms = _FR13_SG_UNIFORMS
    # pre-drawn uniforms [B, row_cap, 3]: (u_source, u_accept, u_residual).
    # Per-request determinism: seed a DEVICE generator from each host
    # generator (depthsync's proven pattern) — never silently fall to the
    # global stream on device mismatch (the tawcg live-FAIL root cause
    # candidate: cuda device + cpu gens dropped per-req seeding entirely).
    if uniforms is None:
        uniforms = torch.empty(nreq, row_cap, 3, device=device)
        if generators:
            for req_i in range(nreq):
                g = generators.get(req_i)
                if g is None:
                    uniforms[req_i] = torch.rand(
                        row_cap, 3, device=device,
                        generator=_fr13_bulk_gen(device))
                elif g.device.type == device.type:
                    uniforms[req_i] = torch.rand(
                        row_cap, 3, device=device, generator=g
                    )
                else:
                    dev_gen = torch.Generator(device=device)
                    seed_t = torch.randint(
                        0, 2 ** 31 - 1, (1,), device=g.device, generator=g
                    )
                    dev_gen.manual_seed(int(seed_t.item()))
                    uniforms[req_i] = torch.rand(
                        row_cap, 3, device=device, generator=dev_gen
                    )
        else:
            uniforms.uniform_(generator=_fr13_bulk_gen(device))
    if _fr13_pin_uniforms():
        uniforms.fill_(0.5)

    cur = torch.full((nreq,), -1, dtype=torch.long, device=device)  # parent node
    alive = torch.ones(nreq, dtype=torch.bool, device=device)
    # +1 trash column: capture-legal masked writes via scatter_ (no
    # boolean-mask indexing => no nonzero host sync inside the capture)
    row_buf = torch.full((nreq, row_cap + 1), -1, dtype=torch.long, device=device)
    row_len = torch.zeros(nreq, dtype=torch.long, device=device)
    _trash = torch.full((nreq,), row_cap, dtype=torch.long, device=device)
    path_buf = torch.full((nreq, row_cap + 1), -1, dtype=torch.long, device=device)
    path_len = torch.zeros(nreq, dtype=torch.long, device=device)
    bonus_flat = bonus_token_ids.reshape(-1).to(device=device, dtype=torch.long)
    ar = torch.arange(nreq, device=device)

    for level in range(row_cap):
        kids = ctab[ar, cur + 1]              # [B, W]
        nk = ccnt[ar, cur + 1]                # [B]
        has_kids = (nk > 0) & alive
        # ---- leaf/bonus rows (alive, no children): self-row sample or bonus id
        leaf = alive & ~ (nk > 0)
        if True:  # capture-legal: leaf block runs unconditionally (mask-internal)
            # careful host-free: compute for ALL rows, apply masked
            cp_ok = (cur >= 0) & (cur < counts_t)
            self_rows = torch.softmax(
                tree_self_logits[(starts_t + cur.clamp(min=0)).clamp(max=tree_self_logits.shape[0] - 1)].to(torch.float32),
                dim=-1,
            )
            tok_self = _fr13_taw_inv_cdf(self_rows, uniforms[:, level, 2])
            tok_bonus = bonus_flat[ar.clamp(max=bonus_flat.numel() - 1)]
            tok_leaf = torch.where(cp_ok, tok_self, tok_bonus)
            emit = leaf
            row_buf.scatter_(1, torch.where(emit, row_len, _trash).unsqueeze(1),
                             tok_leaf.unsqueeze(1))
            row_len = row_len + emit.long()
            alive = alive & ~leaf
        # capture-legal: fixed-depth loop, no data-dependent break
        # ---- parent target row: legacy uses target_logits[start+children[0]]
        first_child = kids[:, 0].clamp(min=0)
        p = torch.softmax(
            target_logits[(starts_t + first_child).clamp(max=target_logits.shape[0] - 1)].to(torch.float32),
            dim=-1,
        )                                      # [B, V]
        p = p / p.sum(dim=-1, keepdim=True)
        kid_tokens = drafts_t[(starts_t.unsqueeze(1) + kids.clamp(min=0)).clamp(max=drafts_t.numel() - 1)]  # [B, W]
        kid_mask = kids >= 0
        overlaps = torch.gather(p, 1, kid_tokens.clamp(min=0)) * kid_mask
        mass = overlaps.sum(dim=-1)            # [B]
        zero_mass = has_kids & (mass <= 0)
        # source draw (inverse-CDF over overlaps)
        src = _fr13_taw_inv_cdf(overlaps, uniforms[:, level, 0])  # [B]
        tok = kid_tokens[ar, src]              # [B]
        # accept prob = min(1, p[tok] / q_mix_tok); q_mix = weight mass of
        # children sharing tok (weights = overlaps / mass)
        same = (kid_tokens == tok.unsqueeze(1)) & kid_mask
        q_mix_tok = (overlaps * same).sum(dim=-1) / mass.clamp(min=1e-30)
        p_tok = torch.gather(p, 1, tok.unsqueeze(1)).squeeze(1)
        acc_p = (p_tok / q_mix_tok.clamp(min=1e-30)).clamp(max=1.0)
        accepted = has_kids & ~zero_mass & (uniforms[:, level, 1] < acc_p)
        # rejected rows: residual sample over full vocab (or plain p on zero mass)
        rejected = has_kids & ~accepted
        if True:  # capture-legal: rejected block runs unconditionally (mask-internal)
            weights = overlaps / mass.clamp(min=1e-30).unsqueeze(-1)
            q_mix_v = torch.zeros_like(p)
            q_mix_v.scatter_add_(1, kid_tokens.clamp(min=0), weights * kid_mask)
            residual = (p - q_mix_v).clamp(min=0)
            rmass = residual.sum(dim=-1, keepdim=True)
            residual = torch.where(rmass > 0, residual / rmass.clamp(min=1e-30), p)
            residual = torch.where(zero_mass.unsqueeze(1), p, residual)
            tok_rej = _fr13_taw_inv_cdf(residual, uniforms[:, level, 2])
            tok = torch.where(rejected, tok_rej, tok)
        # emit token for every has_kids row; record path node for accepted
        row_buf.scatter_(1, torch.where(has_kids, row_len, _trash).unsqueeze(1),
                         tok.unsqueeze(1))
        row_len = row_len + has_kids.long()
        acc_node = kids[ar, src]
        path_buf.scatter_(1, torch.where(accepted, path_len, _trash).unsqueeze(1),
                          acc_node.unsqueeze(1))
        path_len = path_len + accepted.long()
        cur = torch.where(accepted, acc_node, cur)
        alive = alive & accepted

    if defer_materialize:
        # S1 capture mode: everything above is pure tensor ops (zero syncs).
        return row_buf[:, :row_cap], row_len, path_buf[:, :row_cap], path_len

    # ---- materialization shim (ONE batched DtoH; wrapper phase removes this)
    row_buf = row_buf[:, :row_cap]
    path_buf = path_buf[:, :row_cap]
    rb = row_buf.cpu().tolist()
    rl = row_len.cpu().tolist()
    pb = path_buf.cpu().tolist()
    pl = path_len.cpu().tolist()
    out_rows = [r[: int(length)] for r, length in zip(rb, rl)]
    accepted_node_paths = [
        pth[: int(length)] for pth, length in zip(pb, pl)
    ]
    accepted_lens = [int(length) for length in pl]
    accepted_rows = accepted_lens  # legacy alias shape (per-req accepted count)
    accepted_token_rows = [
        r[: int(length)] for r, length in zip(rb, pl)
    ]
    return out_rows, accepted_rows, accepted_lens, accepted_node_paths, accepted_token_rows


def fr13_taw_products_device(row_buf, row_len, path_buf, path_len,
                             output_token_ids, accepted_tree_rows):
    """S1-full (=2) in-capture product consumption: fill the sampler output
    tensors ON DEVICE from the TAW defer products, replacing the host-list
    committer glue (out_rows python loops / .tolist()) that cannot run inside
    a capture. Byte contract vs the host route is gated CPU-side
    (scripts/fr13_taw_products_device_byte_gate.py).

    Returns (gdn_paths, gdn_rows): the +1-shifted 0-padded accepted node paths
    [nreq, k] and per-request last-path-node rows [nreq] the CG committer body
    consumes (host route: _gdn_path = [node+1 ...], row = last or 0)."""
    nreq, cols = output_token_ids.shape
    k = min(cols, row_buf.shape[1])
    ar = torch.arange(k, device=row_buf.device)
    mask = ar.unsqueeze(0) < row_len.unsqueeze(1)
    output_token_ids.fill_(-1)
    output_token_ids[:, :k] = torch.where(
        mask, row_buf[:, :k], torch.full_like(row_buf[:, :k], -1))
    accepted_tree_rows.copy_(path_len)
    pmask = ar.unsqueeze(0) < path_len.unsqueeze(1)
    gdn_paths = torch.where(
        pmask, path_buf[:, :k] + 1, torch.zeros_like(path_buf[:, :k]))
    last_idx = (path_len - 1).clamp(min=0)
    gdn_rows = torch.where(
        path_len > 0,
        gdn_paths.gather(1, last_idx.unsqueeze(1)).squeeze(1),
        torch.zeros_like(path_len))
    return gdn_paths, gdn_rows


def fr13_taw_materialize(row_buf, row_len, path_buf, path_len):
    """Post-replay materialization of TAW device products into the legacy
    host-list contract (ONE batched DtoH). Used by the S1 wrapper."""
    rb = row_buf.cpu().tolist()
    rl = row_len.cpu().tolist()
    pb = path_buf.cpu().tolist()
    pl = path_len.cpu().tolist()
    out_rows = [r[: int(length)] for r, length in zip(rb, rl)]
    accepted_node_paths = [
        pth[: int(length)] for pth, length in zip(pb, pl)
    ]
    accepted_lens = [int(length) for length in pl]
    accepted_rows = accepted_lens
    accepted_token_rows = [
        r[: int(length)] for r, length in zip(rb, pl)
    ]
    return out_rows, accepted_rows, accepted_lens, accepted_node_paths, accepted_token_rows


# ---- S1 wrapper handoff: pre-drawn uniforms static (set by the STEP_GRAPH
# wrapper OUTSIDE the capture; TAW consumes it INSIDE, keeping the captured
# region free of generator seeding .item() syncs). None => TAW draws its own.
_FR13_SG_UNIFORMS = None


def fr13_sg_set_uniforms(u):
    global _FR13_SG_UNIFORMS
    _FR13_SG_UNIFORMS = u


def fr13_sg_fill_uniforms(nreq, row_cap, device, generators=None):
    """Draw the per-step uniforms OUTSIDE the graph into (or as) the static.
    Returns the tensor (callers keep it static and refill per replay)."""
    global _FR13_SG_UNIFORMS
    if (_FR13_SG_UNIFORMS is None
            or _FR13_SG_UNIFORMS.shape != (nreq, row_cap, 3)):
        _FR13_SG_UNIFORMS = torch.empty(nreq, row_cap, 3, device=device)
    u = _FR13_SG_UNIFORMS
    if generators:
        for req_i in range(nreq):
            g = generators.get(req_i)
            if g is None:
                # poison-immune (boot-24: an aborted capture leaves the
                # DEFAULT philox in captured-offset state; the retry's
                # pre-capture draw here then raised "Offset increment
                # outside graph capture" and burned attempt 2 for free)
                u[req_i] = torch.rand(row_cap, 3, device=device,
                                      generator=_fr13_bulk_gen(device))
            elif g.device.type == device.type if hasattr(device, "type") else str(g.device) == str(device):
                u[req_i] = torch.rand(row_cap, 3, device=device, generator=g)
            else:
                dev_gen = torch.Generator(device=device)
                seed_t = torch.randint(0, 2 ** 31 - 1, (1,), device=g.device, generator=g)
                dev_gen.manual_seed(int(seed_t.item()))
                u[req_i] = torch.rand(row_cap, 3, device=device, generator=dev_gen)
    else:
        u.uniform_(generator=_fr13_bulk_gen(u.device))
    if _fr13_pin_uniforms():
        u.fill_(0.5)
    return u


# S1 topology handoff (step-constant; wrapper pre-reads OUTSIDE the capture)
_FR13_SG_TOPOLOGY = None
# S1-full (=2): replay-order permutation static (sampler-row -> spec-row
# order), wrapper-owned per key, refilled pre-replay (never baked stale)
_FR13_SG_PERM = None
# S1-full (=2) pre-capture warmup: forces the defer/device route on an EAGER
# side-stream run (warms Triton configs + allocates every persistent buffer
# OUTSIDE the capture — first-run cudaMalloc inside capture invalidates it)
_FR13_SG_FORCE_DEFER = False


def fr13_sg_set_perm(t):
    global _FR13_SG_PERM
    _FR13_SG_PERM = t


def fr13_sg_set_force_defer(v):
    global _FR13_SG_FORCE_DEFER
    _FR13_SG_FORCE_DEFER = bool(v)


def fr13_sg_set_topology(tree_parent_indices, num_draft_tokens):
    global _FR13_SG_TOPOLOGY
    parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
    if hasattr(num_draft_tokens, "detach"):
        counts = [int(x) for x in num_draft_tokens.detach().cpu().tolist()]
    else:
        counts = [int(x) for x in num_draft_tokens]
    _FR13_SG_TOPOLOGY = (parents_cpu, counts)
    return _FR13_SG_TOPOLOGY


# ---------------------------------------------------------------------------
# S1 v2: capture the TAW walk region ONLY (all our code, RNG-free inside —
# uniforms pre-drawn, no philox) => an aborted capture cannot poison vLLM's
# graph-safe sampler generators (the boot-2 EngineDead). Managed entirely in
# this module; the dispatcher routes here when FR13_STEP_GRAPH=1.
# ---------------------------------------------------------------------------
_FR13_SG_CAP = {}
_FR13_SG_CAP_DEAD = False


def _fr13_s3_setup(nreq, device):
    """=3 capture-prep (walk+products+committer in the LIVE-PROVEN =1 in-
    dispatcher region — zero vLLM sampler code in-graph): replay-order perm
    static + state save (col0 SSM/conv rows + scan flags — the side-stream
    warmup executes a real commit) + module handles. Raises on missing state
    (=> DISABLED => staged, survivable)."""
    import importlib
    from vllm.model_executor.layers.mamba import gdn_linear_attn as g
    import vllm.v1.sample.rejection_sampler as rs
    tk = importlib.import_module("lumo_flywheel_serving.fr10_gdn_tree_kernel")
    rid = getattr(g, "_LUMO_FA_SAMPLER_ROW_REQ_IDS", None)
    sid = getattr(g, "_LUMO_FA_SPEC_ROW_REQ_IDS", None)
    stk = getattr(g, "_FR13_EAGER_PACK_STACKS", None)
    lay = getattr(g, "_FR13_REPLAY_LAYERS", None)
    if (not rid or not sid or len(sid) != nreq or len(rid) < nreq
            or stk is None or not lay):
        raise RuntimeError(
            f"S3 setup: rows/stacks missing rid={rid and len(rid)} "
            f"sid={sid and len(sid)} nreq={nreq} stk={stk is not None}")
    idx = {str(r): i for i, r in enumerate(rid[:nreq])}
    perm_list = [idx.get(str(r)) for r in sid]
    if any(p is None for p in perm_list):
        raise RuntimeError("S3 setup: spec req id missing from sampler rows")
    perm = torch.as_tensor(perm_list, dtype=torch.long, device=device)
    fr13_sg_set_perm(perm)
    saved = []
    for li, pref in enumerate(list(stk["layer_order"])):
        ly = lay[pref]
        c0 = stk["spec_idx"][li, :nreq, 0].to(torch.long)
        cvs = getattr(ly, "_fr13_replay_conv_state", None)
        saved.append((ly, c0, ly._fr13_replay_ssm_state[c0].clone(),
                      cvs[c0].clone() if cvs is not None else None))
    return {"g": g, "rs": rs, "tk": tk, "perm": perm, "saved": saved,
            "flags_sv": stk["flags"][:, 0].clone(), "stk": stk}


def _fr13_s3_restore(s3):
    for ly, c0, sv, cv in s3["saved"]:
        ly._fr13_replay_ssm_state[c0] = sv
        if cv is not None:
            ly._fr13_replay_conv_state[c0] = cv
    s3["stk"]["flags"][:, 0].copy_(s3["flags_sv"])


def _fr13_s3_perm_refill(ent, nreq):
    """Replay-time perm rebuild from CURRENT host row lists. False on any
    composition mismatch => caller falls back to eager for THIS call only."""
    from vllm.model_executor.layers.mamba import gdn_linear_attn as g
    rid = getattr(g, "_LUMO_FA_SAMPLER_ROW_REQ_IDS", None)
    sid = getattr(g, "_LUMO_FA_SPEC_ROW_REQ_IDS", None)
    if not rid or not sid or len(sid) != nreq or len(rid) < nreq:
        return False
    idx = {str(r): i for i, r in enumerate(rid[:nreq])}
    pl = [idx.get(str(r)) for r in sid]
    if any(p is None for p in pl):
        return False
    ent["perm"].copy_(torch.as_tensor(pl, dtype=torch.long))
    return True
_FR13_SG_STREAM = None


def fr13_taw_commit_captured(
    num_draft_tokens, draft_token_ids, tree_parent_indices,
    target_logits, tree_self_logits, bonus_token_ids, max_spec_len,
    *, generators=None,
):
    global _FR13_SG_CAP_DEAD, _FR13_SG_STREAM, _FR13_SG_TOPOLOGY
    device = target_logits.device
    # counts TUPLE in the key (not just nreq): [21,21,21,0] and [21,0,21,21]
    # share every shape (boot-5 lesson generalized) but the captured walk
    # bakes per-request tables — a permuted-zero replay would silently commit
    # the wrong requests' tokens. One small DtoH per call buys soundness.
    if hasattr(num_draft_tokens, "detach"):
        counts_h = tuple(int(x) for x in num_draft_tokens.detach().cpu().tolist())
    else:
        counts_h = tuple(int(x) for x in num_draft_tokens)
    nreq = len(counts_h)
    key = (counts_h, tuple(target_logits.shape), tuple(draft_token_ids.reshape(-1).shape),
           int(bonus_token_ids.numel()))
    ent = _FR13_SG_CAP.get(key)
    row_cap = int(max_spec_len) + 1
    try:
        if ent is None and key not in _FR13_SG_CAP:
            # warmup: eager TAW. Clear the shared uniforms handoff first — a
            # same-shape ent["uni"] left set by another key's replay would be
            # consumed here as ALREADY-USED draws (cross-step RNG reuse).
            fr13_sg_set_uniforms(None)
            _FR13_SG_CAP[key] = False  # warmup done marker
            return fr13_taw_commit(
                num_draft_tokens, draft_token_ids, tree_parent_indices,
                target_logits, tree_self_logits, bonus_token_ids,
                max_spec_len, generators=generators)
        if ent is False or ent is None:
            pass  # uniforms handled per-branch below
        else:
            fr13_sg_set_uniforms(ent["uni"])
        fr13_sg_fill_uniforms(nreq, row_cap, device, generators)
        if ent is False:
            # capture on 2nd call. TOPOLOGY MUST COME FROM THIS CALL'S ARGS:
            # _FR13_SG_TOPOLOGY holds whichever EAGER call ran last, and any
            # other batch shape can run between this key's warmup and now.
            # Boot-7 fatal: a 4-req key captured a 3-req walk off the stale
            # global (counts [21,21,21,0] vs [21,21,21] — same 63-node tpi),
            # and the capture-step return fed 3 products into a 4-request
            # committer. Host reads are legal here (pre-capture).
            _parents_cap = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
            _FR13_SG_TOPOLOGY = (_parents_cap, list(counts_h))
            # per-key uniforms static (so the graph bakes this key's tensor
            # address, immune to other keys' reallocation of the shared global)
            uni = torch.empty(nreq, row_cap, 3, device=device)
            fr13_sg_set_uniforms(uni)
            fr13_sg_fill_uniforms(nreq, row_cap, device, generators)
            statics = {
                "ndt": num_draft_tokens.clone() if hasattr(num_draft_tokens, "clone") else num_draft_tokens,
                "dti": draft_token_ids.clone(),
                "tpi": tree_parent_indices.clone(),
                "tl": target_logits.clone(),
                "sl": tree_self_logits.clone(),
                "bti": bonus_token_ids.clone(),
            }
            _mode3 = os.environ.get("FR13_STEP_GRAPH", "0") == "3"
            _s3 = None
            if _mode3:
                try:
                    _s3 = _fr13_s3_setup(nreq, device)
                except RuntimeError as _s3e:
                    # transient (mixed/transitional step, e.g. sid=1 nreq=3 at
                    # a task boundary — boot-21): eager THIS call, stay armed;
                    # the next same-key step retries the capture.
                    if not globals().get("_FR13_S3_SKIP_SEEN"):
                        globals()["_FR13_S3_SKIP_SEEN"] = True
                        print(f"[FR13_STEP_GRAPH] S3 setup skip (first): {_s3e}",
                              flush=True)
                    return fr13_taw_commit(
                        num_draft_tokens, draft_token_ids, tree_parent_indices,
                        target_logits, tree_self_logits, bonus_token_ids,
                        max_spec_len, generators=generators)

            def _region():
                _o = fr13_taw_commit(
                    statics["ndt"], statics["dti"], statics["tpi"],
                    statics["tl"], statics["sl"], statics["bti"],
                    max_spec_len, generators=None, defer_materialize=True)
                if _mode3:
                    # products + conv col0 + device committer, all device ops
                    # (the =2-era state part, reused verbatim in-region)
                    _s3["rs"]._fr13_sg_commit_state_part(
                        __import__("sys").modules["_fr13_device_multidraft_kernel"],
                        _o)
                return _o

            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            # documented pre-capture warmup ON A SIDE STREAM (classic
            # StreamCaptureInvalidated fix: allocator blocks get stream-
            # assigned before capture)
            if _FR13_SG_STREAM is None:
                _FR13_SG_STREAM = torch.cuda.Stream()
            _FR13_SG_STREAM.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(_FR13_SG_STREAM):
                _warm = _region()
            torch.cuda.current_stream().wait_stream(_FR13_SG_STREAM)
            torch.cuda.synchronize()
            if _mode3:
                # the warmup executed a REAL commit — restore col0 SSM/conv
                # rows + scan flags before the capture replays the same step
                _fr13_s3_restore(_s3)
                torch.cuda.synchronize()
            if int(_warm[0].shape[0]) != nreq:
                raise RuntimeError(
                    f"TAW-capture pre-capture row skew: warm_rows={int(_warm[0].shape[0])} "
                    f"nreq={nreq} key={key}")
            # manual begin/end with g.reset() repair on failure: reset()
            # destroys the underlying graph AND unregisters the philox
            # generators (the ctx manager's __exit__ skips that cleanup when
            # capture_end itself throws => the persistent post-abort poison)
            prev2 = torch.cuda.current_stream()
            torch.cuda.set_stream(_FR13_SG_STREAM)
            try:
                g.capture_begin()
                out = _region()
                g.capture_end()
            except Exception:
                # abort order matters: END the open capture first (else the
                # eager fallback itself runs "while capturing"), THEN reset
                # to unregister philox, THEN restore the stream.
                try:
                    g.capture_end()
                except Exception:
                    pass
                try:
                    g.reset()
                except Exception:
                    pass
                torch.cuda.set_stream(prev2)
                torch.cuda.synchronize()
                raise
            torch.cuda.set_stream(prev2)
            if int(out[0].shape[0]) != nreq:
                # never store or return from a wrong-row graph (the boot-7
                # fatal path was THIS return, unguarded)
                try:
                    g.reset()
                except Exception:
                    pass
                raise RuntimeError(
                    f"TAW-capture captured row skew: rows={int(out[0].shape[0])} "
                    f"nreq={nreq} key={key}")
            g.replay()
            _FR13_SG_CAP[key] = {"g": g, "statics": statics, "out": out,
                                 "uni": uni, "mode3": _mode3,
                                 "perm": (_s3["perm"] if _mode3 else None),
                                 "tk": (_s3["tk"] if _mode3 else None)}
            print(f"[FR13_STEP_GRAPH] TAW-walk captured key={key} "
                  f"mode3={_mode3}", flush=True)
            if _mode3:
                # the graph committed state this step: the staged tail must
                # skip its own scan launch (consume-once flag in the kernel
                # lib entry). Conv col0 copy is idempotent — left alone.
                _s3["tk"]._FR13_S1_STATE_DONE = True
            _mat = fr13_taw_materialize(*out)
            if len(_mat[0]) != nreq:
                raise RuntimeError(
                    f"TAW-capture capture-step product mismatch: products={len(_mat[0])} "
                    f"nreq={nreq} key={key}")
            return _mat
        # replay path
        if ent.get("mode3") and not _fr13_s3_perm_refill(ent, nreq):
            # transient composition mismatch: eager THIS call, stay armed
            return fr13_taw_commit(
                num_draft_tokens, draft_token_ids, tree_parent_indices,
                target_logits, tree_self_logits, bonus_token_ids,
                max_spec_len, generators=generators)
        for name, src in (("ndt", num_draft_tokens), ("dti", draft_token_ids),
                          ("tpi", tree_parent_indices), ("tl", target_logits),
                          ("sl", tree_self_logits), ("bti", bonus_token_ids)):
            dst = ent["statics"][name]
            if hasattr(dst, "copy_"):
                dst.copy_(src, non_blocking=True)
        ent["g"].replay()
        if ent.get("mode3"):
            ent["tk"]._FR13_S1_STATE_DONE = True
        _mat = fr13_taw_materialize(*ent["out"])
        if len(_mat[0]) != nreq:
            raise RuntimeError(
                f"TAW-capture product-shape mismatch: products={len(_mat[0])} "
                f"nreq={nreq} key={key} ent_rows={ent['out'][0].shape} "
                f"counts_head={num_draft_tokens.reshape(-1)[:4].tolist() if hasattr(num_draft_tokens, 'reshape') else num_draft_tokens}")
        return _mat
    except Exception as e:
        _FR13_SG_CAP_DEAD = True
        import traceback as _tb
        _tb.print_exc()
        # philox repair: destroy the failed graph object so its destructor
        # unregisters the generators (else every later eager rand crashes
        # with "Offset increment outside graph capture")
        try:
            import gc as _gc
            for _v in ("g",):
                if _v in dir():
                    pass
            locals_g = locals().get("g")
            if locals_g is not None:
                del locals_g
            _gc.collect()
            torch.cuda.synchronize()
        except Exception:
            pass
        print("[FR13_STEP_GRAPH] TAW-capture DISABLED (eager fallback): "
              + type(e).__name__ + ": " + str(e)[:160], flush=True)
        # clear the shared uniforms handoff: post-DEAD eager steps must draw
        # fresh per-call (a lingering same-shape ent["uni"] would be re-
        # consumed every step => cross-step RNG reuse)
        fr13_sg_set_uniforms(None)
        return fr13_taw_commit(
            num_draft_tokens, draft_token_ids, tree_parent_indices,
            target_logits, tree_self_logits, bonus_token_ids,
            max_spec_len, generators=generators)


# ---------------------------------------------------------------------------
# CPU-only fixed32 contract tests. These execute the production tensor core;
# they do not import vLLM, launch a model, or make performance claims.
# ---------------------------------------------------------------------------
def _fr13_fixed32_test_set_env(topology, mode: str) -> None:
    os.environ["FR13_FIXED32_MODE"] = mode
    os.environ["FR13_FIXED32_VALID_MASK"] = hex(int(topology.VALID_MASK_BY_MODE[mode]))
    os.environ["FR13_FIXED32_ACTIVE_NODES"] = str(
        _fr13_fixed32_expected_active(topology, mode)
    )
    os.environ["FR13_FIXED32_TAW_WALK_CAP"] = str(topology.WALK_CAP)
    os.environ["FR13_TAW"] = "1"
    os.environ["FR13_FIXED32_WORK_CENSUS"] = "1"
    os.environ.pop("LUMO_TREE_SAMPLER_DEBUG_LOG", None)


def _fr13_fixed32_test_fixture(
    topology,
    mode: str,
    batch_size: int,
    *,
    vocab_size: int = 97,
) -> dict[str, Any]:
    physical_drafts = int(topology.PHYSICAL_DRAFTS)
    drafts_one = torch.arange(1, physical_drafts + 1, dtype=torch.int32)
    drafts = drafts_one.repeat(batch_size, 1)
    target_logits = torch.full(
        (batch_size * physical_drafts, vocab_size),
        float("-inf"),
        dtype=torch.float32,
    )
    target_logits[:, 0] = 0.0
    self_logits = torch.full_like(target_logits, float("-inf"))
    self_logits[:, vocab_size - 1] = 0.0
    children = topology.active_child_lists(mode)
    for request_index in range(batch_size):
        start = request_index * physical_drafts
        for child_nodes in children.values():
            first_child = child_nodes[0]
            token = int(drafts_one[first_child])
            target_logits[start + first_child].fill_(float("-inf"))
            target_logits[start + first_child, token] = 0.0
    return {
        "counts": fr13_fixed32_taw_preseeded_counts(
            torch.device("cpu"),
            mode=mode,
            valid_mask=int(topology.VALID_MASK_BY_MODE[mode]),
            batch_size=batch_size,
        ),
        "drafts": drafts.reshape(-1),
        "parents": torch.tensor(
            topology.DRAFT_PARENT,
            dtype=torch.int32,
        ).repeat(batch_size),
        "target": target_logits,
        "self": self_logits,
        "bonus": torch.full(
            (batch_size, 1),
            vocab_size - 2,
            dtype=torch.int32,
        ),
        "uniforms": torch.full(
            (
                batch_size,
                int(topology.WALK_CAP),
                3,
            ),
            0.1,
            dtype=torch.float32,
        ),
    }


def _fr13_fixed32_test_call(
    topology,
    mode: str,
    fixture: dict[str, Any],
):
    return fr13_fixed32_taw_commit(
        fixture["counts"],
        fixture["drafts"],
        fixture["parents"],
        fixture["target"],
        fixture["self"],
        fixture["bonus"],
        int(topology.PHYSICAL_DRAFTS),
        uniforms=fixture["uniforms"],
        mode=mode,
    )


def _fr13_fixed32_expect_runtime_error(
    function: Callable[[], Any],
    *,
    contains: str,
) -> None:
    try:
        function()
    except RuntimeError as error:
        if contains not in str(error):
            raise AssertionError(
                f"expected {contains!r} in RuntimeError: {error}"
            ) from error
    else:
        raise AssertionError(f"expected RuntimeError containing {contains!r}")


def _fr13_fixed32_test_accept_leaf_depth_pad(topology) -> None:
    mode = "tail6_fixed32"
    _fr13_fixed32_test_set_env(topology, mode)
    fixture = _fr13_fixed32_test_fixture(topology, mode, 1)
    output, output_lens, paths, path_lens, last_row = _fr13_fixed32_test_call(
        topology, mode, fixture
    )
    children = topology.active_child_lists(mode)
    expected_nodes = []
    current = -1
    while current in children:
        current = children[current][0]
        expected_nodes.append(current)
    expected_path = torch.tensor(
        [node + 1 for node in expected_nodes],
        dtype=torch.long,
    )
    if not torch.equal(path_lens, torch.tensor([11], dtype=torch.long)):
        raise AssertionError(f"max-depth accepted lens mismatch: {path_lens}")
    if not torch.equal(paths[0, :11], expected_path):
        raise AssertionError("max-depth accepted path mismatch")
    if not torch.equal(last_row, torch.tensor([31], dtype=torch.long)):
        raise AssertionError(f"max-depth last row mismatch: {last_row}")
    if not torch.equal(output_lens, torch.tensor([12], dtype=torch.long)):
        raise AssertionError(f"leaf output lens mismatch: {output_lens}")
    if not torch.equal(
        output[0, 11:12],
        torch.tensor([96], dtype=torch.long),
    ):
        raise AssertionError("leaf did not emit the self-target token")
    if not torch.equal(
        output[0, 12:],
        torch.full((20,), -1, dtype=torch.long),
    ):
        raise AssertionError("fixed output padding is not -1")
    if not torch.equal(
        paths[0, 11:],
        torch.zeros(5, dtype=torch.long),
    ):
        raise AssertionError("fixed accepted-path padding is not zero")


def _fr13_fixed32_test_reject_residual_zero_mass(topology) -> None:
    mode = "tail6_fixed32"
    _fr13_fixed32_test_set_env(topology, mode)
    fixture = _fr13_fixed32_test_fixture(topology, mode, 1)
    children = topology.active_child_lists(mode)[-1]
    child_tokens = [int(fixture["drafts"][child]) for child in children]
    target_row = children[0]
    fixture["target"][target_row].fill_(float("-inf"))
    for token in child_tokens:
        fixture["target"][target_row, token] = math.log(0.05)
    fixture["target"][target_row, 90] = math.log(0.85)
    fixture["uniforms"][0, 0, 1] = 0.9
    output, output_lens, _paths, path_lens, last_row = _fr13_fixed32_test_call(
        topology, mode, fixture
    )
    if not torch.equal(output[0, :1], torch.tensor([90])):
        raise AssertionError("rejection did not sample the canonical residual")
    if not torch.equal(output_lens, torch.tensor([1])):
        raise AssertionError("rejection output length mismatch")
    if not torch.equal(path_lens, torch.tensor([0])):
        raise AssertionError("rejected source entered the accepted path")
    if not torch.equal(last_row, torch.tensor([0])):
        raise AssertionError("rejection last row must be zero")

    zero_fixture = _fr13_fixed32_test_fixture(topology, mode, 1)
    zero_fixture["target"][target_row].fill_(float("-inf"))
    zero_fixture["target"][target_row, 91] = 0.0
    zero_output, _ol, _paths, zero_lens, _last = _fr13_fixed32_test_call(
        topology, mode, zero_fixture
    )
    if not torch.equal(zero_output[0, :1], torch.tensor([91])):
        raise AssertionError("zero-overlap row did not sample target p")
    if not torch.equal(zero_lens, torch.tensor([0])):
        raise AssertionError("zero-overlap row was accepted")


def _fr13_fixed32_test_duplicate_semantics(topology) -> None:
    mode = "tail6_fixed32"
    _fr13_fixed32_test_set_env(topology, mode)
    fixture = _fr13_fixed32_test_fixture(topology, mode, 1)
    root_children = topology.active_child_lists(mode)[-1]
    for child in root_children:
        fixture["drafts"][child] = 5
    target_row = root_children[0]
    fixture["target"][target_row].fill_(float("-inf"))
    fixture["target"][target_row, 5] = math.log(0.4)
    fixture["target"][target_row, 92] = math.log(0.6)
    fixture["uniforms"][0, 0, 0] = 0.5
    fixture["uniforms"][0, 0, 1] = 0.3
    output, _ol, paths, path_lens, _last = _fr13_fixed32_test_call(
        topology, mode, fixture
    )
    if not torch.equal(output[0, :1], torch.tensor([5])):
        raise AssertionError("duplicate child acceptance emitted wrong token")
    if not torch.equal(path_lens, torch.tensor([1])):
        raise AssertionError("duplicate q_mix acceptance probability is wrong")
    if not torch.equal(paths[0, :1], torch.tensor([2])):
        raise AssertionError("duplicate source inverse-CDF selected wrong child")

    fixture["uniforms"][0, 0, 1] = 0.5
    rejected, _ol, _paths, rejected_lens, _last = _fr13_fixed32_test_call(
        topology, mode, fixture
    )
    if not torch.equal(rejected[0, :1], torch.tensor([92])):
        raise AssertionError("duplicate q_mix residual is wrong")
    if not torch.equal(rejected_lens, torch.tensor([0])):
        raise AssertionError("duplicate q_mix rejection entered the path")


def _fr13_fixed32_test_inactive_poison(topology) -> None:
    mode = "tail6_fixed32"
    _fr13_fixed32_test_set_env(topology, mode)
    clean = _fr13_fixed32_test_fixture(topology, mode, 1)
    clean_result = tuple(
        tensor.clone() for tensor in _fr13_fixed32_test_call(topology, mode, clean)
    )
    poisoned = _fr13_fixed32_test_fixture(topology, mode, 1)
    for node in topology.TAIL6_INACTIVE_DRAFT_IDS:
        poisoned["drafts"][node] = 94
        poisoned["target"][node].fill_(12345.0)
        poisoned["self"][node].fill_(-12345.0)
        poisoned["self"][node, 93] = 12345.0
    poison_result = tuple(
        tensor.clone() for tensor in _fr13_fixed32_test_call(topology, mode, poisoned)
    )
    if not all(
        torch.equal(left, right) for left, right in zip(clean_result, poison_result)
    ):
        raise AssertionError("inactive-node poison changed fixed32 results")


def _fr13_fixed32_test_boot_warm_state(topology) -> None:
    mode = "tail6_fixed32"
    _fr13_fixed32_test_set_env(topology, mode)
    valid_mask = int(topology.VALID_MASK_BY_MODE[mode])
    fr13_fixed32_taw_preseed("cpu", mode=mode, valid_mask=valid_mask)
    entries = [
        _FR13_FIXED32_TAW_CACHE[
            fr13_fixed32_taw_cache_key(mode, valid_mask, batch, "cpu")
        ]
        for batch in _FR13_FIXED32_BATCHES
    ]
    mutable_names = (
        "uniforms",
        "draft_tokens",
        "bonus_tokens",
        "output_tokens",
        "output_lens",
        "accepted_path_rows",
        "accepted_lens",
        "last_row",
        "exact_current",
        "exact_alive",
    )
    saved_tensors = tuple(
        {
            name: entry[name].clone()
            for name in mutable_names
        }
        for entry in entries
    )
    generator = _fr13_bulk_gen(torch.device("cpu"))
    saved_generator_state = generator.get_state().clone()
    saved_last_work = _FR13_FIXED32_TAW_LAST_WORK
    saved_announced = _FR13_FIXED32_WORK_ANNOUNCED
    callback_calls = []

    def callback(payload):
        callback_calls.append(payload)

    saved_callback = _FR13_FIXED32_TAW_WORK_CALLBACK
    fr13_fixed32_taw_set_work_callback(callback)
    try:
        result = fr13_fixed32_taw_warm_execute(
            "cpu",
            mode=mode,
            valid_mask=valid_mask,
            max_batch_size=4,
            vocab_size=97,
        )
        if (
            result.get("ready") is not True
            or result.get("classification") != "unmeasured_boot"
            or tuple(result.get("batches", ())) != (1, 2, 3, 4)
            or result.get("executions") != 4
            or callback_calls
            or _FR13_FIXED32_TAW_WORK_CALLBACK is not callback
            or _FR13_FIXED32_TAW_LAST_WORK is not saved_last_work
            or _FR13_FIXED32_WORK_ANNOUNCED is not saved_announced
            or not torch.equal(generator.get_state(), saved_generator_state)
        ):
            raise AssertionError("fixed32 TAW boot warm polluted measured state")
        for entry, saved in zip(entries, saved_tensors, strict=True):
            if any(
                not _fr13_fixed32_tensor_bits_equal(
                    entry[name], saved[name]
                )
                for name in mutable_names
            ):
                raise AssertionError(
                    "fixed32 TAW boot warm did not restore cache staging"
                )
        repeated = fr13_fixed32_taw_warm_execute(
            "cpu",
            mode=mode,
            valid_mask=valid_mask,
            max_batch_size=4,
            vocab_size=97,
        )
        if repeated != result or callback_calls:
            raise AssertionError("fixed32 TAW boot warm is not idempotent")
        saved_output = entries[-1]["output_tokens"]
        entries[-1]["output_tokens"] = saved_output.clone()
        try:
            stale = fr13_fixed32_taw_warmup_counters(
                "cpu",
                mode=mode,
                valid_mask=valid_mask,
                max_batch_size=4,
                vocab_size=97,
            )
            if stale.get("ready") is not False:
                raise AssertionError(
                    "fixed32 TAW warm lease ignored tensor rebinding"
                )
        finally:
            entries[-1]["output_tokens"] = saved_output
    finally:
        fr13_fixed32_taw_set_work_callback(saved_callback)


def _fr13_fixed32_test_bonus_core(topology) -> None:
    mode = "tail6_fixed32"
    _fr13_fixed32_test_set_env(topology, mode)
    fixture = _fr13_fixed32_test_fixture(topology, mode, 1)
    key = fr13_fixed32_taw_cache_key(
        mode,
        int(topology.VALID_MASK_BY_MODE[mode]),
        1,
        torch.device("cpu"),
    )
    entry = dict(_FR13_FIXED32_TAW_CACHE[key])
    entry["child_counts"] = entry["child_counts"].clone()
    entry["child_counts"][:, 0] = 0
    drafts, bonus = _fr13_fixed32_validate_inputs(
        topology,
        entry,
        fixture["counts"],
        fixture["drafts"],
        fixture["parents"],
        fixture["target"],
        fixture["self"],
        fixture["bonus"],
        int(topology.PHYSICAL_DRAFTS),
    )
    result = _fr13_fixed32_taw_execute(
        topology,
        entry,
        drafts,
        fixture["target"],
        fixture["self"],
        bonus,
        fixture["uniforms"],
        walk_cap=int(topology.WALK_CAP),
    )
    output, output_lens, _paths, path_lens, _last, loops = result
    if not torch.equal(output[0, :1], fixture["bonus"].reshape(-1)):
        raise AssertionError("root leaf did not emit the bonus token")
    if not torch.equal(output_lens, torch.tensor([1])):
        raise AssertionError("bonus output length mismatch")
    if not torch.equal(path_lens, torch.tensor([0])):
        raise AssertionError("bonus branch accepted a draft node")
    if loops != 12:
        raise AssertionError("bonus branch did not run all 12 iterations")


def _fr13_fixed32_test_mode_switch_batches(topology) -> None:
    entries = {}
    callback_rows = []
    taw_by_batch = {}
    expected_taw_keys = {
        "route",
        "preseeded_batches",
        "topology_cache_hit",
        "cache_misses",
        "table_shape",
        "buffer_capacity",
        "loop_iterations",
        "uniform_slots",
        "child_lanes",
        "target_rows",
        "self_rows",
        "self_cdf_rows",
        "source_cdf_rows",
        "residual_cdf_rows",
        "qmix_rows",
        "residual_rows",
        "row_scatter_slots",
        "path_scatter_slots",
        "exact_commit_launches",
        "exact_commit_programs",
        "floating_sampling_reimplementation",
        "source_contract_schema",
        "source_contract_sha256",
        "tensor_call_census",
        "count_route",
        "rng_route",
        "vocab_size",
        "count_shape",
        "count_dtype",
        "count_stride",
        "count_contiguous",
        "draft_shape",
        "draft_dtype",
        "draft_stride",
        "draft_contiguous",
        "parent_shape",
        "parent_dtype",
        "parent_stride",
        "parent_contiguous",
        "bonus_shape",
        "bonus_dtype",
        "bonus_stride",
        "bonus_contiguous",
        "target_shape",
        "target_dtype",
        "target_stride",
        "target_contiguous",
        "self_shape",
        "self_dtype",
        "self_stride",
        "self_contiguous",
        "uniform_shape",
        "uniform_dtype",
        "uniform_stride",
        "uniform_contiguous",
        "child_table_shape",
        "child_counts_shape",
        "output_shape",
        "output_lens_shape",
        "accepted_path_shape",
        "accepted_lens_shape",
        "last_row_shape",
        "exact_current_shape",
        "exact_alive_shape",
    }
    fr13_fixed32_taw_set_work_callback(callback_rows.append)
    for mode in ("tail6_fixed32", "hydra27_fixed32"):
        _fr13_fixed32_test_set_env(topology, mode)
        mask = int(topology.VALID_MASK_BY_MODE[mode])
        keys = fr13_fixed32_taw_preseed(
            "cpu",
            mode=mode,
            valid_mask=mask,
        )
        if len(keys) != 4:
            raise AssertionError(f"{mode}: preseed did not build B1..4")
        for batch_size in _FR13_FIXED32_BATCHES:
            key = fr13_fixed32_taw_cache_key(
                mode,
                mask,
                batch_size,
                torch.device("cpu"),
            )
            entry = _FR13_FIXED32_TAW_CACHE[key]
            entries[(mode, batch_size)] = entry
            counts = fr13_fixed32_taw_preseeded_counts(
                torch.device("cpu"),
                mode=mode,
                valid_mask=mask,
                batch_size=batch_size,
            )
            if counts is not entry["draft_counts"]:
                raise AssertionError(f"{mode}/B{batch_size}: count cache identity drifted")
            if not torch.equal(
                counts,
                torch.full((batch_size,), 31, dtype=torch.int32),
            ):
                raise AssertionError(f"{mode}/B{batch_size}: count values drifted")
            if tuple(entry["child_table"].shape) != (
                batch_size,
                32,
                3,
            ):
                raise AssertionError(f"{mode}/B{batch_size}: child-table shape drifted")
            fixture = _fr13_fixed32_test_fixture(
                topology,
                mode,
                batch_size,
            )
            if batch_size == 4:
                fixture["drafts"] = fixture["drafts"].to(torch.int32)
                fixture["bonus"] = fixture["bonus"].to(torch.int32)
            result = tuple(
                tensor.clone()
                for tensor in _fr13_fixed32_test_call(
                    topology,
                    mode,
                    fixture,
                )
            )
            expected_shapes = (
                (batch_size, 32),
                (batch_size,),
                (batch_size, 16),
                (batch_size,),
                (batch_size,),
            )
            if tuple(tuple(tensor.shape) for tensor in result) != (expected_shapes):
                raise AssertionError(f"{mode}/B{batch_size}: product shape drifted")
            work = callback_rows[-1]
            if (
                set(work) != {"mode", "valid_mask", "batch_size", "taw"}
                or set(work["taw"]) != expected_taw_keys
                or work["mode"] != mode
                or work["valid_mask"] != mask
                or work["batch_size"] != batch_size
                or work["taw"]["loop_iterations"] != 12
                or work["taw"]["table_shape"] != [batch_size, 32, 3]
            ):
                raise AssertionError(f"{mode}/B{batch_size}: callback counters drifted")
            expected_layout_values = {
                "count_route": "preseeded_cpu_fixed31_test",
                "rng_route": "provided_uniforms",
                "vocab_size": 97,
                "count_shape": [batch_size],
                "count_dtype": "torch.int32",
                "count_stride": [1],
                "count_contiguous": True,
                "draft_shape": [batch_size * 31],
                "draft_dtype": "torch.int32",
                "draft_stride": [1],
                "draft_contiguous": True,
                "parent_shape": [batch_size * 31],
                "parent_dtype": "torch.int32",
                "parent_stride": [1],
                "parent_contiguous": True,
                "bonus_shape": [batch_size, 1],
                "bonus_dtype": "torch.int32",
                "bonus_stride": [1, 1],
                "bonus_contiguous": True,
                "target_shape": [batch_size * 31, 97],
                "target_dtype": "torch.float32",
                "target_stride": [97, 1],
                "target_contiguous": True,
                "self_shape": [batch_size * 31, 97],
                "self_dtype": "torch.float32",
                "self_stride": [97, 1],
                "self_contiguous": True,
                "uniform_shape": [batch_size, 12, 3],
                "uniform_dtype": "torch.float32",
                "uniform_stride": [36, 3, 1],
                "uniform_contiguous": True,
                "child_table_shape": [batch_size, 32, 3],
                "child_counts_shape": [batch_size, 32],
                "output_shape": [batch_size, 32],
                "output_lens_shape": [batch_size],
                "accepted_path_shape": [batch_size, 16],
                "accepted_lens_shape": [batch_size],
                "last_row_shape": [batch_size],
                "exact_current_shape": [batch_size],
                "exact_alive_shape": [batch_size],
            }
            for name, expected in expected_layout_values.items():
                if work["taw"][name] != expected:
                    raise AssertionError(
                        f"{mode}/B{batch_size}: {name} callback drifted"
                    )
            if (
                work["taw"]["source_contract_schema"]
                != _FR13_FIXED32_TAW_SOURCE_SCHEMA
                or work["taw"]["source_contract_sha256"]
                != _FR13_FIXED32_TAW_SOURCE_SHA256
                or work["taw"]["tensor_call_census"]
                != _FR13_FIXED32_TAW_TENSOR_CALL_CENSUS
            ):
                raise AssertionError(
                    f"{mode}/B{batch_size}: source contract callback drifted"
                )
            prior_mode_work = taw_by_batch.setdefault(batch_size, dict(work["taw"]))
            if work["taw"] != prior_mode_work:
                raise AssertionError(
                    f"{mode}/B{batch_size}: TAW work changed with active-node mask"
                )
    if entries[("tail6_fixed32", 1)] is entries[("hydra27_fixed32", 1)]:
        raise AssertionError("Tail/Hydra cache entries collided")
    tail_key = fr13_fixed32_taw_cache_key(
        "tail6_fixed32",
        int(topology.TAIL6_VALID_MASK),
        1,
        torch.device("cpu"),
    )
    if _FR13_FIXED32_TAW_CACHE[tail_key] is not entries[("tail6_fixed32", 1)]:
        raise AssertionError("Tail-Hydra-Tail mode switch lost Tail cache")
    fr13_fixed32_taw_set_work_callback(None)


def _fr13_fixed32_test_fail_loud(topology) -> None:
    mode = "tail6_fixed32"
    _fr13_fixed32_test_set_env(topology, mode)
    mask = int(topology.VALID_MASK_BY_MODE[mode])
    fixture = _fr13_fixed32_test_fixture(topology, mode, 1)
    fr13_fixed32_taw_set_work_callback(lambda _payload: None)

    os.environ["FR13_FIXED32_VALID_MASK"] = hex(mask ^ 1)
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_test_call(topology, mode, fixture),
        contains="validity mask",
    )
    _fr13_fixed32_test_set_env(topology, mode)
    os.environ["FR13_FIXED32_TAW_WALK_CAP"] = "11"
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_test_call(topology, mode, fixture),
        contains="walk cap",
    )
    _fr13_fixed32_test_set_env(topology, mode)

    wrong_parent = dict(fixture)
    wrong_parent["parents"] = fixture["parents"].clone()
    wrong_parent["parents"][0] = 7
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_test_call(
            topology,
            mode,
            wrong_parent,
        ),
        contains="physical parent",
    )
    mixed = _fr13_fixed32_test_fixture(topology, mode, 2)
    mixed["counts"] = mixed["counts"].clone()
    mixed["counts"][1] = 0
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_test_call(topology, mode, mixed),
        contains="cache-owned preseeded draft-count",
    )
    _fr13_fixed32_expect_runtime_error(
        lambda: fr13_fixed32_taw_commit(
            fixture["counts"],
            fixture["drafts"],
            fixture["parents"],
            fixture["target"],
            fixture["self"],
            fixture["bonus"],
            30,
            uniforms=fixture["uniforms"],
            mode=mode,
        ),
        contains="max_spec_len",
    )

    key = fr13_fixed32_taw_cache_key(
        mode,
        mask,
        1,
        torch.device("cpu"),
    )
    bad_fanout = dict(_FR13_FIXED32_TAW_CACHE[key])
    bad_fanout["child_counts"] = bad_fanout["child_counts"].clone()
    bad_fanout["child_counts"][0, 0] = 4
    drafts, bonus = _fr13_fixed32_validate_inputs(
        topology,
        bad_fanout,
        fixture["counts"],
        fixture["drafts"],
        fixture["parents"],
        fixture["target"],
        fixture["self"],
        fixture["bonus"],
        31,
    )
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_taw_execute(
            topology,
            bad_fanout,
            drafts,
            fixture["target"],
            fixture["self"],
            bonus,
            fixture["uniforms"],
            walk_cap=12,
        ),
        contains="fanout",
    )
    overflow_topology = SimpleNamespace(
        WALK_CAP=17,
        ACCEPTED_PATH_CAPACITY=16,
        OUTPUT_PUBLISH_CAPACITY=32,
        PHYSICAL_DRAFTS=31,
        SAMPLER_MAX_FANOUT=3,
        PHYSICAL_ROWS=32,
    )
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_taw_execute(
            overflow_topology,
            _FR13_FIXED32_TAW_CACHE[key],
            drafts,
            fixture["target"],
            fixture["self"],
            bonus,
            fixture["uniforms"],
            walk_cap=17,
        ),
        contains="overflow",
    )
    fr13_fixed32_taw_set_work_callback(None)


def _fr13_fixed32_test_adversarial_contract(topology) -> None:
    mode = "tail6_fixed32"
    _fr13_fixed32_test_set_env(topology, mode)
    fixture = _fr13_fixed32_test_fixture(topology, mode, 1)
    fr13_fixed32_taw_set_work_callback(lambda _payload: None)

    copied_count = dict(fixture)
    copied_count["counts"] = fixture["counts"].clone()
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_test_call(topology, mode, copied_count),
        contains="cache-owned preseeded draft-count",
    )
    for name in ("drafts", "parents"):
        wrong_integer_dtype = dict(fixture)
        wrong_integer_dtype[name] = fixture[name].to(torch.int64)
        _fr13_fixed32_expect_runtime_error(
            lambda candidate=wrong_integer_dtype: _fr13_fixed32_test_call(
                topology,
                mode,
                candidate,
            ),
            contains="live input layout drift",
        )

    wrong_target_dtype = dict(fixture)
    wrong_target_dtype["target"] = fixture["target"].to(torch.float64)
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_test_call(topology, mode, wrong_target_dtype),
        contains="live input layout drift",
    )
    noncontiguous_target = dict(fixture)
    noncontiguous_target["target"] = fixture["target"].t().contiguous().t()
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_test_call(topology, mode, noncontiguous_target),
        contains="live input layout drift",
    )
    wrong_bonus_shape = dict(fixture)
    wrong_bonus_shape["bonus"] = fixture["bonus"].reshape(-1)
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_test_call(topology, mode, wrong_bonus_shape),
        contains="live input layout drift",
    )
    noncontiguous_uniforms = dict(fixture)
    noncontiguous_uniforms["uniforms"] = torch.full(
        (1, 12, 6),
        0.1,
        dtype=torch.float32,
    )[:, :, ::2]
    _fr13_fixed32_expect_runtime_error(
        lambda: _fr13_fixed32_test_call(topology, mode, noncontiguous_uniforms),
        contains="uniforms layout drift",
    )
    _fr13_fixed32_expect_runtime_error(
        lambda: fr13_fixed32_taw_commit(
            fixture["counts"],
            fixture["drafts"],
            fixture["parents"],
            fixture["target"],
            fixture["self"],
            fixture["bonus"],
            31,
            generators={0: object()},
            mode=mode,
        ),
        contains="forbids per-request generator maps",
    )

    global _FR13_FIXED32_TAW_SOURCE_SHA256
    saved_digest = _FR13_FIXED32_TAW_SOURCE_SHA256
    try:
        _FR13_FIXED32_TAW_SOURCE_SHA256 = "0" * 64
        _fr13_fixed32_expect_runtime_error(
            lambda: _fr13_fixed32_taw_source_contract(topology),
            contains="pinned source digest changed after binding",
        )
    finally:
        _FR13_FIXED32_TAW_SOURCE_SHA256 = saved_digest
    _fr13_fixed32_taw_source_contract(topology)

    global _fr13_taw_inv_cdf
    saved_inv_cdf = _fr13_taw_inv_cdf

    def replacement_inv_cdf(probabilities, uniforms):
        return saved_inv_cdf(probabilities, uniforms)

    try:
        _fr13_taw_inv_cdf = replacement_inv_cdf
        _fr13_fixed32_expect_runtime_error(
            lambda: _fr13_fixed32_taw_source_contract(topology),
            contains="source objects changed after binding",
        )
    finally:
        _fr13_taw_inv_cdf = saved_inv_cdf
    _fr13_fixed32_taw_source_contract(topology)
    fr13_fixed32_taw_set_work_callback(None)


def fr13_fixed32_taw_self_test() -> None:
    """Run the CPU-only production-core fixed32 gate."""
    if torch is None:
        raise RuntimeError("FR13 fixed32 self-test requires torch")
    topology = _fr13_fixed32_topology()
    environment_names = (
        "FR13_FIXED32_MODE",
        "FR13_FIXED32_VALID_MASK",
        "FR13_FIXED32_ACTIVE_NODES",
        "FR13_FIXED32_TAW_WALK_CAP",
        "FR13_TAW",
        "FR13_FIXED32_WORK_CENSUS",
        "LUMO_TREE_SAMPLER_DEBUG_LOG",
    )
    saved_environment = {name: os.environ.get(name) for name in environment_names}
    tests = (
        _fr13_fixed32_test_mode_switch_batches,
        _fr13_fixed32_test_accept_leaf_depth_pad,
        _fr13_fixed32_test_reject_residual_zero_mass,
        _fr13_fixed32_test_duplicate_semantics,
        _fr13_fixed32_test_inactive_poison,
        _fr13_fixed32_test_boot_warm_state,
        _fr13_fixed32_test_bonus_core,
        _fr13_fixed32_test_fail_loud,
        _fr13_fixed32_test_adversarial_contract,
    )
    try:
        for test in tests:
            fr13_fixed32_taw_set_work_callback(lambda _payload: None)
            test(topology)
    finally:
        fr13_fixed32_taw_set_work_callback(None)
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    print(
        f"PASS fr13_fixed32_taw groups={len(tests)} modes=2 batches=1..4 walk=12",
        flush=True,
    )


def _fr13_device_multidraft_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FR13 device multidraft offline gates")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the CPU-only fixed32 TAW production-core gate",
    )
    return parser


def _fr13_device_multidraft_main(
    argv: Sequence[str] | None = None,
) -> int:
    args = _fr13_device_multidraft_parser().parse_args(argv)
    if not args.self_test:
        raise SystemExit("use --self-test")
    fr13_fixed32_taw_self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(_fr13_device_multidraft_main())
