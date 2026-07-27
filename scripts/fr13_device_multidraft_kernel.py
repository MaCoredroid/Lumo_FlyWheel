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

import os
from typing import Sequence

try:
    import torch
except Exception:  # pragma: no cover - torch always present in the vLLM image
    torch = None  # type: ignore

import numpy as np


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
        import torch, json
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
    V = int(target_logits.shape[-1])

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
    out_rows = [r[: int(l)] for r, l in zip(rb, rl)]
    accepted_node_paths = [pth[: int(l)] for pth, l in zip(pb, pl)]
    accepted_lens = [int(l) for l in pl]
    accepted_rows = accepted_lens  # legacy alias shape (per-req accepted count)
    accepted_token_rows = [r[: int(l)] for r, l in zip(rb, pl)]
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
    out_rows = [r[: int(l)] for r, l in zip(rb, rl)]
    accepted_node_paths = [pth[: int(l)] for pth, l in zip(pb, pl)]
    accepted_lens = [int(l) for l in pl]
    accepted_rows = accepted_lens
    accepted_token_rows = [r[: int(l)] for r, l in zip(rb, pl)]
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
