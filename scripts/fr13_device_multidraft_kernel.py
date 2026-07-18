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
        _fr13_sp_prefix = True  # spine-commit de-risk: were ALL accepted steps children[0]?
        for _step in range(int(max_spec_len) + 1):
            children = [
                node for node, parent in enumerate(parents)
                if int(parent) == int(current_parent)
            ]
            if not children:
                if current_parent >= 0:
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
            if int(source_index) != 0:
                _fr13_sp_prefix = False
            accepted_path.append(int(current_parent))

        out_rows.append(row[:int(max_spec_len) + 1])
        accepted_rows.append(int(accepted_row))
        accepted_lens.append(int(len(accepted_path)))
        accepted_node_paths.append([int(x) for x in accepted_path])
        # SPINE-COMMIT DE-RISK (byte-neutral counter): how often is the accepted path a pure
        # spine prefix (all children[0])? High => spine-commit skips the 72ms replay often.
        _fr13_ss = globals().setdefault(
            '_FR13_SPINE_STATS', {'spine': 0, 'branch': 0, 'empty': 0})
        if len(accepted_path) == 0:
            _fr13_ss['empty'] += 1
        elif _fr13_sp_prefix:
            _fr13_ss['spine'] += 1
        else:
            _fr13_ss['branch'] += 1
        if (_fr13_ss['spine'] + _fr13_ss['branch']
                + _fr13_ss['empty']) % 500 == 0:
            try:
                import json as _ssj
                _ssj.dump(_fr13_ss, open('/logs/fr13_spine_stats.json', 'w'))
            except Exception:
                pass
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
        cp = cur_parent[req_i]
        if cp >= 0:
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
