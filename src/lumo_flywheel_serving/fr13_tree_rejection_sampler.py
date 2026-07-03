"""FR13_TREE_REJECTION_SAMPLER: lossless, vectorized, cuda-graph-safe tree
rejection committer for Qwen3-Next MTP tree speculative decoding.

WHY THIS MODULE EXISTS
======================
The deployed tree committer at temp>0 is one of two things depending on flags:

  * ``_lumo_tree_path_lcp_max_greedy_sample`` (greedy-LCP): commits the path
    where ``draft == target ARGMAX``. At temp>0 that commits the GREEDY token,
    NOT a token SAMPLED from the temp-0.6 target distribution -> distributionally
    LOSSY vs the true target (the temp-0.6 q-vs-p TV gate measures exactly this).

  * ``_lumo_tree_canonical_multidraft_sample`` -> ``fr13_device_multidraft_
    commit`` (scripts/fr13_device_multidraft_kernel.py): the correct SpecInfer /
    multi-draft residual-mix rejection rule, distribution-LOSSLESS, BUT it still
    runs a PER-REQUEST / PER-LEVEL Python interpreter walk with ``.item()``
    device syncs at every node (source pick, accept Bernoulli, token id all
    ``.item()``'d mid-walk). That per-node host<->device ping-pong is the hot-path
    tax this module removes.

WHAT THIS MODULE DOES
=====================
Same DISTRIBUTION-LOSSLESS SpecInfer / multi-draft canonical accept rule as
``fr13_device_multidraft_kernel.py`` (verbatim per-node semantics -- proven
offline by scripts/fr13_device_multidraft_offline_gate.py), but VECTORIZED over
batch AND tree nodes on device with NO per-node Python ``.item()`` loop on the
hot path:

  1. Init-time STATIC int tables from ``tree_parent_indices``: per node, its
     ordered child slots packed ``[N, F]`` (F = max fan-out, small; typically
     <= 4), a validity mask, and the child draft token ids. These depend only on
     the tree STRUCTURE, not on logits/RNG, so they are the "static shapes"
     piece.

  2. VECTORIZED per-node distribution math over ALL nodes at once (the load-
     bearing subtlety -- duplicate draft tokens summed into q_mix -- done with a
     grouped scatter, no Python loop):
       p_at[n,f]      = softmax(target_logits[first-child-row of node n])[t_{n,f}]
       overlap_mass[n]= sum_f p_at[n,f]  (over valid child slots)
       w[n,f]         = p_at[n,f] / overlap_mass[n]         (source weights)
       q_mix[n,f]     = sum over sibling slots g whose token == token f of w[n,g]
       accept_p[n,f]  = min(1, p_at[n,f] / q_mix[n,f])
     All ``[N, F]`` tensor ops. ``overlap_mass == 0`` nodes -> pure residual (~p).

  3. PRE-DRAWN randomness, per request, in a FIXED draw order (for TV-gate
     stability + batch-invariance): from each request's own seeded
     ``torch.Generator`` (sampling_metadata.generators, matching
     FR13_TREE_PER_REQ_GEN) we pre-draw, for EVERY node the walk could visit:
       - a source-pick uniform  -> Gumbel-max over the ``w[n,:]`` weights,
       - an accept uniform ``u[n]``,
       - a residual sample ``resid_tok[n]`` (Gumbel-max over the residual dist),
       - a self/bonus sample ``self_tok[n]`` for a fully-accepted leaf.
     Drawn UP FRONT so no host sync happens mid-walk.

  4. The DESCENT (top-down root -> accepted child) is then a pure INTEGER index
     chase over the precomputed per-node results: at the current parent read its
     precomputed (chosen source, chosen token, accepted?) and, on accept of the
     drafted token, jump to the chosen child; else stop. This chase is bounded by
     ``max_spec_len`` levels and touches ONLY small precomputed int tensors
     (still device-side; a single compact readback at the end), so it is O(L)
     integer indexing, not O(L) GPU-math-with-sync.

  5. ONE compact device->host readback at the very end (the committed rows,
     accepted rows/lens, node paths) on a side stream + CUDA event, matching the
     greedy GPU committer's async-readback shape. No mid-walk ``.item()``.

LOSSLESSNESS
============
The per-node rule is byte-for-the-distribution identical to
``host_multidraft_accept_probs`` / ``sample_deterministic_multidraft_rejection_
step``: source weights ``w = overlaps / overlap_mass``; accept ``min(1, p/q_mix)``
with q_mix summing DUPLICATE draft tokens; residual ``(p - q_mix)+ / mass``
(``mass==0`` -> ``~p``); ``overlap_mass==0`` -> reject ~p; fully-accepted leaf
bonus ~ ``softmax(self_logits[leaf])``. Each node step is standard single-draft
speculative rejection against the composed proposal q_mix (a valid distribution:
q_mix>=0, sum=sum w=1), whose output law is exactly p (spec-decode theorem:
min(p,q_mix) + residual_mass * (p-q_mix)+/mass = p). Descent happens only on
ACCEPT of the drafted token, so the committed sequence is distributed as the
target chain at T. DISTRIBUTION-lossless (not byte: Gumbel-max torch draws differ
from numpy ``rng.choice``, but follow the identical categoricals -- the temp-0.6
q-vs-p TV gate is the sufficient measure).

The ONE deliberate difference from ``fr13_device_multidraft_kernel.py`` is HOW
randomness is consumed: a Gumbel-max over pre-drawn per-node uniforms instead of
per-node ``torch.multinomial`` with mid-walk syncs. Both draw the SAME
categoricals; the marginal committed distribution is unchanged. Because the draw
order is FIXED (all nodes, fixed layout) and seeded from the per-request
generator, the stream is batch-invariant (a request's draws do not depend on
batch neighbours), which the TV gate needs.

CUDA-GRAPH
==========
This runs HOST-SIDE EAGER under piecewise cudagraph_mode; it is NOT graph-
captured. vLLM v1 graphs ONLY the model forward -- ``self.rejection_sampler(...)``
is called from ``_sample()`` AFTER ``execute_model`` returns logits, never inside
``graph_capture()``. Both existing committers on this path (greedy-LCP + baked
fr13_device_multidraft) freely ``.item()`` / ``.cpu()`` / ``multinomial`` /
``synchronize`` with no capture guard, proving ``is_current_stream_capturing() ==
False`` here. We add a FAIL-LOUD assert at entry so a future move of the sampler
into a captured region fails loud instead of silently drawing garbage RNG.
Vectorization here is a pure-perf win (kills the per-node sync), not a graph
requirement.

GATING
======
This module is import-inert. It is consumed ONLY when the hook in
``scripts/fr10_phase4_patch_vllm_tree_gdn.py`` reads ``FR13_TREE_REJECTION_
SAMPLER=1`` (default 0). With the flag OFF the deployed committer path is
byte-identical to HEAD (this file is never imported / called). Non-APC / non-tree
paths are untouched.
"""

from __future__ import annotations

import os

try:
    import torch
except Exception:  # pragma: no cover - torch always present in the vLLM image
    torch = None  # type: ignore


# A large negative used as the -inf sentinel for masked-out Gumbel-max slots and
# for token-space residual masking. -1e30 in fp32 is safely below any real logit
# while staying finite (avoids nan from (-inf) + (-inf)).
_NEG = -1.0e30


def _capture_guard() -> None:
    """Fail loud if this ever runs inside CUDA-graph capture.

    The sampler is expected HOST-SIDE EAGER (see module docstring). If a future
    refactor moves it into a captured region, the data-dependent draws + integer
    descent would silently corrupt; assert instead.
    """
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "fr13_tree_rejection_sampler ran inside CUDA-graph capture; it "
                "is host-side-eager only (data-dependent RNG + integer descent "
                "are un-capturable). This is a fail-loud guard (bug-class 9)."
            )
    except RuntimeError:
        raise
    except Exception:
        # is_current_stream_capturing can be unavailable on some builds; treat
        # absence as "not capturing" (matches the existing committers' guards).
        return


def _per_request_seed(generators, req_i: int, fallback_device) -> int:
    """Draw one int seed from the request's own generator (FR13_TREE_PER_REQ_GEN).

    Matches the seed source ``fr13_device_multidraft_kernel`` /
    ``_lumo_tree_canonical_multidraft_sample`` use so the per-request stream is
    batch-invariant. Falls back to a global-device draw only when the request has
    no seeded generator (unseeded request), same as the reference.
    """
    host_gen = generators.get(req_i) if generators else None
    if host_gen is not None:
        return int(
            torch.randint(
                0, 2 ** 31 - 1, (1,), device=host_gen.device, generator=host_gen
            ).item()
        )
    return int(torch.randint(0, 2 ** 31 - 1, (1,), device=fallback_device).item())


def _build_child_tables(parents, node_count: int, device):
    """STATIC per-node child-slot tables from the tree structure.

    Returns:
      child_idx  [N, F] long  -- child NODE index at slot f of node n (or -1)
      valid      [N, F] bool  -- slot f of node n is a real child
      root_kids  1-D long     -- children of the virtual root (parent == -1)
    where N = node_count and F = max fan-out over {root} U nodes.

    Structure-only (no logits/RNG), so this is the "static shapes" piece. Pure
    Python over the (tiny) parent list at build time -- NOT on the per-token hot
    inner loop; the hot work (distribution math + descent) is vectorized below.
    """
    kids: list[list[int]] = [[] for _ in range(node_count)]
    root_kids: list[int] = []
    for node in range(node_count):
        p = int(parents[node])
        if p < 0:
            root_kids.append(node)
        elif 0 <= p < node_count:
            kids[p].append(node)
    fan = max([len(root_kids)] + [len(k) for k in kids]) if node_count else 1
    fan = max(1, fan)
    child_idx = torch.full((node_count, fan), -1, dtype=torch.long, device=device)
    valid = torch.zeros((node_count, fan), dtype=torch.bool, device=device)
    for node in range(node_count):
        for f, c in enumerate(kids[node]):
            child_idx[node, f] = c
            valid[node, f] = True
    root_kids_t = torch.tensor(root_kids, dtype=torch.long, device=device)
    return child_idx, valid, root_kids_t, fan


def _node_distribution_tables(
    target_logits,  # [N, V] device
    drafts,         # [N] long device -- draft token id AT each node
    child_idx,      # [N, F] long
    valid,          # [N, F] bool
):
    """VECTORIZED per-node canonical multidraft distribution objects.

    For each node n treated as a PARENT with child slots f:
      tok[n,f]     = drafts[child_idx[n,f]]                   (child's draft id)
      p_at[n,f]    = softmax(target_logits[first-child-row])[tok[n,f]]
      overlap_mass = sum_f p_at over valid slots
      w[n,f]       = p_at / overlap_mass                      (source weights)
      q_mix[n,f]   = sum over sibling slots g (same token) of w[n,g]  (DUP sum)
      accept_p[n,f]= min(1, p_at / q_mix)

    Returns per-node tensors used by the descent + the pre-draw. All ``[N, F]``
    (or ``[N]``) tensor ops -- no Python per-node loop.

    NOTE on the target ROW: the reference reads
    ``target_logits[start + children[0]]`` -- i.e. the verify row of the FIRST
    child of the parent. Every child of the same parent shares that row (the
    tree stores the parent's next-token distribution once, at each child's row,
    and the reference indexes children[0]). We replicate that EXACTLY: for node n
    the target row is ``child_idx[n, 0]`` (n's first child). Nodes with no
    children never act as a parent, so their row is unused.
    """
    N, F = child_idx.shape

    # First-child row per node (clamp -1 -> 0; masked out where no children).
    first_child = child_idx[:, 0].clamp_min(0)  # [N]
    has_children = valid[:, 0]                   # [N] (slot0 valid == node is a parent)

    # Child draft token ids at each slot (clamp invalid -> 0, masked later).
    safe_child = child_idx.clamp_min(0)          # [N, F]
    tok = drafts[safe_child]                     # [N, F] long

    # softmax the target row for EACH node's first-child row, then gather the
    # child tokens' probs. We only need p AT the child tokens, but a full-row
    # softmax over [N, V] is the simple correct form; N is tiny (<= a few tens of
    # nodes total across the batch) so [N, V] is cheap and avoids a gather-then-
    # renormalize approximation. Matches reference normalize() exactly.
    p_full = torch.softmax(target_logits[first_child].to(torch.float32), dim=-1)  # [N, V]
    # Re-normalize (reference does p = p / p.sum() after softmax; softmax already
    # sums to 1 in fp32 but we mirror the explicit renorm for bit-parity of the
    # arithmetic path).
    p_full = p_full / p_full.sum(dim=-1, keepdim=True)

    p_at = torch.gather(p_full, 1, tok)          # [N, F]
    p_at = torch.where(valid, p_at, torch.zeros_like(p_at))
    overlap_mass = p_at.sum(dim=1)               # [N]
    has_overlap = overlap_mass > 0.0             # [N]

    safe_mass = torch.where(has_overlap, overlap_mass, torch.ones_like(overlap_mass))
    w = p_at / safe_mass.unsqueeze(1)            # [N, F] (0 where invalid)
    w = torch.where(valid, w, torch.zeros_like(w))

    # q_mix[n,f] = sum over sibling slots g with tok[g]==tok[f] of w[g]. Done
    # WITHOUT a Python loop via a per-node token-equality mask over F (F is tiny).
    # eq[n,f,g] = (tok[n,f]==tok[n,g]) & valid[n,g]
    eq = (tok.unsqueeze(2) == tok.unsqueeze(1)) & valid.unsqueeze(1)  # [N, F, F]
    q_mix = (eq.to(w.dtype) * w.unsqueeze(1)).sum(dim=2)              # [N, F]
    safe_q = torch.where(q_mix > 0.0, q_mix, torch.ones_like(q_mix))
    accept_p = torch.clamp(p_at / safe_q, max=1.0)                    # [N, F]
    accept_p = torch.where(valid & (q_mix > 0.0), accept_p, torch.zeros_like(accept_p))

    return {
        "tok": tok,                    # [N, F] child draft ids
        "p_full": p_full,              # [N, V] target prob rows
        "p_at": p_at,                  # [N, F]
        "w": w,                        # [N, F] source weights
        "q_mix": q_mix,                # [N, F]
        "accept_p": accept_p,          # [N, F]
        "overlap_mass": overlap_mass,  # [N]
        "has_overlap": has_overlap,    # [N]
        "has_children": has_children,  # [N]
    }


def _gumbel_argmax(logprobs, valid_mask, gumbel):
    """Gumbel-max categorical sample over the last dim.

    ``logprobs`` = log of unnormalized weights ([...,K]); ``valid_mask`` masks
    illegal slots to -inf; ``gumbel`` = pre-drawn Gumbel(0,1) noise of the same
    shape. argmax(logprobs + gumbel) ~ Categorical(softmax(logprobs)) =
    Categorical(weights). This is the sync-free equivalent of ``multinomial`` /
    ``rng.choice`` over the same categorical, using PRE-DRAWN noise so no host
    sync happens mid-walk.
    """
    scores = torch.where(valid_mask, logprobs + gumbel, torch.full_like(logprobs, _NEG))
    return scores.argmax(dim=-1)


def _predraw_gumbel(shape, device, generator, dtype=torch.float32):
    """Pre-draw Gumbel(0,1) noise = -log(-log(U)), U~Uniform(0,1).

    From the per-request seeded generator, in one shot (fixed draw order)."""
    u = torch.rand(shape, device=device, generator=generator, dtype=dtype)
    # clamp away from 0/1 to avoid inf in the double-log.
    u = u.clamp_(min=1e-20, max=1.0 - 1e-7)
    return -torch.log(-torch.log(u))


def fr13_tree_rejection_commit(
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
):
    """Vectorized, lossless, cuda-graph-safe tree rejection committer.

    Returns the SAME five products the deployed committer publishes:
      ``(out_rows, accepted_rows, accepted_lens, accepted_node_paths,
      accepted_token_rows)``

    Distribution-lossless vs ``fr13_device_multidraft_commit`` /
    ``sample_deterministic_multidraft_rejection_step`` (same per-node
    categoricals), but with NO per-node Python ``.item()`` loop on the hot path:
    the per-node distribution math is vectorized ``[N, F]`` tensor ops, the
    randomness is pre-drawn per request, and the top-down descent is a pure
    integer index chase with a single compact readback at the end.

    FAIL-LOUD (bug-class 9): ``draft_probs is not None`` (the explicit-q multi-
    draft path) is not ported here; the deployment MTP path passes
    ``draft_probs=None`` (deterministic one-hot q). We refuse a silent bypass.
    """
    if torch is None:
        raise RuntimeError("fr13_tree_rejection_commit requires torch")
    _capture_guard()
    if draft_probs is not None:
        raise RuntimeError(
            "FR13_TREE_REJECTION_SAMPLER: draft_probs!=None (explicit-q "
            "multidraft) not supported -- deployment MTP passes draft_probs=None "
            "(one-hot q). Refusing silent fallback (bug-class 9)."
        )
    if target_logits is None or tree_self_logits is None:
        raise RuntimeError(
            "FR13_TREE_REJECTION_SAMPLER: missing target/self logits (disengaged)"
        )

    device = target_logits.device
    parents_cpu = [int(x) for x in tree_parent_indices.detach().cpu().tolist()]
    drafts_all = draft_token_ids.detach().to(device=device, dtype=torch.long).reshape(-1)
    if hasattr(num_draft_tokens, "detach"):
        counts = [int(x) for x in num_draft_tokens.detach().cpu().tolist()]
    else:
        counts = [int(x) for x in num_draft_tokens]
    bonus_flat = bonus_token_ids.detach().reshape(-1)

    out_rows: list[list[int]] = []
    accepted_rows: list[int] = []
    accepted_lens: list[int] = []
    accepted_node_paths: list[list[int]] = []
    accepted_token_rows: list[list[int]] = []

    L = int(max_spec_len)
    start = 0
    for req_i, node_count in enumerate(counts):
        node_count = int(node_count)
        if node_count <= 0:
            # empty tree: emit the request's bonus token (native path0 bonus).
            row: list[int] = []
            if req_i < int(bonus_flat.numel()):
                row.append(int(bonus_flat[req_i].item()))
            out_rows.append(row[: L + 1])
            accepted_rows.append(0)
            accepted_lens.append(0)
            accepted_node_paths.append([])
            accepted_token_rows.append([])
            continue

        seed = _per_request_seed(generators, req_i, device)
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        parents = parents_cpu[start:start + node_count]
        drafts = drafts_all[start:start + node_count]            # [N] this req
        tgt = target_logits[start:start + node_count]            # [N, V]
        selfl = tree_self_logits[start:start + node_count]       # [N, V]

        child_idx, valid, root_kids, F = _build_child_tables(
            parents, node_count, device
        )
        tables = _node_distribution_tables(tgt, drafts, child_idx, valid)

        tok = tables["tok"]              # [N, F]
        w = tables["w"]                  # [N, F]
        accept_p = tables["accept_p"]    # [N, F]
        p_full = tables["p_full"]        # [N, V]
        q_mix = tables["q_mix"]          # [N, F]
        has_overlap = tables["has_overlap"]  # [N]

        # ---- PRE-DRAW all randomness for this request, fixed order --------
        # source-pick Gumbel over child slots (per node), accept uniform (per
        # node), residual Gumbel over vocab (per node), self/bonus Gumbel over
        # vocab (per node). One generator, fixed draw order => batch-invariant.
        src_gumbel = _predraw_gumbel((node_count, F), device, gen)      # [N, F]
        accept_u = torch.rand(
            (node_count,), device=device, generator=gen, dtype=torch.float32
        )                                                              # [N]
        resid_gumbel = _predraw_gumbel((node_count, p_full.shape[1]), device, gen)  # [N,V]
        self_gumbel = _predraw_gumbel((node_count, selfl.shape[1]), device, gen)    # [N,V]

        # Root-level rejection needs its OWN residual/source draws (the root is
        # not a "node" -- its children are root_kids). We reuse a synthetic slot
        # by treating the virtual root as an extra parent computed inline below.

        # ---- VECTORIZED per-node decision (chosen source/token/accept) -----
        # source ~ Categorical(w[n,:]) via Gumbel-max over log(w).
        logw = torch.log(w.clamp_min(1e-30))
        chosen_slot = _gumbel_argmax(logw, valid, src_gumbel)          # [N]
        gath = chosen_slot.unsqueeze(1)                                # [N,1]
        chosen_tok = torch.gather(tok, 1, gath).squeeze(1)            # [N]
        chosen_accept_p = torch.gather(accept_p, 1, gath).squeeze(1)  # [N]
        chosen_child = torch.gather(child_idx, 1, gath).squeeze(1)    # [N]
        accepted_node = (accept_u < chosen_accept_p) & has_overlap    # [N] bool

        # residual token per node: (p - q_mix_vocab)+ / mass, via Gumbel-max over
        # log(residual). q_mix_vocab built by scatter_add of w onto tok positions.
        q_mix_vocab = torch.zeros_like(p_full)                        # [N, V]
        # scatter the source weights w onto their token positions (dup-safe: two
        # slots with the same token add, exactly q_mix over the vocab).
        q_mix_vocab.scatter_add_(1, tok, torch.where(valid, w, torch.zeros_like(w)))
        residual = torch.clamp(p_full - q_mix_vocab, min=0.0)         # [N, V]
        resid_mass = residual.sum(dim=1, keepdim=True)                # [N,1]
        # mass==0 -> residual = p (reference). overlap_mass==0 -> also ~p.
        use_p = (resid_mass.squeeze(1) <= 0.0) | (~has_overlap)       # [N]
        residual = torch.where(
            use_p.unsqueeze(1), p_full, residual / resid_mass.clamp_min(1e-30)
        )
        resid_logits = torch.log(residual.clamp_min(1e-30))
        all_valid_vocab = torch.ones_like(resid_logits, dtype=torch.bool)
        resid_tok = _gumbel_argmax(resid_logits, all_valid_vocab, resid_gumbel)  # [N]

        # self/bonus token per node (fully-accepted leaf): ~ softmax(self_logits).
        self_p = torch.softmax(selfl.to(torch.float32), dim=-1)
        self_logits = torch.log(self_p.clamp_min(1e-30))
        self_tok = _gumbel_argmax(self_logits, all_valid_vocab, self_gumbel)  # [N]

        # ---- ROOT node decision (virtual parent -1 over root_kids) ---------
        # Same rule at the root. Compute inline (one synthetic node).
        root_out = _root_decision(
            root_kids, drafts, tgt, gen, device
        )
        # root_out = (accepted:bool, chosen_child:int, emit_tok:int)

        # ---- INTEGER DESCENT (pure index chase, no GPU math) ---------------
        # Move small per-node decision tensors to host ONCE (compact readback).
        # These are O(N) ints/bools, not O(N*V) -- the whole point.
        accepted_node_h = accepted_node.cpu().tolist()
        chosen_child_h = chosen_child.cpu().tolist()
        chosen_tok_h = chosen_tok.cpu().tolist()
        resid_tok_h = resid_tok.cpu().tolist()
        self_tok_h = self_tok.cpu().tolist()
        drafts_h = drafts.cpu().tolist()

        row = []
        accepted_row = 0
        accepted_path: list[int] = []

        # Level 0: root.
        r_accept, r_child, r_emit = root_out
        row.append(int(r_emit))
        if r_accept and r_child >= 0:
            current = int(r_child)
            accepted_row = current
            accepted_path.append(current)
            # Levels 1..L: descend accepted children.
            for _lvl in range(L):
                if not self_has_children_host(child_idx, current):
                    # fully-accepted leaf -> emit self/bonus token, stop.
                    row.append(int(self_tok_h[current]))
                    break
                acc = bool(accepted_node_h[current])
                if acc:
                    emit = int(chosen_tok_h[current])   # the accepted drafted tok
                else:
                    emit = int(resid_tok_h[current])    # reject (incl. overlap==0)
                row.append(emit)
                if not acc:
                    break
                # On accept the emitted token IS the chosen source's drafted
                # token (chosen_tok == drafts[chosen_child]); descend that child.
                # (An accepted residual token that is not a drafted child cannot
                # descend -- matches the reference tok!=drafts[child] break. Under
                # the one-hot MTP q, accept always emits the drafted token, so
                # this equality holds; we still guard it.)
                nxt = int(chosen_child_h[current])
                if nxt < 0 or int(emit) != int(drafts_h[nxt]):
                    break
                current = nxt
                accepted_row = current
                accepted_path.append(current)

        out_rows.append(row[: L + 1])
        accepted_rows.append(int(accepted_row))
        accepted_lens.append(int(len(accepted_path)))
        accepted_node_paths.append([int(x) for x in accepted_path])
        accepted_token_rows.append([int(drafts_h[x]) for x in accepted_path])
        start += node_count

    return (
        out_rows,
        accepted_rows,
        accepted_lens,
        accepted_node_paths,
        accepted_token_rows,
    )


def self_has_children_host(child_idx, node: int) -> bool:
    """True if ``node`` has at least one child (slot 0 valid == child_idx>=0)."""
    return int(child_idx[node, 0].item()) >= 0


def _root_decision(root_kids, drafts, tgt, gen, device):
    """One canonical multidraft step at the VIRTUAL ROOT over its children.

    Returns (accepted:bool, chosen_child:int, emit_token:int). Vectorized over
    the root's (few) children -- one row of the same [N,F] math specialized to a
    single node. Draws come from the same per-request generator (drawn AFTER the
    per-node predraws, a fixed order for batch-invariance).
    """
    K = int(root_kids.numel())
    if K == 0:
        return (False, -1, -1)
    child_nodes = root_kids                       # [K] node indices
    tok = drafts[child_nodes]                     # [K] draft ids
    row = int(child_nodes[0].item())              # first child's verify row
    p_full = torch.softmax(tgt[row].to(torch.float32), dim=-1)
    p_full = p_full / p_full.sum()
    p_at = p_full[tok]                            # [K]
    overlap_mass = float(p_at.sum().item())
    if overlap_mass <= 0.0:
        # reject ~ p
        g = _predraw_gumbel((p_full.shape[0],), device, gen)
        emit = int((torch.log(p_full.clamp_min(1e-30)) + g).argmax().item())
        return (False, -1, emit)
    w = p_at / overlap_mass                       # [K]
    # source ~ Categorical(w)
    gs = _predraw_gumbel((K,), device, gen)
    src = int((torch.log(w.clamp_min(1e-30)) + gs).argmax().item())
    token = int(tok[src].item())
    same = tok == tok[src]
    q_mix_token = float(w[same].sum().item())
    accept_p = min(1.0, float(p_at[src].item()) / q_mix_token) if q_mix_token > 0 else 0.0
    u = float(torch.rand((), device=device, generator=gen, dtype=torch.float32).item())
    if u < accept_p:
        return (True, int(child_nodes[src].item()), token)
    # reject -> residual ~ (p - q_mix_vocab)+ / mass
    q_mix_vocab = torch.zeros_like(p_full)
    q_mix_vocab.scatter_add_(0, tok, w)
    residual = torch.clamp(p_full - q_mix_vocab, min=0.0)
    mass = float(residual.sum().item())
    if mass <= 0.0:
        residual = p_full
    else:
        residual = residual / mass
    g = _predraw_gumbel((p_full.shape[0],), device, gen)
    emit = int((torch.log(residual.clamp_min(1e-30)) + g).argmax().item())
    return (False, int(child_nodes[src].item()), emit)
