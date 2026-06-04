"""CPU reference for FR10 tree rejection-sampler correctness.

This module is intentionally model-free. The verifier supplies target
probabilities p for candidate nodes; the drafter supplies one or more draft
distributions q. Correctness of the committer is a distributional statement
about p and q, independent of Qwen/MTP details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ChiSquareResult:
    statistic: float
    df: int
    threshold: float
    passed: bool


@dataclass(frozen=True)
class RejectionStep:
    token_id: int
    source_index: int
    accepted: bool


def normalize(probs: Sequence[float] | Array) -> Array:
    arr = np.asarray(probs, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D probability vector, got shape {arr.shape}")
    if np.any(arr < 0):
        raise ValueError("probabilities must be non-negative")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    return arr / total


def canonical_draft_weights(target_p: Sequence[float] | Array, draft_qs: Sequence[Sequence[float] | Array]) -> Array:
    """Choose draft branches by their single-draft overlap with target.

    The weights are an acceptance-aware canonical decomposition: better draft
    branches are selected more often, then the selected branch is verified by
    ordinary speculative rejection sampling against the resulting mixture q.
    """

    p = normalize(target_p)
    qs = np.stack([normalize(q) for q in draft_qs])
    if qs.shape[1] != p.shape[0]:
        raise ValueError("draft and target vocab sizes differ")
    overlaps = np.minimum(qs, p[None, :]).sum(axis=1)
    return normalize(overlaps)


def mixture_distribution(draft_qs: Sequence[Sequence[float] | Array], weights: Sequence[float] | Array) -> Array:
    qs = np.stack([normalize(q) for q in draft_qs])
    w = normalize(weights)
    if qs.shape[0] != w.shape[0]:
        raise ValueError("number of draft distributions and weights differ")
    return normalize((w[:, None] * qs).sum(axis=0))


def residual_distribution(target_p: Sequence[float] | Array, proposal_q: Sequence[float] | Array) -> Array:
    p = normalize(target_p)
    q = normalize(proposal_q)
    if p.shape != q.shape:
        raise ValueError("target and proposal vocab sizes differ")
    residual = np.maximum(p - q, 0.0)
    mass = float(residual.sum())
    if mass == 0.0:
        return p.copy()
    return residual / mass


def exact_rejection_distribution(target_p: Sequence[float] | Array, proposal_q: Sequence[float] | Array) -> Array:
    """Closed-form output distribution of standard speculative rejection."""

    p = normalize(target_p)
    q = normalize(proposal_q)
    accept = np.minimum(p, q)
    residual_mass = 1.0 - float(accept.sum())
    if residual_mass <= 1e-15:
        return normalize(accept)
    return normalize(accept + residual_mass * residual_distribution(p, q))


def exact_multidraft_distribution(
    target_p: Sequence[float] | Array,
    draft_qs: Sequence[Sequence[float] | Array],
    weights: Sequence[float] | Array | None = None,
) -> Array:
    """Canonical decomposition: draft selection + single-draft rejection."""

    w = canonical_draft_weights(target_p, draft_qs) if weights is None else normalize(weights)
    q_mix = mixture_distribution(draft_qs, w)
    return exact_rejection_distribution(target_p, q_mix)


def sample_multidraft_rejection(
    target_p: Sequence[float] | Array,
    draft_qs: Sequence[Sequence[float] | Array],
    *,
    n: int,
    rng: np.random.Generator,
    weights: Sequence[float] | Array | None = None,
) -> Array:
    """Draw samples from the exact multi-draft rejection sampler."""

    p = normalize(target_p)
    qs = np.stack([normalize(q) for q in draft_qs])
    w = canonical_draft_weights(p, qs) if weights is None else normalize(weights)
    q_mix = mixture_distribution(qs, w)
    residual = residual_distribution(p, q_mix)
    sources = rng.choice(qs.shape[0], size=n, p=w)
    tokens = np.empty(n, dtype=np.int64)
    for source in range(qs.shape[0]):
        mask = sources == source
        count = int(mask.sum())
        if count:
            tokens[mask] = rng.choice(p.shape[0], size=count, p=qs[source])
    accept_prob = np.minimum(1.0, p[tokens] / q_mix[tokens])
    accepted = rng.random(n) < accept_prob
    rejected_count = int((~accepted).sum())
    if rejected_count:
        tokens[~accepted] = rng.choice(p.shape[0], size=rejected_count, p=residual)
    return tokens


def sample_multidraft_rejection_step(
    target_p: Sequence[float] | Array,
    draft_qs: Sequence[Sequence[float] | Array],
    *,
    rng: np.random.Generator,
    weights: Sequence[float] | Array | None = None,
) -> RejectionStep:
    """One canonical multi-draft rejection step with source trace.

    This is the serving-facing reference primitive. It uses the same
    decomposition as ``sample_multidraft_rejection`` but returns which draft
    branch was selected and whether the sampled draft token was accepted, so a
    tree committer can descend only when a verified candidate node exists.
    """

    p = normalize(target_p)
    qs = np.stack([normalize(q) for q in draft_qs])
    w = canonical_draft_weights(p, qs) if weights is None else normalize(weights)
    q_mix = mixture_distribution(qs, w)
    source = int(rng.choice(qs.shape[0], p=w))
    token = int(rng.choice(p.shape[0], p=qs[source]))
    accept_prob = min(1.0, float(p[token] / q_mix[token]))
    accepted = bool(rng.random() < accept_prob)
    if accepted:
        return RejectionStep(token_id=token, source_index=source, accepted=True)
    residual = residual_distribution(p, q_mix)
    return RejectionStep(
        token_id=int(rng.choice(p.shape[0], p=residual)),
        source_index=source,
        accepted=False,
    )


def sample_deterministic_multidraft_rejection_step(
    target_p: Sequence[float] | Array,
    draft_token_ids: Sequence[int],
    *,
    rng: np.random.Generator,
) -> RejectionStep:
    """One exact multi-draft step for deterministic proposed tokens.

    vLLM's current MTP path passes ``draft_probs=None`` to the rejection
    sampler, matching its deterministic proposal behavior: each draft node
    proposes a fixed token. This is the one-hot specialization of the
    canonical multi-draft rule and avoids materializing huge vocab-sized q
    vectors in the serving committer.
    """

    p = normalize(target_p)
    tokens = np.asarray(draft_token_ids, dtype=np.int64)
    if tokens.ndim != 1 or tokens.size == 0:
        raise ValueError("draft_token_ids must be a non-empty 1-D sequence")
    if np.any(tokens < 0) or np.any(tokens >= p.shape[0]):
        raise ValueError("draft token id is outside the target vocabulary")

    overlaps = p[tokens].astype(np.float64, copy=True)
    overlap_mass = float(overlaps.sum())
    if overlap_mass <= 0.0:
        return RejectionStep(
            token_id=int(rng.choice(p.shape[0], p=p)),
            source_index=0,
            accepted=False,
        )

    weights = overlaps / overlap_mass
    source = int(rng.choice(tokens.size, p=weights))
    token = int(tokens[source])

    q_mix_token = float(weights[tokens == token].sum())
    accept_prob = min(1.0, float(p[token] / q_mix_token))
    if rng.random() < accept_prob:
        return RejectionStep(token_id=token, source_index=source, accepted=True)

    q_mix = np.zeros_like(p)
    for idx, token_id in enumerate(tokens):
        q_mix[int(token_id)] += weights[idx]
    residual = residual_distribution(p, q_mix)
    return RejectionStep(
        token_id=int(rng.choice(p.shape[0], p=residual)),
        source_index=source,
        accepted=False,
    )


def sample_biased_multidraft_control(
    target_p: Sequence[float] | Array,
    draft_qs: Sequence[Sequence[float] | Array],
    *,
    n: int,
    rng: np.random.Generator,
    weights: Sequence[float] | Array | None = None,
) -> Array:
    """A realistic lossy control: accept against the selected q_i, not q_mix.

    This resembles a wrong multi-draft committer that forgets the canonical
    mixture decomposition. It keeps all tokens in-vocabulary and is much less
    trivial than an out-of-vocabulary garbage control, but it is biased.
    """

    p = normalize(target_p)
    qs = np.stack([normalize(q) for q in draft_qs])
    w = canonical_draft_weights(p, qs) if weights is None else normalize(weights)
    q_mix = mixture_distribution(qs, w)
    residual = residual_distribution(p, q_mix)
    sources = rng.choice(qs.shape[0], size=n, p=w)
    tokens = np.empty(n, dtype=np.int64)
    selected_q = np.empty(n, dtype=np.float64)
    for source in range(qs.shape[0]):
        mask = sources == source
        count = int(mask.sum())
        if count:
            sampled = rng.choice(p.shape[0], size=count, p=qs[source])
            tokens[mask] = sampled
            selected_q[mask] = qs[source, sampled]
    accept_prob = np.minimum(1.0, p[tokens] / selected_q)
    accepted = rng.random(n) < accept_prob
    rejected_count = int((~accepted).sum())
    if rejected_count:
        tokens[~accepted] = rng.choice(p.shape[0], size=rejected_count, p=residual)
    return tokens


def sample_two_step_tree_rejection(
    root_target_p: Sequence[float] | Array,
    root_draft_qs: Sequence[Sequence[float] | Array],
    child_target_ps: Sequence[Sequence[float] | Array],
    child_draft_qs: Sequence[Sequence[Sequence[float] | Array]],
    *,
    n: int,
    rng: np.random.Generator,
) -> Array:
    """Sample a two-level tree sequence with exact per-node rejection."""

    root = sample_multidraft_rejection(root_target_p, root_draft_qs, n=n, rng=rng)
    out = np.empty((n, 2), dtype=np.int64)
    out[:, 0] = root
    for token in range(len(child_target_ps)):
        mask = root == token
        count = int(mask.sum())
        if count:
            out[mask, 1] = sample_multidraft_rejection(
                child_target_ps[token],
                child_draft_qs[token],
                n=count,
                rng=rng,
            )
    return out


def chi_square_gof(
    observed_counts: Sequence[int] | Array,
    expected_probs: Sequence[float] | Array,
    *,
    n_sigma: float = 6.0,
) -> ChiSquareResult:
    observed = np.asarray(observed_counts, dtype=np.float64)
    probs = normalize(expected_probs)
    if observed.shape != probs.shape:
        raise ValueError("observed and expected shapes differ")
    total = float(observed.sum())
    expected = total * probs
    if np.any(expected <= 0):
        raise ValueError("expected probabilities must all be positive")
    stat = float(((observed - expected) ** 2 / expected).sum())
    df = int(len(probs) - 1)
    threshold = float(df + n_sigma * np.sqrt(2.0 * df))
    return ChiSquareResult(statistic=stat, df=df, threshold=threshold, passed=stat <= threshold)


def count_tokens(samples: Array, vocab_size: int) -> Array:
    return np.bincount(np.asarray(samples, dtype=np.int64), minlength=vocab_size)


def count_pairs(samples: Array, vocab_size: int) -> Array:
    flat = np.asarray(samples[:, 0], dtype=np.int64) * vocab_size + np.asarray(
        samples[:, 1], dtype=np.int64
    )
    return np.bincount(flat, minlength=vocab_size * vocab_size)
