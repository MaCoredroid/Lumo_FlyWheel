from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_tree_rejection_sampler import (
    canonical_draft_weights,
    chi_square_gof,
    count_pairs,
    count_tokens,
    exact_multidraft_distribution,
    exact_rejection_distribution,
    mixture_distribution,
    residual_distribution,
    sample_biased_multidraft_control,
    sample_deterministic_multidraft_rejection_step,
    sample_multidraft_rejection,
    sample_multidraft_rejection_step,
    sample_two_step_tree_rejection,
)


def _target() -> np.ndarray:
    return np.array([0.05, 0.12, 0.21, 0.08, 0.19, 0.25, 0.10], dtype=np.float64)


def _drafts() -> list[np.ndarray]:
    return [
        np.array([0.09, 0.10, 0.18, 0.10, 0.16, 0.24, 0.13], dtype=np.float64),
        np.array([0.04, 0.18, 0.24, 0.06, 0.17, 0.21, 0.10], dtype=np.float64),
        np.array([0.06, 0.08, 0.16, 0.12, 0.25, 0.22, 0.11], dtype=np.float64),
    ]


def test_canonical_multidraft_decomposition_is_analytic_identity() -> None:
    p = _target()
    qs = _drafts()
    weights = canonical_draft_weights(p, qs)
    q_mix = mixture_distribution(qs, weights)

    exact_single = exact_rejection_distribution(p, q_mix)
    exact_multi = exact_multidraft_distribution(p, qs, weights)

    assert np.allclose(exact_single, p, atol=1e-12)
    assert np.allclose(exact_multi, p, atol=1e-12)
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)


def test_multidraft_rejection_sampler_converges_to_target_distribution() -> None:
    rng = np.random.default_rng(20260604)
    p = _target()
    samples = sample_multidraft_rejection(p, _drafts(), n=200_000, rng=rng)
    observed = count_tokens(samples, len(p))
    result = chi_square_gof(observed, p)

    assert result.passed, result


def test_multidraft_rejection_step_converges_to_target_distribution() -> None:
    rng = np.random.default_rng(20260607)
    p = _target()
    samples = np.fromiter(
        (
            sample_multidraft_rejection_step(p, _drafts(), rng=rng).token_id
            for _ in range(200_000)
        ),
        dtype=np.int64,
        count=200_000,
    )
    result = chi_square_gof(count_tokens(samples, len(p)), p)

    assert result.passed, result


def test_deterministic_multidraft_step_converges_to_target_distribution() -> None:
    rng = np.random.default_rng(20260608)
    p = _target()
    draft_token_ids = [0, 2, 5]
    samples = np.fromiter(
        (
            sample_deterministic_multidraft_rejection_step(
                p, draft_token_ids, rng=rng
            ).token_id
            for _ in range(200_000)
        ),
        dtype=np.int64,
        count=200_000,
    )
    result = chi_square_gof(count_tokens(samples, len(p)), p)

    assert result.passed, result


def test_deterministic_one_hot_step_accept_rate_and_residual() -> None:
    rng = np.random.default_rng(20260609)
    p = np.array([0.07, 0.20, 0.11, 0.17, 0.31, 0.14], dtype=np.float64)
    draft_token = 1
    n = 100_000

    accepted = 0
    rejected_tokens: list[int] = []
    for _ in range(n):
        step = sample_deterministic_multidraft_rejection_step(
            p, [draft_token], rng=rng
        )
        if step.accepted:
            accepted += 1
            assert step.token_id == draft_token
        else:
            rejected_tokens.append(step.token_id)

    observed_accept = accepted / n
    expected_accept = float(p[draft_token])
    sigma = (expected_accept * (1.0 - expected_accept) / n) ** 0.5
    assert abs(observed_accept - expected_accept) < 5.0 * sigma

    q = np.zeros_like(p)
    q[draft_token] = 1.0
    expected_residual = residual_distribution(p, q)
    observed_residual = count_tokens(
        np.asarray(rejected_tokens, dtype=np.int64), len(p)
    )
    positive = expected_residual > 0
    assert observed_residual[~positive].sum() == 0
    result = chi_square_gof(observed_residual[positive], expected_residual[positive])
    assert result.passed, result


def test_branched_tree_rejection_sampler_converges_to_target_joint() -> None:
    rng = np.random.default_rng(20260605)
    root_p = np.array([0.38, 0.17, 0.29, 0.16], dtype=np.float64)
    root_qs = [
        np.array([0.45, 0.13, 0.24, 0.18], dtype=np.float64),
        np.array([0.32, 0.20, 0.33, 0.15], dtype=np.float64),
    ]
    child_ps = [
        np.array([0.10, 0.55, 0.20, 0.15], dtype=np.float64),
        np.array([0.31, 0.14, 0.40, 0.15], dtype=np.float64),
        np.array([0.25, 0.25, 0.12, 0.38], dtype=np.float64),
        np.array([0.45, 0.20, 0.20, 0.15], dtype=np.float64),
    ]
    child_qs = [
        [
            np.array([0.14, 0.48, 0.23, 0.15], dtype=np.float64),
            np.array([0.08, 0.58, 0.18, 0.16], dtype=np.float64),
        ],
        [
            np.array([0.25, 0.18, 0.42, 0.15], dtype=np.float64),
            np.array([0.36, 0.12, 0.36, 0.16], dtype=np.float64),
        ],
        [
            np.array([0.20, 0.30, 0.10, 0.40], dtype=np.float64),
            np.array([0.31, 0.21, 0.15, 0.33], dtype=np.float64),
        ],
        [
            np.array([0.50, 0.16, 0.18, 0.16], dtype=np.float64),
            np.array([0.41, 0.25, 0.22, 0.12], dtype=np.float64),
        ],
    ]

    samples = sample_two_step_tree_rejection(
        root_p,
        root_qs,
        child_ps,
        child_qs,
        n=250_000,
        rng=rng,
    )
    expected = np.concatenate([root_p[parent] * child_ps[parent] for parent in range(4)])
    result = chi_square_gof(count_pairs(samples, 4), expected)

    assert result.passed, result


def test_biased_multidraft_negative_control_fails_convergence() -> None:
    rng = np.random.default_rng(20260606)
    p = _target()
    samples = sample_biased_multidraft_control(p, _drafts(), n=200_000, rng=rng)
    observed = count_tokens(samples, len(p))
    result = chi_square_gof(observed, p)

    assert not result.passed
    assert result.statistic > result.threshold * 4
