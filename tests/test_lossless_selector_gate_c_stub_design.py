"""Design stub for FR9 Gate C; this is not a selector proof.

The real Gate C must exercise the production multi-draft selector once that
selector exists.  This file only locks down the statistical negative-control
shape from /tmp/fr9_lossless_research.md section 3 so future Gate C work does
not regress to the old tautological placeholder.
"""

from __future__ import annotations

import math
import random
from collections import Counter


_TRIALS = 100_000


def _sample_categorical(rng: random.Random, probs: list[float]) -> int:
    draw = rng.random()
    total = 0.0
    for idx, prob in enumerate(probs):
        total += prob
        if draw < total:
            return idx
    return len(probs) - 1


def _sample_counts(
    seed: int,
    probs: list[float],
    trials: int = _TRIALS,
) -> Counter[int]:
    rng = random.Random(seed)
    return Counter(_sample_categorical(rng, probs) for _ in range(trials))


def _best_of_spines_max_order_counts(
    seed: int,
    target_probs_desc: list[float],
    trials: int = _TRIALS,
) -> Counter[int]:
    """Negative control: emit the higher-probability token of two iid samples."""
    rng = random.Random(seed)
    counts: Counter[int] = Counter()
    for _ in range(trials):
        x1 = _sample_categorical(rng, target_probs_desc)
        x2 = _sample_categorical(rng, target_probs_desc)
        counts[min(x1, x2)] += 1
    return counts


def _max_order_statistic_probs_desc(target_probs_desc: list[float]) -> list[float]:
    """Analytic max-order p'(z)=p(z)*(2*CDF(z)-p(z)).

    The fixture stores probabilities in descending order and the negative
    control emits the lower token id, so CDF is the descending-order tail mass.
    """
    out: list[float] = []
    tail = 1.0
    for prob in target_probs_desc:
        out.append(prob * (2.0 * tail - prob))
        tail -= prob
    total = sum(out)
    return [x / total for x in out]


def _chi_square_stat(counts: Counter[int], expected_probs: list[float]) -> float:
    trials = sum(counts.values())
    return sum(
        (counts[idx] - trials * prob) ** 2 / (trials * prob)
        for idx, prob in enumerate(expected_probs)
    )


def _total_variation(counts: Counter[int], expected_probs: list[float]) -> float:
    trials = sum(counts.values())
    return 0.5 * sum(
        abs(counts[idx] / trials - prob)
        for idx, prob in enumerate(expected_probs)
    )


def _noncentrality(trials: int, actual: list[float], expected: list[float]) -> float:
    return trials * sum(
        (actual[idx] - expected[idx]) ** 2 / expected[idx]
        for idx in range(len(expected))
    )


def test_gate_c_stub_negative_control_matches_analytic_order_statistic():
    target = [0.40, 0.30, 0.15, 0.10, 0.05]
    analytic_bias = _max_order_statistic_probs_desc(target)
    counts = _best_of_spines_max_order_counts(seed=17, target_probs_desc=target)

    # df=4, alpha=0.01 chi-square critical value ~= 13.277.
    assert _chi_square_stat(counts, target) > 13.277
    assert _chi_square_stat(counts, analytic_bias) < 13.277
    assert _total_variation(counts, analytic_bias) < 0.01
    assert _noncentrality(_TRIALS, analytic_bias, target) > 1_000


def test_gate_c_stub_unbiased_target_sampler_sanity_check():
    target = [0.40, 0.30, 0.15, 0.10, 0.05]
    counts = _sample_counts(seed=23, probs=target)

    assert _chi_square_stat(counts, target) < 13.277
    assert _total_variation(counts, target) < 4.0 * math.sqrt(len(target) / _TRIALS)
