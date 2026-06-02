import random
from collections import Counter


def _sample_categorical(rng: random.Random, probs: list[float]) -> int:
    draw = rng.random()
    total = 0.0
    for idx, prob in enumerate(probs):
        total += prob
        if draw < total:
            return idx
    return len(probs) - 1


def _target_sampler(seed: int, target_probs: list[float], trials: int) -> Counter[int]:
    rng = random.Random(seed)
    return Counter(_sample_categorical(rng, target_probs) for _ in range(trials))


def _naive_longest_accepted_selector(
    *,
    target_sampler_seed: int,
    hidden_token: int,
    hidden_accepts_longer: bool,
    trials: int,
) -> Counter[int]:
    rng = random.Random(target_sampler_seed)
    counts: Counter[int] = Counter()
    for _ in range(trials):
        public_token = 0 if rng.random() < 0.70 else 1
        counts[hidden_token if hidden_accepts_longer else public_token] += 1
    return counts


def _freq(counter: Counter[int], token: int, trials: int) -> float:
    return counter[token] / trials


def test_gate_c_target_sampler_converges_on_synthetic_vocab():
    trials = 20_000
    target = _target_sampler(17, [0.70, 0.30], trials)

    assert abs(_freq(target, 0, trials) - 0.70) < 0.02
    assert abs(_freq(target, 1, trials) - 0.30) < 0.02


def test_gate_c_naive_longest_accepted_hidden_winner_fails_negative_control():
    trials = 20_000
    naive = _naive_longest_accepted_selector(
        target_sampler_seed=17,
        hidden_token=1,
        hidden_accepts_longer=True,
        trials=trials,
    )

    assert _freq(naive, 1, trials) > 0.98
    assert abs(_freq(naive, 1, trials) - 0.30) > 0.50
