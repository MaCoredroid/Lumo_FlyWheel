#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_tree_rejection_sampler import (  # noqa: E402
    canonical_draft_weights,
    chi_square_gof,
    count_pairs,
    count_tokens,
    exact_multidraft_distribution,
    mixture_distribution,
    sample_biased_multidraft_control,
    sample_multidraft_rejection,
    sample_multidraft_rejection_step,
    sample_two_step_tree_rejection,
)


def main() -> int:
    out = Path("output/fr10_tree_sampler_convergence/sampler_convergence_summary.json")
    p = np.array([0.05, 0.12, 0.21, 0.08, 0.19, 0.25, 0.10], dtype=np.float64)
    qs = [
        np.array([0.09, 0.10, 0.18, 0.10, 0.16, 0.24, 0.13], dtype=np.float64),
        np.array([0.04, 0.18, 0.24, 0.06, 0.17, 0.21, 0.10], dtype=np.float64),
        np.array([0.06, 0.08, 0.16, 0.12, 0.25, 0.22, 0.11], dtype=np.float64),
    ]
    rng = np.random.default_rng(20260604)
    samples = sample_multidraft_rejection(p, qs, n=200_000, rng=rng)
    convergence = chi_square_gof(count_tokens(samples, len(p)), p)

    rng = np.random.default_rng(20260607)
    step_samples = np.fromiter(
        (
            sample_multidraft_rejection_step(p, qs, rng=rng).token_id
            for _ in range(200_000)
        ),
        dtype=np.int64,
        count=200_000,
    )
    step_convergence = chi_square_gof(count_tokens(step_samples, len(p)), p)

    rng = np.random.default_rng(20260606)
    biased = sample_biased_multidraft_control(p, qs, n=200_000, rng=rng)
    biased_control = chi_square_gof(count_tokens(biased, len(p)), p)

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
    rng = np.random.default_rng(20260605)
    tree_samples = sample_two_step_tree_rejection(
        root_p,
        root_qs,
        child_ps,
        child_qs,
        n=250_000,
        rng=rng,
    )
    expected_joint = np.concatenate(
        [root_p[parent] * child_ps[parent] for parent in range(4)]
    )
    tree_convergence = chi_square_gof(count_pairs(tree_samples, 4), expected_joint)
    weights = canonical_draft_weights(p, qs)
    q_mix = mixture_distribution(qs, weights)
    exact = exact_multidraft_distribution(p, qs, weights)

    payload = {
        "schema": "fr10.tree_sampler_convergence.v1",
        "single_position_multidraft": {
            "samples": 200_000,
            "chi_square": convergence.__dict__,
            "target": p.tolist(),
            "draft_weights": weights.tolist(),
            "mixture_q": q_mix.tolist(),
            "exact_distribution_max_abs_vs_target": float(np.max(np.abs(exact - p))),
        },
        "serving_step_multidraft": {
            "samples": 200_000,
            "chi_square": step_convergence.__dict__,
        },
        "branched_two_step_tree": {
            "samples": 250_000,
            "chi_square": tree_convergence.__dict__,
            "vocab_size": 4,
            "joint_states": 16,
        },
        "biased_negative_control": {
            "samples": 200_000,
            "chi_square": biased_control.__dict__,
            "threshold_ratio": biased_control.statistic / biased_control.threshold,
            "detected": not biased_control.passed,
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return (
        0
        if convergence.passed
        and step_convergence.passed
        and tree_convergence.passed
        and not biased_control.passed
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
