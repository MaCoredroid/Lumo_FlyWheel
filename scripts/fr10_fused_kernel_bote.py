#!/usr/bin/env python3
from __future__ import annotations

import json


def run() -> dict[str, object]:
    p0_step_seconds = 1.203652878065367
    p0_tokens_per_step = 13.45945945945946
    p0_sequence_tokens_per_draft = 4.030150753768844
    gdn_layers = 48
    fla_us = 135.0

    dense_solve_14_us = 636.507314046224
    strict_pair_fraction_14 = 36 / 120
    sparse_solve_us = dense_solve_14_us * strict_pair_fraction_14
    setup_us = 12.0
    output_bytes = 14 * 24 * 1024
    committed_state_bytes = 48 * 128 * 128 * 4
    gb10_observed_state_bw_gb_s = 273.0
    output_commit_us = (output_bytes + committed_state_bytes) / (gb10_observed_state_bw_gb_s * 1_000)
    fused_tree_us = setup_us + sparse_solve_us + output_commit_us

    extra_us_per_layer = fused_tree_us - fla_us
    extra_ms_per_step = extra_us_per_layer * gdn_layers / 1000.0
    extra_step_fraction = (extra_ms_per_step / 1000.0) / p0_step_seconds
    required_sequence_tokens_per_draft = p0_sequence_tokens_per_draft * (1.0 + extra_step_fraction)
    required_extra_accepted_per_draft = required_sequence_tokens_per_draft - p0_sequence_tokens_per_draft
    required_tokens_per_step = p0_tokens_per_step * (1.0 + extra_step_fraction)

    return {
        "schema": "fr10.fused_kernel_bote.v1",
        "inputs": {
            "p0_step_seconds": p0_step_seconds,
            "p0_tokens_per_step": p0_tokens_per_step,
            "p0_sequence_tokens_per_draft": p0_sequence_tokens_per_draft,
            "gdn_layers": gdn_layers,
            "fla_chunk_us": fla_us,
            "dense_solve_14_us": dense_solve_14_us,
            "strict_ancestor_pairs_14": 36,
            "dense_lower_pairs_padded16": 120,
            "strict_pair_fraction_14": strict_pair_fraction_14,
            "setup_us_assumption": setup_us,
            "output_bytes_assumption": output_bytes,
            "committed_state_bytes": committed_state_bytes,
            "gb10_observed_state_bw_gb_s": gb10_observed_state_bw_gb_s,
        },
        "estimate": {
            "sparse_solve_us": sparse_solve_us,
            "output_plus_commit_us": output_commit_us,
            "fused_tree_14_us": fused_tree_us,
            "extra_us_vs_fla_per_layer": extra_us_per_layer,
            "extra_ms_per_step_48_layers": extra_ms_per_step,
            "extra_step_fraction": extra_step_fraction,
            "required_tokens_per_step_to_beat_p0": required_tokens_per_step,
            "required_sequence_tokens_per_draft_to_beat_p0": required_sequence_tokens_per_draft,
            "required_extra_accepted_tokens_per_draft": required_extra_accepted_per_draft,
        },
        "decision": {
            "plausibly_cheap_big_tree_path_exists": True,
            "prototype_condition": (
                "Prototype only a fused sparse tree verifier that avoids spilling all N node states. "
                "A sparse solve plus outputs-only and one canonical committed state is estimated at "
                "about 220us for 14 nodes, only about 0.34% step-time overhead versus FLA across 48 "
                "GDN layers. This clears the cost-gate only if branch acceptance recovers more than "
                "about 0.014 accepted tokens per draft over the P0 MTP-5 spine."
            ),
            "do_not_prototype": (
                "Do not prototype another dense or all-node-state-spilling kernel for >=6 nodes; "
                "state-output alone already exceeds the FLA flat cost."
            ),
        },
    }


def main() -> None:
    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
