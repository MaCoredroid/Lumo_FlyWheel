#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse


def run(decode_forward_ms: float | None, decode_forward_ms_range: tuple[float, float]) -> dict[str, object]:
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

    def requirement(forward_ms: float) -> dict[str, float]:
        overhead_fraction = extra_ms_per_step / forward_ms
        required_sequence_tokens_per_draft = p0_sequence_tokens_per_draft * (1.0 + overhead_fraction)
        return {
            "decode_forward_ms": forward_ms,
            "extra_forward_fraction": overhead_fraction,
            "required_sequence_tokens_per_draft_to_beat_p0": required_sequence_tokens_per_draft,
            "required_extra_accepted_tokens_per_draft": required_sequence_tokens_per_draft - p0_sequence_tokens_per_draft,
        }

    if decode_forward_ms is None:
        requirements = [requirement(decode_forward_ms_range[0]), requirement(decode_forward_ms_range[1])]
        requirement_source = "range_assumption_pending_clean_dgx_steptrace_forward_pass"
    else:
        requirements = [requirement(decode_forward_ms)]
        requirement_source = "explicit_decode_forward_ms"

    return {
        "schema": "fr10.fused_kernel_bote.v1",
        "inputs": {
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
            "decode_forward_ms": decode_forward_ms,
            "decode_forward_ms_range": list(decode_forward_ms_range),
        },
        "estimate": {
            "sparse_solve_us": sparse_solve_us,
            "output_plus_commit_us": output_commit_us,
            "fused_tree_14_us": fused_tree_us,
            "extra_us_vs_fla_per_layer": extra_us_per_layer,
            "extra_ms_per_step_48_layers": extra_ms_per_step,
            "requirement_source": requirement_source,
            "requirements": requirements,
        },
        "decision": {
            "plausibly_cheap_big_tree_path_exists": True,
            "prototype_condition": (
                "Prototype only a fused sparse tree verifier that avoids spilling all N node states. "
                "A sparse solve plus outputs-only and one canonical committed state is estimated at "
                "about 216us for 14 nodes. Do not denominator this against agentic step wall time; "
                "compare extra GDN cost against decode-forward-pass latency and require branch "
                "acceptance to lift sequence tokens per forward enough to pay that overhead."
            ),
            "do_not_prototype": (
                "Do not prototype another dense or all-node-state-spilling kernel for >=6 nodes; "
                "state-output alone already exceeds the FLA flat cost."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decode-forward-ms", type=float, default=None)
    parser.add_argument("--decode-forward-ms-min", type=float, default=25.0)
    parser.add_argument("--decode-forward-ms-max", type=float, default=40.0)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.decode_forward_ms, (args.decode_forward_ms_min, args.decode_forward_ms_max)),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
