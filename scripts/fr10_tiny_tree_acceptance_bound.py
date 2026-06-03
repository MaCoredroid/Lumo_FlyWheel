#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class P0Counters:
    drafts_total: int = 398
    draft_tokens_total: int = 1990
    accepted_tokens_total: int = 1206
    accepted_by_position: tuple[int, ...] = (273, 266, 253, 238, 176)
    window_tokens_per_step: float = 13.45945945945946
    window_accepted_per_draft_token: float = 0.5786666666666667
    window_drafts_per_step: float = 75 / 37


def depth_rows(p0: P0Counters) -> list[dict[str, float | int]]:
    baseline_accepted_per_draft = p0.accepted_tokens_total / p0.drafts_total
    baseline_sequence_tokens_per_draft = 1.0 + baseline_accepted_per_draft
    rows = []
    for depth in range(1, 5):
        accepted = sum(p0.accepted_by_position[:depth])
        accepted_per_draft = accepted / p0.drafts_total
        sequence_tokens_per_draft = 1.0 + accepted_per_draft
        token_step_ratio = sequence_tokens_per_draft / baseline_sequence_tokens_per_draft
        rows.append(
            {
                "max_depth": depth,
                "accepted_positions_included": depth,
                "accepted_tokens": accepted,
                "accepted_per_draft": accepted_per_draft,
                "accepted_per_draft_token": accepted / (p0.drafts_total * 5),
                "sequence_tokens_per_draft": sequence_tokens_per_draft,
                "token_step_ratio_vs_mtp5_spine": token_step_ratio,
                "projected_tokens_per_step_same_step_time": p0.window_tokens_per_step * token_step_ratio,
                "forward_latency_reduction_needed_to_match_baseline": 1.0 - token_step_ratio,
            }
        )
    return rows


def run(decode_forward_ms: float | None, decode_forward_ms_range: tuple[float, float]) -> dict[str, object]:
    p0 = P0Counters()
    rows = depth_rows(p0)
    fla_chunk_us = 135.0
    tiny_tree_us = 45.339
    gdn_layers = 48
    saved_us_per_step = (fla_chunk_us - tiny_tree_us) * gdn_layers
    best_depth4 = rows[3]

    def saving(forward_ms: float) -> dict[str, float]:
        saved_fraction = (saved_us_per_step / 1000.0) / forward_ms
        return {
            "decode_forward_ms": forward_ms,
            "saved_forward_fraction": saved_fraction,
            "net_tps_ratio_if_depth4": best_depth4["token_step_ratio_vs_mtp5_spine"] / (1.0 - saved_fraction),
        }

    if decode_forward_ms is None:
        savings = [saving(decode_forward_ms_range[0]), saving(decode_forward_ms_range[1])]
        saving_source = "range_assumption_pending_clean_dgx_steptrace_forward_pass"
    else:
        savings = [saving(decode_forward_ms)]
        saving_source = "explicit_decode_forward_ms"

    return {
        "schema": "fr10.tiny_tree_acceptance_bound.v1",
        "p0_source": {
            "greedy_sha256": "b8b1ec327f60e34073fcedf54c8dad402bee47264f650888f3e982176c2e9794",
            "drafts_total": p0.drafts_total,
            "draft_tokens_total": p0.draft_tokens_total,
            "accepted_tokens_total": p0.accepted_tokens_total,
            "accepted_by_position": list(p0.accepted_by_position),
            "window_tokens_per_step": p0.window_tokens_per_step,
            "window_accepted_per_draft_token": p0.window_accepted_per_draft_token,
        },
        "baseline": {
            "mtp5_spine_accepted_per_draft": p0.accepted_tokens_total / p0.drafts_total,
            "mtp5_spine_accepted_per_draft_token_full_stream": p0.accepted_tokens_total / p0.draft_tokens_total,
            "mtp5_spine_accepted_per_draft_token_window": p0.window_accepted_per_draft_token,
            "mtp5_spine_sequence_tokens_per_draft": 1.0 + p0.accepted_tokens_total / p0.drafts_total,
            "tokens_per_step": p0.window_tokens_per_step,
        },
        "kernel_savings_bound": {
            "fla_chunk_us_reference": fla_chunk_us,
            "tiny_tree_us_reference_3_node_padded4": tiny_tree_us,
            "gdn_layers": gdn_layers,
            "saved_us_per_step_if_all_layers_replace_fla": saved_us_per_step,
            "saving_source": saving_source,
            "decode_forward_ms": decode_forward_ms,
            "decode_forward_ms_range": list(decode_forward_ms_range),
            "savings": savings,
        },
        "depth_rows": rows,
        "decision": {
            "tiny_tree_depth4_projected_tokens_per_step": best_depth4["projected_tokens_per_step_same_step_time"],
            "tiny_tree_depth4_required_forward_latency_reduction": best_depth4["forward_latency_reduction_needed_to_match_baseline"],
            "depth_bound_clears_mtp5": False,
            "interpretation": (
                "Using P0 spine acceptance counters, a <=4-token spine/suffix upper-bound loses about "
                "11% sequence tokens per draft versus MTP-5. This is the tiny-tree depth kill. "
                "Kernel savings must be denominatored against decode-forward-pass latency, not agentic "
                "step wall time; the depth bound itself remains insufficient without branch acceptance "
                "beyond the observed spine-position counters."
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
