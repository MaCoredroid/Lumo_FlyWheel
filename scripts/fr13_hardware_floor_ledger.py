#!/usr/bin/env python3
"""Emit the explicit FR13 speculative-step mandatory weight-read ledger."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path
from typing import Any


BANDWIDTH_BYTES_PER_S = 273_000_000_000
TARGET_MODEL_BYTES = 24_382_399_488
DRAFT_VOCAB_ROWS = 65_536
FULL_VOCAB_ROWS = 248_320
DRAFTER_HIDDEN_SIZE = 5_120
HEAD_ELEMENT_BYTES = 2
MTP_FORWARD_BYTES_PER_PASS = 477_199_744
INITIAL_MTP_FORWARD_PASSES = 1
POST_ROOT_GRAPH_MTP_FORWARD_PASSES = 4
MTP_FORWARD_PASSES = (
    INITIAL_MTP_FORWARD_PASSES + POST_ROOT_GRAPH_MTP_FORWARD_PASSES
)
FIXED32_SLO_MULTIPLIER = Decimal("1.15")


def _floor_ms(byte_count: int) -> float:
    value = (
        Decimal(byte_count)
        * Decimal(1_000)
        / Decimal(BANDWIDTH_BYTES_PER_S)
    )
    return float(value.quantize(Decimal("0.000000001")))


FULL_HEAD_BYTES = FULL_VOCAB_ROWS * DRAFTER_HIDDEN_SIZE * HEAD_ELEMENT_BYTES
SUBSET_HEAD_BYTES = DRAFT_VOCAB_ROWS * DRAFTER_HIDDEN_SIZE * HEAD_ELEMENT_BYTES
MTP_FORWARD_BYTES = MTP_FORWARD_BYTES_PER_PASS * MTP_FORWARD_PASSES
CURRENT_DRAFTER_HEAD_BYTES = (
    FULL_HEAD_BYTES
    + POST_ROOT_GRAPH_MTP_FORWARD_PASSES * SUBSET_HEAD_BYTES
)
ROOT_64K_DRAFTER_HEAD_BYTES = MTP_FORWARD_PASSES * SUBSET_HEAD_BYTES
LEGACY_MANDATORY_WEIGHT_BYTES = TARGET_MODEL_BYTES + FULL_HEAD_BYTES
CURRENT_MANDATORY_WEIGHT_BYTES = (
    LEGACY_MANDATORY_WEIGHT_BYTES
    + MTP_FORWARD_BYTES
    + CURRENT_DRAFTER_HEAD_BYTES
)
FIXED32_MANDATORY_WEIGHT_BYTES = (
    LEGACY_MANDATORY_WEIGHT_BYTES
    + MTP_FORWARD_BYTES
    + ROOT_64K_DRAFTER_HEAD_BYTES
)
FIXED32_MANDATORY_WEIGHT_FLOOR_MS = _floor_ms(FIXED32_MANDATORY_WEIGHT_BYTES)
FIXED32_SLO_CAP_MS = float(
    (
        Decimal(FIXED32_MANDATORY_WEIGHT_BYTES)
        * Decimal(1_000)
        / Decimal(BANDWIDTH_BYTES_PER_S)
        * FIXED32_SLO_MULTIPLIER
    ).quantize(Decimal("0.000000001"))
)


def build_ledger() -> dict[str, Any]:
    return {
        "schema": "fr13.speculative_step_weight_ledger.v2",
        "bandwidth_bytes_per_s": BANDWIDTH_BYTES_PER_S,
        "formula": "floor_ms = mandatory_weight_bytes * 1000 / bandwidth_bytes_per_s",
        "production_invariants": {
            "initial_mtp_forward_passes_per_event": (
                INITIAL_MTP_FORWARD_PASSES
            ),
            "post_root_graph_mtp_forward_passes_per_event": (
                POST_ROOT_GRAPH_MTP_FORWARD_PASSES
            ),
            "total_mtp_forward_passes_per_event": MTP_FORWARD_PASSES,
            "drafter_head_passes_per_event": MTP_FORWARD_PASSES,
        },
        "components": {
            "target_model": {
                "bytes": TARGET_MODEL_BYTES,
                "source": "logical target-model weight tensor byte ledger",
            },
            "full_bf16_head": {
                "bytes": FULL_HEAD_BYTES,
                "element_bytes": HEAD_ELEMENT_BYTES,
                "hidden_size": DRAFTER_HIDDEN_SIZE,
                "rows": FULL_VOCAB_ROWS,
            },
            "draft_64k_bf16_head": {
                "bytes": SUBSET_HEAD_BYTES,
                "element_bytes": HEAD_ELEMENT_BYTES,
                "hidden_size": DRAFTER_HIDDEN_SIZE,
                "rows": DRAFT_VOCAB_ROWS,
            },
            "mtp_forward": {
                "bytes": MTP_FORWARD_BYTES,
                "bytes_per_pass": MTP_FORWARD_BYTES_PER_PASS,
                "initial_passes": INITIAL_MTP_FORWARD_PASSES,
                "passes": MTP_FORWARD_PASSES,
                "post_root_graph_passes": (
                    POST_ROOT_GRAPH_MTP_FORWARD_PASSES
                ),
                "source": "logical MTP forward tensor byte ledger",
            },
        },
        "scenarios": {
            "legacy_target_plus_verifier_head": {
                "component_formula": "target_model + full_bf16_head",
                "mandatory_weight_bytes": LEGACY_MANDATORY_WEIGHT_BYTES,
                "mandatory_weight_floor_ms": _floor_ms(
                    LEGACY_MANDATORY_WEIGHT_BYTES
                ),
                "is_full_speculative_step_floor": False,
            },
            "current_one_full_plus_four_64k_draft_heads": {
                "component_formula": (
                    "target_model + full_bf16_head + mtp_forward + "
                    "full_bf16_head + 4 * draft_64k_bf16_head"
                ),
                "drafter_head_bytes": CURRENT_DRAFTER_HEAD_BYTES,
                "mandatory_weight_bytes": CURRENT_MANDATORY_WEIGHT_BYTES,
                "mandatory_weight_floor_ms": _floor_ms(
                    CURRENT_MANDATORY_WEIGHT_BYTES
                ),
                "nonweight_costs_included": False,
            },
            "root_64k_five_64k_draft_heads": {
                "component_formula": (
                    "target_model + full_bf16_head + mtp_forward + "
                    "5 * draft_64k_bf16_head"
                ),
                "drafter_head_bytes": ROOT_64K_DRAFTER_HEAD_BYTES,
                "mandatory_weight_bytes": FIXED32_MANDATORY_WEIGHT_BYTES,
                "mandatory_weight_floor_ms": (
                    FIXED32_MANDATORY_WEIGHT_FLOOR_MS
                ),
                "nonweight_costs_included": False,
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rendered = json.dumps(
        build_ledger(), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
