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
MTP_FORWARD_PASSES = 4


def _floor_ms(byte_count: int) -> float:
    value = (
        Decimal(byte_count)
        * Decimal(1_000)
        / Decimal(BANDWIDTH_BYTES_PER_S)
    )
    return float(value.quantize(Decimal("0.000000001")))


def build_ledger() -> dict[str, Any]:
    full_head_bytes = (
        FULL_VOCAB_ROWS * DRAFTER_HIDDEN_SIZE * HEAD_ELEMENT_BYTES
    )
    subset_head_bytes = (
        DRAFT_VOCAB_ROWS * DRAFTER_HIDDEN_SIZE * HEAD_ELEMENT_BYTES
    )
    mtp_forward_bytes = MTP_FORWARD_BYTES_PER_PASS * MTP_FORWARD_PASSES
    current_drafter_head_bytes = full_head_bytes + 4 * subset_head_bytes
    root_64k_drafter_head_bytes = 5 * subset_head_bytes

    legacy_bytes = TARGET_MODEL_BYTES + full_head_bytes
    current_bytes = (
        legacy_bytes + mtp_forward_bytes + current_drafter_head_bytes
    )
    root_64k_bytes = (
        legacy_bytes + mtp_forward_bytes + root_64k_drafter_head_bytes
    )

    return {
        "schema": "fr13.speculative_step_weight_ledger.v1",
        "bandwidth_bytes_per_s": BANDWIDTH_BYTES_PER_S,
        "formula": "floor_ms = mandatory_weight_bytes * 1000 / bandwidth_bytes_per_s",
        "components": {
            "target_model": {
                "bytes": TARGET_MODEL_BYTES,
                "source": "logical target-model weight tensor byte ledger",
            },
            "full_bf16_head": {
                "bytes": full_head_bytes,
                "element_bytes": HEAD_ELEMENT_BYTES,
                "hidden_size": DRAFTER_HIDDEN_SIZE,
                "rows": FULL_VOCAB_ROWS,
            },
            "draft_64k_bf16_head": {
                "bytes": subset_head_bytes,
                "element_bytes": HEAD_ELEMENT_BYTES,
                "hidden_size": DRAFTER_HIDDEN_SIZE,
                "rows": DRAFT_VOCAB_ROWS,
            },
            "mtp_forward": {
                "bytes": mtp_forward_bytes,
                "bytes_per_pass": MTP_FORWARD_BYTES_PER_PASS,
                "passes": MTP_FORWARD_PASSES,
                "source": "logical MTP forward tensor byte ledger",
            },
        },
        "scenarios": {
            "legacy_target_plus_verifier_head": {
                "component_formula": "target_model + full_bf16_head",
                "mandatory_weight_bytes": legacy_bytes,
                "mandatory_weight_floor_ms": _floor_ms(legacy_bytes),
                "is_full_speculative_step_floor": False,
            },
            "current_one_full_plus_four_64k_draft_heads": {
                "component_formula": (
                    "target_model + full_bf16_head + mtp_forward + "
                    "full_bf16_head + 4 * draft_64k_bf16_head"
                ),
                "drafter_head_bytes": current_drafter_head_bytes,
                "mandatory_weight_bytes": current_bytes,
                "mandatory_weight_floor_ms": _floor_ms(current_bytes),
                "nonweight_costs_included": False,
            },
            "root_64k_five_64k_draft_heads": {
                "component_formula": (
                    "target_model + full_bf16_head + mtp_forward + "
                    "5 * draft_64k_bf16_head"
                ),
                "drafter_head_bytes": root_64k_drafter_head_bytes,
                "mandatory_weight_bytes": root_64k_bytes,
                "mandatory_weight_floor_ms": _floor_ms(root_64k_bytes),
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
