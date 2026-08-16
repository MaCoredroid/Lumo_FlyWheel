#!/usr/bin/env python3
"""Emit the explicit FR14 speculative-step mandatory weight-read ledger.

FR14 (2026-08-16) re-derived every byte in this ledger by SUMMING the actual
tensor byte-spans of the served Qwen3.8-27B-NVFP4 checkpoint. Nothing here is a
scaled FR13 constant: the recon's tempting "NVFP4 halves the floor" shortcut is
wrong twice over (the BF16 head does not shrink at all, and the MTP shard got
BIGGER because every serious NVFP4 repack ships MTP in BF16 while the served
3.6 checkpoint carried it in FP8).

Run ``--derive-from-checkpoint`` to reproduce the pinned constants from the
files on disk; it is the arithmetic of record and it fails loud on any drift.
"""

from __future__ import annotations

import argparse
import json
import struct
from decimal import Decimal
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# hardware                                                                     #
# --------------------------------------------------------------------------- #
# GB10 unified-memory bandwidth. Hardware constant, unchanged by the model swap.
BANDWIDTH_BYTES_PER_S = 273_000_000_000

# --------------------------------------------------------------------------- #
# checkpoint (FR14: /models/qwen3.8-27b-nvfp4, post-lm_head-surgery)           #
# --------------------------------------------------------------------------- #
MODEL_ROOT = Path("/models/qwen3.8-27b-nvfp4")
TARGET_SHARD = "model.safetensors"
MTP_SHARD = "model_mtp.safetensors"

# Target-model weight tensor byte ledger: every tensor under
# `model.language_model.` EXCEPT embed_tokens -- i.e. the 64 decoder layers
# (1,616 tensors: NVFP4 W4A4-g16 MLP weights + their fp8 e4m3 block scales and
# fp32 global scales, FP8 attention/GDN projections, BF16 GDN conv/in_proj and
# every norm) plus the final `model.language_model.norm`.
#
# Excluded on purpose, each for its own reason:
#   * lm_head            -- accounted separately as FULL_HEAD_BYTES; it is read
#                           once per verify and again per drafter pass, so it
#                           cannot live inside a single "target model" term.
#   * embed_tokens       -- a row gather of a handful of ids, not a stream.
#   * model.visual.*     -- the vision tower is not on the text decode path.
#
# DERIVED 2026-08-16 from /models/qwen3.8-27b-nvfp4/model.safetensors:
#   model.language_model.layers.*  17_831_778_688   (1,616 tensors)
#   model.language_model.norm              10_240   (5120 x BF16)
#   ------------------------------------------------
#   TARGET_MODEL_BYTES             17_831_788_928
#
# FR13 FOSSIL, recorded so the delta is not mistaken for a model effect: the
# FR13 pin 24_382_399_488 was layers-ONLY and silently omitted the final norm
# (the same rule applied to the 3.6 checkpoint yields 24_382_409_728, a
# +10_240 B difference = 0.0000375 ms of floor). FR14 counts the final norm,
# because it is streamed on every forward pass like any other layer weight.
TARGET_MODEL_BYTES = 17_831_788_928

# The whole MTP shard: 15 BF16 tensors, read in full on every MTP forward pass.
# DERIVED 2026-08-16 by summing model_mtp.safetensors' tensor byte-spans
# (849_398_784 B of tensors inside an 849_400_392 B file; the 1,608 B remainder
# is the safetensors header, which is not streamed per pass).
#
# This term went UP, not down: the served 3.6 MTP shard was FP8 at 477_199_744
# B/pass. Every NVFP4 repack in the field keeps MTP in BF16, so the drafter's
# five passes now cost 4_246_993_920 B instead of 2_385_998_720 B. A local FP8
# requant of the MTP shard is the lever that would give those 1.86 GB back;
# it is deliberately NOT taken here (it would change the served bytes).
MTP_FORWARD_BYTES_PER_PASS = 849_398_784

DRAFT_VOCAB_ROWS = 65_536
FULL_VOCAB_ROWS = 248_320
DRAFTER_HIDDEN_SIZE = 5_120
# lm_head is BF16 in the served checkpoint. It arrived FP8-per-channel from
# unsloth and was dequantised to BF16 by the FR14 lm_head surgery, because this
# vLLM's qwen3_5 builds lm_head as an unquantized ParallelLMHead and refused the
# FP8 tensor set outright (REDTEAM_20260816.md pass 6). So the head bytes are
# byte-for-byte what FR13 measured -- 2 bytes/element -- and the 5.9 GB of head
# traffic in the fixed32 arm did not move at all under NVFP4.
HEAD_ELEMENT_BYTES = 2

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
FULL_VOCAB_DRAFTER_HEAD_BYTES = MTP_FORWARD_PASSES * FULL_HEAD_BYTES
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
FULL_VOCAB_MANDATORY_WEIGHT_BYTES = (
    LEGACY_MANDATORY_WEIGHT_BYTES
    + MTP_FORWARD_BYTES
    + FULL_VOCAB_DRAFTER_HEAD_BYTES
)
# Preserve the full precision used by the full-vocabulary B1 acceptance contract.
FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS = 136.7603064029304
FULL_VOCAB_SLO_CAP_MS = 157.27435236336996
FIXED32_SLO_CAP_MS = float(
    (
        Decimal(FIXED32_MANDATORY_WEIGHT_BYTES)
        * Decimal(1_000)
        / Decimal(BANDWIDTH_BYTES_PER_S)
        * FIXED32_SLO_MULTIPLIER
    ).quantize(Decimal("0.000000001"))
)

# FR14 PROVISIONAL SLO CAP. FR13's cap was 1.15 x floor and the same
# multiplier is carried here so the instrument shape is unchanged, but the
# objective bar for FR14 is Mark's open ruling (results/fr14_nvfp4_port_
# 20260816/README.md "Correctness bar -- PROPOSED, AWAITING MARK"): NVFP4 is
# lossy by construction, so no FR13 acceptance number transfers. Treat every
# *_SLO_CAP_MS / ONE_SIDED_U95_CAP_MS in this train as PROVISIONAL until that
# ruling lands.
FIXED32_SLO_CAP_PROVISIONAL = True


# --------------------------------------------------------------------------- #
# derivation (the arithmetic of record)                                        #
# --------------------------------------------------------------------------- #
def _safetensors_tensor_bytes(path: Path) -> dict[str, int]:
    """Return {tensor_name: byte_span} from a safetensors file's header only.

    Reads the 8-byte little-endian header length plus the JSON header; the
    tensor payload is never touched, so this is cheap on a 23.8 GB shard.
    """
    with path.open("rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    spans: dict[str, int] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        spans[name] = end - start
    return spans


def derive_checkpoint_bytes(model_root: Path = MODEL_ROOT) -> dict[str, Any]:
    """Re-derive the byte terms by summing real tensor spans on disk."""
    target = _safetensors_tensor_bytes(model_root / TARGET_SHARD)
    mtp = _safetensors_tensor_bytes(model_root / MTP_SHARD)

    layers = 0
    final_norm = 0
    lm_head = 0
    embed_tokens = 0
    vision = 0
    unclassified: dict[str, int] = {}
    for name, span in target.items():
        if name.startswith("lm_head"):
            lm_head += span
        elif name.startswith("model.language_model.embed_tokens"):
            embed_tokens += span
        elif name.startswith("model.language_model.layers."):
            layers += span
        elif name.startswith("model.language_model.norm"):
            final_norm += span
        elif name.startswith("model.visual."):
            vision += span
        else:
            unclassified[name] = span
    if unclassified:
        raise SystemExit(
            "fr14 floor derivation refuses to guess: unclassified target "
            f"tensors {sorted(unclassified)}"
        )

    return {
        "model_root": str(model_root),
        "target_shard": TARGET_SHARD,
        "mtp_shard": MTP_SHARD,
        "target_model_bytes": layers + final_norm,
        "target_decoder_layer_bytes": layers,
        "target_final_norm_bytes": final_norm,
        "lm_head_bytes_on_disk": lm_head,
        "embed_tokens_bytes_excluded": embed_tokens,
        "vision_tower_bytes_excluded": vision,
        "mtp_forward_bytes_per_pass": sum(mtp.values()),
        "mtp_tensor_count": len(mtp),
        "target_tensor_count": len(target),
    }


def verify_pinned_constants(model_root: Path = MODEL_ROOT) -> dict[str, Any]:
    """Fail loud if the pinned constants no longer match the checkpoint."""
    derived = derive_checkpoint_bytes(model_root)
    problems: list[str] = []
    if derived["target_model_bytes"] != TARGET_MODEL_BYTES:
        problems.append(
            "TARGET_MODEL_BYTES "
            f"{derived['target_model_bytes']} != {TARGET_MODEL_BYTES}"
        )
    if derived["mtp_forward_bytes_per_pass"] != MTP_FORWARD_BYTES_PER_PASS:
        problems.append(
            "MTP_FORWARD_BYTES_PER_PASS "
            f"{derived['mtp_forward_bytes_per_pass']} "
            f"!= {MTP_FORWARD_BYTES_PER_PASS}"
        )
    if derived["lm_head_bytes_on_disk"] != FULL_HEAD_BYTES:
        problems.append(
            "lm_head is not the BF16 head the floor assumes: "
            f"{derived['lm_head_bytes_on_disk']} != {FULL_HEAD_BYTES}"
        )
    derived["problems"] = problems
    derived["pass"] = not problems
    return derived


def build_ledger() -> dict[str, Any]:
    return {
        "schema": "fr13.speculative_step_weight_ledger.v3",
        "campaign": "fr14_nvfp4_port_20260816",
        "model_root": str(MODEL_ROOT),
        "bandwidth_bytes_per_s": BANDWIDTH_BYTES_PER_S,
        "formula": "floor_ms = mandatory_weight_bytes * 1000 / bandwidth_bytes_per_s",
        "derivation": (
            "every byte term is a SUM of real safetensors tensor spans in "
            "/models/qwen3.8-27b-nvfp4; no FR13 constant was scaled"
        ),
        "slo_cap_provisional": FIXED32_SLO_CAP_PROVISIONAL,
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
                "source": (
                    "summed tensor spans of model.language_model.* minus "
                    "embed_tokens (64 decoder layers + final norm)"
                ),
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
                "source": (
                    "summed tensor spans of model_mtp.safetensors (BF16)"
                ),
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
            "full_vocab_five_full_draft_heads": {
                "component_formula": (
                    "target_model + full_bf16_head + mtp_forward + "
                    "5 * full_bf16_head"
                ),
                "drafter_head_bytes": FULL_VOCAB_DRAFTER_HEAD_BYTES,
                "mandatory_weight_bytes": FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
                "mandatory_weight_floor_ms": (
                    FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS
                ),
                "nonweight_costs_included": False,
            },
        },
        "retired_scenarios": {
            "draft_head_fp8": (
                "FR13_DRAFT_HEAD_FP8 is DEAD under the FR14 checkpoint. The "
                "arm re-read the five K64 draft heads as FP8 qweight + FP32 "
                "scales (1_678_131_200 B instead of 3_355_443_200 B). The "
                "served lm_head is BF16 -- unsloth's FP8 per-channel head was "
                "dequantised to BF16 by the lm_head surgery to make the "
                "checkpoint loadable at all -- so there is no FP8 head to "
                "read and the arm would measure a floor it cannot realise. "
                "The launcher and the floor-timer sequence now refuse it."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--derive-from-checkpoint",
        nargs="?",
        const=MODEL_ROOT,
        type=Path,
        metavar="MODEL_ROOT",
        help=(
            "re-derive the byte terms from the checkpoint on disk, print the "
            "arithmetic, and exit non-zero if the pinned constants drifted"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.derive_from_checkpoint is not None:
        report = verify_pinned_constants(args.derive_from_checkpoint)
        rendered = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if args.output is None:
            print(rendered, end="")
        else:
            args.output.write_text(rendered, encoding="ascii")
        return 0 if report["pass"] else 2
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
