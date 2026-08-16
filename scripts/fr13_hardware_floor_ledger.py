#!/usr/bin/env python3
"""Emit the explicit FR14 speculative-step mandatory weight-read ledger.

FR14 (2026-08-16) re-derived every byte in this ledger by SUMMING the actual
tensor byte-spans of the served checkpoint. Nothing here is a scaled constant.

ARM B -- AGGRESSIVE BYTES (this file's live pin). The served checkpoint is
``RadixArk/Qwen3.8-27B-NVFP4`` @ ``554ebba9b5f1b79dc11246341960360e6ef05ef4``
(ModelOpt recipe): NVFP4 W4A4-g16 across the MLP stack, FP8 attention/GDN
projections, BF16 GDN conv/in_proj and norms, **and an NVFP4 lm_head**. That
last item is the whole ballgame -- see the scenario table below.

ARM A -- CONSERVATIVE BYTES (what this file pinned before, still on disk at
``/models/qwen3.8-27b-nvfp4``): ``unsloth/Qwen3.8-27B-NVFP4``, whose FP8
per-channel lm_head had to be dequantised to BF16 to load at all, pinning the
fixed32 floor at 27_977_022_848 B / 102.479937172 ms. The two arms are the
FR14 bytes ablation: same stack, same workload, aggressive vs conservative
bytes. Re-serving arm A is a REVERT of the arm-B constant-train commit, not an
environment switch -- ~50 timing instruments carry the floor as a shell
literal, so a partial switch would export one arm's floor against the other
arm's weights.

WHAT ACTUALLY MOVED, arm A -> arm B (all sums of real tensor spans):
    target_model              17_831_788_928 -> 16_892_610_688   (-0.94 GB)
    lm_head (verifier head)    2_542_796_800 ->    715_161_608   (-1.83 GB)
    mtp_forward per pass          849_398_784 ->    849_398_784   (identical:
        both repacks ship the 15 MTP tensors in BF16)
    draft_64k head (x5)           671_088_640 ->    671_088_640   (identical:
        PHASE 1 dequants the sliced NVFP4 rows to BF16 at boot, so the sealed
        BF16 K64 GEMV units and the 128-id block map are untouched)
    -----------------------------------------------------------------------
    fixed32 root_64k          27_977_022_848 -> 25_210_209_416
    fixed32 floor_ms             102.479937172 ->   92.345089436   (-10.135 ms)

Note the shape of that table: the aggressive BACKBONE is worth 0.94 GB and the
HEAD is worth 1.83 GB. Serving this checkpoint with a BF16 head would land at
27_037_844_608 B / 99.040 ms -- only 3.44 ms under arm A. Nearly the entire
win is the NVFP4 lm_head, which is why the boot-time lm_head loader patch
(scripts/fr14_patch_nvfp4_lmhead.py) is a hard requirement of this pin and is
fail-closed under FR14_REQUIRE_NVFP4_LMHEAD=1.

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
# checkpoint (FR14 arm B: /models/qwen3.8-27b-nvfp4-radixark, post-KV-surgery)  #
# --------------------------------------------------------------------------- #
MODEL_ROOT = Path("/models/qwen3.8-27b-nvfp4-radixark")

# ARM-B SHARD LAYOUT. RadixArk ships THREE shards
# (model-0000{1,2,3}-of-00003.safetensors) and -- unlike the unsloth repack,
# which carried the drafter in its own `model_mtp.safetensors` -- the 15 `mtp.*`
# tensors live INSIDE model-00003-of-00003.safetensors alongside body tensors.
# A filename-keyed ledger therefore cannot see them at all, which is why the
# derivation below classifies EVERY tensor in EVERY shard BY NAME and never by
# which file it happens to sit in. `SHARD_SUFFIX` is the whole file-level
# assumption that remains, and a missing/extra shard shows up as a byte
# mismatch in verify_pinned_constants rather than as a silent undercount.
SHARD_SUFFIX = ".safetensors"

# Target-model weight tensor byte ledger: every tensor under
# `model.language_model.` EXCEPT embed_tokens -- i.e. the 64 decoder layers
# (1,840 tensors: NVFP4 W4A4-g16 MLP weights as packed U8 + their fp8 e4m3
# per-16 block scales and fp32 global scales, FP8 attention/GDN projections,
# BF16 GDN conv/in_proj and every norm) plus the final
# `model.language_model.norm`.
#
# Excluded on purpose, each for its own reason:
#   * lm_head            -- accounted separately as FULL_HEAD_BYTES; it is read
#                           once per verify and again per drafter pass, so it
#                           cannot live inside a single "target model" term.
#   * mtp.*              -- accounted separately as MTP_FORWARD_BYTES_PER_PASS;
#                           read five times per speculative event, not once.
#   * embed_tokens       -- a row gather of a handful of ids, not a stream.
#   * model.visual.*     -- the vision tower is not on the text decode path.
#
# DERIVED 2026-08-16 from the three /models/qwen3.8-27b-nvfp4-radixark shards:
#   model.language_model.layers.*  16_892_600_448   (1,840 tensors)
#   model.language_model.norm              10_240   (5120 x BF16)
#   ------------------------------------------------
#   TARGET_MODEL_BYTES             16_892_610_688
#
# Cross-check on the whole ledger, not just this term: the sum of all 2,194
# tensor spans is 21_921_428_072 B, which equals RadixArk's own
# qualification.json `checkpoint.output_indexed_payload_bytes` exactly -- our
# ledger and the publisher's audit agree byte-for-byte.
#
# FR13 FOSSIL, recorded so the delta is not mistaken for a model effect: the
# FR13 pin 24_382_399_488 was layers-ONLY and silently omitted the final norm
# (the same rule applied to the 3.6 checkpoint yields 24_382_409_728, a
# +10_240 B difference = 0.0000375 ms of floor). FR14 counts the final norm,
# because it is streamed on every forward pass like any other layer weight.
TARGET_MODEL_BYTES = 16_892_610_688

# The drafter: 15 BF16 `mtp.*` tensors, read in full on every MTP forward pass.
# DERIVED 2026-08-16 by summing their tensor byte-spans inside shard 3.
#
# IDENTICAL to the conservative arm's 849_398_784 -- both repacks leave the MTP
# head in BF16, and this term is the reason neither arm's floor halves. It went
# UP against FR13, not down: the served 3.6 MTP shard was FP8 at 477_199_744
# B/pass, so the drafter's five passes cost 4_246_993_920 B instead of
# 2_385_998_720 B. A local FP8 requant of the MTP tensors is the lever that
# would give those 1.86 GB back (projected 20_675_636_616 B / 75.735 ms when
# stacked on the Phase-2 draft head); it is deliberately NOT taken here,
# because it would change the served bytes.
MTP_FORWARD_BYTES_PER_PASS = 849_398_784

DRAFT_VOCAB_ROWS = 65_536
FULL_VOCAB_ROWS = 248_320
DRAFTER_HIDDEN_SIZE = 5_120
# The K64 draft head is BF16 and stays BF16. PHASE 1 of the DVK port dequants
# the 65_536 sliced NVFP4 rows to BF16 at boot (fr10_phase4_patch_vllm_tree_gdn
# ._fr13_dvk_prepare), so the sealed BF16 GEMV units and the 128-id block map
# are byte-identical to FR13. 2 bytes/element, unchanged from arm A.
#
# PHASE 2 -- reading those five draft-head slices as NVFP4 (188_743_680 B each
# instead of 671_088_640) is the next byte lever and is worth 8.834 ms
# (25_210_209_416 -> 22_798_484_616 B, 92.345 -> 83.511 ms). It needs an FP4
# GEMV unit and carries its own byte gate; it is NOT pinned here.
HEAD_ELEMENT_BYTES = 2

# THE VERIFIER HEAD IS QUANTISED IN ARM B. This is the single largest term that
# moved and it is not derivable from rows x hidden x element_bytes any more, so
# it is pinned as a measured sum of the four shipped `lm_head.*` tensor spans:
#
#   lm_head.weight          U8      [248320, 2560]   635_699_200
#   lm_head.weight_scale    F8_E4M3 [248320,  320]    79_462_400
#   lm_head.weight_scale_2  F32     []                        4
#   lm_head.input_scale     F32     []                        4
#   ----------------------------------------------------------
#   NVFP4_HEAD_BYTES                                 715_161_608
#
# against 2_542_796_800 B for the BF16 head arm A was forced to serve: a
# 1_827_635_192 B saving worth 6.695 ms of floor on its own.
#
# The FP8 fallback head (had the NVFP4 head refused to load) would have been
# 1_271_895_040 B / 94.384 ms -- 2.039 ms worse. It was NOT needed: the NVFP4
# head boots and generates (results/fr14_nvfp4_port_20260816/
# radixark_smoke_serve_PASS.json), so the fallback is deliberately not built.
NVFP4_HEAD_BYTES = 715_161_608

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


# The verifier head as SERVED. Arm A computed this as
# FULL_VOCAB_ROWS * DRAFTER_HIDDEN_SIZE * HEAD_ELEMENT_BYTES because the head
# was a dense BF16 matrix; under arm B it is a 4-tensor NVFP4 set, so the
# product is no longer the byte count and the measured sum is the pin. The
# BF16 product is retained beside it as the reference the ablation is against.
FULL_HEAD_BYTES = NVFP4_HEAD_BYTES
FULL_BF16_HEAD_REFERENCE_BYTES = (
    FULL_VOCAB_ROWS * DRAFTER_HIDDEN_SIZE * HEAD_ELEMENT_BYTES
)
SUBSET_HEAD_BYTES = DRAFT_VOCAB_ROWS * DRAFTER_HIDDEN_SIZE * HEAD_ELEMENT_BYTES
# PHASE 2 PROJECTION ONLY -- what the same 65_536-row slice costs if the draft
# head is READ as NVFP4 instead of dequantised: packed U8 [65536, 2560] plus
# F8_E4M3 per-16 block scales [65536, 320]. Not served; see projected_scenarios.
DRAFT_64K_NVFP4_HEAD_BYTES = (
    DRAFT_VOCAB_ROWS * (DRAFTER_HIDDEN_SIZE // 2)
    + DRAFT_VOCAB_ROWS * (DRAFTER_HIDDEN_SIZE // 16)
)
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
# PROJECTION, not a pin: the Phase-2 FP4 draft-head read.
PHASE2_MANDATORY_WEIGHT_BYTES = (
    LEGACY_MANDATORY_WEIGHT_BYTES
    + MTP_FORWARD_BYTES
    + MTP_FORWARD_PASSES * DRAFT_64K_NVFP4_HEAD_BYTES
)
# Preserve the full precision used by the full-vocabulary B1 acceptance contract.
FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS = 93.15228665201465
FULL_VOCAB_SLO_CAP_MS = 107.12512964981684
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


def _classify_tensor(name: str) -> str | None:
    """Which floor bucket a tensor belongs to, decided BY NAME.

    Deliberately not keyed on the containing file. RadixArk puts the 15 `mtp.*`
    drafter tensors inside model-00003-of-00003.safetensors next to ordinary
    body tensors, so the unsloth-era "the MTP shard is a separate file" rule
    would have silently counted the drafter as target-model bytes (inflating a
    once-per-step term with a five-times-per-step one) or missed it entirely.
    """
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("lm_head"):
        return "lm_head"
    if name.startswith("model.language_model.embed_tokens"):
        return "embed_tokens"
    if name.startswith("model.language_model.layers."):
        return "layers"
    if name.startswith("model.language_model.norm"):
        return "final_norm"
    if name.startswith("model.visual."):
        return "visual"
    return None


def derive_checkpoint_bytes(model_root: Path = MODEL_ROOT) -> dict[str, Any]:
    """Re-derive the byte terms by summing real tensor spans on disk."""
    shards = sorted(
        path
        for path in model_root.iterdir()
        if path.is_file() and path.name.endswith(SHARD_SUFFIX)
    )
    if not shards:
        raise SystemExit(
            f"fr14 floor derivation found no *{SHARD_SUFFIX} shards in {model_root}"
        )

    spans: dict[str, int] = {}
    shard_names: list[str] = []
    for shard in shards:
        shard_names.append(shard.name)
        for name, span in _safetensors_tensor_bytes(shard).items():
            if name in spans:
                raise SystemExit(
                    "fr14 floor derivation refuses to guess: tensor "
                    f"{name!r} appears in more than one shard"
                )
            spans[name] = span

    buckets = dict.fromkeys(
        ("layers", "final_norm", "lm_head", "embed_tokens", "visual", "mtp"), 0
    )
    counts = dict.fromkeys(buckets, 0)
    unclassified: dict[str, int] = {}
    for name, span in spans.items():
        bucket = _classify_tensor(name)
        if bucket is None:
            unclassified[name] = span
            continue
        buckets[bucket] += span
        counts[bucket] += 1
    if unclassified:
        raise SystemExit(
            "fr14 floor derivation refuses to guess: unclassified "
            f"tensors {sorted(unclassified)}"
        )

    return {
        "model_root": str(model_root),
        "shards": shard_names,
        "target_model_bytes": buckets["layers"] + buckets["final_norm"],
        "target_decoder_layer_bytes": buckets["layers"],
        "target_final_norm_bytes": buckets["final_norm"],
        "lm_head_bytes_on_disk": buckets["lm_head"],
        "lm_head_tensor_count": counts["lm_head"],
        "embed_tokens_bytes_excluded": buckets["embed_tokens"],
        "vision_tower_bytes_excluded": buckets["visual"],
        "mtp_forward_bytes_per_pass": buckets["mtp"],
        "mtp_tensor_count": counts["mtp"],
        "target_tensor_count": counts["layers"] + counts["final_norm"],
        "checkpoint_total_tensor_count": len(spans),
        "checkpoint_total_tensor_bytes": sum(spans.values()),
    }


# RadixArk's own qualification.json records this as
# checkpoint.output_indexed_payload_bytes. Cross-checking the WHOLE ledger
# against the publisher's audit catches a truncated or partially-redownloaded
# shard that happens to leave one bucket intact.
CHECKPOINT_TOTAL_TENSOR_BYTES = 21_921_428_072
CHECKPOINT_TOTAL_TENSOR_COUNT = 2_194


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
    if derived["mtp_tensor_count"] != 15:
        problems.append(
            "the drafter is not the 15-tensor BF16 MTP head the floor assumes: "
            f"{derived['mtp_tensor_count']} mtp.* tensors found"
        )
    if derived["lm_head_bytes_on_disk"] != NVFP4_HEAD_BYTES:
        problems.append(
            "lm_head is not the NVFP4 head the floor assumes: "
            f"{derived['lm_head_bytes_on_disk']} != {NVFP4_HEAD_BYTES} "
            f"(BF16 head would be {FULL_BF16_HEAD_REFERENCE_BYTES})"
        )
    if derived["lm_head_tensor_count"] != 4:
        problems.append(
            "lm_head is not the 4-tensor ModelOpt NVFP4 set: "
            f"{derived['lm_head_tensor_count']} lm_head* tensors found"
        )
    if derived["checkpoint_total_tensor_bytes"] != CHECKPOINT_TOTAL_TENSOR_BYTES:
        problems.append(
            "checkpoint total tensor bytes drifted from the publisher's own "
            "qualification.json output_indexed_payload_bytes: "
            f"{derived['checkpoint_total_tensor_bytes']} "
            f"!= {CHECKPOINT_TOTAL_TENSOR_BYTES}"
        )
    if derived["checkpoint_total_tensor_count"] != CHECKPOINT_TOTAL_TENSOR_COUNT:
        problems.append(
            "checkpoint tensor count drifted: "
            f"{derived['checkpoint_total_tensor_count']} "
            f"!= {CHECKPOINT_TOTAL_TENSOR_COUNT}"
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
            "/models/qwen3.8-27b-nvfp4-radixark; no FR13 constant was scaled"
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
            "nvfp4_head": {
                "bytes": FULL_HEAD_BYTES,
                "rows": FULL_VOCAB_ROWS,
                "hidden_size": DRAFTER_HIDDEN_SIZE,
                "tensors": 4,
                "source": (
                    "summed tensor spans of the 4 shipped lm_head.* tensors "
                    "(U8 [248320,2560] packed + F8_E4M3 [248320,320] per-16 "
                    "block scales + 2 F32 global scalars)"
                ),
                "bf16_head_reference_bytes": FULL_BF16_HEAD_REFERENCE_BYTES,
            },
            "draft_64k_bf16_head": {
                "bytes": SUBSET_HEAD_BYTES,
                "element_bytes": HEAD_ELEMENT_BYTES,
                "hidden_size": DRAFTER_HIDDEN_SIZE,
                "rows": DRAFT_VOCAB_ROWS,
                "source": (
                    "PHASE 1: the 65536 NVFP4 rows sliced out of the head are "
                    "dequantised to BF16 at boot, so the K64 read stays BF16"
                ),
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
                    "summed tensor spans of the 15 mtp.* tensors (BF16), "
                    "located BY NAME -- they live inside "
                    "model-00003-of-00003.safetensors, not a separate shard"
                ),
            },
        },
        "scenarios": {
            "legacy_target_plus_verifier_head": {
                "component_formula": "target_model + nvfp4_head",
                "mandatory_weight_bytes": LEGACY_MANDATORY_WEIGHT_BYTES,
                "mandatory_weight_floor_ms": _floor_ms(
                    LEGACY_MANDATORY_WEIGHT_BYTES
                ),
                "is_full_speculative_step_floor": False,
            },
            "current_one_full_plus_four_64k_draft_heads": {
                "component_formula": (
                    "target_model + nvfp4_head + mtp_forward + "
                    "nvfp4_head + 4 * draft_64k_bf16_head"
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
                    "target_model + nvfp4_head + mtp_forward + "
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
                    "target_model + nvfp4_head + mtp_forward + "
                    "5 * nvfp4_head"
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
                "FR13_DRAFT_HEAD_FP8 is DEAD under both FR14 arms. It "
                "re-read the five K64 draft heads as FP8 qweight + FP32 "
                "scales (1_678_131_200 B instead of 3_355_443_200 B). Arm A "
                "serves a BF16 head (unsloth's FP8 per-channel head was "
                "dequantised to make the checkpoint loadable); arm B serves "
                "an NVFP4 head whose sliced rows PHASE 1 dequantises to BF16 "
                "at boot. Neither leaves an FP8 head to read, so the arm "
                "would pin a floor the hardware cannot realise. The launcher "
                "and the floor-timer sequence refuse it. The live successor "
                "is the NVFP4 draft-head read below, which is a real 8.834 ms "
                "and needs an FP4 GEMV unit, not a requant."
            ),
        },
        "projected_scenarios": {
            "root_64k_five_nvfp4_draft_heads": {
                "component_formula": (
                    "target_model + nvfp4_head + mtp_forward + "
                    "5 * draft_64k_nvfp4_head"
                ),
                "draft_64k_nvfp4_head_bytes": DRAFT_64K_NVFP4_HEAD_BYTES,
                "mandatory_weight_bytes": PHASE2_MANDATORY_WEIGHT_BYTES,
                "mandatory_weight_floor_ms": _floor_ms(
                    PHASE2_MANDATORY_WEIGHT_BYTES
                ),
                "nonweight_costs_included": False,
                "status": (
                    "PROJECTION, NOT PINNED. Needs an FP4 draft-head GEMV "
                    "unit; it is a byte lever with its own gate, and no "
                    "instrument may quote it as a floor until that lands."
                ),
            },
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
