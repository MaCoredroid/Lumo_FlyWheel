#!/usr/bin/env python3
"""Fail-closed selector for the fixed32 K64 block-FP8 draft head."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


class SelectorError(RuntimeError):
    """The K64 block-FP8 draft-head selector contract is not exact."""


SCHEMA = "fr13.fixed32.dfwd_k64_fp8_static_selector.v1"
SOURCE_REL = "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
SOURCE_SHA256 = "99c85c64834c7021367de1f8735a8ec1f307649ff3f494741c9236832c10f54a"
SMOKE_SOURCE_REL = "scripts/fr13_draft_head_fp8_sm121_smoke.py"
SMOKE_SOURCE_SHA256 = "68af218f3ffc5c13789e73a32bf4f30410eed3182a3d2aec5cbc188b8374b048"
SMOKE_RESULT_REL = "results/fr13_draft_head_fp8_sm121_smoke_20260803/result.json"
SMOKE_RESULT_SHA256 = "545cb56e858c6d182db243735520d7af8324082a67c4569dd69fec2f865b4bed"
BLOCKS_CONTAINER_PATH = "/workspace/scripts/fr13_dvk_subset_blocks.json"

_EXACT_ENV = {
    "FR13_DRAFT_HEAD_FP8": "1",
    "FR13_DRAFT_HEAD_FP8_STATIC_IO": "1",
    "FR13_DRAFT_HEAD_FP8_ENGAGEMENT_JSON": (
        "/logs/fr13_draft_head_fp8.engagement.json"
    ),
    "FR13_FIXED32_MODE": "hydra27_fixed32",
    "ENFORCE_EAGER": "0",
    "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
    "FR13_DRAFT_VOCAB_ROOT": "1",
    "FR13_DRAFT_VOCAB_K": "65536",
    "FR13_DRAFT_VOCAB_BLOCKS": BLOCKS_CONTAINER_PATH,
    "FR13_DRAFTER_SINGLE_LOGITS": "1",
    "FR13_FIXED32_PHYSICAL_DRAFTS": "31",
    "FR13_FIXED32_ACTIVE_NODES": "27",
    "NUM_SPECULATIVE_TOKENS": "31",
    "FR13_MANDATORY_WEIGHT_BYTES": "30989326208",
    "FR13_WEIGHT_FLOOR_MS": "113.514015414",
}

_DISABLED_ENV = {
    "FR13_DRAFT_HEAD_K64_TC": "0",
    "FR13_DRAFT_HEAD_B14_WARP4_PAIR8": "0",
    "FR13_DRAFT_HEAD_PAD_ROWS": "0",
    "FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB": "0",
    "FR13_DRAFT_HEAD_M32_LIVE_AB": "0",
    "FR13_DRAFT_HEAD_M32_PRODUCTION": "0",
    "FR13_DRAFT_HEAD_M32_TIMING_ARM": "0",
    "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB": "0",
    "FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE": "0",
    "FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE": "0",
    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION": "0",
    "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB": "0",
    "FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE": "0",
    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION": "0",
    "FR13_DFWD_K64_TOP3": "0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _authenticated_file(repo: Path, rel: str, expected_sha: str) -> Path:
    path = repo / rel
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise SelectorError(f"selector input is missing: {rel}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise SelectorError(f"selector input must be a regular non-symlink: {rel}")
    actual = _sha256(path)
    if actual != expected_sha:
        raise SelectorError(
            f"selector input SHA-256 drifted: {rel}: {actual} != {expected_sha}"
        )
    return path


def _validate_smoke(payload: object) -> None:
    if not isinstance(payload, dict):
        raise SelectorError("block-FP8 smoke artifact is malformed")
    traffic = payload.get("traffic")
    batches = payload.get("batches")
    qweight = payload.get("qweight")
    scale = payload.get("weight_scale")
    if (
        payload.get("schema") != "fr13.draft_head_fp8_sm121_smoke.v1"
        or payload.get("status") != "PASS"
        or payload.get("classification") != "kernel_smoke_not_acceptance"
        or payload.get("production_default_enabled") is not False
        or payload.get("lossless_acceptance_claimed") is not False
        or payload.get("device_capability") != [12, 1]
        or payload.get("cutlass_block_fp8_supported") is not True
        or payload.get("script_sha256") != SMOKE_SOURCE_SHA256
        or not isinstance(traffic, dict)
        or traffic.get("calls_per_event") != 5
        or traffic.get("bf16_weight_bytes_per_call_removed") != 671_088_640
        or traffic.get("fp8_weight_bytes_per_call") != 335_544_320
        or traffic.get("fp32_weight_scale_bytes_per_call") != 81_920
        or traffic.get("mandatory_bytes_per_event") != 1_678_131_200
        or traffic.get("candidate_full_step_mandatory_bytes") != 30_989_326_208
        or traffic.get("candidate_weight_floor_ms") != 113.514015414
        or traffic.get("one_sided_u95_cap_ms") != 130.541117726
        or not isinstance(qweight, dict)
        or qweight.get("shape") != [65_536, 5_120]
        or qweight.get("stride") != [5_120, 1]
        or qweight.get("dtype") != "torch.float8_e4m3fn"
        or not isinstance(scale, dict)
        or scale.get("shape") != [512, 40]
        or scale.get("stride") != [40, 1]
        or not isinstance(batches, dict)
    ):
        raise SelectorError("block-FP8 smoke artifact contract drifted")
    for batch in (1, 4):
        result = batches.get(str(batch))
        if (
            not isinstance(result, dict)
            or result.get("activation_q", {}).get("shape") != [batch, 5_120]
            or result.get("activation_q", {}).get("stride") != [5_120, 1]
            or result.get("activation_scale", {}).get("shape") != [batch, 40]
            or result.get("activation_scale", {}).get("stride") != [1, batch]
            or result.get("output", {}).get("shape") != [batch, 65_536]
            or result.get("output", {}).get("stride") != [65_536, 1]
        ):
            raise SelectorError(f"block-FP8 B{batch} static geometry drifted")


def validate_environment(
    env: Mapping[str, str], repo: Path, source_commit: str
) -> dict[str, object]:
    repo = repo.resolve()
    if len(source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in source_commit
    ):
        raise SelectorError("source commit must be exactly 40 lowercase hex characters")
    if env.get("FR13_DRAFT_HEAD_FP8_SOURCE_COMMIT", "") != source_commit:
        raise SelectorError(
            "selector source commit does not match the launched checkout"
        )
    arm = env.get("FR13_DRAFT_HEAD_FP8_ARM", "")
    if (
        not arm
        or len(arm) > 200
        or any(
            c
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for c in arm
        )
    ):
        raise SelectorError("FR13_DRAFT_HEAD_FP8_ARM must be canonical")

    for key, expected in _EXACT_ENV.items():
        if env.get(key, "") != expected:
            raise SelectorError(f"{key} must be exactly {expected!r}")
    for key, expected in _DISABLED_ENV.items():
        if env.get(key, expected) != expected:
            raise SelectorError(f"competing draft-head mode must be disabled: {key}")

    batch = env.get("MAX_NUM_SEQS", "")
    if batch not in {"1", "4"} or env.get("SWE_CONCURRENCY", "") != batch:
        raise SelectorError("selector requires matching exact B1 or B4 concurrency")

    source = _authenticated_file(repo, SOURCE_REL, SOURCE_SHA256)
    _authenticated_file(repo, SMOKE_SOURCE_REL, SMOKE_SOURCE_SHA256)
    result_path = _authenticated_file(repo, SMOKE_RESULT_REL, SMOKE_RESULT_SHA256)
    try:
        smoke = json.loads(result_path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectorError("block-FP8 smoke artifact JSON is malformed") from error
    _validate_smoke(smoke)

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "source_commit": source_commit,
        "mode": "hydra27_fixed32",
        "batch_size": int(batch),
        "physical_drafts": 31,
        "active_nodes": 27,
        "draft_vocab_k": 65_536,
        "operation": "vllm_cutlass_block_fp8_scaled_mm_static_io",
        "identities": {
            "source_sha256": _sha256(source),
            "static_smoke_sha256": SMOKE_RESULT_SHA256,
        },
        "five_call_mandatory_bytes": 1_678_131_200,
        "full_step_mandatory_bytes": 30_989_326_208,
        "weight_floor_ms": 113.514015414,
        "one_sided_u95_cap_ms": 130.541117726,
        "proposal_only": True,
        "target_authority_changed": False,
        "static_smoke_is_performance_evidence": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=args.repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if resolved != args.source_commit:
        raise SelectorError("--source-commit does not match repository HEAD")
    print(
        json.dumps(
            validate_environment(os.environ, args.repo, resolved), sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
