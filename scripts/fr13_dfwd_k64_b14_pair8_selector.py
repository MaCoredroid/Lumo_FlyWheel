#!/usr/bin/env python3
"""Fail-closed selector validation for the fixed32 packed K64 draft head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path


class SelectorError(RuntimeError):
    """The packed draft-head selector contract is not exact."""


SCHEMA = "fr13.fixed32.dfwd_k64_b14_warp4_pair8_selector.v1"
SOURCE_REL = "csrc/fr13_bf16_gemvx_k64_b14_warp4_pair8.cu"
ARTIFACT_REL = "results/fr13_fixed32_dfwd_k64_b14_warp4_pair8_sm121a_20260805"
SO_REL = f"{ARTIFACT_REL}/fr13_bf16_k64_b14_warp4_pair8.abi3.so"
BUILD_REL = f"{ARTIFACT_REL}/build_attestation.json"
MANIFEST_REL = f"{ARTIFACT_REL}/manifest.json"
SOURCE_SHA256 = "01fd0ec526afcfa3e3a4a42ec95b95dfa04d566fa8431eee84d72d7eada9ed3f"
SO_SHA256 = "dcd9301f447d1ef4a81240027af3a008558dfa080b8660ac9dba7b2c5800ec7f"
SO_BYTES = 161056
BUILD_SHA256 = "3ca9dda783db5fb8a0cabd05ef9671f6a6d2063c8cc979d4c4694d9530dc848f"
MANIFEST_SHA256 = "577e780472bbc0af53217f5789bb38145bee22b7e9fc962ecc1a1d58f46b4e3e"
BLOCKS_CONTAINER_PATH = "/workspace/scripts/fr13_dvk_subset_blocks.json"

_EXACT_ENV = {
    "FR13_DRAFT_HEAD_B14_WARP4_PAIR8": "1",
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
}

_DISABLED_ENV = {
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
    "FR13_DRAFT_HEAD_FP8": "0",
    "FR13_DRAFT_HEAD_FP8_STATIC_IO": "0",
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


def validate_environment(
    env: Mapping[str, str], repo: Path, source_commit: str
) -> dict[str, object]:
    repo = repo.resolve()
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise SelectorError("source commit must be exactly 40 lowercase hex characters")
    if env.get("FR13_DRAFT_HEAD_B14_WARP4_PAIR8_SOURCE_COMMIT", "") != source_commit:
        raise SelectorError("selector source commit does not match the launched checkout")

    for key, expected in _EXACT_ENV.items():
        if env.get(key, "") != expected:
            raise SelectorError(f"{key} must be exactly {expected!r}")
    for key, expected in _DISABLED_ENV.items():
        if env.get(key, expected) != expected:
            raise SelectorError(f"competing draft-head mode must be disabled: {key}")
    if env.get("FR13_DRAFT_HEAD_FP8_ARM", ""):
        raise SelectorError("FR13_DRAFT_HEAD_FP8_ARM must be empty")

    batch = env.get("MAX_NUM_SEQS", "")
    if batch not in {"1", "4"} or env.get("SWE_CONCURRENCY", "") != batch:
        raise SelectorError("selector requires matching exact B1 or B4 concurrency")

    source = _authenticated_file(repo, SOURCE_REL, SOURCE_SHA256)
    binary = _authenticated_file(repo, SO_REL, SO_SHA256)
    build_path = _authenticated_file(repo, BUILD_REL, BUILD_SHA256)
    manifest_path = _authenticated_file(repo, MANIFEST_REL, MANIFEST_SHA256)
    if binary.stat().st_size != SO_BYTES:
        raise SelectorError("candidate binary size drifted")

    try:
        build = json.loads(build_path.read_text(encoding="ascii"))
        manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SelectorError("candidate attestation JSON is malformed") from error
    contract = build.get("kernel_contract", {})
    if (
        build.get("schema")
        != "fr13.fixed32.dfwd_k64_b14_warp4_pair8_sm121a_build.v1"
        or build.get("status") != "BUILT_UNQUALIFIED"
        or build.get("source")
        != {"path": SOURCE_REL, "sha256": SOURCE_SHA256}
        or build.get("binary", {}).get("sha256") != SO_SHA256
        or build.get("binary", {}).get("bytes") != SO_BYTES
        or contract.get("batch_scopes") != [1, 4]
        or contract.get("input") != "BF16[B,5120] contiguous, B in {1,4}"
        or contract.get("weight") != "BF16[65536,5120] contiguous"
        or contract.get("output") != "BF16[B,65536] contiguous"
        or contract.get("proposal_only") is not True
        or contract.get("target_authority_changed") is not False
    ):
        raise SelectorError("candidate build attestation contract drifted")

    candidate = manifest.get("candidate", {})
    file_hashes = {
        item.get("path"): item.get("sha256")
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    if (
        manifest.get("artifact_schema")
        != "fr13.fixed32.dfwd_k64_b14_warp4_pair8_sm121a_artifact.v1"
        or candidate.get("batch_scopes") != [1, 4]
        or candidate.get("proposal_only") is not True
        or candidate.get("target_authority_changed") is not False
        or candidate.get("vocabulary") != 65536
        or candidate.get("hidden") != 5120
        or file_hashes.get(SOURCE_REL) != SOURCE_SHA256
        or file_hashes.get(SO_REL) != SO_SHA256
    ):
        raise SelectorError("candidate artifact manifest contract drifted")

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "source_commit": source_commit,
        "mode": "hydra27_fixed32",
        "batch_size": int(batch),
        "physical_drafts": 31,
        "active_nodes": 27,
        "draft_vocab_k": 65536,
        "operation": (
            "gemvx_m1_warp4_pair8_out"
            if batch == "1"
            else "gemvx_m4_warp4_pair8_out"
        ),
        "identities": {
            "source_sha256": _sha256(source),
            "candidate_so_sha256": _sha256(binary),
            "candidate_so_bytes": binary.stat().st_size,
            "build_attestation_sha256": BUILD_SHA256,
            "artifact_manifest_sha256": MANIFEST_SHA256,
        },
        "proposal_only": True,
        "target_authority_changed": False,
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
    print(json.dumps(validate_environment(os.environ, args.repo, resolved), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
