#!/usr/bin/env python3
"""Validate a Git- and profile-bound grouped-GQA production credential."""

from __future__ import annotations

import argparse
import json
import re
import stat
from pathlib import Path
from typing import Any


SCHEMA = "fr13.fixed32.gdn_single_launch.real_task_credential.v3"
CANDIDATE = "fixed32_gdn_single_launch_gqa_group3_v1"
REFERENCE = "fixed32_gdn_two_launch_reference_v1"
RUNTIME_PROFILE = "fixed32"
RUNTIME_SEQUENCE = "scripts/fr13_fixed32_floor_timers_seq.sh"
BLOCK_MAP_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
B1_TASK_IDS = ["astropy__astropy-12907"]
EXACT4_TASK_IDS = [
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
]
MODE = {
    "tail6_fixed32": {
        "slug": "tail23",
        "logical_topology": "Tail23",
        "logical_drafts": 23,
        "valid_mask": 0x7A9CE7FF,
        "batches": frozenset({4}),
    },
    "hydra27_fixed32": {
        "slug": "hydra27",
        "logical_topology": "Hydra27",
        "logical_drafts": 27,
        "valid_mask": 0x7ABDFFFF,
        "batches": frozenset({1, 4}),
    },
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_CREDENTIAL_BYTES = 16 << 20


class CredentialError(RuntimeError):
    """The credential is not valid for this exact production process."""


def _duplicate_checked(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CredentialError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise CredentialError(f"non-finite JSON constant {value!r}")


def _load(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
    except OSError as error:
        raise CredentialError(f"credential is missing: {path}") from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise CredentialError(
            "credential must be a singly linked regular non-symlink file"
        )
    if info.st_size <= 0 or info.st_size > MAX_CREDENTIAL_BYTES:
        raise CredentialError(
            f"credential size must be in [1, {MAX_CREDENTIAL_BYTES}] bytes"
        )
    try:
        raw = path.read_bytes()
        if len(raw) != info.st_size:
            raise CredentialError("credential size changed while it was read")
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_duplicate_checked,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CredentialError(f"credential is not strict ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CredentialError("credential must be a JSON object")
    return payload


def _require_exact(
    payload: dict[str, Any], expected: dict[str, object]
) -> None:
    for key, value in expected.items():
        actual = payload.get(key)
        if type(actual) is not type(value) or actual != value:
            raise CredentialError(
                f"credential {key} is not exactly bound to this production process"
            )


def validate_credential(
    path: Path,
    *,
    source_commit: str,
    profile: str,
    mode: str,
    batch: int,
) -> dict[str, Any]:
    if HEX40.fullmatch(source_commit) is None:
        raise CredentialError("current HEAD is not a full lowercase Git object ID")
    if profile != RUNTIME_PROFILE:
        raise CredentialError(f"unsupported runtime profile: {profile!r}")
    contract = MODE.get(mode)
    if contract is None or type(batch) is not int or batch not in contract["batches"]:
        raise CredentialError(
            f"unsupported grouped-GQA production mode/batch: {mode!r}/B{batch}"
        )
    run_classification = (
        "one_real_swe_verified_b1_graph_byte_diagnostic"
        if batch == 1
        else "real_swe_verified_exact4_b4_graph_byte_diagnostic"
    )
    payload = _load(path)
    _require_exact(
        payload,
        {
            "schema": SCHEMA,
            "status": "PASS",
            "credential_scope": f"{contract['slug']}:b{batch}",
            "run_classification": run_classification,
            "runtime_profile": profile,
            "runtime_sequence": RUNTIME_SEQUENCE,
            "candidate": CANDIDATE,
            "reference": REFERENCE,
            "mode": mode,
            "logical_topology": contract["logical_topology"],
            "logical_drafts": contract["logical_drafts"],
            "valid_mask": contract["valid_mask"],
            "expected_batch": batch,
            "batch_size": batch,
            "concurrency": batch,
            "task_ids": B1_TASK_IDS if batch == 1 else EXACT4_TASK_IDS,
            "physical_rows": 32,
            "draft_vocab_k": 65536,
            "draft_vocab_root": 1,
            "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
            "reference_physical_launches_per_request_layer": 2,
            "candidate_physical_launches_per_request_layer": 1,
            "reference_served": True,
            "state_restored": True,
            "raw_byte_equal": True,
            "task_completion_verified": True,
            "finalized_ingress_verified": True,
            "runtime_launch_end_stable": True,
            "external_launch_end_stable": True,
            "graph_work_census_verified": True,
            "batch_specific_pass_verified": True,
            "production_enabled": False,
            "performance_measurement": False,
            "acceptance_valid": False,
            "floor_acceptance_eligible": False,
            "source_commit": source_commit,
            "runtime_manifest_source_commit": source_commit,
            "arm_git_head_source_commit_match": True,
            "source_commit_git_show_verified": True,
        },
    )
    for key in (
        "source_closure_sha256",
        "kernel_source_sha256",
        "candidate_source_sha256",
        "gqa_group3_source_sha256",
        "runtime_manifest_canonical_sha256",
        "runtime_manifest_sha256",
        "entrypoint_sha256",
        "common_runner_sha256",
        "reducer_sha256",
    ):
        value = payload.get(key)
        if not isinstance(value, str) or HEX64.fullmatch(value) is None:
            raise CredentialError(f"credential {key} is not a lowercase SHA-256")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credential", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--profile", choices=(RUNTIME_PROFILE,), required=True)
    parser.add_argument("--mode", choices=tuple(MODE), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 4), required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        validate_credential(
            args.credential,
            source_commit=args.source_commit,
            profile=args.profile,
            mode=args.mode,
            batch=args.batch,
        )
    except CredentialError as error:
        raise SystemExit(f"GDN GQA-group3 production credential rejected: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
