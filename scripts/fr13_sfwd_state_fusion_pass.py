#!/usr/bin/env python3
"""Validate the source-bound SFWD state-fusion B1 live PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any


CANDIDATE = "fixed32_sfwd_state_fusion_rowgroup8_v3"
LIVE_SCHEMA = "fr13.fixed32.sfwd_state_fusion.live_pass.v1"
ENGAGEMENT_SCHEMA = (
    "fr13.fixed32.sfwd_state_fusion.production_engagement.v1"
)
TASK_MARKER = "swe_verified:astropy__astropy-12907"
RUN_CLASSIFICATION = "one_real_swe_verified_k64_root_b1_byte_diagnostic"
DRAFT_VOCAB_K = 65536
DRAFT_VOCAB_ROOT = 1
DRAFT_VOCAB_BLOCKS_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)


class PassError(ValueError):
    """The qualification artifact is missing or violates its closed contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise PassError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> None:
    raise PassError(f"non-finite JSON value: {value}")


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise PassError(f"{label} does not exist: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise PassError(f"{label} is not a regular non-symlink file: {path}")
    raw = path.read_bytes()
    if not raw or len(raw) > 131072:
        raise PassError(f"{label} is empty or exceeds 128 KiB")
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PassError(f"{label} is not strict ASCII JSON") from error
    if not isinstance(payload, dict):
        raise PassError(f"{label} must contain a JSON object")
    return payload, raw


def _require_sha256(value: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PassError(f"{label} must be a lowercase SHA-256")


def validate_live_result(
    payload: dict[str, Any], *, source_sha256: str
) -> dict[str, Any]:
    _require_sha256(source_sha256, "kernel source identity")
    required = {
        "schema": LIVE_SCHEMA,
        "status": "byte_pass_source_only",
        "run_classification": RUN_CLASSIFICATION,
        "candidate": CANDIDATE,
        "source_sha256": source_sha256,
        "task_marker": TASK_MARKER,
        "batch": 1,
        "draft_vocab_k": DRAFT_VOCAB_K,
        "draft_vocab_root": DRAFT_VOCAB_ROOT,
        "draft_vocab_blocks_sha256": DRAFT_VOCAB_BLOCKS_SHA256,
        "layer_count": 48,
        "physical_rows_per_request": 32,
        "candidate_conv_launches_per_layer": 1,
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "real_task_authenticated": True,
        "reference_always_served": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise PassError(
                f"live PASS {key} mismatch: {payload.get(key)!r} != {expected!r}"
            )
    layer_keys = payload.get("layer_keys")
    if (
        not isinstance(layer_keys, list)
        or len(layer_keys) != 48
        or len(set(layer_keys)) != 48
        or any(
            not isinstance(key, str)
            or re.fullmatch(r"0x[0-9a-f]+", key) is None
            for key in layer_keys
        )
    ):
        raise PassError("live PASS does not cover 48 unique layer keys")
    return payload


def validate_live_path(
    live_result: Path,
    *,
    expected_live_sha256: str,
    kernel_source: Path,
) -> dict[str, Any]:
    _require_sha256(expected_live_sha256, "expected live PASS identity")
    payload, raw = load_json(live_result, "live PASS")
    actual_live_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_live_sha256 != expected_live_sha256:
        raise PassError("live PASS raw SHA-256 mismatch")
    try:
        source_info = kernel_source.lstat()
    except FileNotFoundError as error:
        raise PassError(f"kernel source does not exist: {kernel_source}") from error
    if kernel_source.is_symlink() or not stat.S_ISREG(source_info.st_mode):
        raise PassError("kernel source must be a regular non-symlink file")
    source_sha256 = sha256(kernel_source)
    validate_live_result(payload, source_sha256=source_sha256)
    return {
        "schema": "fr13.fixed32.sfwd_state_fusion.production_binding.v1",
        "status": "pass",
        "candidate": CANDIDATE,
        "task_marker": TASK_MARKER,
        "batch_size": 1,
        "draft_vocab_k": DRAFT_VOCAB_K,
        "draft_vocab_root": DRAFT_VOCAB_ROOT,
        "draft_vocab_blocks_sha256": DRAFT_VOCAB_BLOCKS_SHA256,
        "physical_rows_per_request": 32,
        "layer_count": 48,
        "live_pass_sha256": actual_live_sha256,
        "source_sha256": source_sha256,
        "candidate_serving_permitted": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }


def validate_engagement(
    payload: dict[str, Any], *, live_pass_sha256: str, source_sha256: str
) -> dict[str, Any]:
    required = {
        "schema": ENGAGEMENT_SCHEMA,
        "status": "engaged",
        "run_classification": (
            "one_real_swe_verified_k64_root_b1_production_timing_diagnostic"
        ),
        "candidate": CANDIDATE,
        "task_marker": TASK_MARKER,
        "batch_size": 1,
        "draft_vocab_k": DRAFT_VOCAB_K,
        "draft_vocab_root": DRAFT_VOCAB_ROOT,
        "draft_vocab_blocks_sha256": DRAFT_VOCAB_BLOCKS_SHA256,
        "layer_count": 48,
        "live_pass_sha256": live_pass_sha256,
        "source_sha256": source_sha256,
        "candidate_served": True,
        "real_task_bound": True,
        "physical_rows_per_request": 32,
        "source_rows_per_request": 36,
        "candidate_conv_launches_per_layer": 1,
        "incumbent_conv_launches_per_layer": 0,
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "real_task_pass_bound": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise PassError(
                f"engagement {key} mismatch: {payload.get(key)!r} != {expected!r}"
            )
    launches = payload.get("launches_observed")
    layers = payload.get("layer_keys")
    if (
        isinstance(launches, bool)
        or not isinstance(launches, int)
        or launches < 48
        or not isinstance(layers, list)
        or len(layers) != 48
        or len(set(layers)) != 48
        or any(
            not isinstance(key, str)
            or re.fullmatch(r"0x[0-9a-f]+", key) is None
            for key in layers
        )
    ):
        raise PassError("engagement lacks 48 unique served layer launches")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="ascii", dir=path.parent, delete=False
    ) as handle:
        handle.write(raw)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--live-result", type=Path, required=True)
    validate.add_argument("--expected-live-sha256", required=True)
    validate.add_argument("--kernel-source", type=Path, required=True)
    validate.add_argument("--out", type=Path)
    engagement = commands.add_parser("verify-engagement")
    engagement.add_argument("--engagement", type=Path, required=True)
    engagement.add_argument("--expected-live-sha256", required=True)
    engagement.add_argument("--kernel-source", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate":
        binding = validate_live_path(
            args.live_result,
            expected_live_sha256=args.expected_live_sha256,
            kernel_source=args.kernel_source,
        )
        if args.out is not None:
            _write_json(args.out, binding)
        print(json.dumps(binding, ensure_ascii=True, sort_keys=True))
        return 0
    _require_sha256(args.expected_live_sha256, "expected live PASS identity")
    source_sha256 = sha256(args.kernel_source)
    engagement, _ = load_json(args.engagement, "production engagement")
    validate_engagement(
        engagement,
        live_pass_sha256=args.expected_live_sha256,
        source_sha256=source_sha256,
    )
    print(json.dumps(engagement, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
