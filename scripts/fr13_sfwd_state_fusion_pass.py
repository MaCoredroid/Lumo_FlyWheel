#!/usr/bin/env python3
"""Validate the source-bound SFWD state-fusion B1 live PASS."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any


CANDIDATE = "fixed32_sfwd_state_fusion_v1"
LIVE_SCHEMA = "fr13.fixed32.sfwd_state_fusion.live_pass.v1"
ENGAGEMENT_SCHEMA = (
    "fr13.fixed32.sfwd_state_fusion.production_engagement.v1"
)
TASK_MARKER = "swe_verified:astropy__astropy-12907"
RUN_CLASSIFICATION = (
    "one_real_swe_verified_full_vocab_b1_byte_timing_diagnostic"
)
QUALIFIED_SOURCE_SHA256 = (
    "a9decdbe60db4227e4128ea05bfa405e09e284bfbc7c3ed216cd73d3f72d35ec"
)
QUALIFIED_CLOSURE_SHA256 = (
    "792c14941d933e9978811cd7dba33e9c9877638ed7f8fca4aa2044643cfbf038"
)
CLOSURE_NAMES = (
    "_FR13_FIXED32_MODES",
    "_FR13_FIXED32_PARENT",
    "_FR13_FIXED32_SFWD_STATE_FUSION_CANDIDATE_ID",
    "_FR13_FIXED32_SFWD_CONV_STATE_LEN",
    "fixed32_sfwd_state_fusion_contract",
    "_fr13_fixed32_sfwd_state_fusion_kernel",
    "_fr13_fixed32_conv_source_flat_expected",
    "launch_fixed32_sfwd_state_fusion",
)


class PassError(ValueError):
    """The qualification artifact is missing or violates its closed contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_closure_sha256(path: Path) -> str:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=os.fspath(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise PassError(f"kernel candidate closure cannot be parsed: {error}") from error
    members: dict[str, str] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, ast.Assign):
            names = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            ]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
        for name in names:
            if name in CLOSURE_NAMES:
                if name in members:
                    raise PassError(f"kernel candidate closure duplicates {name}")
                members[name] = ast.dump(
                    node, annotate_fields=True, include_attributes=False
                )
    missing = tuple(name for name in CLOSURE_NAMES if name not in members)
    if missing:
        raise PassError(f"kernel candidate closure is missing {missing!r}")
    canonical = "".join(
        f"{name}\0{members[name]}\0" for name in CLOSURE_NAMES
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


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
    runtime_source_sha256 = sha256(kernel_source)
    closure_sha256 = candidate_closure_sha256(kernel_source)
    if closure_sha256 != QUALIFIED_CLOSURE_SHA256:
        raise PassError("kernel candidate closure drifted from qualified source")
    validate_live_result(payload, source_sha256=QUALIFIED_SOURCE_SHA256)
    return {
        "schema": "fr13.fixed32.sfwd_state_fusion.production_binding.v1",
        "status": "pass",
        "candidate": CANDIDATE,
        "task_marker": TASK_MARKER,
        "batch_size": 1,
        "physical_rows_per_request": 32,
        "layer_count": 48,
        "live_pass_sha256": actual_live_sha256,
        "qualified_source_sha256": QUALIFIED_SOURCE_SHA256,
        "runtime_source_sha256": runtime_source_sha256,
        "candidate_closure_sha256": closure_sha256,
        "candidate_serving_permitted": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }


def validate_engagement(
    payload: dict[str, Any],
    *,
    live_pass_sha256: str,
    runtime_source_sha256: str,
    closure_sha256: str,
) -> dict[str, Any]:
    required = {
        "schema": ENGAGEMENT_SCHEMA,
        "status": "engaged",
        "run_classification": "real_swe_verified_exact4_k64_b1_kernel_stack",
        "candidate": CANDIDATE,
        "batch_size": 1,
        "layer_count": 48,
        "live_pass_sha256": live_pass_sha256,
        "source_sha256": runtime_source_sha256,
        "qualified_source_sha256": QUALIFIED_SOURCE_SHA256,
        "candidate_closure_sha256": closure_sha256,
        "candidate_served": True,
        "physical_rows_per_request": 32,
        "source_rows_per_request": 36,
        "candidate_conv_launches_per_layer": 1,
        "incumbent_conv_launches_per_layer": 0,
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "real_task_pass_bound": True,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "timing_eligible": True,
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
    runtime_source_sha256 = sha256(args.kernel_source)
    closure_sha256 = candidate_closure_sha256(args.kernel_source)
    if closure_sha256 != QUALIFIED_CLOSURE_SHA256:
        raise PassError("kernel candidate closure drifted from qualified source")
    engagement, _ = load_json(args.engagement, "production engagement")
    validate_engagement(
        engagement,
        live_pass_sha256=args.expected_live_sha256,
        runtime_source_sha256=runtime_source_sha256,
        closure_sha256=closure_sha256,
    )
    print(json.dumps(engagement, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
