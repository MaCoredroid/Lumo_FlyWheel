#!/usr/bin/env python3
"""Fail-closed static-build and real exact4 gates for the FA2 qrow32 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fr13_patch_fa2_tree_bias import (  # noqa: E402
    FIXED32_QUERY_TILE32_API_DECLARATION,
    FIXED32_QUERY_TILE32_API_GATE,
    FIXED32_QUERY_TILE32_TRANSLATION_UNIT,
)


FA2_COMMIT = "29210221863736a08f71a866459e368ad1ac4a95"
EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
TARGET_LAYERS = tuple(
    f"language_model.model.layers.{index}.self_attn.attn"
    for index in range(3, 64, 4)
)
EXACT_SAFE_SOURCE_SHA256 = {
    "flash_fwd_launch_template.h": (
        "d9e9f4b92cb731d7955b514449e59b8e411bf7a0c929aafb454f2402d41fe976"
    ),
    "flash_fwd_kernel.h": (
        "934e8c6c2e72c667f3cb0a8dc53b11c16a4eba8e3ac2b5811c882eff399ac3de"
    ),
    "flash_fwd_split_hdim256_bf16_sm80.cu": (
        "53ac045b6b8a960a3134b4538cbde7cdca07fd8e454a39348a7b3da91f4207e0"
    ),
}
EXACT_SAFE_STOCK_OBJECT_SHA256 = (
    "fc31f75bf88bf318cd9530745734c2f7fee6d755b372fefdcf11d157f014f389"
)


class GateError(RuntimeError):
    pass


def _regular(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"{label} is missing, non-regular, or a symlink: {path}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str, label: str) -> str:
    actual = _sha256(_regular(path, label))
    if actual != expected:
        raise GateError(f"{label} SHA-256 drifted: {actual} != {expected}")
    return actual


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(_regular(path, label).read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not canonical ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"{label} root is not an object")
    return payload


def _json_exact(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _json_exact(value[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _json_exact(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return value == expected


def verify_build(args: argparse.Namespace) -> dict[str, Any]:
    fa2 = args.fa2_src.resolve(strict=True)
    source_dir = fa2 / "csrc/flash_attn/src"
    source_hashes: dict[str, str] = {}
    for name, expected in EXACT_SAFE_SOURCE_SHA256.items():
        source_hashes[name] = _require_sha256(
            source_dir / name,
            expected,
            f"exact-safe {name}",
        )

    qrow_source = _regular(
        source_dir / "flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu",
        "qrow32 translation unit",
    )
    if qrow_source.read_text(encoding="ascii") != FIXED32_QUERY_TILE32_TRANSLATION_UNIT:
        raise GateError("qrow32 generated translation unit differs from the patcher")
    kernel_text = (source_dir / "flash_fwd_kernel.h").read_text(encoding="ascii")
    if kernel_text.count("FR13_FA2_TREE_BIAS_TILE_EARLYOUT") != 1:
        raise GateError("exact-safe suffix tile early-out is absent or duplicated")
    api_text = (fa2 / "csrc/flash_attn/flash_api.cpp").read_text(encoding="ascii")
    if (
        api_text.count(FIXED32_QUERY_TILE32_API_DECLARATION.strip()) != 1
        or api_text.count(FIXED32_QUERY_TILE32_API_GATE.strip()) != 1
    ):
        raise GateError("qrow32 gate-only API dispatch is absent or duplicated")

    cmake_text = _regular(fa2 / "CMakeLists.txt", "FA2 CMakeLists").read_text(
        encoding="ascii"
    )
    glob = 'file(GLOB FA2_GEN_SRCS "csrc/flash_attn/src/flash_fwd_*.cu")'
    if cmake_text.count(glob) != 1:
        raise GateError("pinned FA2 source glob drifted")
    if "CONFIGURE_DEPENDS" in cmake_text[cmake_text.index(glob) : cmake_text.index(glob) + 160]:
        raise GateError("unexpected automatic CMake source rediscovery contract")

    qrow_object = _regular(args.qrow_object, "qrow32 CUDA object")
    stock_object_sha256 = _require_sha256(
        args.stock_object,
        EXACT_SAFE_STOCK_OBJECT_SHA256,
        "stock HD256 BF16 CUDA object",
    )
    route: str
    evidence: dict[str, Any]
    if args.build_manifest is not None:
        manifest = _regular(args.build_manifest, "fresh-configure build manifest")
        manifest_text = manifest.read_text(encoding="utf-8", errors="strict")
        if qrow_source.name not in manifest_text:
            raise GateError("configured build manifest does not discover qrow32")
        if manifest.stat().st_mtime_ns < qrow_source.stat().st_mtime_ns:
            raise GateError("configured build manifest predates qrow32 source")
        initial = _regular(args.initial_dry_run, "initial Ninja dry run").read_text(
            encoding="utf-8", errors="strict"
        )
        if initial.count("Building CUDA object") != 1:
            raise GateError("fresh build must schedule exactly one CUDA object")
        if initial.count("Linking CXX shared library") != 1:
            raise GateError("fresh build must schedule exactly one shared-library relink")
        if qrow_source.stem not in initial:
            raise GateError("fresh build does not schedule the qrow32 CUDA object")
        route = "fresh_configure_discovered_object"
        evidence = {
            "build_manifest": str(manifest.resolve()),
            "initial_dry_run": str(args.initial_dry_run.resolve()),
        }
    else:
        compile_log = _regular(args.explicit_compile_log, "explicit compile log")
        compile_text = compile_log.read_text(encoding="utf-8", errors="strict")
        compile_tokens = shlex.split(compile_text)
        if sum(Path(token).name == qrow_source.name for token in compile_tokens) != 1:
            raise GateError("explicit compile must name qrow32 source exactly once")
        if (
            sum(Path(token).name == qrow_object.name for token in compile_tokens) != 1
            or "-c" not in compile_tokens
        ):
            raise GateError("explicit compile does not produce the qrow32 object")
        link_log = _regular(args.explicit_link_log, "explicit link log")
        link_text = link_log.read_text(encoding="utf-8", errors="strict")
        link_tokens = shlex.split(link_text)
        if sum(Path(token).name == qrow_object.name for token in link_tokens) != 1:
            raise GateError("explicit relink must append qrow32 object exactly once")
        if "-shared" not in link_text or "_vllm_fa2_C" not in link_text:
            raise GateError("explicit qrow32 relink is not the FA2 shared-library link")
        route = "explicit_object_compile_and_append"
        evidence = {
            "explicit_compile_log": str(compile_log.resolve()),
            "explicit_link_log": str(link_log.resolve()),
        }

    final_dry_run = _regular(args.final_dry_run, "final Ninja dry run")
    final_text = final_dry_run.read_text(encoding="utf-8", errors="strict")
    if "ninja: no work to do." not in final_text:
        raise GateError("final Ninja dry run is not clean")

    result = {
        "schema": "fr13.fixed32.fa2_qrow32_build_contract.v1",
        "status": "PASS",
        "fa2_commit": FA2_COMMIT,
        "required_patch_flags": [
            "--tree-bias-tile-earlyout",
            "--fixed32-query-tile32",
        ],
        "source_hashes": source_hashes,
        "qrow_source_sha256": _sha256(qrow_source),
        "stock_object_sha256": stock_object_sha256,
        "qrow_object_sha256": _sha256(qrow_object),
        "object_discovery_route": route,
        "discovery_evidence": evidence,
        "final_dry_run": str(final_dry_run.resolve()),
        "performance_measurement": False,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    return result


def verify_live(args: argparse.Namespace) -> dict[str, Any]:
    result = _load_json(args.result, "qrow32 live result")
    candidate_sha256 = _sha256(_regular(args.candidate_so, "candidate FA2 SO"))
    source_commit = args.source_commit
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise GateError("expected source commit is not a full lowercase SHA-1")
    expected = {
        "schema": "fr13.fixed32.fa2_qrow32_live_paged_exact4_ab.v1",
        "status": "PASS",
        "suite": "SWE-Verified",
        "task_ids": list(TASK_IDS),
        "subset_sha256": EXACT4_SUBSET_SHA256,
        "concurrency": 4,
        "batch_size": 4,
        "physical_rows_per_slot": 32,
        "total_query_rows": 128,
        "fixed32_mode": args.fixed32_mode,
        "candidate_so_sha256": candidate_sha256,
        "source_commit": source_commit,
        "runtime_mode": "FULL",
        "layer_count": 16,
        "target_layers": list(TARGET_LAYERS),
        "stock_calls": 16,
        "candidate_calls": 16,
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "served_return": "stock captured graph output unchanged",
        "fallback_allowed": False,
        "performance_measurement": False,
    }
    for key, value in expected.items():
        if not _json_exact(result.get(key), value):
            raise GateError(f"qrow32 live result {key} drifted")
    if args.fixed32_mode not in ("tail6_fixed32", "hydra27_fixed32"):
        raise GateError("qrow32 live mode is not Tail23 or Hydra27")
    operands = result.get("operands")
    if not isinstance(operands, dict) or (
        not _json_exact(operands.get("query_shape"), [128, 24, 256])
        or not _json_exact(
            operands.get("query_start_loc"), [0, 32, 64, 96, 128]
        )
        or not _json_exact(operands.get("slot_coverage"), [0, 1, 2, 3])
        or not _json_exact(
            operands.get("key_cache_tail_shape"), [1024, 4, 256]
        )
        or not isinstance(operands.get("seq_lens"), list)
        or len(operands["seq_lens"]) != 4
        or any(type(value) is not int for value in operands["seq_lens"])
        or not isinstance(operands.get("suffix_start_mod64"), list)
        or len(operands["suffix_start_mod64"]) != 4
        or any(type(value) is not int for value in operands["suffix_start_mod64"])
    ):
        raise GateError("qrow32 live operand coverage drifted")
    layers = result.get("layers")
    if not isinstance(layers, list) or len(layers) != 16:
        raise GateError("qrow32 live result does not contain 16 layers")
    for expected_layer, layer in zip(TARGET_LAYERS, layers, strict=True):
        if not isinstance(layer, dict) or layer.get("layer_name") != expected_layer:
            raise GateError("qrow32 live layer order or identity drifted")
        if not _json_exact(
            layer.get("output", {}).get("raw_byte_mismatches"), 0
        ):
            raise GateError(f"qrow32 output mismatch at {expected_layer}")
        if not _json_exact(layer.get("lse", {}).get("raw_byte_mismatches"), 0):
            raise GateError(f"qrow32 LSE mismatch at {expected_layer}")
        slots = layer.get("slots")
        if not isinstance(slots, list) or not _json_exact(
            [slot.get("slot") for slot in slots], [0, 1, 2, 3]
        ):
            raise GateError(f"qrow32 slot coverage drifted at {expected_layer}")
        if any(
            not _json_exact(slot.get(kind, {}).get("raw_byte_mismatches"), 0)
            for slot in slots
            for kind in ("output", "lse")
        ):
            raise GateError(f"qrow32 per-slot byte mismatch at {expected_layer}")

    campaign_arm = _load_json(args.campaign_arm, "exact4 campaign arm")
    expected_marker = f"swe_verified:campaign4_{EXACT4_SUBSET_SHA256}"
    if (
        campaign_arm.get("schema") != "fr13-fixed32-taw-campaign-arm-v1"
        or campaign_arm.get("state") != "ended"
        or campaign_arm.get("run_classification") != "b4_taw_diagnostic"
        or not _json_exact(campaign_arm.get("batch_size"), 4)
        or not _json_exact(campaign_arm.get("concurrency"), 4)
        or not _json_exact(campaign_arm.get("task_count"), 4)
        or campaign_arm.get("subset_sha256") != EXACT4_SUBSET_SHA256
        or not _json_exact(campaign_arm.get("task_ids"), list(TASK_IDS))
        or campaign_arm.get("marker") != expected_marker
    ):
        raise GateError("qrow32 gate is not bound to the canonical exact4 arm")

    campaign = _load_json(args.campaign_provenance, "exact4 campaign provenance")
    if (
        campaign.get("schema") != "fr13-fixed32-qwen-campaign-provenance-v1"
        or campaign.get("metric_scope") != "concurrent_campaign_union"
        or not _json_exact(campaign.get("concurrency"), 4)
        or not _json_exact(campaign.get("task_ids"), list(TASK_IDS))
    ):
        raise GateError("qrow32 gate is not bound to canonical real exact4 provenance")
    return {
        "schema": "fr13.fixed32.fa2_qrow32_live_gate_verification.v1",
        "status": "PASS",
        "fixed32_mode": args.fixed32_mode,
        "candidate_so_sha256": candidate_sha256,
        "source_commit": source_commit,
        "task_ids": list(TASK_IDS),
        "layer_count": 16,
        "slot_coverage": [0, 1, 2, 3],
        "performance_measurement": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("verify-build")
    build.add_argument("--fa2-src", type=Path, required=True)
    build.add_argument("--stock-object", type=Path, required=True)
    build.add_argument("--qrow-object", type=Path, required=True)
    route = build.add_mutually_exclusive_group(required=True)
    route.add_argument("--build-manifest", type=Path)
    route.add_argument("--explicit-compile-log", type=Path)
    build.add_argument("--initial-dry-run", type=Path)
    build.add_argument("--explicit-link-log", type=Path)
    build.add_argument("--final-dry-run", type=Path, required=True)
    build.add_argument("--output", type=Path)

    live = subparsers.add_parser("verify-live")
    live.add_argument("--result", type=Path, required=True)
    live.add_argument("--campaign-arm", type=Path, required=True)
    live.add_argument("--campaign-provenance", type=Path, required=True)
    live.add_argument("--candidate-so", type=Path, required=True)
    live.add_argument(
        "--fixed32-mode",
        choices=("tail6_fixed32", "hydra27_fixed32"),
        required=True,
    )
    live.add_argument("--source-commit", required=True)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.command == "verify-build":
        if args.build_manifest is not None and args.initial_dry_run is None:
            parser.error("fresh-configure route requires --initial-dry-run")
        if args.explicit_compile_log is not None and args.explicit_link_log is None:
            parser.error("explicit-object route requires --explicit-link-log")
        result = verify_build(args)
    else:
        result = verify_live(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
