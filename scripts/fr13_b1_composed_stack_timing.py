#!/usr/bin/env python3
"""Reduce final-HEAD candidate-only exact4 timing for the composed B1 stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
from pathlib import Path
from typing import Any

import fr13_qrow32_split2_timing as qrow_timing


SCHEMA = "fr13.fixed32.b1_composed_stack.exact4_timing.v1"
TARGET_SELECTOR = "identity_wide256_fullgrid_b1"
TARGET_SHA256 = "85937b5c35ec87bce12e4b5d677dd67f63004f9a9d9fb6d64473a5bd3b53b2da"
DFWD_MARKERS = (
    "[FR13_DFWD_K64_TOP3] ready B1 K64 mapped width3",
    "[FR13_DFWD_K64_TOP3] engaged stock_argmax_topk_map_copy=0",
    "[FR13_DFWD_K64_TOP3] graph captured_calls=4",
)
SFWD_MARKER = "[FR13_SFWD_CONV_POSTPREP] production engaged layer="


def _regular(path: Path) -> bytes:
    info = path.lstat()
    raw = path.read_bytes()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or not raw:
        raise ValueError(f"composed evidence is not a regular nonempty file: {path}")
    return raw


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path)
    payload = json.loads(raw.decode("ascii"))
    if not isinstance(payload, dict):
        raise ValueError(f"composed evidence is not a JSON object: {path}")
    return payload, raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _number(payload: dict[str, Any], key: str, *, positive: bool = False) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"measure {key} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"measure {key} is not finite and valid")
    return result


def _expect_sha(raw: bytes, expected: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None or _sha(raw) != expected:
        raise ValueError(f"{label} SHA-256 drifted")
    return expected


def reduce_composed(args: argparse.Namespace) -> dict[str, Any]:
    base = qrow_timing.reduce_timing(
        subset_path=args.subset,
        measure_path=args.measure,
        baseline_path=args.baseline,
        engagement_path=args.engagement,
        health_path=args.health,
        traffic_audit_path=args.traffic_audit,
        source_commit=args.source_commit,
        patch_source_sha256=args.patch_source_sha256,
        pass_sha256=args.pass_sha256,
        pass_sidecar_sha256=args.pass_sidecar_sha256,
        runner_sha256=args.runner_sha256,
        block_map_sha256=args.block_map_sha256,
        floor_ms=args.floor_ms,
        cap_ms=args.cap_ms,
        arm=args.arm,
    )
    measure, measure_raw = _load(args.measure)
    container_raw = _regular(args.container_env)
    docker_raw = _regular(args.docker_log)
    gqa_raw = _regular(args.gqa3_production_credential)
    gqa_arm_raw = _regular(args.gqa3_production_arm)
    gqa_batch_raw = _regular(args.gqa3_production_batch)
    target_sidecar_raw = _regular(args.target_production_sidecar)
    target_binary, target_binary_raw = _load(args.target_binary_record)
    sfwd_pass_raw = _regular(args.sfwd_production_pass)
    sfwd_manifest_raw = _regular(args.sfwd_production_manifest)
    qrow_composed_raw = _regular(args.qrow_composed_credential)
    dfwd_credential_raw = _regular(args.dfwd_credential)
    eager_summary_raw = _regular(args.target_sfwd_combined_summary)
    if gqa_arm_raw != b"1\n" or gqa_batch_raw != b"1\n":
        raise ValueError("GQA3 production arm or batch marker drifted")
    evidence = {
        "gqa3_production_credential_sha256": _expect_sha(
            gqa_raw, args.gqa3_pass_sha256, "GQA3 production credential"
        ),
        "target_production_sidecar_sha256": _expect_sha(
            target_sidecar_raw,
            args.target_production_sidecar_sha256,
            "target production sidecar",
        ),
        "sfwd_production_pass_sha256": _expect_sha(
            sfwd_pass_raw, args.sfwd_pass_sha256, "SFWD production PASS"
        ),
        "sfwd_production_manifest_sha256": _expect_sha(
            sfwd_manifest_raw,
            args.sfwd_manifest_sha256,
            "SFWD production manifest",
        ),
        "qrow_composed_credential_sha256": _expect_sha(
            qrow_composed_raw,
            args.qrow_composed_credential_sha256,
            "Qrow composed credential",
        ),
        "dfwd_credential_sha256": _expect_sha(
            dfwd_credential_raw,
            args.dfwd_credential_sha256,
            "DFWD top3 credential",
        ),
        "target_sfwd_combined_summary_sha256": _expect_sha(
            eager_summary_raw,
            args.target_sfwd_combined_summary_sha256,
            "target/SFWD combined summary",
        ),
    }
    if (
        target_binary.get("schema") != "fr13.fixed32.cutlass_streamk_binary.v2"
        or target_binary.get("selector") != TARGET_SELECTOR
        or target_binary.get("production_enabled") is not True
        or target_binary.get("qualification_profile") != "k64_root"
        or not isinstance(target_binary.get("source"), dict)
        or target_binary["source"].get("sha256") != TARGET_SHA256
        or not isinstance(target_binary.get("destination"), dict)
        or target_binary["destination"].get("sha256") != TARGET_SHA256
        or target_binary.get("installed_mode") != "0555"
    ):
        raise ValueError("target production binary evidence drifted")
    container_lines = container_raw.decode("ascii").splitlines()
    required_env = (
        "FR13_FIXED32_MODE=hydra27_fixed32",
        "FR13_DRAFT_VOCAB_ROOT=1",
        "FR13_DRAFT_VOCAB_K=65536",
        "MAX_NUM_SEQS=1",
        "SWE_CONCURRENCY=1",
        "ENFORCE_EAGER=0",
        "CUDAGRAPH_MODE=FULL_AND_PIECEWISE",
        "FR10_METRICS=1",
        "FR13_FA2_QROW32_B1_PRODUCTION_ARM=split2",
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=1",
        "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH=1",
        "FR13_DFWD_K64_TOP3=1",
        f"FR13_FIXED32_CUTLASS_WAVE={TARGET_SELECTOR}",
        "FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=1",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=1",
        "FR13_CONV_WB_BATCHED=1",
        "FR13_SFWD_GPU_TIMER=1",
        "FR13_DFWD_GPU_TIMER=1",
        "FR13_CFWD_GPU_TIMER=1",
    )
    missing = [line for line in required_env if container_lines.count(line) != 1]
    if missing:
        raise ValueError(f"composed container environment drifted: {missing!r}")
    docker_text = docker_raw.decode("utf-8", errors="replace")
    if any(marker not in docker_text for marker in DFWD_MARKERS):
        raise ValueError("DFWD top3 ready/engaged/captured evidence is incomplete")
    sfwd_layers = set(
        re.findall(
            r"\[FR13_SFWD_CONV_POSTPREP\] production engaged "
            r"layer=([^ ]+) B=1 rows=32",
            docker_text,
        )
    )
    if len(sfwd_layers) != 48 or docker_text.count(SFWD_MARKER) != 48:
        raise ValueError("SFWD conv/post-prep did not engage exactly 48 layers")
    step_wall_ms = _number(measure, "step_wall_ms", positive=True)
    wall_tps = _number(measure, "measured_tps_fullstep_wall", positive=True)
    sfwd_ms = 1000.0 * _number(measure, "s_per_fwd_gpu", positive=True)
    dfwd_ms = _number(measure, "drafter_gpu_ms_per_step", positive=True)
    cfwd_ms = _number(measure, "committer_gpu_ms_per_step", positive=True)
    other_ms = _number(measure, "overhead_other_ms_per_event")
    accept = _number(measure, "accept_per_event", positive=True)
    committed = _number(measure, "committed_per_event", positive=True)
    floor_ratio = _number(measure, "floor_ratio", positive=True)
    gpu_tps = _number(measure, "derived_tps_fullstep_gpu", positive=True)
    if not math.isclose(step_wall_ms / args.floor_ms, floor_ratio, rel_tol=1e-9):
        raise ValueError("measure floor ratio is inconsistent")
    u95 = base["descriptive_equal_task_one_sided_u95"]
    return {
        **base,
        "schema": SCHEMA,
        "run_classification": "real_swe_verified_exact4_b1_composed_kernel_stack",
        "composed_stack": {
            "qrow32_split2": True,
            "gdn_gqa_group3": True,
            "dfwd_k64_top3": True,
            "target_gemm_selector": TARGET_SELECTOR,
            "sfwd_conv_postprep": True,
        },
        "phase_breakdown_ms_per_event": {
            "sfwd_verify_gpu": sfwd_ms,
            "dfwd_drafter_gpu": dfwd_ms,
            "cfwd_committer_gpu": cfwd_ms,
            "host_and_unattributed": other_ms,
            "wall_full_step": step_wall_ms,
        },
        "full_step_tps": {
            "wall": wall_tps,
            "gpu_components": gpu_tps,
        },
        "acceptance": {
            "accepted_drafts_per_event": accept,
            "committed_tokens_per_event": committed,
            "weight_read_floor_ms": args.floor_ms,
            "one_sided_u95_cap_ms": args.cap_ms,
            "equal_task_mean_ms": u95["mean_ms"],
            "equal_task_u95_ms": u95["u95_ms"],
            "mean_floor_ratio": u95["mean_ms"] / args.floor_ms,
            "u95_floor_ratio": u95["u95_ms"] / args.floor_ms,
            "mean_gap_to_floor_ms": u95["mean_ms"] - args.floor_ms,
            "u95_gap_to_cap_ms": u95["u95_ms"] - args.cap_ms,
            "descriptive_screen_pass": u95["u95_ms"] <= args.cap_ms,
        },
        "production_evidence": {
            **evidence,
            "qrow_timing_core_sha256": _sha(
                (
                    json.dumps(
                        base, ensure_ascii=True, separators=(",", ":"), sort_keys=True
                    )
                    + "\n"
                ).encode("ascii")
            ),
            "measure_sha256": _sha(measure_raw),
            "container_env_sha256": _sha(container_raw),
            "docker_log_sha256": _sha(docker_raw),
            "target_binary_record_sha256": _sha(target_binary_raw),
            "gqa3_production_arm_sha256": _sha(gqa_arm_raw),
            "gqa3_production_batch_sha256": _sha(gqa_batch_raw),
            "sfwd_engaged_layer_count": len(sfwd_layers),
            "dfwd_ready_engaged_captured": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "subset",
        "measure",
        "baseline",
        "engagement",
        "health",
        "traffic-audit",
        "container-env",
        "docker-log",
        "gqa3-production-credential",
        "gqa3-production-arm",
        "gqa3-production-batch",
        "target-production-sidecar",
        "target-binary-record",
        "sfwd-production-pass",
        "sfwd-production-manifest",
        "qrow-composed-credential",
        "dfwd-credential",
        "target-sfwd-combined-summary",
    ):
        result.add_argument(f"--{name}", type=Path, required=True)
    for name in (
        "source-commit",
        "patch-source-sha256",
        "pass-sha256",
        "pass-sidecar-sha256",
        "runner-sha256",
        "block-map-sha256",
        "arm",
        "gqa3-pass-sha256",
        "target-production-sidecar-sha256",
        "sfwd-pass-sha256",
        "sfwd-manifest-sha256",
        "qrow-composed-credential-sha256",
        "dfwd-credential-sha256",
        "target-sfwd-combined-summary-sha256",
    ):
        result.add_argument(f"--{name}", required=True)
    result.add_argument("--floor-ms", type=float, required=True)
    result.add_argument("--cap-ms", type=float, required=True)
    result.add_argument("--out", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    output = reduce_composed(args)
    args.out.write_text(
        json.dumps(output, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(output, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
