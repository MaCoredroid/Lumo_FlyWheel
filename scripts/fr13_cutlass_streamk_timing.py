#!/usr/bin/env python3
"""Reduce a real B1 stock/Stream-K full-wall timing pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

import fr13_cutlass_streamk_pass as qualification
import fr13_cutlass_wave_binary as binary
import fr13_qrow16_pass_sidecar as qrow


SCHEMA = "fr13.fixed32.cutlass_streamk.b1_full_wall_timing_pair.v5"
MEASURE_SCHEMA = "fr13.measure.deploy_speed.v1"
EXPECTED_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
EXPECTED_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
ONE_TASK_SUBSET_SHA256 = (
    "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
)
ONE_TASK_IDS = (EXPECTED_TASK_IDS[0],)
TASK_SET_CONTRACTS = {
    "exact4": {
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "task_ids": EXPECTED_TASK_IDS,
        "run_classification": (
            "real_swe_verified_exact4_b1_hydra27_qrow16_streamk_timing"
        ),
        "timing_eligible": True,
        "timing_claim_source": "paired exact4 real SWE-Verified full-wall arms",
    },
    "one": {
        "subset_sha256": ONE_TASK_SUBSET_SHA256,
        "task_ids": ONE_TASK_IDS,
        "run_classification": (
            "one_real_swe_verified_b1_hydra27_qrow16_streamk_timing_diagnostic"
        ),
        "timing_eligible": False,
        "timing_claim_source": (
            "paired one-task real SWE-Verified full-wall diagnostic arms"
        ),
    },
}
COMMON_EXPECTED_ENV = (
    "FR13_FA2_QROW16_PRODUCTION=1",
    "FR13_FA2_QROW16_SO_SHA256="
    "1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86",
)
EXPECTED_ENV = (
    "FR13_DRAFT_VOCAB_ROOT=0",
    "FR13_DRAFT_VOCAB_K=0",
    *COMMON_EXPECTED_ENV,
)
QROW16_SHA256 = "1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86"
QROW16_BYTES = 299_507_792
QROW16_LIVE_RESULT_SHA256 = (
    "36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77"
)


class TimingError(ValueError):
    """The timing pair is incomplete or has mismatched provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise TimingError(f"{label} does not exist") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise TimingError(f"{label} is not a regular non-symlink file")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise TimingError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> None:
    raise TimingError(f"non-finite JSON value: {value}")


def _load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular(path, label)
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimingError(f"{label} is not canonical ASCII JSON") from error
    if not isinstance(payload, dict):
        raise TimingError(f"{label} must contain a JSON object")
    return payload, raw


def _positive(record: dict[str, Any], key: str, label: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimingError(f"{label} lacks numeric {key}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise TimingError(f"{label} {key} is not finite and positive")
    return value


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise TimingError(f"{label} mismatch: {actual!r} != {expected!r}")


def _validate_measure(
    record: dict[str, Any],
    label: str,
    task_ids: tuple[str, ...],
    profile: dict[str, object],
) -> dict[str, float]:
    required = {
        "schema": MEASURE_SCHEMA,
        "regime": "deployment",
        "instrument": "OFF",
        "batch_size": 1,
        "n_tasks": len(task_ids),
        "draft_vocab_k": profile["draft_vocab_k"],
        "draft_vocab_root": profile["draft_vocab_root"],
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "floor_is_full_step_hardware_floor": False,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise TimingError(
                f"{label} {key} mismatch: {record.get(key)!r} != {expected!r}"
            )
    if sorted(record.get("task_instance_ids", [])) != sorted(task_ids):
        raise TimingError(f"{label} is not bound to the selected timing task set")
    numeric = {}
    for key in (
        "measured_tps_fullstep_wall",
        "step_wall_ms",
        "accept_per_event",
        "committed_per_event",
        "wall_steps_measured",
        "events_per_step",
        "s_per_fwd_gpu",
        "drafter_gpu_ms_per_step",
        "committer_gpu_ms_per_step",
        "weight_floor_ms",
        "floor_ms",
        "floor_ratio",
    ):
        numeric[key] = _positive(record, key, label)
    _close(
        numeric["weight_floor_ms"],
        float(profile["mandatory_weight_floor_ms"]),
        f"{label} mandatory-weight floor",
    )
    _close(
        numeric["floor_ms"],
        float(profile["mandatory_weight_floor_ms"]),
        f"{label} active floor",
    )
    _close(
        numeric["floor_ratio"],
        numeric["step_wall_ms"] / numeric["floor_ms"],
        f"{label} floor ratio",
    )
    return numeric


def _validate_container_env(
    path: Path,
    label: str,
    *,
    cutlass_selector: str,
    cutlass_production: int,
    qualification_profile: str,
    diagnostic_task_profile: str,
    profile: dict[str, object],
) -> str:
    _regular(path, label)
    raw = path.read_bytes()
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise TimingError(f"{label} is not ASCII") from error
    expected_lines = (
        f"FR13_DRAFT_VOCAB_ROOT={profile['draft_vocab_root']}",
        f"FR13_DRAFT_VOCAB_K={profile['draft_vocab_k']}",
        *COMMON_EXPECTED_ENV,
        "FR13_FIXED32_MODE=hydra27_fixed32",
        "FR13_FA2_QROW16_LIVE_PAGED_AB=0",
        f"FR13_FIXED32_CUTLASS_WAVE={cutlass_selector}",
        f"FR13_FIXED32_CUTLASS_WAVE_PRODUCTION={cutlass_production}",
        (
            "FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE="
            f"{qualification_profile}"
        ),
    )
    if profile["requires_block_map"]:
        expected_lines = (
            *expected_lines,
            "FR13_DRAFT_VOCAB_BLOCKS="
            f"{qualification.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH}",
        )
    for expected in expected_lines:
        if lines.count(expected) != 1:
            raise TimingError(f"{label} lacks exact timing pin {expected}")
    environment_prefixes = (
        "FR13_DRAFT_VOCAB_ROOT=",
        "FR13_DRAFT_VOCAB_K=",
        "FR13_FA2_QROW16_PRODUCTION=",
        "FR13_FA2_QROW16_SO_SHA256=",
        "FR13_FA2_QROW16_LIVE_PAGED_AB=",
        "FR13_FIXED32_MODE=",
        "FR13_FIXED32_CUTLASS_WAVE=",
        "FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=",
        "FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE=",
    )
    if profile["requires_block_map"]:
        environment_prefixes += ("FR13_DRAFT_VOCAB_BLOCKS=",)
    for prefix in environment_prefixes:
        matches = [line for line in lines if line.startswith(prefix)]
        if len(matches) != 1:
            raise TimingError(f"{label} has ambiguous {prefix[:-1]}")
    diagnostic_prefix = "FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE="
    diagnostic_matches = [
        line for line in lines if line.startswith(diagnostic_prefix)
    ]
    expected_diagnostic = f"{diagnostic_prefix}{diagnostic_task_profile}"
    if diagnostic_matches == [expected_diagnostic]:
        pass
    elif (
        not diagnostic_matches
        and diagnostic_task_profile
        == qualification.DEFAULT_DIAGNOSTIC_TASK_PROFILE
    ):
        pass
    else:
        raise TimingError(
            f"{label} lacks exact timing pin {expected_diagnostic}"
        )
    return hashlib.sha256(raw).hexdigest()


def _validate_qrow16_engagement(
    sidecar_path: Path,
    capture_path: Path,
    qrow16_so: Path,
    label: str,
) -> dict[str, Any]:
    _regular(sidecar_path, f"{label} Qrow16 production sidecar")
    sidecar_sha256 = _sha256(sidecar_path)
    try:
        sidecar = qrow.verify_sidecar(
            sidecar_path=sidecar_path,
            expected_sidecar_sha256=sidecar_sha256,
            candidate_so=qrow16_so,
            expected_candidate_sha256=QROW16_SHA256,
        )
    except (OSError, ValueError) as error:
        raise TimingError(
            f"{label} Qrow16 sidecar validation failed: {error}"
        ) from error
    if sidecar.get("live_result_sha256") != QROW16_LIVE_RESULT_SHA256:
        raise TimingError(
            f"{label} Qrow16 sidecar is not bound to the pinned live PASS"
        )
    capture, capture_raw = _load(capture_path, f"{label} Qrow16 capture")
    required = {
        "schema": "fr13.fixed32.fa2_qrow16_production_capture.v1",
        "status": "ENGAGED",
        "runtime_mode": "FULL",
        "batch_size": 1,
        "layer_count": 16,
        "candidate_so_sha256": QROW16_SHA256,
        "pass_sidecar_sha256": sidecar_sha256,
        "dispatch": "qrow16 exact geometry; no fallback",
    }
    if set(capture) != {*required, "graph_id", "graph_signature", "layers"}:
        raise TimingError(f"{label} Qrow16 capture key set drifted")
    for key, expected in required.items():
        if capture.get(key) != expected:
            raise TimingError(
                f"{label} Qrow16 capture {key} mismatch: "
                f"{capture.get(key)!r} != {expected!r}"
            )
    graph_id = capture.get("graph_id")
    graph_signature = capture.get("graph_signature")
    layers = capture.get("layers")
    if (
        isinstance(graph_id, bool)
        or not isinstance(graph_id, int)
        or graph_id <= 0
        or not isinstance(graph_signature, str)
        or re.fullmatch(r"[0-9a-f]{64}", graph_signature) is None
        or not isinstance(layers, list)
        or len(layers) != 16
        or any(not isinstance(layer, str) or not layer for layer in layers)
        or len(set(layers)) != 16
    ):
        raise TimingError(f"{label} Qrow16 capture graph/layer identity drifted")
    return {
        "candidate_so_sha256": QROW16_SHA256,
        "candidate_so_bytes": QROW16_BYTES,
        "live_result_sha256": QROW16_LIVE_RESULT_SHA256,
        "production_sidecar_sha256": sidecar_sha256,
        "production_capture_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "graph_signature": graph_signature,
        "layer_count": 16,
        "dispatch": required["dispatch"],
    }


def reduce_pair(
    subset: Path,
    stock_measure: Path,
    candidate_measure: Path,
    stock_container_env: Path,
    candidate_container_env: Path,
    stock_qrow16_sidecar: Path,
    candidate_qrow16_sidecar: Path,
    stock_qrow16_capture: Path,
    candidate_qrow16_capture: Path,
    qrow16_so: Path,
    production_binding: Path,
    candidate_so: Path,
    source_commit: str,
    *,
    candidate_selector: str = "streamk_coop128",
    qualification_profile: str = "full_vocab",
    diagnostic_task_profile: str = qualification.DEFAULT_DIAGNOSTIC_TASK_PROFILE,
    task_set: str = "exact4",
) -> dict[str, Any]:
    try:
        task_contract = TASK_SET_CONTRACTS[task_set]
    except KeyError as error:
        raise TimingError(f"unsupported timing task set: {task_set!r}") from error
    if candidate_selector not in qualification.CANDIDATE_CONTRACTS:
        raise TimingError(
            f"unsupported Stream-K timing candidate: {candidate_selector!r}"
        )
    try:
        profile = qualification._qualification_profile(
            candidate_selector, qualification_profile
        )
        diagnostic_profile = qualification._diagnostic_task_profile(
            candidate_selector, diagnostic_task_profile
        )
    except qualification.QualificationError as error:
        raise TimingError(str(error)) from error
    task_ids = tuple(task_contract["task_ids"])
    candidate_sha256, candidate_bytes, candidate_family = binary.candidate_identity(
        candidate_selector
    )
    diagnostic_selector = qualification.CANDIDATE_CONTRACTS[candidate_selector][
        "diagnostic_selector"
    ]
    source_contract = qualification._source_contract(candidate_selector)
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise TimingError("timing source commit is invalid")
    _regular(subset, "timing subset")
    if _sha256(subset) != task_contract["subset_sha256"]:
        raise TimingError("canonical timing subset SHA-256 drift")
    subset_payload, _ = _load(subset, "timing subset")
    if sorted(subset_payload.get("instance_ids", [])) != sorted(task_ids):
        raise TimingError("canonical timing subset task IDs drift")
    stock, _ = _load(stock_measure, "stock full-wall measurement")
    candidate, _ = _load(candidate_measure, "candidate full-wall measurement")
    binding, binding_raw = _load(production_binding, "production binding")
    stock_values = _validate_measure(stock, "stock", task_ids, profile)
    candidate_values = _validate_measure(candidate, "candidate", task_ids, profile)
    stock_env_sha256 = _validate_container_env(
        stock_container_env,
        "stock container environment",
        cutlass_selector="stock",
        cutlass_production=0,
        qualification_profile=qualification_profile,
        diagnostic_task_profile=diagnostic_task_profile,
        profile=profile,
    )
    candidate_env_sha256 = _validate_container_env(
        candidate_container_env,
        "candidate container environment",
        cutlass_selector=candidate_selector,
        cutlass_production=1,
        qualification_profile=qualification_profile,
        diagnostic_task_profile=diagnostic_task_profile,
        profile=profile,
    )
    _regular(qrow16_so, "Qrow16 candidate SO")
    qrow16_info = qrow16_so.lstat()
    if (
        qrow16_so.is_symlink()
        or not stat.S_ISREG(qrow16_info.st_mode)
        or qrow16_info.st_size != QROW16_BYTES
        or _sha256(qrow16_so) != QROW16_SHA256
    ):
        raise TimingError("Qrow16 candidate SO identity drifted")
    stock_qrow16 = _validate_qrow16_engagement(
        stock_qrow16_sidecar,
        stock_qrow16_capture,
        qrow16_so,
        "stock",
    )
    candidate_qrow16 = _validate_qrow16_engagement(
        candidate_qrow16_sidecar,
        candidate_qrow16_capture,
        qrow16_so,
        "candidate",
    )
    for key in (
        "candidate_so_sha256",
        "candidate_so_bytes",
        "live_result_sha256",
        "production_sidecar_sha256",
        "graph_signature",
        "layer_count",
        "dispatch",
    ):
        if stock_qrow16[key] != candidate_qrow16[key]:
            raise TimingError(f"Qrow16 {key} differs across timing arms")
    expected_binding = {
        "schema": profile["binding_schema"],
        "status": "BOUND",
        "selector": candidate_selector,
        "diagnostic_selector": diagnostic_selector,
        "candidate_family": candidate_family,
        "candidate_sha256": candidate_sha256,
        "candidate_bytes": candidate_bytes,
        "patch_source_sha256": source_contract["patch_source_sha256"],
        "qualification_source_commit": source_commit,
        "qualification_task_marker": diagnostic_profile["task_marker"],
        "qualified_draft_vocab_root": profile["draft_vocab_root"],
        "qualified_draft_vocab_k": profile["draft_vocab_k"],
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "production_default_enabled": False,
    }
    if qualification_profile == "k64_root":
        expected_binding.update(
            {
                "qualification_profile": qualification_profile,
                "qualified_draft_vocab_blocks": (
                    qualification.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH
                ),
                "qualified_draft_vocab_blocks_sha256": (
                    qualification.DRAFT_VOCAB_BLOCKS_SHA256
                ),
                "qualified_comparison_call_limit": qualification.MAX_COMPARISONS,
            }
        )
    if qualification.CANDIDATE_CONTRACTS[candidate_selector].get(
        "source_binding"
    ) == "required":
        try:
            if candidate_selector == "identity_onen_b1":
                source_identity = qualification.validate_source_commit_binding(
                    source_commit
                )
            else:
                source_identity = qualification.validate_source_commit_binding(
                    source_commit, candidate_selector=candidate_selector
                )
            expected_binding["qualification_source_identity"] = source_identity
        except qualification.QualificationError as error:
            raise TimingError(str(error)) from error
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise TimingError(
                f"production binding {key} mismatch: "
                f"{binding.get(key)!r} != {expected!r}"
            )
    for key, expected in (
        ("qualification_task_profile", diagnostic_task_profile),
        ("qualification_task_ids", list(diagnostic_profile["task_ids"])),
    ):
        actual = binding.get(key)
        if actual == expected:
            continue
        if (
            actual is None
            and diagnostic_task_profile
            == qualification.DEFAULT_DIAGNOSTIC_TASK_PROFILE
        ):
            continue
        raise TimingError(
            f"production binding {key} mismatch: {actual!r} != {expected!r}"
        )
    for key in (
        "production_sidecar_sha256",
        "live_result_sha256",
        "binary_attestation_sha256",
        "real_task_arm_sha256",
        "container_env_sha256",
    ):
        value = binding.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise TimingError(f"production binding {key} is not SHA-256")
    actual_candidate = binary.verify_candidate(
        candidate_so,
        candidate_selector,
        qualification_profile=qualification_profile,
    )
    if actual_candidate["sha256"] != binding["candidate_sha256"]:
        raise TimingError("production binding and candidate binary disagree")

    stock_wall = stock_values["step_wall_ms"]
    candidate_wall = candidate_values["step_wall_ms"]
    stock_tps = stock_values["measured_tps_fullstep_wall"]
    candidate_tps = candidate_values["measured_tps_fullstep_wall"]

    def arm(
        selector: str, values: dict[str, float], container_env_sha256: str
    ) -> dict[str, Any]:
        return {
            "cutlass_selector": selector,
            "step_wall_ms": values["step_wall_ms"],
            "measured_tps_fullstep_wall": values["measured_tps_fullstep_wall"],
            "accepted_drafts_per_event": values["accept_per_event"],
            "committed_tokens_per_event": values["committed_per_event"],
            "s_fwd_gpu_ms_per_step": values["s_per_fwd_gpu"] * 1000.0,
            "drafter_gpu_ms_per_step": values["drafter_gpu_ms_per_step"],
            "committer_gpu_ms_per_step": values["committer_gpu_ms_per_step"],
            "step_wall_to_mandatory_weight_floor_ratio": values["floor_ratio"],
            "wall_steps_measured": values["wall_steps_measured"],
            "events_per_step": values["events_per_step"],
            "container_env_sha256": container_env_sha256,
        }

    return {
        "schema": SCHEMA,
        "status": "complete",
        "run_classification": task_contract["run_classification"],
        "topology": "hydra27_fixed32",
        "lineage": "successor_to_legacy_hydra23_not_same_topology",
        "task_set": task_set,
        "task_count": len(task_ids),
        "batch_size": 1,
        "concurrency": 1,
        "task_ids": sorted(task_ids),
        "source_commit": source_commit,
        "decision_metric": "measured_tps_fullstep_wall",
        "qualification_profile": qualification_profile,
        "qualification_task_profile": diagnostic_task_profile,
        "draft_vocab_root": profile["draft_vocab_root"],
        "draft_vocab_k": profile["draft_vocab_k"],
        "common_kernel_stack": {
            "fa2_selector": "qrow16 production",
            "stock_arm": stock_qrow16,
            "candidate_arm": candidate_qrow16,
            "identical_in_both_arms": True,
        },
        "only_arm_delta": f"CUTLASS stock to {candidate_selector}",
        "stock_reference": arm("stock", stock_values, stock_env_sha256),
        "candidate": {
            **arm(candidate_selector, candidate_values, candidate_env_sha256),
            "candidate_family": candidate_family,
            "diagnostic_selector": diagnostic_selector,
            "candidate_sha256": candidate_sha256,
            "candidate_bytes": candidate_bytes,
            "patch_source_sha256": source_contract["patch_source_sha256"],
            "production_binding_sha256": hashlib.sha256(binding_raw).hexdigest(),
            "live_result_sha256": binding["live_result_sha256"],
            "production_sidecar_sha256": binding["production_sidecar_sha256"],
            "real_task_arm_sha256": binding["real_task_arm_sha256"],
        },
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "mandatory_weight_floor_is_complete_step_floor": False,
        "candidate_to_stock_full_wall_tps_ratio": candidate_tps / stock_tps,
        "stock_to_candidate_step_wall_ratio": stock_wall / candidate_wall,
        "candidate_step_wall_delta_ms": candidate_wall - stock_wall,
        "comparator_gate_timing_eligible": False,
        "timing_eligible": task_contract["timing_eligible"],
        "timing_claim_source": task_contract["timing_claim_source"],
        "floor_acceptance_eligible": False,
        "formal_floor_acceptance_eligible": False,
        "formal_floor_acceptance_reason": (
            "Hydra27 same-topology kernel timing candidate only; the canonical "
            "Tail6/Hydra27 one-sided U95 floor gate was not run"
        ),
        "production_default_enabled": False,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--stock-measure", type=Path, required=True)
    parser.add_argument("--candidate-measure", type=Path, required=True)
    parser.add_argument("--stock-container-env", type=Path, required=True)
    parser.add_argument("--candidate-container-env", type=Path, required=True)
    parser.add_argument("--stock-qrow16-sidecar", type=Path, required=True)
    parser.add_argument("--candidate-qrow16-sidecar", type=Path, required=True)
    parser.add_argument("--stock-qrow16-capture", type=Path, required=True)
    parser.add_argument("--candidate-qrow16-capture", type=Path, required=True)
    parser.add_argument("--qrow16-so", type=Path, required=True)
    parser.add_argument("--production-binding", type=Path, required=True)
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--candidate-selector",
        choices=tuple(qualification.CANDIDATE_CONTRACTS),
        default="streamk_coop128",
    )
    parser.add_argument(
        "--qualification-profile",
        choices=tuple(qualification.QUALIFICATION_PROFILES),
        default="full_vocab",
    )
    parser.add_argument(
        "--diagnostic-task-profile",
        choices=tuple(qualification.DIAGNOSTIC_TASK_PROFILES),
        default=qualification.DEFAULT_DIAGNOSTIC_TASK_PROFILE,
    )
    parser.add_argument(
        "--task-set", choices=tuple(TASK_SET_CONTRACTS), default="exact4"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = reduce_pair(
        args.subset,
        args.stock_measure,
        args.candidate_measure,
        args.stock_container_env,
        args.candidate_container_env,
        args.stock_qrow16_sidecar,
        args.candidate_qrow16_sidecar,
        args.stock_qrow16_capture,
        args.candidate_qrow16_capture,
        args.qrow16_so,
        args.production_binding,
        args.candidate_so,
        args.source_commit,
        candidate_selector=args.candidate_selector,
        qualification_profile=args.qualification_profile,
        diagnostic_task_profile=args.diagnostic_task_profile,
        task_set=args.task_set,
    )
    _write(args.out, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
