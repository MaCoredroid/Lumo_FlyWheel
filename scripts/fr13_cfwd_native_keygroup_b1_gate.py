#!/usr/bin/env python3
"""Validate one resolved real-SWE K64/root1 B1 native key-group CFWD gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lumo_flywheel_serving import (  # noqa: E402
    fr13_cfwd_native_fullvalue_cuda as candidate,
)

import fr13_cfwd_native_keygroup_binary as binary_gate  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402


SCHEMA = "fr13.fixed32.cfwd_native_keygroup_k64_root_b1_gate.v1"
TASK_ID = "astropy__astropy-12907"
SUBSET_SHA256 = "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
BLOCK_MAP_SHA256 = "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
ARM_MARKER = f"swe_verified:{TASK_ID}"
ARM_MARKER_SHA256 = (
    "04fe7f61a0e0bbd48bf28127385c481b85550b291535f3705511494ba24c8463"
)
FULL_MASK = 0x0FFF
EXPECTED_LENGTHS = list(range(12))
ROUTE = "native_keygroup_precompute_cuda"
REQUIRED_MANIFEST_PATHS = {
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr13_cfwd_native_keygroup_b1_gate.py",
    "scripts/fr13_cfwd_native_keygroup_binary.py",
    "scripts/fr13_floor_gate.py",
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr13_patch_vllm_cfwd_native_fullvalue_cuda.py",
    "scripts/run_swe_bench_q36_a.py",
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py",
    "src/lumo_flywheel_serving/fr13_cfwd_native_fullvalue_cuda.py",
}


class GateError(RuntimeError):
    """A live gate input failed its exact evidence contract."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read(path: Path, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise GateError(f"required artifact is not a regular non-symlink: {path}")
    payload = path.read_bytes()
    if not payload or len(payload) > max_bytes:
        raise GateError(f"required artifact size is invalid: {path}")
    return payload


def _json(path: Path) -> dict[str, object]:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise GateError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise GateError(f"non-finite JSON constant in {path}: {value}")

    try:
        payload = json.loads(
            _read(path).decode("ascii"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid ASCII JSON: {path}") from error
    if not isinstance(payload, dict):
        raise GateError(f"JSON root must be an object: {path}")
    return payload


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise GateError(f"{label} must be an object")
    return value


def _exact_sha(value: object, label: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise GateError(f"{label} is not SHA-256")
    return value


def _validate_manifest_pair(launch: Path, end: Path) -> str:
    launch_bytes = _read(launch)
    if launch_bytes != _read(end):
        raise GateError("runtime manifest changed during the real task")
    manifest = _json(launch)
    if (
        manifest.get("schema") != "fr13-runtime-manifest-v1"
        or manifest.get("profile") != "fixed32"
        or manifest.get("sequence")
        != "scripts/fr13_run_b1_cfwd_native_keygroup_gate.sh"
    ):
        raise GateError("runtime manifest identity drift")
    closures = _mapping(manifest.get("closures"), "runtime manifest closures")
    observed: set[str] = set()
    for records in closures.values():
        if not isinstance(records, list):
            raise GateError("runtime manifest closure is not a list")
        for record in records:
            observed.add(str(_mapping(record, "runtime manifest record").get("path")))
    missing = REQUIRED_MANIFEST_PATHS - observed
    if missing:
        raise GateError(f"runtime manifest omits gate source: {sorted(missing)}")
    return _sha256(launch_bytes)


def _validate_snapshot_reference(
    reference: object, arm_dir: Path, label: str
) -> dict[str, object]:
    record = _mapping(reference, label)
    path = Path(str(record.get("path"))).resolve(strict=True)
    logs = (arm_dir / "logs").resolve(strict=True)
    if path.parent != logs:
        raise GateError(f"{label} is outside the arm logs directory")
    payload = _read(path)
    if _sha256(payload) != _exact_sha(record.get("sha256"), f"{label}.sha256"):
        raise GateError(f"{label} digest drift")
    snapshot = _json(path)
    if (
        snapshot.get("schema") != "fr13-fixed32-boundary-snapshot-v4"
        or snapshot.get("generation") != record.get("generation")
        or snapshot.get("mode") != "hydra27_fixed32"
    ):
        raise GateError(f"{label} identity drift")
    return snapshot


def _validate_candidate_maps(
    committer: Mapping[str, object], binary_sha256: str, label: str
) -> None:
    expected = {"1": ROUTE}
    if committer.get("candidate_routes_by_batch") != expected:
        raise GateError(f"{label} candidate route drift")
    if committer.get("candidate_source_sha256_by_batch") != {
        "1": candidate.CUDA_SOURCE_SHA256
    }:
        raise GateError(f"{label} candidate source drift")
    if committer.get("candidate_binary_sha256_by_batch") != {
        "1": binary_sha256
    }:
        raise GateError(f"{label} candidate binary drift")


def validate_gate(
    *,
    repo: Path,
    arm_dir: Path,
    source_commit: str,
    candidate_so: Path,
    binding_path: Path,
    manifest_launch: Path,
    manifest_end: Path,
) -> dict[str, object]:
    repo = repo.resolve(strict=True)
    arm_dir = arm_dir.resolve(strict=True)
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise GateError("source commit is invalid")
    actual_commit = subprocess.run(
        ("git", "-C", str(repo), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != source_commit:
        raise GateError("source commit changed during the real task")

    manifest_sha256 = _validate_manifest_pair(manifest_launch, manifest_end)
    binding = binary_gate.verify_binding(binding_path, candidate_so, repo)
    binary_identity = _mapping(binding.get("binary"), "binding binary")
    binary_sha256 = _exact_sha(binary_identity.get("sha256"), "binding binary")
    installed_binding_path = (
        arm_dir
        / "logs/fr13_fixed32_cfwd_native_keygroup_precompute.binding.json"
    )
    if _read(binding_path) != _read(installed_binding_path):
        raise GateError("installed binary binding differs from the issued binding")
    installed_binding = binary_gate.verify_binding(
        installed_binding_path, candidate_so, repo
    )
    if installed_binding != binding:
        raise GateError("installed binary binding normalized identity drift")
    selector_arm_path = (
        arm_dir / "logs/fr13_fixed32_cfwd_native_keygroup_precompute.arm"
    )
    selector_arm_metadata = selector_arm_path.lstat()
    if (
        stat.S_IMODE(selector_arm_metadata.st_mode) != 0o400
        or selector_arm_metadata.st_nlink != 1
        or _read(selector_arm_path) != b"diagnostic\n"
    ):
        raise GateError("installed native key-group selector arm drift")

    install_path = (
        arm_dir / "logs/fr13_fixed32_cfwd_native_keygroup_precompute.install.json"
    )
    install = _json(install_path)
    if (
        install.get("schema")
        != "fr13.fixed32.cfwd_native_keygroup_install.v1"
        or install.get("candidate") != candidate.CANDIDATE
        or install.get("destination") != "vllm/_C.abi3.so"
        or install.get("binary") != binary_identity
        or install.get("binding_sha256") != _sha256(_read(binding_path))
        or install.get("default_on") is not False
        or install.get("production_authorized") is not False
        or install.get("timing_eligible") is not False
    ):
        raise GateError("installed full vLLM extension identity drift")

    diagnostic = _json(arm_dir / "fixed32_b1_diagnostic.json")
    if (
        diagnostic.get("schema") != "fr13-fixed32-b1-diagnostic-v1"
        or diagnostic.get("run_classification") != "b1_diagnostic"
        or diagnostic.get("subset_sha256") != SUBSET_SHA256
        or diagnostic.get("task_ids") != [TASK_ID]
        or diagnostic.get("max_num_seqs") != 1
        or diagnostic.get("swe_concurrency") != 1
        or diagnostic.get("gate_eligible") is not False
        or diagnostic.get("floor_acceptance_eligible") is not False
    ):
        raise GateError("real SWE B1 diagnostic binding drift")

    task_dir = arm_dir / f"swe_out/verified/per_task/{TASK_ID}"
    arm = _json(task_dir / "fixed32_committer_layer_batch_real_task_arm.json")
    if (
        arm.get("schema")
        != "fr13-fixed32-committer-layer-batch-real-task-arm-v1"
        or arm.get("run_classification")
        != "cfwd_layer_batch_real_swe_qualification"
        or arm.get("instance_id") != TASK_ID
        or arm.get("marker") != ARM_MARKER
        or arm.get("marker_bytes") != len((ARM_MARKER + "\n").encode("ascii"))
        or arm.get("marker_sha256") != ARM_MARKER_SHA256
        or arm.get("state") != "ended"
        or not arm.get("started_at")
        or not arm.get("ended_at")
        or arm.get("performance_measurement") is not False
        or arm.get("timing_eligible") is not False
        or arm.get("gate_eligible") is not False
        or arm.get("floor_acceptance_eligible") is not False
        or arm.get("process_local_qualification_only") is not True
        or arm.get("durable_production_pass") is not False
        or arm.get("timing_requires_same_server_process") is not True
        or arm.get("same_process_timing_handoff_implemented") is not False
    ):
        raise GateError("real SWE CFWD arm did not close exactly")

    boundary_path = task_dir / "fixed32_task_boundary.json"
    boundary_bytes = _read(boundary_path)
    boundary = _json(boundary_path)
    coverage = _mapping(
        boundary.get("qualification_coverage"), "qualification coverage"
    )
    expected_coverage = {
        "accepted_length_full_mask": FULL_MASK,
        "pre_attempts_by_batch": {"1": 0},
        "pre_coverage_mask_by_batch": {"1": 0},
        "post_attempts_by_batch": {"1": 12},
        "post_coverage_mask_by_batch": {"1": FULL_MASK},
        "attempt_delta_by_batch": {"1": 12},
        "new_coverage_mask_by_batch": {"1": FULL_MASK},
        "newly_covered_lengths_by_batch": {"1": EXPECTED_LENGTHS},
        "remaining_coverage_mask_by_batch": {"1": 0},
        "coverage_complete": True,
        "shadow_reference_replays": 12,
        "shadow_candidate_replays": 12,
        "new_depth_reference_served_replays": 12,
        "new_depth_served_route": "native_reference",
        "formal_work_census_eligible": False,
    }
    if (
        boundary.get("schema") != "fr13-fixed32-task-boundary-v1"
        or boundary.get("instance_id") != TASK_ID
        or boundary.get("mode") != "hydra27_fixed32"
        or boundary.get("run_classification")
        != "cfwd_layer_batch_real_swe_qualification"
        or boundary.get("acceptance_valid") is not False
        or boundary.get("performance_measurement") is not False
        or boundary.get("timing_eligible") is not False
        or boundary.get("gate_eligible") is not False
        or boundary.get("floor_acceptance_eligible") is not False
        or boundary.get("process_local_qualification_only") is not True
        or boundary.get("durable_production_pass") is not False
        or boundary.get("timing_requires_same_server_process") is not True
        or boundary.get("same_process_timing_handoff_implemented") is not False
        or dict(coverage) != expected_coverage
    ):
        raise GateError("all-depth reference-served byte coverage is incomplete")
    identity = _mapping(boundary.get("candidate_identity"), "candidate identity")
    for phase in ("pre", "post"):
        if identity.get(f"{phase}_routes_by_batch") != {"1": ROUTE}:
            raise GateError(f"{phase} candidate route drift")
        if identity.get(f"{phase}_source_sha256_by_batch") != {
            "1": candidate.CUDA_SOURCE_SHA256
        }:
            raise GateError(f"{phase} candidate source drift")
        if identity.get(f"{phase}_binary_sha256_by_batch") != {
            "1": binary_sha256
        }:
            raise GateError(f"{phase} candidate binary drift")

    pre_snapshot = _validate_snapshot_reference(
        boundary.get("pre_runtime_snapshot"), arm_dir, "pre snapshot"
    )
    post_snapshot = _validate_snapshot_reference(
        boundary.get("post_runtime_snapshot"), arm_dir, "post snapshot"
    )
    pre_committer = _mapping(
        _mapping(pre_snapshot.get("metrics"), "pre metrics").get("committer"),
        "pre committer",
    )
    post_committer = _mapping(
        _mapping(post_snapshot.get("metrics"), "post metrics").get("committer"),
        "post committer",
    )
    _validate_candidate_maps(pre_committer, binary_sha256, "pre snapshot")
    _validate_candidate_maps(post_committer, binary_sha256, "post snapshot")
    if (
        pre_committer.get("layer_batch_gate_attempts_by_batch") != {"1": 0}
        or pre_committer.get("layer_batch_gate_coverage_mask_by_batch")
        != {"1": 0}
        or pre_committer.get("layer_batch_gate_passed_by_batch") != {"1": 0}
        or post_committer.get("layer_batch_gate_attempts_by_batch") != {"1": 12}
        or post_committer.get("layer_batch_gate_coverage_mask_by_batch")
        != {"1": FULL_MASK}
        or post_committer.get("layer_batch_gate_passed_by_batch") != {"1": 1}
    ):
        raise GateError("raw snapshot byte-gate transition drift")
    pre_gate_precompute = pre_committer.get("gate_precompute_launches")
    if (
        type(pre_gate_precompute) is not int
        or pre_gate_precompute <= 0
        or post_committer.get("gate_precompute_launches") != pre_gate_precompute
    ):
        raise GateError("event-independent gate precompute identity drift")

    runner_metadata = _json(task_dir / "runner_metadata.json")
    if runner_metadata.get("fixed32_task_boundary") != boundary:
        raise GateError("runner metadata does not bind the task boundary")
    audit_path = arm_dir / "fixed32_chat_traffic_audit.json"
    audit = _json(audit_path)
    try:
        subset = floor_gate.validate_fixed32_run_subset(
            repo / "config/fr13_fixed32/subset_b1_diagnostic_one.json",
            b1_diagnostic=True,
        )
        dataset_record_digests = floor_gate.pinned_dataset_record_digests(
            str(repo)
        )
        expected_audit = floor_gate.build_fixed32_chat_traffic_audit(
            arm_dir,
            mode="hydra27_fixed32",
            subset=subset,
            dataset_record_digests=dataset_record_digests,
            concurrency=1,
        )
    except floor_gate.GateError as error:
        raise GateError(f"cannot reconstruct authenticated chat audit: {error}") from error
    if audit != expected_audit:
        raise GateError("authenticated chat audit differs from raw evidence")
    audit = expected_audit
    task_audit = _mapping(
        _mapping(audit.get("tasks"), "chat audit tasks").get(TASK_ID),
        "chat audit task",
    )
    terminal = _mapping(task_audit.get("terminal"), "chat audit terminal")
    if (
        audit.get("schema") != "fr13-fixed32-chat-task-provenance-audit-v3"
        or audit.get("mode") != "hydra27_fixed32"
        or audit.get("dataset_name") != "princeton-nlp/SWE-bench_Verified"
        or audit.get("subset")
        != {"sha256": SUBSET_SHA256, "task_count": 1, "task_ids": [TASK_ID]}
        or not isinstance(audit.get("checks"), Mapping)
        or not audit["checks"]
        or any(value is not True for value in audit["checks"].values())
        or terminal.get("agent")
        != {
            "exit_code": 0,
            "timed_out": False,
            "offloaded": True,
            "network_drop": False,
        }
        or terminal.get("eval")
        != {"verdict": "resolved", "passed": True, "harness_exit_code": 0}
    ):
        raise GateError("authenticated real SWE task did not resolve cleanly")
    audit_boundary = _mapping(task_audit.get("boundary"), "chat audit boundary")
    if (
        audit_boundary.get("sha256") != _sha256(boundary_bytes)
        or audit_boundary.get("bytes") != len(boundary_bytes)
    ):
        raise GateError("chat audit does not bind the exact task boundary")
    task_auth = _mapping(task_audit.get("task_auth"), "chat audit task auth")
    if (
        type(task_auth.get("completed_logical_model_requests")) is not int
        or task_auth["completed_logical_model_requests"] <= 0
        or task_auth.get("aborted_logical_requests") != 0
        or task_auth.get("failed_attempts") != 0
    ):
        raise GateError("authenticated real task request stream is incomplete")

    return {
        "schema": SCHEMA,
        "status": "pass",
        "classification": "one_real_swe_verified_k64_root_b1_byte_gate",
        "task_id": TASK_ID,
        "task_resolved": True,
        "subset_sha256": SUBSET_SHA256,
        "draft_vocab": {
            "k": 65536,
            "root_reduced": True,
            "block_map_sha256": BLOCK_MAP_SHA256,
        },
        "candidate": candidate.CANDIDATE,
        "candidate_route": ROUTE,
        "source_commit": source_commit,
        "source_sha256": candidate.CUDA_SOURCE_SHA256,
        "binary_sha256": binary_sha256,
        "binding_sha256": _sha256(_read(binding_path)),
        "runtime_manifest_sha256": manifest_sha256,
        "accepted_lengths": EXPECTED_LENGTHS,
        "all_48_layers_byte_equal": True,
        "reference_served_for_every_new_depth": True,
        "full_vllm_extension_replaced": True,
        "performance_measurement": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_authorized": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
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
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--runtime-manifest-launch", type=Path, required=True)
    parser.add_argument("--runtime-manifest-end", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    payload = validate_gate(
        repo=arguments.repo,
        arm_dir=arguments.arm_dir,
        source_commit=arguments.source_commit,
        candidate_so=arguments.candidate_so,
        binding_path=arguments.binding,
        manifest_launch=arguments.runtime_manifest_launch,
        manifest_end=arguments.runtime_manifest_end,
    )
    _write_json(arguments.output, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
