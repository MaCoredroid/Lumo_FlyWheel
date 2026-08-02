from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import fr13_cfwd_native_keygroup_b1_gate as gate  # noqa: E402
import fr13_runtime_manifest as runtime_manifest  # noqa: E402


def _write_json(path: Path, payload: object) -> bytes:
    encoded = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded


def _candidate_maps(binary_sha256: str) -> dict[str, object]:
    return {
        "candidate_routes_by_batch": {"1": gate.ROUTE},
        "candidate_source_sha256_by_batch": {
            "1": gate.candidate.CUDA_SOURCE_SHA256
        },
        "candidate_binary_sha256_by_batch": {"1": binary_sha256},
    }


def _patch_exact_audit(
    monkeypatch: pytest.MonkeyPatch,
    arm_dir: Path,
    *,
    expected: dict[str, object] | None = None,
) -> None:
    audit = json.loads(
        (arm_dir / "fixed32_chat_traffic_audit.json").read_text(
            encoding="ascii"
        )
    )
    monkeypatch.setattr(
        gate.floor_gate,
        "validate_fixed32_run_subset",
        lambda *_args, **_kwargs: {
            "sha256": gate.SUBSET_SHA256,
            "task_ids": [gate.TASK_ID],
        },
    )
    monkeypatch.setattr(
        gate.floor_gate,
        "pinned_dataset_record_digests",
        lambda *_args, **_kwargs: {gate.TASK_ID: "a" * 64},
    )
    monkeypatch.setattr(
        gate.floor_gate,
        "build_fixed32_chat_traffic_audit",
        lambda *_args, **_kwargs: audit if expected is None else expected,
    )


def _gate_fixture(tmp_path: Path, binary_sha256: str) -> dict[str, Path | str]:
    arm_dir = tmp_path / "arm"
    logs = arm_dir / "logs"
    task_dir = arm_dir / f"swe_out/verified/per_task/{gate.TASK_ID}"
    logs.mkdir(parents=True)
    task_dir.mkdir(parents=True)

    binding_path = tmp_path / "binding.json"
    binding_bytes = b'{"fixture":"binding"}\n'
    binding_path.write_bytes(binding_bytes)
    binding_path.chmod(0o400)
    installed_binding = (
        logs / "fr13_fixed32_cfwd_native_keygroup_precompute.binding.json"
    )
    installed_binding.write_bytes(binding_bytes)
    installed_binding.chmod(0o400)
    selector_arm = logs / "fr13_fixed32_cfwd_native_keygroup_precompute.arm"
    selector_arm.write_bytes(b"diagnostic\n")
    selector_arm.chmod(0o400)

    binary_identity = {"sha256": binary_sha256, "bytes": 123456}
    _write_json(
        logs / "fr13_fixed32_cfwd_native_keygroup_precompute.install.json",
        {
            "schema": "fr13.fixed32.cfwd_native_keygroup_install.v1",
            "candidate": gate.candidate.CANDIDATE,
            "destination": "vllm/_C.abi3.so",
            "binary": binary_identity,
            "binding_sha256": hashlib.sha256(binding_bytes).hexdigest(),
            "default_on": False,
            "production_authorized": False,
            "timing_eligible": False,
        },
    )
    _write_json(
        arm_dir / "fixed32_b1_diagnostic.json",
        {
            "schema": "fr13-fixed32-b1-diagnostic-v1",
            "run_classification": "b1_diagnostic",
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
            "max_num_seqs": 1,
            "swe_concurrency": 1,
            "subset_sha256": gate.SUBSET_SHA256,
            "task_ids": [gate.TASK_ID],
        },
    )
    _write_json(
        task_dir / "fixed32_committer_layer_batch_real_task_arm.json",
        {
            "schema": "fr13-fixed32-committer-layer-batch-real-task-arm-v1",
            "run_classification": "cfwd_layer_batch_real_swe_qualification",
            "instance_id": gate.TASK_ID,
            "marker": gate.ARM_MARKER,
            "marker_bytes": len((gate.ARM_MARKER + "\n").encode("ascii")),
            "marker_sha256": gate.ARM_MARKER_SHA256,
            "state": "ended",
            "started_at": "2026-08-02T00:00:00Z",
            "ended_at": "2026-08-02T00:01:00Z",
            "performance_measurement": False,
            "timing_eligible": False,
            "gate_eligible": False,
            "floor_acceptance_eligible": False,
            "process_local_qualification_only": True,
            "durable_production_pass": False,
            "timing_requires_same_server_process": True,
            "same_process_timing_handoff_implemented": False,
        },
    )

    def snapshot(generation: int, attempts: int, coverage: int, passed: int) -> Path:
        path = logs / f"fr13_fixed32_boundary_snapshot.{generation}.json"
        _write_json(
            path,
            {
                "schema": "fr13-fixed32-boundary-snapshot-v4",
                "generation": generation,
                "mode": "hydra27_fixed32",
                "metrics": {
                    "committer": {
                        **_candidate_maps(binary_sha256),
                        "layer_batch_gate_attempts_by_batch": {"1": attempts},
                        "layer_batch_gate_coverage_mask_by_batch": {"1": coverage},
                        "layer_batch_gate_passed_by_batch": {"1": passed},
                        "gate_precompute_launches": 1,
                    }
                },
            },
        )
        return path

    pre_snapshot = snapshot(1, 0, 0, 0)
    post_snapshot = snapshot(2, 12, gate.FULL_MASK, 1)

    def snapshot_ref(path: Path, generation: int) -> dict[str, object]:
        return {
            "schema": "fr13-fixed32-boundary-snapshot-v4",
            "generation": generation,
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    coverage = {
        "accepted_length_full_mask": gate.FULL_MASK,
        "pre_attempts_by_batch": {"1": 0},
        "pre_coverage_mask_by_batch": {"1": 0},
        "post_attempts_by_batch": {"1": 12},
        "post_coverage_mask_by_batch": {"1": gate.FULL_MASK},
        "attempt_delta_by_batch": {"1": 12},
        "new_coverage_mask_by_batch": {"1": gate.FULL_MASK},
        "newly_covered_lengths_by_batch": {"1": gate.EXPECTED_LENGTHS},
        "remaining_coverage_mask_by_batch": {"1": 0},
        "coverage_complete": True,
        "shadow_reference_replays": 12,
        "shadow_candidate_replays": 12,
        "new_depth_reference_served_replays": 12,
        "new_depth_served_route": "native_reference",
        "formal_work_census_eligible": False,
    }
    candidate_identity = {
        "pre_routes_by_batch": {"1": gate.ROUTE},
        "pre_source_sha256_by_batch": {"1": gate.candidate.CUDA_SOURCE_SHA256},
        "pre_binary_sha256_by_batch": {"1": binary_sha256},
        "post_routes_by_batch": {"1": gate.ROUTE},
        "post_source_sha256_by_batch": {"1": gate.candidate.CUDA_SOURCE_SHA256},
        "post_binary_sha256_by_batch": {"1": binary_sha256},
    }
    boundary = {
        "schema": "fr13-fixed32-task-boundary-v1",
        "instance_id": gate.TASK_ID,
        "mode": "hydra27_fixed32",
        "run_classification": "cfwd_layer_batch_real_swe_qualification",
        "acceptance_valid": False,
        "performance_measurement": False,
        "timing_eligible": False,
        "gate_eligible": False,
        "floor_acceptance_eligible": False,
        "process_local_qualification_only": True,
        "durable_production_pass": False,
        "timing_requires_same_server_process": True,
        "same_process_timing_handoff_implemented": False,
        "candidate_identity": candidate_identity,
        "qualification_coverage": coverage,
        "pre_runtime_snapshot": snapshot_ref(pre_snapshot, 1),
        "post_runtime_snapshot": snapshot_ref(post_snapshot, 2),
    }
    boundary_bytes = _write_json(task_dir / "fixed32_task_boundary.json", boundary)
    _write_json(task_dir / "runner_metadata.json", {"fixed32_task_boundary": boundary})
    _write_json(
        arm_dir / "fixed32_chat_traffic_audit.json",
        {
            "schema": "fr13-fixed32-chat-task-provenance-audit-v3",
            "mode": "hydra27_fixed32",
            "dataset_name": "princeton-nlp/SWE-bench_Verified",
            "subset": {
                "sha256": gate.SUBSET_SHA256,
                "task_count": 1,
                "task_ids": [gate.TASK_ID],
            },
            "checks": {"fixture": True},
            "tasks": {
                gate.TASK_ID: {
                    "terminal": {
                        "agent": {
                            "exit_code": 0,
                            "timed_out": False,
                            "offloaded": True,
                            "network_drop": False,
                        },
                        "eval": {
                            "verdict": "resolved",
                            "passed": True,
                            "harness_exit_code": 0,
                        },
                    },
                    "boundary": {
                        "sha256": hashlib.sha256(boundary_bytes).hexdigest(),
                        "bytes": len(boundary_bytes),
                    },
                    "task_auth": {
                        "completed_logical_model_requests": 1,
                        "aborted_logical_requests": 0,
                        "failed_attempts": 0,
                    },
                }
            },
        },
    )

    manifest = {
        "schema": "fr13-runtime-manifest-v1",
        "profile": "fixed32",
        "sequence": "scripts/fr13_run_b1_cfwd_native_keygroup_gate.sh",
        "closures": {
            "fixture": [
                {"path": path} for path in sorted(gate.REQUIRED_MANIFEST_PATHS)
            ]
        },
    }
    manifest_launch = tmp_path / "manifest.launch.json"
    manifest_end = tmp_path / "manifest.end.json"
    manifest_bytes = _write_json(manifest_launch, manifest)
    manifest_end.write_bytes(manifest_bytes)
    candidate_so = tmp_path / "candidate._C.abi3.so"
    candidate_so.write_bytes(b"fixture")
    source_commit = subprocess.run(
        ("git", "-C", str(ROOT), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "repo": ROOT,
        "arm_dir": arm_dir,
        "source_commit": source_commit,
        "candidate_so": candidate_so,
        "binding_path": binding_path,
        "manifest_launch": manifest_launch,
        "manifest_end": manifest_end,
    }


def test_gate_accepts_exact_source_bound_all_depth_real_b1_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary_sha256 = "d" * 64
    paths = _gate_fixture(tmp_path, binary_sha256)
    binding = {"binary": {"sha256": binary_sha256, "bytes": 123456}}
    monkeypatch.setattr(gate.binary_gate, "verify_binding", lambda *_: binding)
    _patch_exact_audit(monkeypatch, Path(paths["arm_dir"]))

    report = gate.validate_gate(**paths)

    assert report["status"] == "pass"
    assert report["task_resolved"] is True
    assert report["accepted_lengths"] == list(range(12))
    assert report["all_48_layers_byte_equal"] is True
    assert report["reference_served_for_every_new_depth"] is True
    assert report["performance_measurement"] is False
    assert report["floor_acceptance_eligible"] is False


def test_gate_rejects_selector_arm_or_candidate_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary_sha256 = "d" * 64
    paths = _gate_fixture(tmp_path, binary_sha256)
    binding = {"binary": {"sha256": binary_sha256, "bytes": 123456}}
    monkeypatch.setattr(gate.binary_gate, "verify_binding", lambda *_: binding)
    _patch_exact_audit(monkeypatch, Path(paths["arm_dir"]))
    selector_arm = (
        Path(paths["arm_dir"])
        / "logs/fr13_fixed32_cfwd_native_keygroup_precompute.arm"
    )
    selector_arm.chmod(stat.S_IRUSR | stat.S_IWUSR)
    selector_arm.write_bytes(b"production\n")
    selector_arm.chmod(stat.S_IRUSR)
    with pytest.raises(gate.GateError, match="selector arm drift"):
        gate.validate_gate(**paths)

    selector_arm.chmod(stat.S_IRUSR | stat.S_IWUSR)
    selector_arm.write_bytes(b"diagnostic\n")
    selector_arm.chmod(stat.S_IRUSR)
    task_dir = (
        Path(paths["arm_dir"])
        / f"swe_out/verified/per_task/{gate.TASK_ID}"
    )
    boundary_path = task_dir / "fixed32_task_boundary.json"
    boundary = json.loads(boundary_path.read_text(encoding="ascii"))
    boundary["candidate_identity"]["post_binary_sha256_by_batch"] = {
        "1": "e" * 64
    }
    _write_json(boundary_path, boundary)
    _write_json(task_dir / "runner_metadata.json", {"fixed32_task_boundary": boundary})
    with pytest.raises(gate.GateError, match="post candidate binary drift"):
        gate.validate_gate(**paths)


def test_gate_rejects_published_audit_that_differs_from_raw_reconstruction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary_sha256 = "d" * 64
    paths = _gate_fixture(tmp_path, binary_sha256)
    binding = {"binary": {"sha256": binary_sha256, "bytes": 123456}}
    monkeypatch.setattr(gate.binary_gate, "verify_binding", lambda *_: binding)
    _patch_exact_audit(
        monkeypatch,
        Path(paths["arm_dir"]),
        expected={"schema": "reconstructed-from-raw-evidence"},
    )

    with pytest.raises(gate.GateError, match="differs from raw evidence"):
        gate.validate_gate(**paths)


def test_gate_runner_and_launcher_are_closed_over_exact_b1_contract() -> None:
    runner = (ROOT / "scripts/fr13_run_b1_cfwd_native_keygroup_gate.sh").read_text(
        encoding="utf-8"
    )
    serve = (ROOT / "scripts/fr13_bigdenom_swe_serve_variant.sh").read_text(
        encoding="utf-8"
    )
    launcher = (
        ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
    ).read_text(encoding="utf-8")

    assert f"TASK_ID={gate.TASK_ID}" in runner
    assert f"SUBSET_SHA256={gate.SUBSET_SHA256}" in runner
    assert f"DRAFT_VOCAB_BLOCKS_SHA256={gate.BLOCK_MAP_SHA256}" in runner
    assert "export FR13_DRAFT_VOCAB_K=65536" in runner
    assert "export FR13_DRAFT_VOCAB_ROOT=1" in runner
    assert "export FR13_FIXED32_B1_DIAGNOSTIC=1" in runner
    assert "export FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION=1" in runner
    assert "export FR13_FIXED32_CFWD_NATIVE_KEYGROUP_PRECOMPUTE_CUDA=diagnostic" in runner
    assert "export FR13_FIXED32_CONV_SOURCE_BATCH=0" in runner
    assert "export FR13_DRAFT_HEAD_PAD_ROWS=0" in runner
    assert "export FR13_DRAFT_HEAD_M32_TIMING_ARM=0" in runner
    assert "export FR13_SFWD_GPU_TIMER=1" in runner
    assert "export FR13_DFWD_GPU_TIMER=1" in runner
    assert "export FR13_CFWD_GPU_TIMER=1" in runner
    assert "export FR13_GRAPH_TIMER=0" in runner
    assert "export LUMO_SWE_AUTOCOMMIT=0" in runner
    assert "fr13_cfwd_native_keygroup_b1_gate.py" in runner

    assert "native key-group CFWD requires layer-batch qualification" in serve
    assert "native key-group CFWD requires boundary counters and forbids auxiliary timing" in serve
    assert "native key-group CFWD byte qualification forbids task autocommit" in serve
    assert "native key-group CFWD must be the only kernel candidate" in launcher
    assert "/tmp/fr13_cfwd_native_keygroup._C.abi3.so:ro" in launcher
    assert "fr13_cfwd_native_keygroup_binary.py install" in launcher
    assert "--destination /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so" in launcher
    assert "_C_stable_libtorch.abi3.so" not in launcher[
        launcher.index("fr13_cfwd_native_keygroup_binary.py install") :
        launcher.index("fr13_cfwd_native_keygroup_binary.py install") + 500
    ]

    manifest_sources = {
        *runtime_manifest.FIXED32_HOST_SCRIPT_SOURCE,
        *runtime_manifest.FIXED32_PYTHON_PACKAGE_SOURCE,
        *runtime_manifest.FIXED32_VERDICT_TOOLS,
    }
    assert gate.REQUIRED_MANIFEST_PATHS <= manifest_sources
