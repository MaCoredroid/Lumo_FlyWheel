from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
BLOCK_MAP = SCRIPTS / "fr13_dvk_subset_blocks.json"
SELECTOR = "identity_hybrid_n5120_b4"
DIAGNOSTIC_SELECTOR = "identity_hybrid_n5120_b4_byte_ab"


def _load(name: str, filename: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load("fr13_hybrid_n5120_dual_pass_test", "fr13_cutlass_b4_pass.py")
    candidate_bytes = b"hybrid N5120 dual-topology candidate\n"
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_bytes)
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    monkeypatch.setattr(
        module.binary,
        "IDENTITY_HYBRID_N5120_B4_CANDIDATE_SIZE",
        len(candidate_bytes),
    )
    monkeypatch.setattr(
        module.binary,
        "IDENTITY_HYBRID_N5120_B4_CANDIDATE_SHA256",
        candidate_sha256,
    )

    patch_bytes = b"hybrid N5120 patch source\n"
    patch_source = tmp_path / "patch.py"
    patch_source.write_bytes(patch_bytes)
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    monkeypatch.setattr(
        module, "IDENTITY_HYBRID_N5120_PATCH_SOURCE_SHA256", patch_sha256
    )

    live_paths: dict[str, Path] = {}
    live_hashes: dict[str, str] = {}
    for index, mode in enumerate(module.QUALIFIED_FIXED32_MODES):
        payload = {
            "schema": module.IDENTITY_HYBRID_N5120_K64_ROOT_LIVE_SCHEMA,
            "status": "pass",
            "run_classification": (
                "real_swe_verified_exact4_b4_k64_root_byte_diagnostic"
            ),
            "acceptance_valid": False,
            "task_count": 4,
            "task_ids": list(module.EXPECTED_TASK_IDS),
            "topology": mode,
            "task_marker": f"swe_verified:{module.EXPECTED_TASK_IDS[index]}",
            "draft_vocab_root": 1,
            "draft_vocab_k": 65_536,
            "mandatory_weight_bytes": module.K64_ROOT_MANDATORY_WEIGHT_BYTES,
            "mandatory_weight_floor_ms": module.K64_ROOT_MANDATORY_WEIGHT_FLOOR_MS,
            "one_sided_u95_cap_ms": module.K64_ROOT_SLO_CAP_MS,
            "comparator_timing_eligible": False,
            "batch_size": 4,
            "concurrency": 4,
            "fixed_rows": 128,
            "eager_builder_capacity": 128,
            "candidate": SELECTOR,
            "diagnostic_selector": DIAGNOSTIC_SELECTOR,
            "served_result": "stock",
            "production_enabled": False,
            "comparison_call_limit": module.MAX_COMPARISONS,
            "comparisons": 5,
            "observed_m_values": [128],
            "observed_projection_nk": [
                list(shape) for shape in module.EXPECTED_PROJECTION_NK
            ],
            "mismatching_comparisons": 0,
            "differing_bytes": 0,
            "candidate_family": SELECTOR,
            "candidate_sha256": candidate_sha256,
            "candidate_bytes": len(candidate_bytes),
            "patch_source_sha256": patch_sha256,
            "vllm_base_commit": module.VLLM_BASE_COMMIT,
            "patched_dispatch_sha256": (
                module.IDENTITY_HYBRID_N5120_PATCHED_DISPATCH_SHA256
            ),
            "source_commit": "c" * 40,
            "binary_attestation_sha256": f"{index + 1}" * 64,
            "real_task_arm_sha256": f"{index + 3}" * 64,
            "container_env_sha256": f"{index + 5}" * 64,
            "qualification_profile": "k64_root",
            "draft_vocab_blocks": module.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
            "draft_vocab_blocks_sha256": module.DRAFT_VOCAB_BLOCKS_SHA256,
            "errors": [],
        }
        path = tmp_path / f"{mode}.json"
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
        live_paths[mode] = path
        live_hashes[mode] = hashlib.sha256(path.read_bytes()).hexdigest()
    return module, candidate, patch_source, live_paths, live_hashes


def test_hybrid_binary_requires_explicit_k64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("fr13_hybrid_n5120_binary_test", "fr13_cutlass_wave_binary.py")
    payload = b"hybrid N5120 candidate\n"
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(payload)
    monkeypatch.setattr(
        module, "IDENTITY_HYBRID_N5120_B4_CANDIDATE_SIZE", len(payload)
    )
    monkeypatch.setattr(
        module,
        "IDENTITY_HYBRID_N5120_B4_CANDIDATE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    with pytest.raises(ValueError, match="requires a k64_root qualification"):
        module.verify_candidate(candidate, DIAGNOSTIC_SELECTOR)
    with pytest.raises(ValueError, match="requires a k64_root qualification"):
        module.verify_candidate(
            candidate, DIAGNOSTIC_SELECTOR, qualification_profile="full_vocab"
        )

    record = module.verify_candidate(
        candidate, DIAGNOSTIC_SELECTOR, qualification_profile="k64_root"
    )
    assert record["candidate_family"] == SELECTOR
    assert record["qualification_profile"] == "k64_root"


def test_hybrid_dual_sidecar_binds_tail23_and_hydra27(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, paths, hashes = _fixture(tmp_path, monkeypatch)
    sidecar = tmp_path / "hybrid-dual-sidecar.json"
    issued = module.issue_dual_sidecar(
        paths["tail6_fixed32"],
        hashes["tail6_fixed32"],
        paths["hydra27_fixed32"],
        hashes["hydra27_fixed32"],
        candidate,
        sidecar,
        patch_source,
        expected_source_commit="c" * 40,
        draft_vocab_blocks=BLOCK_MAP,
        candidate_selector=SELECTOR,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    verified = module.verify_dual_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        BLOCK_MAP,
        candidate_selector=SELECTOR,
    )

    assert verified == issued
    assert issued["schema"] == (
        module.IDENTITY_HYBRID_N5120_DUAL_K64_ROOT_SIDECAR_SCHEMA
    )
    assert issued["qualification_profile"] == "k64_root"
    assert {
        mode: record["live_result_sha256"]
        for mode, record in issued["topology_qualifications"].items()
    } == hashes


def test_hybrid_production_install_is_fail_closed_and_preserves_dual_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("fr13_hybrid_n5120_install_test", "fr13_cutlass_wave_binary.py")
    payload = b"hybrid N5120 production candidate\n"
    candidate_sha256 = hashlib.sha256(payload).hexdigest()
    source = tmp_path / "candidate.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    sidecar = tmp_path / "dual-sidecar.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock\n")
    sidecar.write_text("{}\n", encoding="ascii")
    monkeypatch.setattr(
        module, "IDENTITY_HYBRID_N5120_B4_CANDIDATE_SIZE", len(payload)
    )
    monkeypatch.setattr(
        module, "IDENTITY_HYBRID_N5120_B4_CANDIDATE_SHA256", candidate_sha256
    )

    with pytest.raises(ValueError, match="requires a k64_root qualification"):
        module.install_candidate(source, destination, attestation, SELECTOR)
    with pytest.raises(ValueError, match="requires a pinned production sidecar"):
        module.install_candidate(
            source,
            destination,
            attestation,
            SELECTOR,
            qualification_profile="k64_root",
        )
    assert destination.read_bytes() == b"stock\n"

    topology_records = {
        mode: {
            "topology": mode,
            "live_result_sha256": f"{index + 1}" * 64,
            "binary_attestation_sha256": f"{index + 3}" * 64,
            "qualification_task_marker": f"swe_verified:task-{index}",
            "real_task_arm_sha256": f"{index + 5}" * 64,
            "container_env_sha256": f"{index + 7}" * 64,
        }
        for index, mode in enumerate(("tail6_fixed32", "hydra27_fixed32"))
    }
    qualification = {
        "candidate_sha256": candidate_sha256,
        "patch_source_sha256": "a" * 64,
        "qualification_source_commit": "b" * 40,
        "qualification_profile": "k64_root",
        "qualification_topologies": ["tail6_fixed32", "hydra27_fixed32"],
        "qualification_task_ids": ["task-a", "task-b", "task-c", "task-d"],
        "topology_qualifications": topology_records,
        "qualified_draft_vocab_root": 1,
        "qualified_draft_vocab_k": 65_536,
        "qualified_comparison_call_limit": 320,
        "qualified_eager_builder_capacity": 128,
        "qualified_fixed_rows": 128,
        "qualified_projection_nk": [[5120, 6144]],
        "qualified_draft_vocab_blocks": "/workspace/scripts/blocks.json",
        "qualified_draft_vocab_blocks_sha256": "c" * 64,
        "mandatory_weight_bytes": 1,
        "mandatory_weight_floor_ms": 1.0,
        "one_sided_u95_cap_ms": 1.15,
    }
    monkeypatch.setattr(
        module,
        "_verify_production_qualification",
        lambda *args, **kwargs: qualification,
    )
    record = module.install_candidate(
        source,
        destination,
        attestation,
        SELECTOR,
        qualification_profile="k64_root",
        production_sidecar=sidecar,
        expected_production_sidecar_sha256="d" * 64,
    )

    assert destination.read_bytes() == payload
    assert record["production_enabled"] is True
    assert record["qualification_profile"] == "k64_root"
    assert record["qualification"]["topology_qualifications"] == topology_records


def test_hybrid_production_verification_uses_b4_dual_credential_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("fr13_hybrid_n5120_module_dispatch", "fr13_cutlass_wave_binary.py")
    calls: list[tuple[str, str]] = []

    def verify_dual_sidecar(*args, candidate_selector: str, **kwargs):
        calls.append(("b4", candidate_selector))
        return {"status": "QUALIFIED"}

    def wrong_verify_dual_sidecar(*args, candidate_selector: str, **kwargs):
        calls.append(("b1", candidate_selector))
        raise AssertionError("hybrid credential verification used the B1 module")

    monkeypatch.setitem(
        sys.modules,
        "fr13_cutlass_b4_pass",
        types.SimpleNamespace(verify_dual_sidecar=verify_dual_sidecar),
    )
    monkeypatch.setitem(
        sys.modules,
        "fr13_cutlass_streamk_pass",
        types.SimpleNamespace(verify_dual_sidecar=wrong_verify_dual_sidecar),
    )

    result = module._verify_production_qualification(
        tmp_path / "sidecar.json",
        "a" * 64,
        tmp_path / "candidate.so",
        tmp_path / "patch.py",
        SELECTOR,
        "hydra27_fixed32",
    )

    assert result == {"status": "QUALIFIED"}
    assert calls == [("b4", SELECTOR)]


def test_hybrid_full_vocab_is_rejected_before_live_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, paths, hashes = _fixture(tmp_path, monkeypatch)
    with pytest.raises(module.QualificationError, match="requires qualification profile"):
        module.validate_live_result(
            paths["tail6_fixed32"],
            hashes["tail6_fixed32"],
            candidate,
            patch_source,
            candidate_selector=SELECTOR,
            qualification_profile="full_vocab",
            fixed32_mode="tail6_fixed32",
        )


def test_hybrid_launcher_wiring_is_selector_specific() -> None:
    launcher = (SCRIPTS / "fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )
    live_gate = (
        SCRIPTS / "fr13_run_b4_cutlass_persistent_m128_live_gate.sh"
    ).read_text(encoding="utf-8")
    timing = (SCRIPTS / "fr13_run_b4_cutlass_persistent_m128_timing.sh").read_text(
        encoding="utf-8"
    )

    for name in (
        "FR13_FIXED32_CUTLASS_HYBRID_N5120_TAIL23_LIVE_PASS_JSON",
        "FR13_FIXED32_CUTLASS_HYBRID_N5120_TAIL23_LIVE_PASS_SHA256",
        "FR13_FIXED32_CUTLASS_HYBRID_N5120_HYDRA27_LIVE_PASS_JSON",
        "FR13_FIXED32_CUTLASS_HYBRID_N5120_HYDRA27_LIVE_PASS_SHA256",
    ):
        assert name in launcher
    assert 'FR13_FIXED32_CUTLASS_WAVE" == "identity_hybrid_n5120_b4"' in launcher
    assert "CUTLASS hybrid N5120 Tail23/Hydra27 PASS identity mismatch" in launcher
    assert "identity_hybrid_n5120_b4)" in live_gate
    assert "DIAGNOSTIC_SELECTOR=identity_hybrid_n5120_b4_byte_ab" in live_gate
    assert "hybrid N5120 identity gate requires k64_root" in live_gate
    assert "identity_hybrid_n5120_b4)" in timing
    assert (
        "FR13_FIXED32_CUTLASS_HYBRID_N5120_TAIL23_LIVE_PASS_JSON" in timing
    )
    assert "fr13_cutlass_b4_pass.py dual-validate" in timing
    assert "fr13_cutlass_b4_pass.py dual-verify" in timing
