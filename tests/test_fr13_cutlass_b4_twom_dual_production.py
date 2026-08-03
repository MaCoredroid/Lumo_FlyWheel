from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
BLOCK_MAP = SCRIPTS / "fr13_dvk_subset_blocks.json"
SELECTOR = "identity_twom_b4"
DIAGNOSTIC_SELECTOR = "identity_twom_b4_byte_ab"


def _load(name: str, filename: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load("fr13_twom_dual_pass_test", "fr13_cutlass_b4_pass.py")
    candidate_bytes = b"two-M dual topology candidate\n"
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_bytes)
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    monkeypatch.setattr(
        module.binary, "IDENTITY_TWOM_B4_CANDIDATE_SIZE", len(candidate_bytes)
    )
    monkeypatch.setattr(
        module.binary, "IDENTITY_TWOM_B4_CANDIDATE_SHA256", candidate_sha256
    )

    patch_bytes = b"two-M patch source\n"
    patch_source = tmp_path / "patch.py"
    patch_source.write_bytes(patch_bytes)
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    monkeypatch.setattr(
        module, "IDENTITY_STOCKSHAPE_STAGE2_PATCH_SOURCE_SHA256", patch_sha256
    )

    live_paths: dict[str, Path] = {}
    live_hashes: dict[str, str] = {}
    for index, mode in enumerate(module.QUALIFIED_FIXED32_MODES):
        payload = {
            "schema": module.IDENTITY_TWOM_K64_ROOT_LIVE_SCHEMA,
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
                module.IDENTITY_STOCKSHAPE_STAGE2_PATCHED_DISPATCH_SHA256
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


def _issue(fixture, tmp_path: Path):
    module, candidate, patch_source, paths, hashes = fixture
    sidecar = tmp_path / "two-M-dual-sidecar.json"
    record = module.issue_dual_sidecar(
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
    return record, sidecar, hashlib.sha256(sidecar.read_bytes()).hexdigest()


def test_twom_dual_sidecar_binds_both_exact4_topologies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    module, candidate, patch_source, _, hashes = fixture
    issued, sidecar, sidecar_sha256 = _issue(fixture, tmp_path)

    verified = module.verify_dual_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        BLOCK_MAP,
        candidate_selector=SELECTOR,
    )

    assert verified == issued
    assert issued["schema"] == module.IDENTITY_TWOM_DUAL_K64_ROOT_SIDECAR_SCHEMA
    assert issued["candidate_selector"] == SELECTOR
    assert issued["diagnostic_selector"] == DIAGNOSTIC_SELECTOR
    assert issued["production_default_enabled"] is False
    assert {
        mode: record["live_result_sha256"]
        for mode, record in issued["topology_qualifications"].items()
    } == hashes


def test_twom_dual_sidecar_is_not_transferable_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    module, candidate, patch_source, paths, hashes = fixture
    _, sidecar, sidecar_sha256 = _issue(fixture, tmp_path)
    monkeypatch.setattr(
        module.binary,
        "IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SIZE",
        candidate.stat().st_size,
    )
    monkeypatch.setattr(
        module.binary,
        "IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SHA256",
        hashlib.sha256(candidate.read_bytes()).hexdigest(),
    )

    with pytest.raises(module.QualificationError, match="schema mismatch"):
        module.verify_dual_sidecar(
            sidecar,
            sidecar_sha256,
            candidate,
            patch_source,
            BLOCK_MAP,
            candidate_selector=module.STAGE2_DUAL_PRODUCTION_SELECTOR,
        )
    with pytest.raises(module.QualificationError, match="topology"):
        module.validate_dual_live_results(
            paths["hydra27_fixed32"],
            hashes["hydra27_fixed32"],
            paths["tail6_fixed32"],
            hashes["tail6_fixed32"],
            candidate,
            patch_source,
            "c" * 40,
            BLOCK_MAP,
            SELECTOR,
        )

    payload = json.loads(sidecar.read_text(encoding="ascii"))
    payload["candidate_selector"] = module.STAGE2_DUAL_PRODUCTION_SELECTOR
    sidecar.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    tampered_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    with pytest.raises(module.QualificationError, match="candidate_selector mismatch"):
        module.verify_dual_sidecar(
            sidecar,
            tampered_sha256,
            candidate,
            patch_source,
            BLOCK_MAP,
            candidate_selector=SELECTOR,
        )


def test_twom_install_is_fail_closed_then_preserves_dual_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _load("fr13_twom_dual_binary_test", "fr13_cutlass_wave_binary.py")
    payload = b"two-M production candidate\n"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(binary, "IDENTITY_TWOM_B4_CANDIDATE_SIZE", len(payload))
    monkeypatch.setattr(binary, "IDENTITY_TWOM_B4_CANDIDATE_SHA256", digest)
    source = tmp_path / "candidate.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    sidecar = tmp_path / "dual-sidecar.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock\n")
    sidecar.write_text("{}\n", encoding="ascii")

    with pytest.raises(ValueError, match="requires a pinned production sidecar"):
        binary.install_candidate(source, destination, attestation, SELECTOR)
    assert destination.read_bytes() == b"stock\n"

    topology_records = {
        mode: {
            "topology": mode,
            "live_result_sha256": f"{index + 1}" * 64,
            "binary_attestation_sha256": f"{index + 3}" * 64,
            "qualification_task_marker": f"swe_verified:astropy__astropy-{index}",
            "real_task_arm_sha256": f"{index + 5}" * 64,
            "container_env_sha256": f"{index + 7}" * 64,
        }
        for index, mode in enumerate(("tail6_fixed32", "hydra27_fixed32"))
    }
    qualification = {
        "candidate_sha256": digest,
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
        binary,
        "_verify_production_qualification",
        lambda *args, **kwargs: qualification,
    )
    record = binary.install_candidate(
        source,
        destination,
        attestation,
        SELECTOR,
        production_sidecar=sidecar,
        expected_production_sidecar_sha256="d" * 64,
    )

    assert destination.read_bytes() == payload
    assert record["production_enabled"] is True
    assert record["candidate_family"] == SELECTOR
    assert record["qualification"]["topology_qualifications"] == topology_records


def test_twom_production_attestation_preserves_both_live_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    module, candidate, _, _, hashes = fixture
    issued, _, sidecar_sha256 = _issue(fixture, tmp_path)
    identity = {
        "bytes": candidate.stat().st_size,
        "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "regular": True,
        "symlink": False,
    }
    qualification_keys = (
        "candidate_sha256",
        "patch_source_sha256",
        "qualification_source_commit",
        "qualification_profile",
        "qualification_topologies",
        "qualification_task_ids",
        "topology_qualifications",
        "qualified_draft_vocab_root",
        "qualified_draft_vocab_k",
        "qualified_comparison_call_limit",
        "qualified_eager_builder_capacity",
        "qualified_fixed_rows",
        "qualified_projection_nk",
        "qualified_draft_vocab_blocks",
        "qualified_draft_vocab_blocks_sha256",
        "mandatory_weight_bytes",
        "mandatory_weight_floor_ms",
        "one_sided_u95_cap_ms",
    )
    attestation_payload = {
        "schema": module.ATTESTATION_SCHEMA,
        "selector": SELECTOR,
        "source": {"path": str(module.binary.CONTAINER_SOURCE), **identity},
        "destination": {"path": str(module.binary.CONTAINER_DESTINATION), **identity},
        "installed_mode": "0555",
        "production_enabled": True,
        "candidate_family": SELECTOR,
        "qualification": {
            "sidecar_sha256": sidecar_sha256,
            **{key: issued[key] for key in qualification_keys},
        },
    }
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(attestation_payload, sort_keys=True) + "\n", encoding="ascii"
    )

    binding = module.validate_production_attestation(
        attestation,
        sidecar_sha256,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert binding["schema"] == module.IDENTITY_TWOM_DUAL_K64_ROOT_BINDING_SCHEMA
    assert binding["selector"] == SELECTOR
    assert binding["diagnostic_selector"] == DIAGNOSTIC_SELECTOR
    assert binding["live_result_sha256_by_topology"] == hashes


def test_twom_launcher_and_timing_wiring_is_selector_specific_and_exact4() -> None:
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
        "FR13_FIXED32_CUTLASS_TWOM_TAIL23_LIVE_PASS_JSON",
        "FR13_FIXED32_CUTLASS_TWOM_TAIL23_LIVE_PASS_SHA256",
        "FR13_FIXED32_CUTLASS_TWOM_HYDRA27_LIVE_PASS_JSON",
        "FR13_FIXED32_CUTLASS_TWOM_HYDRA27_LIVE_PASS_SHA256",
    ):
        assert name in launcher
    assert 'FR13_FIXED32_CUTLASS_WAVE" == "identity_twom_b4"' in launcher
    assert '--candidate-selector "$FR13_FIXED32_CUTLASS_WAVE"' in launcher
    assert "CUTLASS two-M Tail23/Hydra27 PASS identity mismatch" in launcher
    assert "identity_twom_b4)" in live_gate
    assert "DIAGNOSTIC_SELECTOR=identity_twom_b4_byte_ab" in live_gate
    assert "identity_twom_b4)" in timing
    assert "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4" in timing
    assert "subset_b4_four.json" in timing
    assert "fr13_measure.py deploy-speed" in timing
    assert "fr13_cutlass_b4_pass.py dual-validate" in timing
    assert "fr13_cutlass_b4_pass.py dual-verify" in timing
