from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
BLOCK_MAP = SCRIPTS / "fr13_dvk_subset_blocks.json"
SELECTOR = "identity_hybrid_n5120_b4"
DIAGNOSTIC_SELECTOR = "identity_hybrid_n5120_b4_byte_ab"
ARTIFACT = (
    REPO
    / "results"
    / "fr13_fixed32_cutlass_b4_hybrid_n5120_production_path_20260803"
)


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

    source_root = tmp_path / "mounted-source"
    source_files: dict[str, dict[str, object]] = {}
    for index, relative in enumerate(module.SOURCE_BINDING_PATHS):
        path = source_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = (
            BLOCK_MAP.read_bytes()
            if relative == str(module.DRAFT_VOCAB_BLOCKS_SOURCE)
            else f"hybrid source file {index}: {relative}\n".encode("ascii")
        )
        path.write_bytes(contents)
        source_files[relative] = {
            "bytes": len(contents),
            "sha256": hashlib.sha256(contents).hexdigest(),
        }
    patch_source = source_root / module.PATCH_SOURCE
    patch_bytes = patch_source.read_bytes()
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    monkeypatch.setattr(
        module, "IDENTITY_HYBRID_N5120_PATCH_SOURCE_SHA256", patch_sha256
    )
    source_identity = {
        "schema": module.SOURCE_BINDING_SCHEMA,
        "source_commit": "c" * 40,
        "files": source_files,
    }
    monkeypatch.setattr(
        module,
        "validate_source_commit_binding",
        lambda source_commit, patch_source, candidate_selector: source_identity,
    )

    live_paths: dict[str, Path] = {}
    live_hashes: dict[str, str] = {}
    for index, mode in enumerate(module.QUALIFIED_FIXED32_MODES):
        topology = module.FIXED32_TOPOLOGY_CONTRACTS[mode]
        payload = {
            "schema": module.IDENTITY_HYBRID_N5120_K64_ROOT_LIVE_SCHEMA,
            "status": "pass",
            "run_classification": (
                "real_swe_verified_exact4_b4_k64_root_byte_diagnostic"
            ),
            "acceptance_valid": False,
            "task_count": 4,
            "task_ids": list(module.EXPECTED_TASK_IDS),
            "authenticated_task_count": 4,
            "authenticated_task_ids": list(module.EXPECTED_TASK_IDS),
            "authenticated_task_set_sha256": module.EXPECTED_TASK_SET_SHA256,
            "engine_ingress_accepted_task_key_ids": list(
                module.EXPECTED_TASK_KEY_IDS
            ),
            "engine_ingress_completed_task_key_ids": list(
                module.EXPECTED_TASK_KEY_IDS
            ),
            "topology": mode,
            "logical_topology": topology["logical_topology"],
            "active_drafts": topology["active_drafts"],
            "valid_mask": topology["valid_mask"],
            "physical_drafts": 31,
            "physical_rows_root_inclusive": 32,
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
            "source_identity": source_identity,
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
    monkeypatch.setattr(
        module,
        "validate_source_commit_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module._GitUnavailableError("git is unavailable for source binding")
        ),
    )
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
    assert issued["qualification_source_identity"]["source_commit"] == "c" * 40
    assert issued["authenticated_task_ids"] == list(module.EXPECTED_TASK_IDS)
    for record in issued["topology_qualifications"].values():
        assert record["engine_ingress_accepted_task_key_ids"] == list(
            module.EXPECTED_TASK_KEY_IDS
        )
        assert record["engine_ingress_completed_task_key_ids"] == list(
            module.EXPECTED_TASK_KEY_IDS
        )


def test_hybrid_single_sidecar_verifies_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, paths, hashes = _fixture(tmp_path, monkeypatch)
    sidecar = tmp_path / "hybrid-single-sidecar.json"
    issued = module.issue_sidecar(
        paths["hydra27_fixed32"],
        hashes["hydra27_fixed32"],
        candidate,
        sidecar,
        patch_source,
        expected_source_commit="c" * 40,
        candidate_selector=SELECTOR,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
        fixed32_mode="hydra27_fixed32",
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    monkeypatch.setattr(
        module,
        "validate_source_commit_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module._GitUnavailableError("git is unavailable for source binding")
        ),
    )

    verified = module.verify_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        candidate_selector=SELECTOR,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
        fixed32_mode="hydra27_fixed32",
    )

    assert verified == issued


@pytest.mark.parametrize(
    ("failure", "match"),
    (
        ("tampered", "SHA-256 mismatch"),
        ("missing", "does not exist"),
        ("symlink", "not a regular non-symlink file"),
        ("wrong_commit", "identity commit mismatch"),
    ),
)
def test_hybrid_no_git_runtime_binding_rejects_mounted_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    match: str,
) -> None:
    module, _, patch_source, _, _ = _fixture(tmp_path, monkeypatch)
    source_identity = module.validate_source_commit_binding(
        "c" * 40, patch_source, SELECTOR
    )
    target = patch_source.parents[1] / module.SOURCE_BINDING_PATHS[-2]
    source_commit = "c" * 40
    if failure == "tampered":
        original = target.read_bytes()
        target.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
    elif failure == "missing":
        target.unlink()
    elif failure == "symlink":
        target.unlink()
        target.symlink_to(patch_source)
    else:
        source_commit = "d" * 40
    monkeypatch.setattr(
        module,
        "validate_source_commit_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module._GitUnavailableError("git is unavailable for source binding")
        ),
    )

    with pytest.raises(module.QualificationError, match=match):
        module._validate_runtime_source_commit_binding(
            source_commit,
            source_identity,
            patch_source,
            SELECTOR,
        )


def test_hybrid_installed_attestation_preserves_exact_head_and_exact4(
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
    destination = tmp_path / "installed.so"
    destination.write_bytes(b"stock\n")
    (tmp_path / "fr13_dvk_subset_blocks.json").write_bytes(BLOCK_MAP.read_bytes())
    attestation = tmp_path / "attestation.json"
    monkeypatch.setitem(sys.modules, "fr13_cutlass_b4_pass", module)
    monkeypatch.setattr(module.binary, "CONTAINER_SOURCE", candidate)
    monkeypatch.setattr(module.binary, "CONTAINER_DESTINATION", destination)
    monkeypatch.setattr(
        module,
        "validate_source_commit_binding",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            module._GitUnavailableError("git is unavailable for source binding")
        ),
    )

    module.binary.install_candidate(
        candidate,
        destination,
        attestation,
        SELECTOR,
        qualification_profile="k64_root",
        production_sidecar=sidecar,
        expected_production_sidecar_sha256=sidecar_sha256,
        patch_source=patch_source,
        fixed32_mode="hydra27_fixed32",
    )
    binding = module.validate_production_attestation(
        attestation,
        sidecar_sha256,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
        fixed32_mode="hydra27_fixed32",
        patch_source=patch_source,
    )

    assert binding["qualification_source_identity"] == issued[
        "qualification_source_identity"
    ]
    assert binding["authenticated_task_set_sha256"] == (
        module.EXPECTED_TASK_SET_SHA256
    )
    assert binding["qualification_topologies"] == list(
        module.QUALIFIED_FIXED32_MODES
    )


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
        "qualification_source_identity": {
            "schema": "source-binding-v1",
            "source_commit": "b" * 40,
            "files": {},
        },
        "authenticated_task_count": 4,
        "authenticated_task_ids": ["task-a", "task-b", "task-c", "task-d"],
        "authenticated_task_set_sha256": "e" * 64,
        "engine_ingress_accepted_task_key_ids": ["f" * 64],
        "engine_ingress_completed_task_key_ids": ["f" * 64],
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
    assert record["qualification"]["qualification_source_identity"] == qualification[
        "qualification_source_identity"
    ]


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


def test_hybrid_live_pass_rejects_incomplete_authenticated_exact4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, paths, _hashes = _fixture(tmp_path, monkeypatch)
    live_path = paths["tail6_fixed32"]
    payload = json.loads(live_path.read_text(encoding="ascii"))
    payload["authenticated_task_ids"] = payload["authenticated_task_ids"][:1]
    live_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")

    with pytest.raises(module.QualificationError, match="authenticated_task_ids"):
        module.validate_live_result(
            live_path,
            hashlib.sha256(live_path.read_bytes()).hexdigest(),
            candidate,
            patch_source,
            expected_source_commit="c" * 40,
            candidate_selector=SELECTOR,
            qualification_profile="k64_root",
            draft_vocab_blocks=BLOCK_MAP,
            fixed32_mode="tail6_fixed32",
        )


def test_hybrid_source_binding_requires_clean_exact_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("fr13_hybrid_n5120_source_binding", "fr13_cutlass_b4_pass.py")
    repo = tmp_path / "repo"
    originals: dict[str, bytes] = {}
    for relative in module.SOURCE_BINDING_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        originals[relative] = f"{relative}\n".encode("ascii")
        path.write_bytes(originals[relative])
    patch_source = repo / module.PATCH_SOURCE
    monkeypatch.setattr(
        module,
        "IDENTITY_HYBRID_N5120_PATCH_SOURCE_SHA256",
        hashlib.sha256(patch_source.read_bytes()).hexdigest(),
    )
    subprocess.run(["git", "init", "-q", repo], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", repo, "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", repo, "add", "."], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", "source"], check=True)
    commit = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"], text=True
    ).strip()

    identity = module.validate_source_commit_binding(commit, patch_source, SELECTOR)
    assert identity["source_commit"] == commit
    assert set(identity["files"]) == set(module.SOURCE_BINDING_PATHS)

    tracked = repo / module.SOURCE_BINDING_PATHS[1]
    tracked.write_bytes(b"dirty\n")
    with pytest.raises(module.QualificationError, match="dirty tracked working tree"):
        module.validate_source_commit_binding(commit, patch_source, SELECTOR)
    tracked.write_bytes(originals[module.SOURCE_BINDING_PATHS[1]])
    subprocess.run(["git", "-C", repo, "commit", "--allow-empty", "-qm", "new head"], check=True)
    with pytest.raises(module.QualificationError, match="runtime commit mismatch"):
        module.validate_source_commit_binding(commit, patch_source, SELECTOR)


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
    assert "source-binding" in live_gate
    assert "engine_ingress_accepted_task_key_ids" in live_gate
    assert "engine_ingress_completed_task_key_ids" in live_gate
    assert 'QUALIFICATION_SOURCE_COMMIT" == "$TIMING_HARNESS_COMMIT' in timing
    assert "work_census_summary = None" in timing
    assert "if all_parent_production:" in timing


def test_hybrid_production_path_artifact_is_sanitized_and_pinned() -> None:
    binary = _load("fr13_hybrid_n5120_artifact_binary", "fr13_cutlass_wave_binary.py")
    qualification = _load(
        "fr13_hybrid_n5120_artifact_pass", "fr13_cutlass_b4_pass.py"
    )
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="ascii"))
    resource = json.loads(
        (ARTIFACT / "resource_gate.json").read_text(encoding="ascii")
    )

    assert manifest["status"] == "host_ready_live_qualification_required"
    assert manifest["acceptance_valid"] is False
    assert manifest["production_available"] is False
    assert manifest["production_default_enabled"] is False
    assert manifest["candidate"]["binary_sha256"] == (
        binary.IDENTITY_HYBRID_N5120_B4_CANDIDATE_SHA256
    )
    assert manifest["candidate"]["binary_bytes"] == (
        binary.IDENTITY_HYBRID_N5120_B4_CANDIDATE_SIZE
    )
    assert manifest["candidate"]["patch_source_sha256"] == (
        qualification.IDENTITY_HYBRID_N5120_PATCH_SOURCE_SHA256
    )
    assert manifest["candidate"]["patched_dispatch_sha256"] == (
        qualification.IDENTITY_HYBRID_N5120_PATCHED_DISPATCH_SHA256
    )
    assert manifest["live_gate_contract"]["task_ids"] == list(
        qualification.EXPECTED_TASK_IDS
    )
    assert manifest["live_gate_contract"]["topologies"] == list(
        qualification.QUALIFIED_FIXED32_MODES
    )
    assert len(resource["kernel_records"]) == 4
    assert all(record["stack_bytes_per_thread"] == 0 for record in resource["kernel_records"])
    assert all(record["local_bytes_per_thread"] == 0 for record in resource["kernel_records"])

    sums = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    for name in ("README.md", "manifest.json", "resource_gate.json"):
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == sums[name]
