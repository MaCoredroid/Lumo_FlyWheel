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


def _load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "fr13_cutlass_streamk_pass.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _k64_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load("fr13_cutlass_streamk_k64_profile_test")
    candidate_bytes = b"wide256 cap320 candidate\n"
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_bytes)
    patch_bytes = b"cap320 patch source\n"
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    patch_source = tmp_path / "patch.py"
    patch_source.write_bytes(patch_bytes)
    monkeypatch.setattr(module.binary, "WIDE256_CANDIDATE_SIZE", len(candidate_bytes))
    monkeypatch.setattr(module.binary, "WIDE256_CANDIDATE_SHA256", candidate_sha256)
    monkeypatch.setattr(module, "PATCH_SOURCE_SHA256", patch_sha256)
    profile = module.QUALIFICATION_PROFILES["k64_root"]
    live = {
        "schema": module.K64_ROOT_LIVE_SCHEMA,
        "status": "pass",
        "run_classification": profile["run_classification"],
        "acceptance_valid": False,
        "task_count": 1,
        "task_ids": list(module.EXPECTED_TASK_IDS),
        "task_marker": module.EXPECTED_TASK_MARKER,
        "qualification_profile": "k64_root",
        "draft_vocab_root": 1,
        "draft_vocab_k": 65_536,
        "draft_vocab_blocks": module.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
        "draft_vocab_blocks_sha256": module.DRAFT_VOCAB_BLOCKS_SHA256,
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "comparator_timing_eligible": False,
        "batch_size": 1,
        "concurrency": 1,
        "fixed_rows": 32,
        "candidate": "streamk_force_wide256",
        "diagnostic_selector": "streamk_force_wide256_byte_ab",
        "served_result": "stock",
        "production_enabled": False,
        "comparison_call_limit": module.MAX_COMPARISONS,
        "comparisons": 257,
        "observed_m_values": [32],
        "observed_projection_nk": [
            list(shape) for shape in module.EXPECTED_PROJECTION_NK
        ],
        "mismatching_comparisons": 0,
        "differing_bytes": 0,
        "candidate_family": "streamk_force_wide256",
        "candidate_sha256": candidate_sha256,
        "candidate_bytes": len(candidate_bytes),
        "patch_source_sha256": patch_sha256,
        "vllm_base_commit": module.VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": module.PATCHED_DISPATCH_SHA256,
        "source_commit": "c" * 40,
        "binary_attestation_sha256": "d" * 64,
        "real_task_arm_sha256": "e" * 64,
        "container_env_sha256": "f" * 64,
        "errors": [],
    }
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(live, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live_path.read_bytes()).hexdigest()
    return module, candidate, patch_source, live_path, live_sha256


def test_k64_root_live_pass_issues_and_verifies_distinct_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _k64_fixture(
        tmp_path, monkeypatch
    )
    sidecar = tmp_path / "sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector="streamk_force_wide256",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    verified = module.verify_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        candidate_selector="streamk_force_wide256",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert verified == issued
    assert issued["schema"] == module.K64_ROOT_SIDECAR_SCHEMA
    assert issued["qualification_profile"] == "k64_root"
    assert issued["qualified_draft_vocab_root"] == 1
    assert issued["qualified_draft_vocab_k"] == 65_536
    assert issued["qualified_comparison_call_limit"] == 320
    assert (
        issued["qualified_draft_vocab_blocks_sha256"]
        == module.DRAFT_VOCAB_BLOCKS_SHA256
    )


def test_k64_root_accepts_static_persistent_stocktile_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, _ = _k64_fixture(tmp_path, monkeypatch)
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    monkeypatch.setattr(
        module.binary,
        "STATIC_PERSISTENT_B1_CANDIDATE_SIZE",
        len(candidate.read_bytes()),
    )
    monkeypatch.setattr(
        module.binary,
        "STATIC_PERSISTENT_B1_CANDIDATE_SHA256",
        candidate_sha256,
    )
    payload = json.loads(live.read_text(encoding="ascii"))
    payload.update(
        {
            "schema": module.STATIC_PERSISTENT_K64_ROOT_LIVE_SCHEMA,
            "candidate": "static_persistent_stocktile",
            "candidate_family": "static_persistent_stocktile",
            "diagnostic_selector": "static_persistent_stocktile_byte_ab",
        }
    )
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    sidecar = tmp_path / "static-sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector="static_persistent_stocktile",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert issued["candidate_selector"] == "static_persistent_stocktile"
    assert issued["diagnostic_selector"] == "static_persistent_stocktile_byte_ab"
    assert issued["qualification_profile"] == "k64_root"
    assert issued["qualified_comparison_call_limit"] == 320


def test_k64_root_accepts_divisor_static_stocktile_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, _ = _k64_fixture(tmp_path, monkeypatch)
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    monkeypatch.setattr(
        module.binary,
        "DIVISOR_STATIC_B1_CANDIDATE_SIZE",
        len(candidate.read_bytes()),
    )
    monkeypatch.setattr(
        module.binary,
        "DIVISOR_STATIC_B1_CANDIDATE_SHA256",
        candidate_sha256,
    )
    payload = json.loads(live.read_text(encoding="ascii"))
    payload.update(
        {
            "schema": module.DIVISOR_STATIC_K64_ROOT_LIVE_SCHEMA,
            "candidate": "divisor_static_stocktile",
            "candidate_family": "divisor_static_stocktile",
            "diagnostic_selector": "divisor_static_stocktile_byte_ab",
        }
    )
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    sidecar = tmp_path / "divisor-static-sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector="divisor_static_stocktile",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert issued["candidate_selector"] == "divisor_static_stocktile"
    assert issued["diagnostic_selector"] == "divisor_static_stocktile_byte_ab"
    assert issued["qualification_profile"] == "k64_root"
    assert issued["qualified_comparison_call_limit"] == 320


def test_k64_root_rejects_non_wide_candidate_and_block_map_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _k64_fixture(
        tmp_path, monkeypatch
    )

    with pytest.raises(module.QualificationError, match="restricted to"):
        module.validate_live_result(
            live,
            live_sha256,
            candidate,
            patch_source,
            candidate_selector="streamk_coop128",
            qualification_profile="k64_root",
        )

    drifted = tmp_path / "blocks.json"
    drifted.write_text("{}\n", encoding="ascii")
    with pytest.raises(module.QualificationError, match="block-map SHA-256 mismatch"):
        module.validate_live_result(
            live,
            live_sha256,
            candidate,
            patch_source,
            candidate_selector="streamk_force_wide256",
            qualification_profile="k64_root",
            draft_vocab_blocks=drifted,
        )


def test_k64_root_attestation_preserves_profile_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _k64_fixture(
        tmp_path, monkeypatch
    )
    sidecar = tmp_path / "sidecar.json"
    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector="streamk_force_wide256",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    qualification = dict(issued)
    qualification["sidecar_sha256"] = sidecar_sha256
    identity = {
        "path": str(module.binary.CONTAINER_SOURCE),
        "bytes": len(candidate.read_bytes()),
        "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "regular": True,
        "symlink": False,
    }
    destination = dict(identity)
    destination["path"] = str(module.binary.CONTAINER_DESTINATION)
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema": module.ATTESTATION_SCHEMA,
                "selector": "streamk_force_wide256",
                "source": identity,
                "destination": destination,
                "installed_mode": "0555",
                "production_enabled": True,
                "qualification": qualification,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )

    result = module.validate_production_attestation(
        attestation,
        sidecar_sha256,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert result["schema"].endswith("k64_root.production_binding.v1")
    assert result["qualification_profile"] == "k64_root"
    assert result["qualified_comparison_call_limit"] == 320
