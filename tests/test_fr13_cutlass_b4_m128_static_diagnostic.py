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
STATIC_RESOURCE_CREDENTIAL = (
    REPO
    / "results"
    / "fr13_fixed32_cutlass_b4_m128_static_host_build_20260802"
    / "build_manifest.json"
)


def _load():
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "fr13_cutlass_b4_pass.py"
    spec = importlib.util.spec_from_file_location(
        "fr13_cutlass_b4_m128_static_diagnostic_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load()
    candidate_bytes = b"persistent b4 m128 static candidate\n"
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_bytes)
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    monkeypatch.setattr(
        module.binary, "STATIC_B4_M128_CANDIDATE_SIZE", len(candidate_bytes)
    )
    monkeypatch.setattr(
        module.binary, "STATIC_B4_M128_CANDIDATE_SHA256", candidate_sha256
    )

    credential_payload = json.loads(
        STATIC_RESOURCE_CREDENTIAL.read_text(encoding="ascii")
    )
    credential_payload["outputs"]["candidate_binary"]["sha256"] = candidate_sha256
    credential_payload["outputs"]["candidate_binary"]["bytes"] = len(
        candidate_bytes
    )
    credential_raw = (
        json.dumps(credential_payload, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("ascii")
    resource_credential = tmp_path / "static-resource.json"
    resource_credential.write_bytes(credential_raw)
    resource_sha256 = hashlib.sha256(credential_raw).hexdigest()
    monkeypatch.setattr(
        module.binary,
        "STATIC_B4_M128_RESOURCE_CREDENTIAL_SHA256",
        resource_sha256,
    )
    monkeypatch.setattr(
        module.binary,
        "STATIC_B4_M128_RESOURCE_CREDENTIAL_SIZE",
        len(credential_raw),
    )

    patch_bytes = b"static cutlass patch\n"
    patch_source = tmp_path / "patch.py"
    patch_source.write_bytes(patch_bytes)
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    monkeypatch.setattr(module, "STATIC_PATCH_SOURCE_SHA256", patch_sha256)

    identity = module.binary.verify_candidate(
        candidate,
        "persistent_b4_m128_static_byte_ab",
        resource_credential=resource_credential,
        expected_resource_credential_sha256=resource_sha256,
    )
    profile = module.QUALIFICATION_PROFILES["k64_root"]
    task_marker = f"swe_verified:{module.EXPECTED_TASK_IDS[0]}"
    live_payload = {
        "schema": module.STATIC_K64_ROOT_LIVE_SCHEMA,
        "status": "pass",
        "run_classification": profile["run_classification"],
        "acceptance_valid": False,
        "task_count": 4,
        "task_ids": list(module.EXPECTED_TASK_IDS),
        "topology": "hydra27_fixed32",
        "task_marker": task_marker,
        "qualification_profile": "k64_root",
        "draft_vocab_root": 1,
        "draft_vocab_k": 65_536,
        "draft_vocab_blocks": module.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
        "draft_vocab_blocks_sha256": module.DRAFT_VOCAB_BLOCKS_SHA256,
        "mandatory_weight_bytes": module.K64_ROOT_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": module.K64_ROOT_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": module.K64_ROOT_SLO_CAP_MS,
        "comparator_timing_eligible": False,
        "batch_size": 4,
        "concurrency": 4,
        "fixed_rows": 128,
        "eager_builder_capacity": 128,
        "candidate": "persistent_b4_m128_static",
        "diagnostic_selector": "persistent_b4_m128_static_byte_ab",
        "served_result": "stock",
        "production_enabled": False,
        "comparison_call_limit": module.MAX_COMPARISONS,
        "comparisons": module.MAX_COMPARISONS,
        "observed_m_values": [128],
        "observed_projection_nk": [
            list(shape) for shape in module.EXPECTED_PROJECTION_NK
        ],
        "mismatching_comparisons": 0,
        "differing_bytes": 0,
        "candidate_family": "persistent_b4_m128_static",
        "candidate_sha256": candidate_sha256,
        "candidate_bytes": len(candidate_bytes),
        "patch_source_sha256": patch_sha256,
        "vllm_base_commit": module.VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": module.STATIC_PATCHED_DISPATCH_SHA256,
        "source_commit": "c" * 40,
        "binary_attestation_sha256": "d" * 64,
        "real_task_arm_sha256": "e" * 64,
        "container_env_sha256": "f" * 64,
        "errors": [],
        **module._resource_binding(identity),
    }
    live = tmp_path / "live.json"
    live.write_text(
        json.dumps(live_payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    return (
        module,
        candidate,
        patch_source,
        resource_credential,
        resource_sha256,
        live,
        live_sha256,
    )


def test_static_m128_exact4_k64_gate_issues_stock_served_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        module,
        candidate,
        patch_source,
        resource_credential,
        resource_sha256,
        live,
        live_sha256,
    ) = _fixture(tmp_path, monkeypatch)
    sidecar = tmp_path / "sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        expected_source_commit="c" * 40,
        candidate_selector="persistent_b4_m128_static",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
        resource_credential=resource_credential,
        expected_resource_credential_sha256=resource_sha256,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    verified = module.verify_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        candidate_selector="persistent_b4_m128_static",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
        resource_credential=resource_credential,
        expected_resource_credential_sha256=resource_sha256,
    )

    assert issued == verified
    assert issued["schema"] == module.STATIC_K64_ROOT_SIDECAR_SCHEMA
    assert issued["candidate_selector"] == "persistent_b4_m128_static"
    assert issued["diagnostic_selector"] == "persistent_b4_m128_static_byte_ab"
    assert issued["served_result_during_qualification"] == "stock"
    assert issued["production_default_enabled"] is False
    assert issued["qualified_comparison_call_limit"] == 320
    assert issued["resource_credential_sha256"] == resource_sha256
    assert issued["candidate_registers_per_thread"] == 168
    assert issued["candidate_stack_bytes_per_thread"] == 0


def test_static_m128_gate_rejects_resource_binding_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        module,
        candidate,
        patch_source,
        resource_credential,
        resource_sha256,
        live,
        _,
    ) = _fixture(tmp_path, monkeypatch)
    payload = json.loads(live.read_text(encoding="ascii"))
    payload["candidate_registers_per_thread"] = 169
    live.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="ascii",
    )
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()

    with pytest.raises(
        module.QualificationError, match="candidate_registers_per_thread"
    ):
        module.validate_live_result(
            live,
            live_sha256,
            candidate,
            patch_source,
            expected_source_commit="c" * 40,
            candidate_selector="persistent_b4_m128_static",
            qualification_profile="k64_root",
            draft_vocab_blocks=BLOCK_MAP,
            resource_credential=resource_credential,
            expected_resource_credential_sha256=resource_sha256,
        )


def test_static_m128_production_and_timing_remain_unauthorized(
    tmp_path: Path,
) -> None:
    module = _load()
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps({"selector": "persistent_b4_m128_static"}) + "\n",
        encoding="ascii",
    )

    with pytest.raises(module.QualificationError, match="Tail23 and Hydra27"):
        module.validate_production_attestation(attestation, "a" * 64)

    timing = (
        SCRIPTS / "fr13_run_b4_cutlass_persistent_m128_timing.sh"
    ).read_text(encoding="utf-8")
    assert "persistent_b4_m128_static" not in timing


def test_static_m128_selector_is_wired_only_through_exact_b4_diagnostic() -> None:
    sources = {
        name: (SCRIPTS / name).read_text(encoding="utf-8")
        for name in (
            "fr13_run_b4_cutlass_persistent_m128_live_gate.sh",
            "fr13_launch_forked_fa2_tree_server.sh",
            "fr13_bigdenom_swe_serve_variant.sh",
            "run_swe_bench_q36_a.py",
            "fr10_phase4_patch_vllm_tree_gdn.py",
        )
    }

    for source in sources.values():
        assert "persistent_b4_m128_static_byte_ab" in source
    gate = sources["fr13_run_b4_cutlass_persistent_m128_live_gate.sh"]
    launcher = sources["fr13_launch_forked_fa2_tree_server.sh"]
    assert "CUTLASS_B4_CANDIDATE_SELECTOR" in gate
    assert "COMPARISON_CALL_LIMIT=320" in gate
    assert "resource-credential" in gate
    assert "resource-credential" in launcher
    assert "static M128 production remains unavailable" in launcher
