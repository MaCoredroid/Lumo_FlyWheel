from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
STATIC_RESOURCE_CREDENTIAL = (
    REPO
    / "results"
    / "fr13_fixed32_cutlass_b4_m128_static_host_build_20260802"
    / "build_manifest.json"
)


def _module():
    path = Path("scripts/fr13_cutlass_wave_binary.py")
    spec = importlib.util.spec_from_file_location("fr13_cutlass_wave_binary", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_binary_identity_and_selectors() -> None:
    module = _module()

    assert module.CANDIDATE_SHA256 == (
        "f9bbbb8dc4ffc2227a71d2bc7b260e586ffbdc0fd946749e4f69e322c46a362d"
    )
    assert module.CANDIDATE_SIZE == 111_417_328
    assert module.CANDIDATE_SELECTORS == {
        "streamk_coop128",
        "streamk_coop128_byte_ab",
        "streamk_force_wide256",
        "streamk_force_wide256_byte_ab",
        "static_persistent_stocktile",
        "static_persistent_stocktile_byte_ab",
        "m32_static_linear",
        "m32_static_linear_byte_ab",
        "persistent_b4_m128",
        "persistent_b4_m128_byte_ab",
        "persistent_b4_m128_static",
        "persistent_b4_m128_static_byte_ab",
    }
    assert module.WIDE256_CANDIDATE_SHA256 == (
        "503277a2dca6784502b709007adfe45f42d0f1a1851107e7b913e1e85a00de5a"
    )
    assert module.WIDE256_CANDIDATE_SIZE == 113_079_680
    assert module.candidate_identity("streamk_force_wide256") == (
        module.WIDE256_CANDIDATE_SHA256,
        module.WIDE256_CANDIDATE_SIZE,
        "streamk_force_wide256",
    )
    assert module.candidate_identity("static_persistent_stocktile_byte_ab") == (
        module.STATIC_PERSISTENT_B1_CANDIDATE_SHA256,
        module.STATIC_PERSISTENT_B1_CANDIDATE_SIZE,
        "static_persistent_stocktile",
    )
    assert module.candidate_identity("m32_static_linear_byte_ab") == (
        module.M32_STATIC_LINEAR_CANDIDATE_SHA256,
        module.M32_STATIC_LINEAR_CANDIDATE_SIZE,
        "m32_static_linear",
    )
    assert module.M32_STATIC_LINEAR_CANDIDATE_SHA256 == (
        "079d82d60426411bf403eb96f4869cb8d3872a4a68d49e9c336a55a90d571f91"
    )
    assert module.M32_STATIC_LINEAR_CANDIDATE_SIZE == 113_809_232
    assert module.B4_M128_CANDIDATE_SHA256 == (
        "895495fe82cb0e0278d3b0a39b8e57e1281aa73a10bbba01a94085733c81d64f"
    )
    assert module.B4_M128_CANDIDATE_SIZE == 112_698_512
    assert module.candidate_identity("persistent_b4_m128_byte_ab") == (
        module.B4_M128_CANDIDATE_SHA256,
        module.B4_M128_CANDIDATE_SIZE,
        "persistent_b4_m128",
    )
    assert module.candidate_identity("persistent_b4_m128_static_byte_ab") == (
        module.STATIC_B4_M128_CANDIDATE_SHA256,
        module.STATIC_B4_M128_CANDIDATE_SIZE,
        "persistent_b4_m128_static",
    )


def _static_resource_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module,
    candidate_sha256: str,
    candidate_size: int,
) -> tuple[Path, str]:
    payload = json.loads(STATIC_RESOURCE_CREDENTIAL.read_text(encoding="ascii"))
    payload["outputs"]["candidate_binary"]["sha256"] = candidate_sha256
    payload["outputs"]["candidate_binary"]["bytes"] = candidate_size
    raw = (json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n").encode(
        "ascii"
    )
    credential = tmp_path / "static-resource.json"
    credential.write_bytes(raw)
    credential_sha256 = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(
        module, "STATIC_B4_M128_CANDIDATE_SHA256", candidate_sha256
    )
    monkeypatch.setattr(module, "STATIC_B4_M128_CANDIDATE_SIZE", candidate_size)
    monkeypatch.setattr(
        module,
        "STATIC_B4_M128_RESOURCE_CREDENTIAL_SHA256",
        credential_sha256,
    )
    monkeypatch.setattr(
        module, "STATIC_B4_M128_RESOURCE_CREDENTIAL_SIZE", len(raw)
    )
    return credential, credential_sha256


def test_static_resource_credential_is_exactly_pinned() -> None:
    module = _module()

    binding = module.verify_static_m128_resource_credential(
        STATIC_RESOURCE_CREDENTIAL,
        module.STATIC_B4_M128_RESOURCE_CREDENTIAL_SHA256,
    )

    assert binding["sha256"] == module.STATIC_B4_M128_RESOURCE_CREDENTIAL_SHA256
    assert binding["candidate_sha256"] == module.STATIC_B4_M128_CANDIDATE_SHA256
    assert binding["candidate_bytes"] == module.STATIC_B4_M128_CANDIDATE_SIZE
    assert binding["resource_records"] == 309
    assert binding["registers_per_thread"] == 168
    assert binding["stack_bytes_per_thread"] == 0
    assert binding["local_bytes_per_thread"] == 0


def test_static_m128_diagnostic_install_binds_resource_and_stays_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    payload = b"static-m128-candidate\n"
    digest = hashlib.sha256(payload).hexdigest()
    credential, credential_sha256 = _static_resource_fixture(
        tmp_path, monkeypatch, module, digest, len(payload)
    )
    source = tmp_path / "static-m128.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    record = module.install_candidate(
        source,
        destination,
        attestation,
        "persistent_b4_m128_static_byte_ab",
        resource_credential=credential,
        expected_resource_credential_sha256=credential_sha256,
    )

    assert destination.read_bytes() == payload
    assert record["production_enabled"] is False
    assert record["candidate_family"] == "persistent_b4_m128_static"
    assert record["source"]["resource_credential"]["sha256"] == credential_sha256
    assert record["destination"]["resource_credential"] == record["source"][
        "resource_credential"
    ]


def test_static_m128_install_fails_closed_without_gate_or_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    payload = b"static-m128-candidate\n"
    digest = hashlib.sha256(payload).hexdigest()
    credential, credential_sha256 = _static_resource_fixture(
        tmp_path, monkeypatch, module, digest, len(payload)
    )
    source = tmp_path / "static-m128.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    with pytest.raises(ValueError, match="pinned resource credential"):
        module.install_candidate(
            source,
            destination,
            attestation,
            "persistent_b4_m128_static_byte_ab",
        )
    with pytest.raises(ValueError, match="Tail23 and Hydra27"):
        module.install_candidate(
            source,
            destination,
            attestation,
            "persistent_b4_m128_static",
            resource_credential=credential,
            expected_resource_credential_sha256=credential_sha256,
        )
    assert destination.read_bytes() == b"stock-extension\n"


def test_wide256_diagnostic_install_uses_its_own_pinned_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    payload = b"wide256-candidate-extension\n"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module, "WIDE256_CANDIDATE_SIZE", len(payload))
    monkeypatch.setattr(module, "WIDE256_CANDIDATE_SHA256", digest)
    source = tmp_path / "wide256.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    record = module.install_candidate(
        source,
        destination,
        attestation,
        "streamk_force_wide256_byte_ab",
    )

    assert destination.read_bytes() == payload
    assert record["production_enabled"] is False
    assert record["candidate_family"] == "streamk_force_wide256"
    assert record["source"]["sha256"] == digest


def test_static_b1_diagnostic_installs_but_production_stays_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    payload = b"static-persistent-b1-candidate\n"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        module, "STATIC_PERSISTENT_B1_CANDIDATE_SIZE", len(payload)
    )
    monkeypatch.setattr(module, "STATIC_PERSISTENT_B1_CANDIDATE_SHA256", digest)
    source = tmp_path / "static-b1.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    record = module.install_candidate(
        source,
        destination,
        attestation,
        "static_persistent_stocktile_byte_ab",
    )

    assert destination.read_bytes() == payload
    assert record["production_enabled"] is False
    assert record["candidate_family"] == "static_persistent_stocktile"

    destination.chmod(0o644)
    destination.write_bytes(b"stock-extension\n")
    with pytest.raises(ValueError, match="K64/root raw-byte gate"):
        module.install_candidate(
            source,
            destination,
            attestation,
            "static_persistent_stocktile",
        )
    assert destination.read_bytes() == b"stock-extension\n"


def test_install_is_exact_attested_and_production_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    payload = b"candidate-extension\n"
    monkeypatch.setattr(module, "CANDIDATE_SIZE", len(payload))
    monkeypatch.setattr(module, "CANDIDATE_SHA256", hashlib.sha256(payload).hexdigest())
    source = tmp_path / "candidate.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    record = module.install_candidate(
        source, destination, attestation, "streamk_coop128_byte_ab"
    )

    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o555
    assert attestation.stat().st_mode & 0o777 == 0o444
    assert record["production_enabled"] is False
    assert record["schema"] == "fr13.fixed32.cutlass_streamk_binary.v2"
    assert json.loads(attestation.read_text(encoding="ascii")) == record


def test_direct_selector_requires_and_binds_production_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    payload = b"candidate-extension\n"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module, "CANDIDATE_SIZE", len(payload))
    monkeypatch.setattr(module, "CANDIDATE_SHA256", digest)
    source = tmp_path / "candidate.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    with pytest.raises(ValueError, match="requires a pinned production sidecar"):
        module.install_candidate(source, destination, attestation, "streamk_coop128")
    assert destination.read_bytes() == b"stock-extension\n"

    qualification = {
        "live_result_sha256": "a" * 64,
        "candidate_sha256": digest,
        "patch_source_sha256": "b" * 64,
        "qualification_source_commit": "c" * 40,
        "qualification_task_marker": "swe_verified:astropy__astropy-12907",
        "real_task_arm_sha256": "e" * 64,
        "container_env_sha256": "f" * 64,
        "qualified_draft_vocab_root": 0,
        "qualified_draft_vocab_k": 0,
        "mandatory_weight_bytes": 42_025_179_008,
        "mandatory_weight_floor_ms": 153.9383846446886,
        "one_sided_u95_cap_ms": 177.0291423413919,
    }
    monkeypatch.setattr(
        module,
        "_verify_production_qualification",
        lambda *args: qualification,
    )
    record = module.install_candidate(
        source,
        destination,
        attestation,
        "streamk_coop128",
        production_sidecar=tmp_path / "pass.json",
        expected_production_sidecar_sha256="d" * 64,
    )

    assert record["production_enabled"] is True
    assert record["qualification"] == {
        "sidecar_sha256": "d" * 64,
        **qualification,
    }


def test_verify_rejects_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    target = tmp_path / "candidate.so"
    target.write_bytes(b"x")
    monkeypatch.setattr(module, "CANDIDATE_SIZE", 1)
    monkeypatch.setattr(module, "CANDIDATE_SHA256", hashlib.sha256(b"x").hexdigest())
    alias = tmp_path / "alias.so"
    alias.symlink_to(target)

    with pytest.raises(ValueError, match="non-symlink"):
        module.verify_candidate(alias)
