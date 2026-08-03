from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
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
        "divisor_static_stocktile",
        "divisor_static_stocktile_byte_ab",
        "identity_stage2_static",
        "identity_stage2_static_byte_ab",
        "identity_stage2_pingpong_b1",
        "identity_stage2_pingpong_b1_byte_ab",
        "identity_onen_b1",
        "identity_onen_b1_byte_ab",
        "identity_onen_n5120_single_b1",
        "identity_onen_n5120_single_b1_byte_ab",
        "identity_onen_n5120_fullgrid_b1",
        "identity_onen_n5120_fullgrid_b1_byte_ab",
        "identity_stockshape_b4",
        "identity_stockshape_b4_byte_ab",
        "identity_stockshape_stage2_b4",
        "identity_stockshape_stage2_b4_byte_ab",
        "identity_twom_b4",
        "identity_twom_b4_byte_ab",
        "identity_hybrid_n5120_b4",
        "identity_hybrid_n5120_b4_byte_ab",
        "mtp_m1m4_direct_byte_ab",
        "identity_divisor_b4",
        "identity_divisor_b4_byte_ab",
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
    assert module.candidate_identity("divisor_static_stocktile_byte_ab") == (
        module.DIVISOR_STATIC_B1_CANDIDATE_SHA256,
        module.DIVISOR_STATIC_B1_CANDIDATE_SIZE,
        "divisor_static_stocktile",
    )
    assert module.candidate_identity("identity_stage2_static_byte_ab") == (
        module.IDENTITY_STAGE2_CANDIDATE_SHA256,
        module.IDENTITY_STAGE2_CANDIDATE_SIZE,
        "identity_stage2_static",
    )
    assert module.candidate_identity("identity_stage2_pingpong_b1_byte_ab") == (
        module.IDENTITY_STAGE2_PINGPONG_B1_CANDIDATE_SHA256,
        module.IDENTITY_STAGE2_PINGPONG_B1_CANDIDATE_SIZE,
        "identity_stage2_pingpong_b1",
    )
    assert module.candidate_identity("identity_onen_b1_byte_ab") == (
        module.IDENTITY_ONEN_B1_CANDIDATE_SHA256,
        module.IDENTITY_ONEN_B1_CANDIDATE_SIZE,
        "identity_onen_b1",
    )
    assert module.IDENTITY_ONEN_B1_CANDIDATE_SHA256 == (
        "17af1975b1e26cd3d4c3e614bfcab8aa1b0dc031ea5107004b0cc25890fc2b15"
    )
    assert module.IDENTITY_ONEN_B1_CANDIDATE_SIZE == 118_166_088
    assert "identity_onen_b1" in module.PRODUCTION_SELECTORS
    assert "identity_onen_b1_byte_ab" not in module.PRODUCTION_SELECTORS
    assert module.candidate_identity(
        "identity_onen_n5120_single_b1_byte_ab"
    ) == (
        module.IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SHA256,
        module.IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SIZE,
        "identity_onen_n5120_single_b1",
    )
    assert module.IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SHA256 == (
        "876a3d6a0c972926131b1e447ffba80e345979f2d6de3bfa7bf083e862469367"
    )
    assert module.IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SIZE == 118_468_696
    assert "identity_onen_n5120_single_b1" in module.PRODUCTION_SELECTORS
    assert (
        "identity_onen_n5120_single_b1_byte_ab"
        not in module.PRODUCTION_SELECTORS
    )
    assert module.candidate_identity(
        "identity_onen_n5120_fullgrid_b1_byte_ab"
    ) == (
        module.IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SHA256,
        module.IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SIZE,
        "identity_onen_n5120_fullgrid_b1",
    )
    assert module.IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SHA256 == (
        "65250ccb46057e4726f68b6056eab3e46f71a1bee2ce25eca306d4d889a66ecc"
    )
    assert module.IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SIZE == 119_471_552
    assert "identity_onen_n5120_fullgrid_b1" in module.PRODUCTION_SELECTORS
    assert (
        "identity_onen_n5120_fullgrid_b1_byte_ab"
        not in module.PRODUCTION_SELECTORS
    )
    assert module.IDENTITY_B4_CANDIDATE_SHA256 == (
        "d7771d5a95a34d6072a796d520e8f2fa500aeccc900d57e1477941b966ea77a9"
    )
    assert module.IDENTITY_B4_CANDIDATE_SIZE == 116_284_480
    assert module.candidate_identity("identity_stockshape_b4_byte_ab") == (
        module.IDENTITY_B4_CANDIDATE_SHA256,
        module.IDENTITY_B4_CANDIDATE_SIZE,
        "identity_stockshape_b4",
    )
    assert module.candidate_identity("identity_divisor_b4_byte_ab") == (
        module.IDENTITY_B4_CANDIDATE_SHA256,
        module.IDENTITY_B4_CANDIDATE_SIZE,
        "identity_divisor_b4",
    )
    assert module.candidate_identity("identity_stockshape_stage2_b4_byte_ab") == (
        module.IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SHA256,
        module.IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SIZE,
        "identity_stockshape_stage2_b4",
    )
    assert module.IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SHA256 == (
        "c5da32258e678494cd2b6b34da0b2aa96e70096b215db0938ed1e0750aa43d29"
    )
    assert module.IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SIZE == 117_488_608
    assert module.candidate_identity("identity_twom_b4_byte_ab") == (
        module.IDENTITY_TWOM_B4_CANDIDATE_SHA256,
        module.IDENTITY_TWOM_B4_CANDIDATE_SIZE,
        "identity_twom_b4",
    )
    assert module.IDENTITY_TWOM_B4_CANDIDATE_SHA256 == (
        "c5da32258e678494cd2b6b34da0b2aa96e70096b215db0938ed1e0750aa43d29"
    )
    assert module.IDENTITY_TWOM_B4_CANDIDATE_SIZE == 117_488_608
    assert module.candidate_identity("identity_hybrid_n5120_b4_byte_ab") == (
        module.IDENTITY_HYBRID_N5120_B4_CANDIDATE_SHA256,
        module.IDENTITY_HYBRID_N5120_B4_CANDIDATE_SIZE,
        "identity_hybrid_n5120_b4",
    )
    assert module.IDENTITY_HYBRID_N5120_B4_CANDIDATE_SHA256 == (
        "65250ccb46057e4726f68b6056eab3e46f71a1bee2ce25eca306d4d889a66ecc"
    )
    assert module.IDENTITY_HYBRID_N5120_B4_CANDIDATE_SIZE == 119_471_552
    assert "identity_hybrid_n5120_b4" in module.PRODUCTION_SELECTORS
    assert "identity_hybrid_n5120_b4_byte_ab" not in module.PRODUCTION_SELECTORS
    assert module.candidate_identity("mtp_m1m4_direct_byte_ab") == (
        module.MTP_M1M4_DIRECT_CANDIDATE_SHA256,
        module.MTP_M1M4_DIRECT_CANDIDATE_SIZE,
        "mtp_m1m4_direct",
    )
    assert module.MTP_M1M4_DIRECT_CANDIDATE_SHA256 == (
        "65250ccb46057e4726f68b6056eab3e46f71a1bee2ce25eca306d4d889a66ecc"
    )
    assert module.MTP_M1M4_DIRECT_CANDIDATE_SIZE == 119_471_552
    assert "mtp_m1m4_direct_byte_ab" not in module.PRODUCTION_SELECTORS
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
    monkeypatch.setattr(module, "STATIC_B4_M128_CANDIDATE_SHA256", candidate_sha256)
    monkeypatch.setattr(module, "STATIC_B4_M128_CANDIDATE_SIZE", candidate_size)
    monkeypatch.setattr(
        module,
        "STATIC_B4_M128_RESOURCE_CREDENTIAL_SHA256",
        credential_sha256,
    )
    monkeypatch.setattr(module, "STATIC_B4_M128_RESOURCE_CREDENTIAL_SIZE", len(raw))
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
    assert (
        record["destination"]["resource_credential"]
        == record["source"]["resource_credential"]
    )


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


def test_onen_b1_diagnostic_installs_but_direct_requires_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    payload = b"onen-b1-candidate-extension\n"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module, "IDENTITY_ONEN_B1_CANDIDATE_SIZE", len(payload))
    monkeypatch.setattr(module, "IDENTITY_ONEN_B1_CANDIDATE_SHA256", digest)
    source = tmp_path / "onen-b1.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    record = module.install_candidate(
        source,
        destination,
        attestation,
        "identity_onen_b1_byte_ab",
        qualification_profile="k64_root",
    )

    assert destination.read_bytes() == payload
    assert record["production_enabled"] is False
    assert record["candidate_family"] == "identity_onen_b1"
    assert record["qualification_profile"] == "k64_root"
    assert record["source"]["qualification_profile"] == "k64_root"
    assert record["destination"]["qualification_profile"] == "k64_root"

    destination.chmod(0o644)
    destination.write_bytes(b"stock-extension\n")
    with pytest.raises(ValueError, match="requires a pinned production sidecar"):
        module.install_candidate(
            source,
            destination,
            attestation,
            "identity_onen_b1",
            qualification_profile="k64_root",
        )
    assert destination.read_bytes() == b"stock-extension\n"


def test_onen_n5120_single_diagnostic_installs_but_direct_requires_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    payload = b"onen-n5120-single-candidate-extension\n"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        module, "IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SIZE", len(payload)
    )
    monkeypatch.setattr(
        module, "IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SHA256", digest
    )
    source = tmp_path / "onen-n5120-single.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    record = module.install_candidate(
        source,
        destination,
        attestation,
        "identity_onen_n5120_single_b1_byte_ab",
        qualification_profile="k64_root",
    )

    assert destination.read_bytes() == payload
    assert record["production_enabled"] is False
    assert record["candidate_family"] == "identity_onen_n5120_single_b1"
    assert record["qualification_profile"] == "k64_root"

    destination.chmod(0o644)
    destination.write_bytes(b"stock-extension\n")
    with pytest.raises(ValueError, match="requires a pinned production sidecar"):
        module.install_candidate(
            source,
            destination,
            attestation,
            "identity_onen_n5120_single_b1",
            qualification_profile="k64_root",
        )
    assert destination.read_bytes() == b"stock-extension\n"


@pytest.mark.parametrize(
    ("selector", "qualification_profile"),
    (
        ("identity_onen_b1", None),
        ("identity_onen_b1", "full_vocab"),
        ("identity_onen_b1_byte_ab", None),
        ("identity_onen_b1_byte_ab", "full_vocab"),
        ("identity_onen_n5120_single_b1", None),
        ("identity_onen_n5120_single_b1", "full_vocab"),
        ("identity_onen_n5120_single_b1_byte_ab", None),
        ("identity_onen_n5120_single_b1_byte_ab", "full_vocab"),
        ("identity_onen_n5120_fullgrid_b1", None),
        ("identity_onen_n5120_fullgrid_b1", "full_vocab"),
        ("identity_onen_n5120_fullgrid_b1_byte_ab", None),
        ("identity_onen_n5120_fullgrid_b1_byte_ab", "full_vocab"),
    ),
)
def test_onen_b1_binary_verification_rejects_non_k64_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    qualification_profile: str | None,
) -> None:
    module = _module()
    payload = b"onen-b1-candidate-extension\n"
    monkeypatch.setattr(module, "IDENTITY_ONEN_B1_CANDIDATE_SIZE", len(payload))
    monkeypatch.setattr(
        module,
        "IDENTITY_ONEN_B1_CANDIDATE_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(payload)

    with pytest.raises(
        ValueError,
        match="binary verification requires a k64_root qualification",
    ):
        module.verify_candidate(
            candidate,
            selector,
            qualification_profile=qualification_profile,
        )


def test_onen_b1_production_uses_k64_streamk_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    observed: dict[str, object] = {}

    def verify_sidecar(*args, **kwargs):
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {"status": "QUALIFIED", "qualification_profile": "k64_root"}

    qualification = types.SimpleNamespace(verify_sidecar=verify_sidecar)
    monkeypatch.setitem(sys.modules, "fr13_cutlass_streamk_pass", qualification)
    sidecar = tmp_path / "sidecar.json"
    candidate = tmp_path / "candidate.so"
    patch_source = tmp_path / "patch.py"

    result = module._verify_production_qualification(
        sidecar,
        "a" * 64,
        candidate,
        patch_source,
        "identity_onen_b1",
        "hydra27_fixed32",
    )

    assert result == {
        "status": "QUALIFIED",
        "qualification_profile": "k64_root",
    }
    assert observed["args"] == (
        sidecar,
        "a" * 64,
        candidate,
        patch_source,
    )
    assert observed["kwargs"] == {
        "candidate_selector": "identity_onen_b1",
        "qualification_profile": "k64_root",
        "diagnostic_task_profile": "astropy12907",
        "draft_vocab_blocks": tmp_path / "fr13_dvk_subset_blocks.json",
    }


def test_onen_b1_production_rejects_verifier_profile_downgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    qualification = types.SimpleNamespace(
        verify_sidecar=lambda *_args, **_kwargs: {
            "status": "QUALIFIED",
            "qualification_profile": "full_vocab",
        }
    )
    monkeypatch.setitem(sys.modules, "fr13_cutlass_streamk_pass", qualification)

    with pytest.raises(ValueError, match="requires a k64_root qualification"):
        module._verify_production_qualification(
            tmp_path / "sidecar.json",
            "a" * 64,
            tmp_path / "candidate.so",
            tmp_path / "patch.py",
            "identity_onen_b1",
            "hydra27_fixed32",
        )


def test_onen_b1_direct_install_binds_k64_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    payload = b"onen-b1-production-candidate\n"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module, "IDENTITY_ONEN_B1_CANDIDATE_SIZE", len(payload))
    monkeypatch.setattr(module, "IDENTITY_ONEN_B1_CANDIDATE_SHA256", digest)
    qualification = {
        "live_result_sha256": "a" * 64,
        "candidate_sha256": digest,
        "patch_source_sha256": "b" * 64,
        "qualification_source_commit": "c" * 40,
        "qualification_task_marker": "swe_verified:astropy__astropy-12907",
        "real_task_arm_sha256": "d" * 64,
        "container_env_sha256": "e" * 64,
        "qualified_draft_vocab_root": 1,
        "qualified_draft_vocab_k": 65_536,
        "mandatory_weight_bytes": 32_666_638_208,
        "mandatory_weight_floor_ms": 119.658015414,
        "one_sided_u95_cap_ms": 137.6067177261,
        "qualification_profile": "k64_root",
        "qualified_draft_vocab_blocks": (
            "/workspace/scripts/fr13_dvk_subset_blocks.json"
        ),
        "qualified_draft_vocab_blocks_sha256": "f" * 64,
        "qualified_comparison_call_limit": 320,
        "qualified_fixed_rows": 32,
        "qualified_projection_nk": [[34816, 5120]],
    }
    monkeypatch.setattr(
        module,
        "_verify_production_qualification",
        lambda *_args: qualification,
    )
    source = tmp_path / "candidate.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    record = module.install_candidate(
        source,
        destination,
        attestation,
        "identity_onen_b1",
        qualification_profile="k64_root",
        production_sidecar=tmp_path / "sidecar.json",
        expected_production_sidecar_sha256="1" * 64,
    )

    assert record["production_enabled"] is True
    assert record["qualification_profile"] == "k64_root"
    assert record["source"]["qualification_profile"] == "k64_root"
    assert record["destination"]["qualification_profile"] == "k64_root"
    assert record["qualification"]["qualification_profile"] == "k64_root"


@pytest.mark.parametrize(
    ("size_attr", "sha_attr", "diagnostic_selector", "production_selector", "family"),
    (
        (
            "STATIC_PERSISTENT_B1_CANDIDATE_SIZE",
            "STATIC_PERSISTENT_B1_CANDIDATE_SHA256",
            "static_persistent_stocktile_byte_ab",
            "static_persistent_stocktile",
            "static_persistent_stocktile",
        ),
        (
            "DIVISOR_STATIC_B1_CANDIDATE_SIZE",
            "DIVISOR_STATIC_B1_CANDIDATE_SHA256",
            "divisor_static_stocktile_byte_ab",
            "divisor_static_stocktile",
            "divisor_static_stocktile",
        ),
        (
            "IDENTITY_STAGE2_CANDIDATE_SIZE",
            "IDENTITY_STAGE2_CANDIDATE_SHA256",
            "identity_stage2_static_byte_ab",
            "identity_stage2_static",
            "identity_stage2_static",
        ),
        (
            "IDENTITY_STAGE2_PINGPONG_B1_CANDIDATE_SIZE",
            "IDENTITY_STAGE2_PINGPONG_B1_CANDIDATE_SHA256",
            "identity_stage2_pingpong_b1_byte_ab",
            "identity_stage2_pingpong_b1",
            "identity_stage2_pingpong_b1",
        ),
    ),
)
def test_static_b1_diagnostic_installs_but_production_stays_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    size_attr: str,
    sha_attr: str,
    diagnostic_selector: str,
    production_selector: str,
    family: str,
) -> None:
    module = _module()
    payload = f"{family}-candidate\n".encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(module, size_attr, len(payload))
    monkeypatch.setattr(module, sha_attr, digest)
    source = tmp_path / "static-b1.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    record = module.install_candidate(
        source,
        destination,
        attestation,
        diagnostic_selector,
    )

    assert destination.read_bytes() == payload
    assert record["production_enabled"] is False
    assert record["candidate_family"] == family

    destination.chmod(0o644)
    destination.write_bytes(b"stock-extension\n")
    with pytest.raises(ValueError, match="K64/root raw-byte gate"):
        module.install_candidate(
            source,
            destination,
            attestation,
            production_selector,
        )
    assert destination.read_bytes() == b"stock-extension\n"


@pytest.mark.parametrize(
    ("diagnostic_selector", "production_selector", "family"),
    (
        (
            "identity_stockshape_b4_byte_ab",
            "identity_stockshape_b4",
            "identity_stockshape_b4",
        ),
        (
            "identity_divisor_b4_byte_ab",
            "identity_divisor_b4",
            "identity_divisor_b4",
        ),
        (
            "identity_stockshape_stage2_b4_byte_ab",
            "identity_stockshape_stage2_b4",
            "identity_stockshape_stage2_b4",
        ),
        (
            "identity_twom_b4_byte_ab",
            "identity_twom_b4",
            "identity_twom_b4",
        ),
    ),
)
def test_identity_b4_diagnostic_installs_but_production_stays_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic_selector: str,
    production_selector: str,
    family: str,
) -> None:
    module = _module()
    payload = f"{family}-candidate\n".encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    if family == "identity_stockshape_stage2_b4":
        monkeypatch.setattr(
            module, "IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SIZE", len(payload)
        )
        monkeypatch.setattr(
            module, "IDENTITY_STOCKSHAPE_STAGE2_B4_CANDIDATE_SHA256", digest
        )
    elif family == "identity_twom_b4":
        monkeypatch.setattr(module, "IDENTITY_TWOM_B4_CANDIDATE_SIZE", len(payload))
        monkeypatch.setattr(module, "IDENTITY_TWOM_B4_CANDIDATE_SHA256", digest)
    else:
        monkeypatch.setattr(module, "IDENTITY_B4_CANDIDATE_SIZE", len(payload))
        monkeypatch.setattr(module, "IDENTITY_B4_CANDIDATE_SHA256", digest)
    source = tmp_path / "identity-b4.so"
    destination = tmp_path / "installed.so"
    attestation = tmp_path / "attestation.json"
    source.write_bytes(payload)
    destination.write_bytes(b"stock-extension\n")

    record = module.install_candidate(
        source,
        destination,
        attestation,
        diagnostic_selector,
    )

    assert destination.read_bytes() == payload
    assert record["production_enabled"] is False
    assert record["candidate_family"] == family

    destination.chmod(0o644)
    destination.write_bytes(b"stock-extension\n")
    expected_error = (
        "requires a pinned production sidecar"
        if family in {"identity_stockshape_stage2_b4", "identity_twom_b4"}
        else "Tail23 and Hydra27 raw-byte gates"
    )
    with pytest.raises(ValueError, match=expected_error):
        module.install_candidate(
            source,
            destination,
            attestation,
            production_selector,
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
