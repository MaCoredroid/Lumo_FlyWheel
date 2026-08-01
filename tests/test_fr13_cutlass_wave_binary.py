from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


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
    }


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
